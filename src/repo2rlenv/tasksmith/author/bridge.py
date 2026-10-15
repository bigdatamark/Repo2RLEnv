# [MODULE DOC] Loopback HTTP bridge between an external (Node) author runtime and the Anthropic API.
# It exposes an aiohttp server on 127.0.0.1 (random port) with three routes: POST /v1/messages
# (proxied, metered model calls), GET /v1/models (single-model listing) and POST /tool (dispatch to
# Python tool handlers). The runtime authenticates with a per-bridge random token; the real
# ANTHROPIC_API_KEY is only read here. Every model call reserves a cost ceiling against an
# AuthorBudget before forwarding and settles actual usage cost afterwards; incomplete streams keep
# their reservation. All requests/responses are appended to a JSONL trace file.
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

from repo2rlenv.campaigns.budget import BudgetExceeded
from repo2rlenv.tasksmith.author.budget import AuthorBudget as Budget
from repo2rlenv.tasksmith.author.inference import (
    MAX_OUTPUT_TOKENS,
    MODEL_TIMEOUT_SEC,
    anthropic_options,
    inference_settings,
)


class ProviderOutputError(RuntimeError):
    # [CLASS DOC] RuntimeError subclass raised when a completed, already-metered provider response
    # is unusable (truncated or contains no text/tool call); callers may safely retry.
    """Completed, metered response was unusable; a bounded new attempt is safe."""


class AgentBridge:
    # [CLASS DOC] Async context manager that runs the loopback bridge server for one author run.
    # Enforces: Anthropic-only models, a max number of model turns, a per-run USD cap (relative to
    # the budget's spend at construction), a tool allowlist, and fixed inference settings. Tracks
    # in-flight tool and model request tasks so it can cancel/drain them on stop. The first
    # unexpected error is stored in `failure` and signalled through the `failed` asyncio.Event.
    """Only model inference and explicitly registered remote tools cross this API.

    Runtime processes receive a short-lived loopback token, never provider keys.
    Requests are metered before forwarding; incomplete streams retain reservations.
    """

    def __init__(self, *, model, budget: Budget, tools, handlers, trace: Path, max_turns, max_cost):
        # [FUNCTION DOC] Stores configuration and initialises state. Inputs (keyword-only): model
        # (str, must start with "anthropic/"), budget (AuthorBudget used for reserve/settle and
        # `spent`), tools (list of OpenAI-style tool schemas with function.name), handlers (dict
        # name -> async callable taking keyword arguments, returning str), trace (Path of the JSONL
        # trace), max_turns (int model-call cap), max_cost (USD cap for this run). Creates a random
        # url-safe auth token, an Event `failed`, locks for tools and models, and empty task sets.
        # Raises ValueError for non-Anthropic models. No I/O.
        if not model.startswith("anthropic/"):
            raise ValueError("The matched runtime experiment currently supports Anthropic models")
        self.model, self.budget = model, budget
        self.tools, self.handlers, self.trace = tools, handlers, trace
        self.max_turns, self.max_cost = max_turns, max_cost
        self.turns, self.cost = 0, 0.0
        self.start_spend = budget.spent
        self.token = secrets.token_urlsafe(32)
        self.failure: BaseException | None = None
        self.failed = asyncio.Event()
        self.tool_lock = asyncio.Lock()
        self.model_lock = asyncio.Lock()
        self._closing = False
        self._tool_tasks: set[asyncio.Task] = set()
        self._model_tasks: set[asyncio.Task] = set()

    def record(self, kind, **data):
        # [FUNCTION DOC] Appends one JSON line {"kind": kind, **data} to the trace file (non-JSON
        # values stringified). Side effect: file append. Returns None.
        with self.trace.open("a") as f:
            f.write(json.dumps({"kind": kind, **data}, default=str) + "\n")

    async def __aenter__(self):
        # [FUNCTION DOC] Starts the bridge: creates the trace's parent directory, an aiohttp
        # ClientSession with MODEL_TIMEOUT_SEC total timeout, an aiohttp Application (8 MiB body
        # limit) with the three routes, and binds a TCPSite on 127.0.0.1 port 0. Sets self.http,
        # self.runner and self.url ("http://127.0.0.1:<port>"). Returns self.
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        self.http = ClientSession(timeout=ClientTimeout(total=MODEL_TIMEOUT_SEC))
        app = web.Application(client_max_size=8 * 1024 * 1024)
        app.router.add_post("/v1/messages", self.messages)
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/tool", self.tool)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = self.runner.addresses[0][1]
        self.url = f"http://127.0.0.1:{port}"
        return self

    def stop(self, *, cancel_models=False):
        # [FUNCTION DOC] Marks the bridge closing so new requests get 503, and cancels in-flight
        # tool tasks (and model tasks too when cancel_models=True, default False), never the current
        # task. Returns None.
        """Reject new work and cancel tool effects before terminating the runtime."""
        self._closing = True
        tasks = self._tool_tasks | (self._model_tasks if cancel_models else set())
        current = asyncio.current_task()
        for task in tasks:
            if task is not current:
                task.cancel()

    async def __aexit__(self, exc_type, exc, tb):
        # [FUNCTION DOC] Stops the bridge (cancelling model tasks only if exiting due to an
        # exception), waits for all remaining tool/model tasks (exceptions swallowed) so late stream
        # settlements are recorded, then cleans up the web runner and closes the HTTP client.
        # Returns None (does not suppress exceptions).
        self.stop(cancel_models=exc_type is not None)
        try:
            # A runtime can exit as soon as it sees message_stop, before the
            # upstream stream reaches EOF and its final charge is recorded.
            await asyncio.gather(*self._tool_tasks, *self._model_tasks, return_exceptions=True)
        finally:
            try:
                await self.runner.cleanup()
            finally:
                await self.http.close()

    def ensure_open(self):
        # [FUNCTION DOC] HELPER: Raises aiohttp HTTPServiceUnavailable (503) if stop() has been
        # called; otherwise returns None.
        if self._closing:
            raise web.HTTPServiceUnavailable(text="Runtime bridge is stopping")

    def authenticate(self, request):
        # [FUNCTION DOC] HELPER: Reads the token from the x-api-key header or from an
        # "Authorization: Bearer" header and compares it to self.token in constant time. Raises
        # HTTPUnauthorized (401) on mismatch.
        token = request.headers.get("x-api-key") or request.headers.get(
            "Authorization", ""
        ).removeprefix("Bearer ")
        if not secrets.compare_digest(token, self.token):
            raise web.HTTPUnauthorized()

    def fail(self, exc):
        # [FUNCTION DOC] Records the first fatal error: sets self.failure, sets the `failed` event
        # and writes a "runtime_error" trace record (only once). Always then calls
        # stop(cancel_models=True).
        if self.failure is None:
            self.failure = exc
            self.failed.set()
            self.record("runtime_error", error=f"{type(exc).__name__}: {exc}")
        self.stop(cancel_models=True)

    async def models(self, request):
        # [FUNCTION DOC] GET /v1/models handler. Authenticates and returns JSON listing exactly the
        # configured model id (provider prefix stripped) with has_more False.
        self.authenticate(request)
        return web.json_response(
            {"data": [{"id": self.model.split("/", 1)[1], "type": "model"}], "has_more": False}
        )

    async def tool(self, request):
        # [FUNCTION DOC] POST /tool handler. Authenticates, rejects if closing, registers itself as
        # a tool task. Body JSON must have "name" (a registered handler) and dict "arguments", else
        # 400. Before executing it waits (shielded) for all model requests that were in flight when
        # the call arrived, so their cost is settled first; then, under the tool lock, runs the
        # handler. In the artifact stage (a "submit_artifact" handler exists), "shell" calls are
        # refused with an explanatory message when two or fewer model turns remain.
        # ValueError/TypeError/KeyError from the handler become a "Tool input error" output; other
        # exceptions call fail() and return HTTP 409 with {"error"}. Outputs longer than 24000 chars
        # are truncated to first 8000 + last 16000 chars. In the artifact stage, shell outputs get a
        # remaining-budget note appended. Records a "tool" trace entry and returns {"output": str}.
        # Always unregisters the task.
        self.authenticate(request)
        self.ensure_open()
        preceding_models = tuple(self._model_tasks)
        task = asyncio.current_task()
        self._tool_tasks.add(task)
        try:
            data = await request.json()
            name, arguments = data.get("name"), data.get("arguments", {})
            if name not in self.handlers or not isinstance(arguments, dict):
                raise web.HTTPBadRequest(text="Unknown tool or malformed arguments")
            artifact_stage = "submit_artifact" in self.handlers
            remaining = self.max_turns - self.turns
            try:
                # Native runtimes can dispatch tools at message_stop before
                # provider EOF. Match LangGraph: meter and validate that model
                # response before allowing any effect. Cancelling a waiting
                # tool must not cancel a model that still needs settlement.
                await asyncio.gather(
                    *(asyncio.shield(model) for model in preceding_models),
                    return_exceptions=True,
                )
                async with self.tool_lock:
                    self.ensure_open()
                    remaining = self.max_turns - self.turns
                    if artifact_stage and name == "shell" and remaining <= 2:
                        output = (
                            "Command not executed: the final two model calls are reserved for "
                            "artifact submission and schema correction. Submit the best supported "
                            "complete artifact using the evidence already collected."
                        )
                    else:
                        output = await self.handlers[name](**arguments)
            except web.HTTPException:
                raise
            except (ValueError, TypeError, KeyError) as exc:
                output = f"Tool input error: {exc}"
            except Exception as exc:
                self.fail(exc)
                return web.json_response({"error": str(exc)}, status=409)
            if len(output) > 24000:
                output = output[:8000] + "\n[truncated]\n" + output[-16000:]
            if artifact_stage and name == "shell":
                output += (
                    f"\nController budget: {max(0, remaining)} model calls remain. "
                    "Reserve the final two for submit_artifact and any needed revise_artifact."
                )
            self.record("tool", name=name, arguments=arguments, output=output)
            return web.json_response({"output": output})
        finally:
            self._tool_tasks.discard(task)

    async def messages(self, request):
        # [FUNCTION DOC] POST /v1/messages handler. Authenticates, rejects if closing, tracks the
        # request as a model task and delegates to _messages. On cancellation calls fail() and re-
        # raises; HTTP exceptions propagate; any other exception calls fail() and returns an
        # Anthropic-style error JSON with status 429 for BudgetExceeded, otherwise 502. Always
        # unregisters the task.
        self.authenticate(request)
        self.ensure_open()
        task = asyncio.current_task()
        self._model_tasks.add(task)
        try:
            return await self._messages(request)
        except asyncio.CancelledError:
            self.fail(RuntimeError("Provider request cancelled before completion"))
            raise
        except web.HTTPException:
            raise
        except Exception as exc:
            self.fail(exc)
            return web.json_response(
                {"type": "error", "error": {"type": "api_error", "message": str(exc)}},
                status=429 if isinstance(exc, BudgetExceeded) else 502,
            )
        finally:
            self._model_tasks.discard(task)

    async def _messages(self, request):
        # [FUNCTION DOC] HELPER: Validates, meters and forwards one Anthropic Messages request.
        # Checks the body's model equals the configured model and that all requested tools are in
        # the allowlist (ValueError otherwise). Forces max_tokens=MAX_OUTPUT_TOKENS, strips thinking
        # / temperature / top_p / top_k / output_config and applies anthropic_options; normalises
        # cache_control to ephemeral. Looks up per-token prices via litellm.get_model_info
        # (ValueError if missing). Ceiling = (body bytes + 4096) * input_rate * 1.25 + max_tokens *
        # output_rate. Under the model lock: raises BudgetExceeded if the turn cap is reached or run
        # spend + ceiling > max_cost, otherwise reserves the ceiling and increments turns. Logs a
        # "model_request" trace record, then POSTs to https://api.anthropic.com/v1/messages with the
        # real API key (env ANTHROPIC_API_KEY, KeyError if absent) and forwards any anthropic-beta
        # header. Non-200: settles 0 for 4xx (5xx keeps the reservation) and raises RuntimeError.
        # Non-streaming: computes usage_cost, settles, adds to self.cost, records, validates content
        # and returns the JSON. Streaming: relays raw chunks to the client as text/event-stream
        # while parsing "data:" SSE lines, reconstructs usage, content blocks and stop_reason; if no
        # message_stop event or usage incomplete, records with cost None and raises (reservation
        # retained). Otherwise settles, records, validates and writes EOF. Raises
        # ProviderOutputError from validation after settlement.
        import litellm

        body = await request.json()
        model = self.model.split("/", 1)[1]
        if body.get("model") != model:
            raise ValueError("Runtime requested a different model than the matched experiment")
        allowed = {t["function"]["name"] for t in self.tools}
        if any(t.get("name") not in allowed for t in body.get("tools", [])):
            raise ValueError("Runtime exposed tools outside the matched allowlist")
        body["max_tokens"] = MAX_OUTPUT_TOKENS
        # Runtime defaults cannot change the matched inference policy. Preserve
        # signed/opaque thinking blocks in the conversation exactly as supplied.
        for key in ("thinking", "temperature", "top_p", "top_k", "output_config"):
            body.pop(key, None)
        body.update(anthropic_options(self.model))
        normalize_cache(body)
        pricing = litellm.get_model_info(self.model)
        input_rate, output_rate = pricing["input_cost_per_token"], pricing["output_cost_per_token"]
        if not input_rate or not output_rate:
            raise ValueError("Unknown model price")
        ceiling = (len(json.dumps(body).encode()) + 4096) * input_rate * 1.25 + body[
            "max_tokens"
        ] * output_rate
        async with self.model_lock:
            self.ensure_open()
            if self.turns >= self.max_turns:
                raise BudgetExceeded(f"Runtime model-call limit reached: {self.max_turns}")
            spent = self.budget.spent - self.start_spend
            if spent + ceiling > self.max_cost:
                raise BudgetExceeded(
                    f"Runtime agent budget ${self.max_cost}: ${spent:.4f} committed; "
                    f"need ${ceiling:.4f}"
                )
            reservation = self.budget.reserve(ceiling, f"llm:{self.model}")
            self.turns += 1
            turn = self.turns
        self.record(
            "model_request",
            turn=turn,
            model=model,
            system=body.get("system"),
            messages=body.get("messages", [])[-1:],
            tools=[t.get("name") for t in body.get("tools", [])],
            tool_choice=body.get("tool_choice"),
            inference=inference_settings(self.model),
            timeout_sec=MODEL_TIMEOUT_SEC,
        )
        headers = {
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if request.headers.get("anthropic-beta"):
            headers["anthropic-beta"] = request.headers["anthropic-beta"]
        async with self.http.post(
            "https://api.anthropic.com/v1/messages", json=body, headers=headers
        ) as upstream:
            if upstream.status != 200:
                error = await upstream.text()
                if 400 <= upstream.status < 500:
                    self.budget.settle(reservation, 0)
                raise RuntimeError(f"Anthropic HTTP {upstream.status}: {error[:1500]}")
            if not body.get("stream"):
                result = await upstream.json()
                usage = result.get("usage", {})
                try:
                    cost = usage_cost(usage, input_rate, output_rate)
                except ValueError:
                    self.record(
                        "model",
                        turn=turn,
                        message=result,
                        usage=usage,
                        cost_usd=None,
                        reservation_usd=ceiling,
                    )
                    raise
                self.budget.settle(reservation, cost)
                self.cost += cost
                self.record("model", turn=turn, message=result, usage=usage, cost_usd=cost)
                validate_provider_response(result.get("content", []), result.get("stop_reason"))
                return web.json_response(result)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            events, pending = [], b""
            async for chunk in upstream.content.iter_any():
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if line.startswith(b"data: "):
                        with contextlib.suppress(ValueError):
                            events.append(json.loads(line[6:]))
                with contextlib.suppress(ConnectionError):
                    await response.write(chunk)
            usage = {}
            content = []
            stop_reason = None
            for event in events:
                if event.get("type") == "message_start":
                    message = event.get("message", {})
                    usage.update(message.get("usage", {}))
                    content.extend(message.get("content", []))
                    stop_reason = message.get("stop_reason") or stop_reason
                elif event.get("type") == "message_delta":
                    usage.update(event.get("usage", {}))
                    stop_reason = event.get("delta", {}).get("stop_reason") or stop_reason
                elif event.get("type") == "content_block_start":
                    content.append(event.get("content_block", {}))
                elif event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        content.append({"type": "text", "text": delta.get("text", "")})
            try:
                if not any(e.get("type") == "message_stop" for e in events):
                    raise RuntimeError("Incomplete provider stream; reservation retained")
                cost = usage_cost(usage, input_rate, output_rate)
            except (ValueError, RuntimeError):
                self.record(
                    "model",
                    turn=turn,
                    events=events,
                    usage=usage,
                    cost_usd=None,
                    reservation_usd=ceiling,
                )
                raise
            self.budget.settle(reservation, cost)
            self.cost += cost
            self.record("model", turn=turn, events=events, usage=usage, cost_usd=cost)
            validate_provider_response(content, stop_reason)
            with contextlib.suppress(ConnectionError):
                await response.write_eof()
            return response


def validate_provider_response(content, stop_reason):
    # [FUNCTION DOC] Raises ProviderOutputError if stop_reason is "max_tokens" or "length", or if no
    # content block is either non-blank text or a tool_use. Inputs: content (list of dict blocks),
    # stop_reason (str or None). Returns None.
    """A metered model turn needs usable text or a tool call, not just thinking."""
    if stop_reason in {"max_tokens", "length"}:
        raise ProviderOutputError(f"Provider response truncated: stop_reason={stop_reason}")
    if not any(
        (block.get("type") == "text" and (block.get("text") or "").strip())
        or block.get("type") == "tool_use"
        for block in content
    ):
        raise ProviderOutputError(
            "Provider response contained no text or tool calls (thinking-only or empty)"
        )


def normalize_cache(value):
    # [FUNCTION DOC] Recursively walks dicts/lists in place, replacing every dict-valued
    # "cache_control" with {"type": "ephemeral"} (5-minute cache, dropping e.g. ttl). Mutates input;
    # returns None.
    """Use the same five-minute cache pricing ceiling as the LangGraph baseline."""
    if isinstance(value, dict):
        if isinstance(value.get("cache_control"), dict):
            value["cache_control"] = {"type": "ephemeral"}
        for child in value.values():
            normalize_cache(child)
    elif isinstance(value, list):
        for child in value:
            normalize_cache(child)


def usage_cost(usage, input_rate, output_rate):
    # [FUNCTION DOC] Computes USD cost of an Anthropic usage dict. Requires input_tokens and
    # output_tokens (ValueError otherwise). Cache writes: 1h tokens at 2x input rate, 5m tokens at
    # 1.25x (from cache_creation.ephemeral_5m_input_tokens, defaulting to
    # cache_creation_input_tokens minus 1h tokens); cache reads at 0.1x. Returns (input + weighted
    # cache tokens) * input_rate + output_tokens * output_rate as float.
    if "input_tokens" not in usage or "output_tokens" not in usage:
        raise ValueError("Provider returned incomplete usage; retain reservation")
    cache = usage.get("cache_creation", {})
    long = cache.get("ephemeral_1h_input_tokens", 0)
    short = cache.get(
        "ephemeral_5m_input_tokens", max(0, usage.get("cache_creation_input_tokens", 0) - long)
    )
    return (
        usage["input_tokens"]
        + short * 1.25
        + long * 2
        + usage.get("cache_read_input_tokens", 0) * 0.1
    ) * input_rate + usage["output_tokens"] * output_rate
