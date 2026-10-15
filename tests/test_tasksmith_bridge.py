# [MODULE DOC] Tests for the TaskSmith author AgentBridge (tasksmith.author.bridge), the local HTTP proxy that
# external agent runtimes call for model turns (/v1/messages) and tools (/tool). Covers usage cost
# and cache normalization, auth and model matching, turn and cost admission (including in-flight
# reservations and global/candidate scopes), tool sequencing and cancellation, runtime inference-
# setting overrides, failure on empty/truncated turns after metering, and gating tool execution on
# model EOF/validation. Uses a deterministic in-process LocalProvider instead of the Anthropic
# API; skipped without aiohttp. One test runs the real opencode runtime when available.
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")

from aiohttp import ClientSession, web

from repo2rlenv.campaigns.budget import BudgetExceeded, BudgetLedger
from repo2rlenv.quality.loop.client import RunBudget
from repo2rlenv.tasksmith.author import bridge as bridge_module
from repo2rlenv.tasksmith.author import external_agent
from repo2rlenv.tasksmith.author.bridge import AgentBridge, normalize_cache, usage_cost
from repo2rlenv.tasksmith.author.budget import AuthorBudget
from repo2rlenv.tasksmith.author.inference import (
    MAX_OUTPUT_TOKENS,
    MODEL_TIMEOUT_SEC,
    anthropic_options,
)


def Budget(path, limit, scope=None, scope_limit=None):
    # [FUNCTION DOC] HELPER: Builds an AuthorBudget for tests: a BudgetLedger at path with the given USD limit,
    # wrapped in a RunBudget scope 'bridge' whose limit is scope_limit when scope is truthy else
    # limit, with charges stored under path.parent/charges and label 'test'. scope only acts as a
    # flag; its value is not used as a name. Output: AuthorBudget. Side effects: creates the
    # ledger file.
    ledger = BudgetLedger(path, limit_usd=str(limit))
    budget = RunBudget(ledger, "bridge", str(scope_limit if scope else limit))
    return AuthorBudget(budget, path.parent / "charges", "test")


def test_provider_usage_accounts_cache_without_double_charging():
    # [FUNCTION DOC] Asserts usage_cost charges input, output, cache creation (1h tier weighted higher) and cache
    # reads at the expected total 265 for prices 1/2, raises ValueError 'incomplete usage' without
    # input_tokens, and normalize_cache strips ttl from cache_control.
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 30,
        "cache_read_input_tokens": 200,
        "cache_creation": {"ephemeral_1h_input_tokens": 10},
    }
    assert usage_cost(usage, 1, 2) == pytest.approx(265)
    with pytest.raises(ValueError, match="incomplete usage"):
        usage_cost({"output_tokens": 50}, 1, 2)
    data = {"system": [{"text": "a", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}
    normalize_cache(data)
    assert "ttl" not in data["system"][0]["cache_control"]


@pytest.mark.asyncio
async def test_bridge_denies_unauthenticated_tools_and_unmatched_models(tmp_path):
    # [FUNCTION DOC] Over real HTTP: a /tool call without a bearer token gets 401, an authenticated one runs the
    # handler, and a /v1/messages call naming a different model gets 502 and sets bridge.failed.
    # Asserts only one effect and zero spend.
    effects = []

    async def remote(command):
        # [FUNCTION DOC] HELPER: fake shell handler recording the command and returning 'remote response'.
        effects.append(command)
        return "remote response"

    budget = Budget(tmp_path / "budget.json", 1)
    async with (
        AgentBridge(
            model="anthropic/claude-sonnet-5",
            budget=budget,
            tools=[{"function": {"name": "shell"}}],
            handlers={"shell": remote},
            trace=tmp_path / "trace.jsonl",
            max_turns=2,
            max_cost=1,
        ) as bridge,
        ClientSession() as client,
    ):
        async with client.post(bridge.url + "/tool", json={"name": "shell"}) as r:
            assert r.status == 401
        headers = {"Authorization": "Bearer " + bridge.token}
        async with client.post(
            bridge.url + "/tool",
            headers=headers,
            json={"name": "shell", "arguments": {"command": "pwd"}},
        ) as r:
            assert (await r.json())["output"] == "remote response"
        async with client.post(
            bridge.url + "/v1/messages", headers=headers, json={"model": "different-model"}
        ) as r:
            assert r.status == 502
        assert bridge.failed.is_set()
    assert effects == ["pwd"]
    assert budget.spent == 0


class BridgeRequest:
    # [CLASS DOC] Minimal stand-in for an aiohttp request: holds a JSON body, an x-api-key header from token
    # and an optional Event set when the body is parsed.
    def __init__(self, body, *, token="", parsed=None):
        # [FUNCTION DOC] HELPER: stores body and parsed event and sets headers {'x-api-key': token}. Inputs: body
        # dict; keyword token (default ''), parsed (asyncio.Event or None).
        self.body, self.parsed = body, parsed
        self.headers = {"x-api-key": token}

    async def json(self):
        # Both concurrent requests finish the admission preamble before either
        # JSON body is available, reproducing the original admission race.
        # [FUNCTION DOC] HELPER: yields to the event loop once (to reproduce concurrent admission races), sets
        # the parsed event if given, and returns a shallow copy of the body.
        await asyncio.sleep(0)
        if self.parsed is not None:
            self.parsed.set()
        return self.body.copy()


@pytest.mark.asyncio
async def test_artifact_stage_reserves_final_calls_for_submission(tmp_path):
    # [FUNCTION DOC] Calls bridge.tool directly with varying bridge.turns under max_turns 5: early shell calls
    # run and report remaining model calls, near the limit shell is refused ('Command not
    # executed'), while submit_artifact still runs. Asserts recorded effects.
    effects = []

    async def shell(command):
        # [FUNCTION DOC] HELPER: fake shell handler recording the command and returning file contents.
        effects.append(command)
        return "file contents"

    async def submit(**artifact):
        # [FUNCTION DOC] HELPER: fake submit_artifact handler recording kwargs and returning 'Artifact
        # committed'.
        effects.append(artifact)
        return "Artifact committed"

    bridge = AgentBridge(
        model="anthropic/claude-sonnet-4-6",
        budget=Budget(tmp_path / "budget.json", 1),
        tools=[],
        handlers={"shell": shell, "submit_artifact": submit},
        trace=tmp_path / "trace.jsonl",
        max_turns=5,
        max_cost=1,
    )
    request = BridgeRequest(
        {"name": "shell", "arguments": {"command": "read metadata"}}, token=bridge.token
    )
    bridge.turns = 1
    response = json.loads((await bridge.tool(request)).text)
    assert "4 model calls remain" in response["output"]
    bridge.turns = 3
    response = json.loads((await bridge.tool(request)).text)
    assert "Command not executed" in response["output"]
    bridge.turns = 4
    response = await bridge.tool(
        BridgeRequest(
            {"name": "submit_artifact", "arguments": {"profile": "complete"}}, token=bridge.token
        )
    )
    assert json.loads(response.text)["output"] == "Artifact committed"
    assert effects == ["read metadata", {"profile": "complete"}]


class LocalProvider:
    # [CLASS DOC] Deterministic fake for the aiohttp ClientSession and response used by the bridge: records
    # posted request bodies, returns configurable usage/blocks/stop_reason as JSON or SSE, and can
    # hold the stream open until `finish` is set. No network.
    """Deterministic provider responses; no upstream HTTP requests are made."""

    status = 200

    def __init__(self):
        # [FUNCTION DOC] HELPER: initializes requests list, called/finish events (finish set), default usage 100
        # in/50 out, one text block, stop_reason end_turn, and content=self so
        # response.content.iter_any works.
        self.requests = []
        self.called = asyncio.Event()
        self.finish = asyncio.Event()
        self.finish.set()
        self.usage = {"input_tokens": 100, "output_tokens": 50}
        self.blocks = [{"type": "text", "text": "Completed response"}]
        self.stop_reason = "end_turn"
        self.content = self

    def post(self, url, *, json, headers):
        # [FUNCTION DOC] HELPER: asserts the Anthropic messages URL and dummy API key header, records the JSON
        # body, sets called, and returns self as the response context manager.
        assert url == "https://api.anthropic.com/v1/messages"
        assert headers["x-api-key"] == "dummy-provider-key"
        self.requests.append(json)
        self.called.set()
        return self

    async def __aenter__(self):
        # [FUNCTION DOC] HELPER: returns self.
        return self

    async def __aexit__(self, *args):
        # [FUNCTION DOC] HELPER: no-op.
        pass

    async def close(self):
        # [FUNCTION DOC] HELPER: no-op session close.
        pass

    async def json(self):
        # [FUNCTION DOC] HELPER: waits for finish, then returns a non-streaming message dict with usage, content
        # blocks and stop_reason.
        await self.finish.wait()
        return {"usage": self.usage, "content": self.blocks, "stop_reason": self.stop_reason}

    async def iter_any(self):
        # [FUNCTION DOC] HELPER: async generator yielding one bytes chunk of SSE events (message_start, a
        # content_block_start per block, message_delta with stop_reason, message_stop), then waits
        # for finish before ending.
        events = [
            {"type": "message_start", "message": {"usage": self.usage}},
            *[
                {"type": "content_block_start", "index": i, "content_block": block}
                for i, block in enumerate(self.blocks)
            ],
            {
                "type": "message_delta",
                "usage": {"output_tokens": 50},
                "delta": {"stop_reason": self.stop_reason},
            },
            {"type": "message_stop"},
        ]
        yield "".join("data: " + json.dumps(event) + "\n\n" for event in events).encode()
        await self.finish.wait()


@pytest.fixture
def local_provider(monkeypatch):
    # [FUNCTION DOC] Pytest fixture installing a LocalProvider as bridge_module.ClientSession, setting
    # ANTHROPIC_API_KEY to a dummy, and patching litellm.get_model_info prices (1e-5 in, 2e-5 out
    # per token). Output: the provider.
    import litellm

    provider = LocalProvider()
    monkeypatch.setattr(bridge_module, "ClientSession", lambda **kwargs: provider)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-provider-key")
    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda model: {"input_cost_per_token": 0.00001, "output_cost_per_token": 0.00002},
    )
    return provider


def make_bridge(tmp_path, **overrides):
    # [FUNCTION DOC] HELPER: Creates an AgentBridge with defaults (model anthropic/claude-mock, 10 USD Budget, no
    # tools/handlers, trace file, max_turns 2, max_cost 1) overridden by keyword overrides. Not
    # entered; caller manages lifecycle.
    config = {
        "model": "anthropic/claude-mock",
        "budget": Budget(tmp_path / "budget.json", 10),
        "tools": [],
        "handlers": {},
        "trace": tmp_path / "trace.jsonl",
        "max_turns": 2,
        "max_cost": 1,
    }
    return AgentBridge(**(config | overrides))


def model_request():
    # [FUNCTION DOC] HELPER: Returns a BridgeRequest for a claude-mock message with empty messages and max_tokens
    # 100 (no token).
    return BridgeRequest({"model": "claude-mock", "messages": [], "max_tokens": 100})


@pytest.mark.asyncio
async def test_model_admission_enforces_turn_limit_after_reading_concurrent_bodies(
    tmp_path, local_provider
):
    # [FUNCTION DOC] With max_turns 1, two concurrent _messages calls yield exactly one BudgetExceeded and one
    # provider request.
    async with make_bridge(tmp_path, max_turns=1) as bridge:
        replies = await asyncio.gather(
            bridge._messages(model_request()),
            bridge._messages(model_request()),
            return_exceptions=True,
        )
    assert sum(isinstance(reply, BudgetExceeded) for reply in replies) == 1
    assert len(local_provider.requests) == bridge.turns == 1


@pytest.mark.asyncio
async def test_agent_cost_limit_includes_the_next_reservation(tmp_path, local_provider):
    # [FUNCTION DOC] With max_cost 0.01 the next reservation does not fit; expects BudgetExceeded 'Runtime agent
    # budget', no spend, no turns, no provider request.
    async with make_bridge(tmp_path, max_cost=0.01) as bridge:
        with pytest.raises(BudgetExceeded, match="Runtime agent budget"):
            await bridge._messages(model_request())
    assert bridge.budget.spent == 0
    assert bridge.turns == 0
    assert local_provider.requests == []


@pytest.mark.asyncio
async def test_agent_cost_limit_counts_an_inflight_model_reservation(tmp_path, local_provider):
    # [FUNCTION DOC] Holds the first model call in flight; a second call raises BudgetExceeded because both
    # reservations exceed max_cost 0.40. Asserts one provider request and final spend 0.002.
    local_provider.finish.clear()
    # With the fixture's prices, one request fits but two reservations do not.
    async with make_bridge(tmp_path, max_cost=0.40) as bridge:
        first = asyncio.create_task(bridge._messages(model_request()))
        try:
            await asyncio.wait_for(local_provider.called.wait(), 2)
            with pytest.raises(BudgetExceeded, match="Runtime agent budget"):
                await bridge._messages(model_request())
        finally:
            local_provider.finish.set()
            await first
    assert len(local_provider.requests) == 1
    assert bridge.budget.spent == pytest.approx(0.002)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget_options",
    [{"limit": 0.01}, {"limit": 10, "scope": "candidate", "scope_limit": 0.01}],
)
async def test_bridge_keeps_global_and_candidate_reservation_limits(
    tmp_path, local_provider, budget_options
):
    # [FUNCTION DOC] Parametrized over a tiny ledger limit or a tiny run-scope limit; expects BudgetExceeded with
    # no spend and no provider request.
    budget = Budget(tmp_path / "limited.json", **budget_options)
    async with make_bridge(tmp_path, budget=budget) as bridge:
        with pytest.raises(BudgetExceeded):
            await bridge._messages(model_request())
    assert budget.spent == 0
    assert local_provider.requests == []


@pytest.mark.asyncio
async def test_cancelling_bridge_owner_cancels_active_and_queued_tool_effects(tmp_path):
    # [FUNCTION DOC] Cancels the task owning the bridge context while one tool call is active and another is
    # queued; asserts both are cancelled, the active handler's finally ran, no effects recorded,
    # and later tool calls raise HTTPServiceUnavailable.
    started, cancelled, queued, ready = (asyncio.Event() for _ in range(4))
    effects = []

    async def remote(command):
        # [FUNCTION DOC] HELPER: fake shell that signals start and blocks forever; sets cancelled in finally.
        started.set()
        try:
            await asyncio.Event().wait()
            effects.append(command)
        finally:
            cancelled.set()

    bridge = make_bridge(tmp_path, handlers={"shell": remote})

    async def owner():
        # [FUNCTION DOC] HELPER: enters the bridge context, signals ready and waits forever.
        async with bridge:
            ready.set()
            await asyncio.Event().wait()

    owner_task = asyncio.create_task(owner())
    await asyncio.wait_for(ready.wait(), 2)
    active = asyncio.create_task(
        bridge.tool(
            BridgeRequest({"name": "shell", "arguments": {"command": "first"}}, token=bridge.token)
        )
    )
    waiting = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        waiting = asyncio.create_task(
            bridge.tool(
                BridgeRequest(
                    {"name": "shell", "arguments": {"command": "second"}},
                    token=bridge.token,
                    parsed=queued,
                )
            )
        )
        await asyncio.wait_for(queued.wait(), 2)
    finally:
        owner_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(owner_task, 2)
        await asyncio.wait_for(
            asyncio.gather(active, *([waiting] if waiting else []), return_exceptions=True), 2
        )
    assert cancelled.is_set()
    assert active.cancelled() and waiting.cancelled()
    assert effects == []
    with pytest.raises(web.HTTPServiceUnavailable):
        await bridge.tool(BridgeRequest({"name": "shell", "arguments": {}}, token=bridge.token))


@pytest.mark.asyncio
async def test_tool_failure_rejects_queued_effects_and_preserves_the_first_error(tmp_path):
    # [FUNCTION DOC] First tool raises RuntimeError after a queued second call arrives; asserts first reply
    # status 409, the queued call is cancelled without effect, and bridge.failure keeps the first
    # error.
    started, release, queued = (asyncio.Event() for _ in range(3))
    effects = []

    async def remote(command):
        # [FUNCTION DOC] HELPER: for 'first', signals start, waits for release, then raises RuntimeError;
        # otherwise records an effect.
        if command == "first":
            started.set()
            await release.wait()
            raise RuntimeError("candidate deferred")
        effects.append(command)
        return "unexpected queued effect"

    async with make_bridge(tmp_path, handlers={"shell": remote}) as bridge:
        first = asyncio.create_task(
            bridge.tool(
                BridgeRequest(
                    {"name": "shell", "arguments": {"command": "first"}}, token=bridge.token
                )
            )
        )
        await asyncio.wait_for(started.wait(), 2)
        second = asyncio.create_task(
            bridge.tool(
                BridgeRequest(
                    {"name": "shell", "arguments": {"command": "second"}},
                    token=bridge.token,
                    parsed=queued,
                )
            )
        )
        await asyncio.wait_for(queued.wait(), 2)
        release.set()
        replies = await asyncio.gather(first, second, return_exceptions=True)
    assert replies[0].status == 409
    assert isinstance(replies[1], asyncio.CancelledError)
    assert str(bridge.failure) == "candidate deferred"
    assert effects == []


@pytest.fixture
def local_runtime(monkeypatch, tmp_path, local_provider):
    # [FUNCTION DOC] Pytest fixture replacing the external runtime process: patches external_agent.AgentBridge
    # with DrainBridge, runtime_path, and asyncio.create_subprocess_exec with a fake that
    # validates the runtime config file (no API key in env, max tokens, timeout, inference
    # options) and returns a RuntimeProcess. Clears provider finish so EOF stays pending.
    """An in-process runtime exits at message_stop while provider EOF is pending."""
    local_provider.finish.clear()

    class DrainBridge(AgentBridge):
        # [CLASS DOC] AgentBridge subclass that releases the provider stream when the bridge context exits.
        async def __aexit__(self, *args):
            # [FUNCTION DOC] HELPER: sets local_provider.finish, then delegates to AgentBridge.__aexit__.
            local_provider.finish.set()
            return await super().__aexit__(*args)

    class RuntimeProcess:
        # [CLASS DOC] Fake subprocess that streams one model call through the bridge over HTTP.
        returncode = None

        def __init__(self, config):
            # [FUNCTION DOC] HELPER: stores the runtime config dict.
            self.config = config

        async def communicate(self):
            # [FUNCTION DOC] HELPER: posts a streaming /v1/messages request to the bridge, reads until
            # message_stop, sets returncode 0 and returns (JSON result bytes with turns 1 and cost
            # 0, None).
            async with ClientSession() as client:
                async with client.post(
                    self.config["bridge_url"] + "/v1/messages",
                    headers={"x-api-key": self.config["bridge_token"]},
                    json={"model": self.config["model"], "messages": [], "stream": True},
                ) as response:
                    assert response.status == 200
                    async for line in response.content:
                        if b"message_stop" in line:
                            break
            self.returncode = 0
            return json.dumps({"messages": [], "turns": 1, "cost": 0}).encode(), None

    async def create_runtime(*args, **kwargs):
        # [FUNCTION DOC] HELPER: fake create_subprocess_exec; asserts the config passed as the third argument and
        # returns RuntimeProcess.
        assert "ANTHROPIC_API_KEY" not in kwargs["env"]
        config = json.loads(Path(args[2]).read_text())
        assert config["max_tokens"] == MAX_OUTPUT_TOKENS
        assert config["model_timeout_sec"] == MODEL_TIMEOUT_SEC
        assert config["inference_options"] == anthropic_options("anthropic/" + config["model"])
        return RuntimeProcess(config)

    monkeypatch.setattr(external_agent, "AgentBridge", DrainBridge)
    monkeypatch.setattr(external_agent, "runtime_path", lambda engine: tmp_path / "runtime.mjs")
    monkeypatch.setattr(external_agent.asyncio, "create_subprocess_exec", create_runtime)


@pytest.mark.asyncio
async def test_runtime_result_cost_waits_for_stream_settlement(
    tmp_path, local_provider, local_runtime
):
    # [FUNCTION DOC] run_external_agent returns cost equal to budget spend 0.002 (settled after the stream ends,
    # not the runtime's reported 0) and turns 1.
    budget = Budget(tmp_path / "budget.json", 10)
    result = await external_agent.run_external_agent(
        engine="pi",
        model="anthropic/claude-mock",
        system="system",
        prompt="prompt",
        budget=budget,
        tools=[],
        handlers={},
        trace=tmp_path / "trace.jsonl",
        max_turns=1,
    )
    assert result["cost"] == budget.spent == pytest.approx(0.002)
    assert result["turns"] == 1


@pytest.mark.asyncio
async def test_runtime_propagates_accounting_errors_after_message_stop(
    tmp_path, local_provider, local_runtime
):
    # [FUNCTION DOC] With usage missing input_tokens, run_external_agent raises ValueError 'incomplete usage';
    # the reservation remains (spent > 0, operations reserved).
    local_provider.usage = {"output_tokens": 50}
    budget = Budget(tmp_path / "budget.json", 10)
    with pytest.raises(ValueError, match="incomplete usage"):
        await external_agent.run_external_agent(
            engine="pi",
            model="anthropic/claude-mock",
            system="system",
            prompt="prompt",
            budget=budget,
            tools=[],
            handlers={},
            trace=tmp_path / "trace.jsonl",
            max_turns=1,
        )
    assert budget.spent > 0
    ledger = budget.budget.ledger.status()
    assert {entry["status"] for entry in ledger["operations"]} == {"reserved"}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-opus-4-6", "claude-mock"])
async def test_bridge_overrides_runtime_inference_settings_and_preserves_opaque_thinking(
    tmp_path, local_provider, model
):
    # [FUNCTION DOC] Parametrized over models. Asserts the forwarded request uses MAX_OUTPUT_TOKENS, keeps
    # thinking/redacted_thinking blocks verbatim, drops temperature/top_p/top_k, applies
    # anthropic_options thinking/output_config, keeps tool_choice, and logs inference settings and
    # timeout in the trace.
    opaque = [
        {"type": "thinking", "thinking": "", "signature": "opaque-signed-reasoning"},
        {"type": "redacted_thinking", "data": "opaque-redacted-reasoning"},
        {"type": "text", "text": "Previous response"},
    ]
    async with make_bridge(tmp_path, model="anthropic/" + model) as bridge:
        body = {
            "model": model,
            "messages": [{"role": "assistant", "content": opaque}],
            "max_tokens": 1,
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "high"},
            "temperature": 0.3,
            "top_p": 0.1,
            "top_k": 2,
        }
        await bridge._messages(BridgeRequest(body))
    forwarded = local_provider.requests[0]
    assert forwarded["max_tokens"] == MAX_OUTPUT_TOKENS
    assert forwarded["messages"][0]["content"] == opaque
    assert not {"temperature", "top_p", "top_k"} & forwarded.keys()
    for key in ("thinking", "output_config"):
        assert forwarded.get(key) == anthropic_options("anthropic/" + model).get(key)
    events = [json.loads(line) for line in bridge.trace.read_text().splitlines()]
    assert events[0]["inference"]["max_tokens"] == MAX_OUTPUT_TOKENS
    assert events[0]["timeout_sec"] == MODEL_TIMEOUT_SEC
    assert events[0]["tool_choice"] == forwarded["tool_choice"] == {"type": "auto"}
    assert events[0]["inference"].get("thinking") == forwarded.get("thinking")


@pytest.fixture
def in_memory_sse(monkeypatch):
    # [FUNCTION DOC] Pytest fixture replacing bridge_module.web.StreamResponse with a no-op stub so streaming
    # replies need no real HTTP transport.
    class StreamResponse:
        # [CLASS DOC] No-op stub of aiohttp StreamResponse.
        def __init__(self, **kwargs):
            # [FUNCTION DOC] HELPER: ignores kwargs.
            pass

        async def prepare(self, request):
            # [FUNCTION DOC] HELPER: no-op.
            pass

        async def write(self, chunk):
            # [FUNCTION DOC] HELPER: no-op; discards the chunk.
            pass

        async def write_eof(self):
            # [FUNCTION DOC] HELPER: no-op.
            pass

    monkeypatch.setattr(bridge_module.web, "StreamResponse", StreamResponse)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure", ["max_tokens", "length", "empty", "thinking_only"])
async def test_empty_and_truncated_provider_turns_fail_after_metering_and_trace(
    tmp_path, local_provider, in_memory_sse, stream, failure
):
    # [FUNCTION DOC] Parametrized over stream mode and failure (max_tokens/length stop, empty content, thinking-
    # only). Asserts reply 502, bridge failed, cost 0.002 metered and settled, trace kinds
    # model_request/model/runtime_error with usage and preserved signature, and the error mentions
    # truncation or missing text/tool calls.
    if failure in {"max_tokens", "length"}:
        local_provider.stop_reason = failure
    else:
        local_provider.blocks = (
            []
            if failure == "empty"
            else [{"type": "thinking", "thinking": "", "signature": "retained-opaque-signature"}]
        )
    async with make_bridge(tmp_path) as bridge:
        reply = await bridge.messages(
            BridgeRequest(
                {"model": "claude-mock", "messages": [], "stream": stream},
                token=bridge.token,
            )
        )
        assert reply.status == 502
        assert bridge.failed.is_set()
        assert bridge.budget.spent == bridge.cost == pytest.approx(0.002)
    events = [json.loads(line) for line in bridge.trace.read_text().splitlines()]
    assert [event["kind"] for event in events] == ["model_request", "model", "runtime_error"]
    assert events[1]["cost_usd"] == pytest.approx(0.002)
    assert events[1]["usage"]["output_tokens"] == 50
    if failure == "thinking_only":
        assert "retained-opaque-signature" in json.dumps(events[1])
    assert (
        "truncated" in events[2]["error"]
        if failure in {"max_tokens", "length"}
        else "no text or tool calls" in events[2]["error"]
    )
    ledger = bridge.budget.budget.ledger.status()
    assert {entry["status"] for entry in ledger["operations"]} == {"settled"}


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_tool_only_provider_turn_is_usable(tmp_path, local_provider, in_memory_sse, stream):
    # [FUNCTION DOC] A turn with only a tool_use block is accepted (no failure) and costs 0.002, streaming or
    # not.
    local_provider.blocks = [{"type": "tool_use", "id": "tool-1", "name": "shell", "input": {}}]
    local_provider.stop_reason = "tool_use"
    async with make_bridge(tmp_path) as bridge:
        await bridge._messages(
            BridgeRequest({"model": "claude-mock", "messages": [], "stream": stream})
        )
        assert bridge.failure is None
        assert bridge.cost == pytest.approx(0.002)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["tool_use", "max_tokens"])
async def test_tool_waits_for_delayed_model_eof_and_validation(
    tmp_path, local_provider, in_memory_sse, stop_reason
):
    # [FUNCTION DOC] Parametrized stop_reason. A tool call arriving before the model stream ends waits (no
    # effect, cost 0). After EOF: tool_use runs the tool once with cost already settled;
    # max_tokens cancels the tool and records a truncation failure. Checks trace kinds.
    local_provider.finish.clear()
    local_provider.stop_reason = stop_reason
    local_provider.blocks = [{"type": "tool_use", "id": "tool-1", "name": "shell", "input": {}}]
    effects, observed_costs = [], []
    parsed = asyncio.Event()

    async def shell():
        # [FUNCTION DOC] HELPER: fake shell recording execution and the bridge cost observed at that moment;
        # returns 'done'.
        effects.append("executed")
        observed_costs.append(bridge.cost)
        return "done"

    async with make_bridge(tmp_path, handlers={"shell": shell}) as bridge:
        model = asyncio.create_task(
            bridge.messages(
                BridgeRequest(
                    {"model": "claude-mock", "stream": True},
                    token=bridge.token,
                )
            )
        )
        await asyncio.wait_for(local_provider.called.wait(), 2)
        tool = asyncio.create_task(
            bridge.tool(
                BridgeRequest(
                    {"name": "shell", "arguments": {}},
                    token=bridge.token,
                    parsed=parsed,
                )
            )
        )
        try:
            await asyncio.wait_for(parsed.wait(), 2)
            assert effects == []
            assert not tool.done()
            assert bridge.cost == 0
        finally:
            local_provider.finish.set()
            replies = await asyncio.wait_for(asyncio.gather(model, tool, return_exceptions=True), 2)
        assert bridge.cost == pytest.approx(0.002)
        if stop_reason == "max_tokens":
            assert effects == []
            assert isinstance(replies[1], asyncio.CancelledError)
            assert "truncated" in str(bridge.failure)
        else:
            assert effects == ["executed"]
            assert observed_costs == [pytest.approx(0.002)]
            assert replies[1].status == 200
    records = [json.loads(line) for line in bridge.trace.read_text().splitlines()]
    assert [record["kind"] for record in records] == [
        "model_request",
        "model",
        "runtime_error" if stop_reason == "max_tokens" else "tool",
    ]


@pytest.mark.asyncio
async def test_cancelling_waiting_tool_does_not_cancel_model_settlement(
    tmp_path, local_provider, in_memory_sse
):
    # [FUNCTION DOC] Cancelling a tool call gated on model EOF does not cancel the model task or set a failure;
    # after EOF cost is 0.002 and no effect ran.
    local_provider.finish.clear()
    parsed = asyncio.Event()
    effects = []

    async def shell():
        # [FUNCTION DOC] HELPER: fake shell that must not run; records an effect if it does.
        effects.append("must not execute")
        return "unexpected"

    async with make_bridge(tmp_path, handlers={"shell": shell}) as bridge:
        model = asyncio.create_task(
            bridge.messages(
                BridgeRequest(
                    {"model": "claude-mock", "stream": True},
                    token=bridge.token,
                )
            )
        )
        await asyncio.wait_for(local_provider.called.wait(), 2)
        tool = asyncio.create_task(
            bridge.tool(
                BridgeRequest(
                    {"name": "shell", "arguments": {}},
                    token=bridge.token,
                    parsed=parsed,
                )
            )
        )
        try:
            await asyncio.wait_for(parsed.wait(), 2)
            tool.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(tool, 2)
            assert not model.done()
            assert bridge.failure is None
        finally:
            local_provider.finish.set()
            await asyncio.wait_for(model, 2)
        assert bridge.cost == pytest.approx(0.002)
        assert effects == []


@pytest.mark.asyncio
async def test_bridge_shutdown_cancels_model_and_gated_tool_without_deadlock(
    tmp_path, local_provider, in_memory_sse
):
    # [FUNCTION DOC] Cancelling the bridge owner while a model stream and gated tool are pending cancels both
    # without deadlock, clears internal task sets, runs no effect and keeps the uncertain
    # reservation (spent > 0).
    local_provider.finish.clear()
    ready, parsed = asyncio.Event(), asyncio.Event()
    effects = []

    async def shell():
        # [FUNCTION DOC] HELPER: fake shell that must not run.
        effects.append("must not execute")
        return "unexpected"

    bridge = make_bridge(tmp_path, handlers={"shell": shell})

    async def owner():
        # [FUNCTION DOC] HELPER: enters the bridge context, signals ready and waits forever.
        async with bridge:
            ready.set()
            await asyncio.Event().wait()

    owning = asyncio.create_task(owner())
    await asyncio.wait_for(ready.wait(), 2)
    model = asyncio.create_task(
        bridge.messages(
            BridgeRequest(
                {"model": "claude-mock", "stream": True},
                token=bridge.token,
            )
        )
    )
    await asyncio.wait_for(local_provider.called.wait(), 2)
    tool = asyncio.create_task(
        bridge.tool(
            BridgeRequest(
                {"name": "shell", "arguments": {}},
                token=bridge.token,
                parsed=parsed,
            )
        )
    )
    try:
        await asyncio.wait_for(parsed.wait(), 2)
    finally:
        owning.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(owning, 2)
        await asyncio.wait_for(asyncio.gather(model, tool, return_exceptions=True), 2)
    assert model.cancelled() and tool.cancelled()
    assert not bridge._model_tasks and not bridge._tool_tasks
    assert effects == []
    assert bridge.budget.spent > 0  # Uncertain provider work retains its reservation.


@pytest.mark.asyncio
async def test_actual_opencode_truncated_tool_response_cannot_execute_before_eof(
    tmp_path, local_provider, monkeypatch
):
    # [FUNCTION DOC] Skips unless the opencode runtime is installed. Streams a max_tokens-truncated tool_use
    # through the real runtime; asserts the tool request arrives but does not execute before EOF,
    # the run raises RuntimeError 'truncated', spend is 0.002 and trace kinds are
    # input/model_request/model/runtime_error. Spawns a real subprocess.
    try:
        external_agent.runtime_path("opencode")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    local_provider.finish.clear()
    arrived = asyncio.Event()
    effects = []
    bridges = []

    class ObservedBridge(AgentBridge):
        # [CLASS DOC] AgentBridge subclass recording instances and signalling when a tool request arrives.
        def __init__(self, **kwargs):
            # [FUNCTION DOC] HELPER: initializes the AgentBridge and appends self to the outer bridges list.
            super().__init__(**kwargs)
            bridges.append(self)

        async def tool(self, request):
            # [FUNCTION DOC] HELPER: sets the arrived event then delegates to AgentBridge.tool.
            arrived.set()
            return await super().tool(request)

    async def stream():
        # [FUNCTION DOC] HELPER: replacement iter_any yielding a full Anthropic SSE sequence for a tool_use
        # truncated by max_tokens, then waiting for finish.
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "message-mock",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-mock",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 100, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool-mock",
                    "name": "shell",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"command":"echo remote-only"}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
                "usage": {"output_tokens": 50},
            },
            {"type": "message_stop"},
        ]
        yield "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        ).encode()
        await local_provider.finish.wait()

    async def shell(command):
        # [FUNCTION DOC] HELPER: fake shell that must not run; records the command.
        effects.append(command)
        return "must not execute"

    monkeypatch.setattr(local_provider, "iter_any", stream)
    monkeypatch.setattr(external_agent, "AgentBridge", ObservedBridge)
    budget = Budget(tmp_path / "budget.json", 10)
    running = asyncio.create_task(
        external_agent.run_external_agent(
            engine="opencode",
            model="anthropic/claude-mock",
            system="Use the supplied remote tool.",
            prompt="Call shell once.",
            budget=budget,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "description": "Run remotely",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ],
            handlers={"shell": shell},
            trace=tmp_path / "trace.jsonl",
            max_turns=1,
        )
    )
    try:
        await asyncio.wait_for(arrived.wait(), 20)
        assert effects == []
        assert bridges[0].cost == 0
    finally:
        local_provider.finish.set()
        with pytest.raises(RuntimeError, match="truncated"):
            await asyncio.wait_for(running, 20)
    assert effects == []
    assert budget.spent == pytest.approx(0.002)
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [event["kind"] for event in events] == [
        "input",
        "model_request",
        "model",
        "runtime_error",
    ]
    assert events[2]["cost_usd"] == pytest.approx(0.002)
