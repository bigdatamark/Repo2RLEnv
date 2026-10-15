# [MODULE DOC] Durable, typed artifact authoring stages for Tasksmith. artifact_stage runs an author
# agent that must produce a Pydantic-schema artifact through tools (submit_artifact full
# replacement, revise_artifact JSON merge patch, optional shell and extra tools), validates it, and
# commits it atomically to <root>/artifact.json keyed by a digest of all inputs, so completed stages
# are reused and never re-rolled. In-progress workers leave operation.json and must be reconciled,
# not replayed. Limits: MAX_ARTIFACT_BYTES 512000, MAX_PATCH_BYTES 64000, MAX_JSON_DEPTH 32; tool
# protocol 2.
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Awaitable, Callable
from copy import deepcopy
from pathlib import Path

from pydantic import BaseModel, ValidationError

from repo2rlenv.campaigns.budget import BudgetExceeded
from repo2rlenv.tasksmith.author.agent import SHELL_TOOL, run_agent
from repo2rlenv.tasksmith.author.bridge import ProviderOutputError
from repo2rlenv.tasksmith.author.budget import AuthorBudget as Budget

ARTIFACT_TOOL_PROTOCOL = 2
MAX_ARTIFACT_BYTES = 512_000
MAX_PATCH_BYTES = 64_000
MAX_JSON_DEPTH = 32


def _bounded_object(value: object, *, max_bytes: int) -> dict:
    # [FUNCTION DOC] HELPER: Validates that value is a JSON-like object and returns a detached deep
    # copy. Iteratively checks nesting depth <= MAX_JSON_DEPTH, string keys, and only
    # str/int/float/bool/None/list/dict values; serializes (NaN disallowed) and enforces max_bytes
    # on the UTF-8 size; returns the re-parsed copy. Raises ValueError on any violation (json may
    # raise ValueError for NaN).
    """Validate JSON before recursive merging, and detach it from caller-owned objects."""
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            raise ValueError(f"JSON nesting exceeds {MAX_JSON_DEPTH} levels")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("Only JSON values are supported")
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode()) > max_bytes:
        raise ValueError(f"JSON object exceeds {max_bytes} UTF-8 bytes")
    return json.loads(encoded)


def _merge_patch(target: object, patch: object) -> object:
    # [FUNCTION DOC] HELPER: Applies an RFC 7396 JSON merge patch recursively: a non-dict patch
    # replaces target; for dicts, null values delete keys and other values merge into a shallow copy
    # of target (a non-dict target becomes empty). Output: new merged value; does not mutate target
    # at top level.
    """RFC 7396 semantics; the public tool additionally requires an object root."""
    if not isinstance(patch, dict):
        return patch
    result = dict(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = _merge_patch(result.get(key), value)
    return result


def canonical_digest(value: object) -> str:
    # [FUNCTION DOC] Returns the SHA-256 hex digest of value's canonical JSON (sorted keys, compact
    # separators, non-JSON values via str, NaN disallowed). Raises ValueError for NaN/inf.
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False
        ).encode()
    ).hexdigest()


def save_json(path: Path, value: object) -> None:
    # [FUNCTION DOC] Atomically writes value as indented JSON to path: creates parent dirs, writes
    # to "<path>.tmp", flushes and fsyncs, then replaces path. Non-JSON values are stringified. Side
    # effect: filesystem write.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, default=str, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


async def artifact_stage[Artifact: BaseModel](
    *,
    schema: type[Artifact],
    stage: str,
    inputs: dict,
    system: str,
    prompt: str,
    root: Path,
    budget: Budget,
    model: str,
    runtime: str,
    max_cost: float,
    max_turns: int,
    deadline: float,
    shell: Callable[..., Awaitable[str]] | None = None,
    extra_tools: list[dict] | None = None,
    extra_handlers: dict[str, Callable[..., Awaitable[str]]] | None = None,
    validate: Callable[[Artifact], Awaitable[None]] | None = None,
    initial_draft_key: str | None = None,
) -> Artifact:
    # [FUNCTION DOC] Runs (or restores) one durable authoring stage and returns the validated
    # artifact. Inputs (keyword): schema (Pydantic model class), stage name, inputs dict,
    # system/prompt text, root dir, budget (spent tracking), model, runtime ("openai" or external
    # engine), max_cost (USD for this stage), max_turns, deadline (epoch seconds), optional shell
    # callable, extra_tools/extra_handlers, validate (async operator check), initial_draft_key (an
    # inputs key holding a draft to seed revisions). Behavior: validates any initial draft against
    # the schema and adds guidance to the prompt; builds the tool list and rejects duplicate names
    # or handlers overriding built-ins; computes an identity digest of inputs, tools, protocol and
    # limits. If artifact.json exists, returns it after checking its digest (protocol 1 artifacts
    # use the legacy digest), else raises ValueError. If operation.json exists, or retained draft
    # files exist with an initial draft, raises RuntimeError (needs reconciliation). Raises
    # TimeoutError if the deadline passed. Writes operation.json ("running") and seed draft files,
    # then runs the agent under an asyncio timeout until the deadline, with one recovery retry after
    # a ProviderOutputError (re-prompting with remaining turns/cost and the retained draft). Raises
    # BudgetExceeded if recovery has no allowance and ValueError if no artifact was committed. On
    # any exception, records status/error; returns an accepted artifact anyway for
    # BudgetExceeded/ValueError. Always records finish time and spend in operation.json. Side
    # effects: files under root (artifact.json, operation.json, draft.json, initial-draft.json,
    # trace files), LLM calls and budget spend.
    """Commit typed output before graph advancement; never reroll unchanged completed stages.

    A crashed worker is deliberately not replayed blindly. If it submitted its artifact,
    recovery uses it. Otherwise its in-progress marker requires explicit reconciliation.
    A named input may seed a schema-valid draft; only a model submission or revision
    can commit it after all normal validation. The original seed remains preserved.
    """
    initial_draft = None
    if initial_draft_key is not None:
        if not isinstance(initial_draft_key, str) or initial_draft_key not in inputs:
            raise ValueError(f"{stage}: initial draft key must identify an existing input")
        try:
            initial_draft = _bounded_object(inputs[initial_draft_key], max_bytes=MAX_ARTIFACT_BYTES)
            schema.model_validate(deepcopy(initial_draft))
        except ValueError as exc:
            raise ValueError(
                f"{stage}: invalid initial draft in inputs[{initial_draft_key!r}]: {exc}"
            ) from exc
        prompt += (
            f"\nThe controller has loaded inputs[{initial_draft_key!r}] as an unvalidated draft. "
            "Use revise_artifact with only changed fields, or submit_artifact for a complete "
            "replacement. Nothing is committed yet; your submission must pass all schema "
            "and operator validation. The original draft is preserved."
        )
    root.mkdir(parents=True, exist_ok=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "submit_artifact",
                "description": "Submit a complete replacement artifact for validation and durable commit. After rejection, use revise_artifact to send only changed fields instead of repeating this full object.",
                "parameters": schema.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "revise_artifact",
                "description": f"Correct the most recent rejected draft using a JSON merge patch (up to {MAX_PATCH_BYTES} UTF-8 bytes, {MAX_JSON_DEPTH} nesting levels). Omit unchanged fields. Objects merge recursively; null deletes a dictionary key; arrays replace entirely. All schema and operator validation runs again. A rejected revision becomes the next unvalidated draft. Use submit_artifact for a complete replacement or to assign a literal null value.",
                "parameters": {
                    "type": "object",
                    "properties": {"patch": {"type": "object", "additionalProperties": True}},
                    "required": ["patch"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    if initial_draft is not None:
        tool = tools[1]["function"]
        tool["description"] = tool["description"].replace(
            "most recent rejected draft",
            "controller-provided initial draft or most recent rejected draft",
        )
    if shell:
        tools.append(SHELL_TOOL)
    tools.extend(extra_tools or [])
    names = [tool["function"]["name"] for tool in tools]
    if len(set(names)) != len(names) or set(extra_handlers or {}) & {
        "submit_artifact",
        "revise_artifact",
        "shell",
    }:
        raise ValueError("Extra tools/handlers cannot replace built-in artifact validation tools")
    legacy_inputs = {
        "stage": stage,
        "inputs": inputs,
        "system": system,
        "prompt": prompt,
        "schema": schema.model_json_schema(),
        "model": model,
        "runtime": runtime,
    }
    if initial_draft_key is not None:
        legacy_inputs["initial_draft_key"] = initial_draft_key
    identity = canonical_digest(
        {
            **legacy_inputs,
            "tool_protocol": ARTIFACT_TOOL_PROTOCOL,
            "tools": tools,
            "artifact_limits": {
                "max_bytes": MAX_ARTIFACT_BYTES,
                "patch_bytes": MAX_PATCH_BYTES,
                "max_depth": MAX_JSON_DEPTH,
            },
        }
    )
    result_path, operation_path = root / "artifact.json", root / "operation.json"
    if result_path.exists():
        stored = json.loads(result_path.read_text())
        # Markerless completed artifacts retain their original identity contract.
        # They are not rewritten or silently upgraded to this new tool protocol.
        protocol = stored.get("tool_protocol", 1)
        expected = canonical_digest(legacy_inputs) if protocol == 1 else identity
        if protocol not in (1, ARTIFACT_TOOL_PROTOCOL) or stored["input_digest"] != expected:
            raise ValueError(f"{stage}: saved artifact belongs to different inputs")
        return schema.model_validate(stored["artifact"])
    prior = json.loads(operation_path.read_text()) if operation_path.exists() else None
    if prior:
        raise RuntimeError(
            f"{stage}: retained {prior['status']} worker needs reconciliation; "
            "preserve its trace, remote state and charges instead of restarting it"
        )
    if initial_draft is not None and any(
        (root / name).exists() for name in ("initial-draft.json", "draft.json")
    ):
        raise RuntimeError(f"{stage}: retained draft without an operation needs reconciliation")
    if deadline <= time.time():
        raise TimeoutError("Candidate deadline exhausted")
    accepted: Artifact | None = None
    draft: dict | None = deepcopy(initial_draft)
    draft_number = 0
    lock = asyncio.Lock()
    operation = {
        "input_digest": identity,
        "tool_protocol": ARTIFACT_TOOL_PROTOCOL,
        "stage": stage,
        "status": "running",
        "started_at": time.time(),
        "deadline": deadline,
        "starting_spend": budget.spent,
    }
    save_json(operation_path, operation)
    if initial_draft is not None:
        retained_seed = {
            "input_digest": identity,
            "tool_protocol": ARTIFACT_TOOL_PROTOCOL,
            "status": "unvalidated",
            "number": 0,
            "origin": "initial_draft",
            "input_key": initial_draft_key,
            "artifact": initial_draft,
        }
        save_json(root / "initial-draft.json", retained_seed)
        save_json(root / "draft.json", retained_seed)

    async def validate_and_commit(payload: dict, origin: str) -> str:
        # [FUNCTION DOC] HELPER: Bounds the payload, records it as the next numbered unvalidated
        # draft in draft.json, validates it against the schema and the optional validate callback,
        # and on success writes artifact.json and sets accepted. Output: a message string for the
        # model describing rejection (with error text truncated to 6000 chars) or commit. Mutates
        # enclosing accepted/draft/number.
        nonlocal accepted, draft, draft_number
        try:
            payload = _bounded_object(payload, max_bytes=MAX_ARTIFACT_BYTES)
        except ValueError as exc:
            return f"Artifact rejected; existing draft unchanged: {exc}"
        draft, draft_number = payload, draft_number + 1
        retained = {
            "input_digest": identity,
            "tool_protocol": ARTIFACT_TOOL_PROTOCOL,
            "status": "unvalidated",
            "number": draft_number,
            "origin": origin,
            "artifact": draft,
        }
        save_json(root / "draft.json", retained)
        try:
            value = schema.model_validate(deepcopy(payload))
            if validate:
                await validate(value)
        except (ValidationError, ValueError) as exc:
            save_json(root / "draft.json", {**retained, "validation_error": str(exc)})
            return f"Artifact rejected; call revise_artifact with only changed fields, or submit_artifact for a complete replacement: {str(exc)[:6000]}"
        save_json(
            result_path,
            {
                "input_digest": identity,
                "tool_protocol": ARTIFACT_TOOL_PROTOCOL,
                "artifact": value.model_dump(mode="json"),
            },
        )
        accepted = value
        return "Artifact committed. Do not call more tools; finish with a brief summary."

    async def submit_artifact(**payload) -> str:
        # [FUNCTION DOC] HELPER: Tool handler for submit_artifact: under the lock, returns an
        # "already committed" message if accepted, else validates and commits the full payload
        # (kwargs).
        async with lock:
            if accepted is not None:
                return "Artifact already committed. Finish with a brief summary."
            return await validate_and_commit(payload, "submit_artifact")

    async def revise_artifact(patch=None, **unexpected) -> str:
        # [FUNCTION DOC] HELPER: Tool handler for revise_artifact: under the lock, rejects if
        # already committed, if there is no draft yet, or if extra arguments or an invalid patch
        # (bounded to MAX_PATCH_BYTES) are given; otherwise merges patch into the current draft and
        # validates/commits it. Output: message string.
        async with lock:
            if accepted is not None:
                return "Artifact already committed. Finish with a brief summary."
            if draft is None:
                return "Artifact revision rejected: no prior draft. Call submit_artifact with a complete artifact first."
            try:
                if unexpected:
                    raise ValueError("revise_artifact accepts only the patch argument")
                patch = _bounded_object(patch, max_bytes=MAX_PATCH_BYTES)
            except ValueError as exc:
                return f"Artifact revision rejected; existing draft unchanged: {exc}"
            return await validate_and_commit(_merge_patch(draft, patch), "revise_artifact")

    async def remote_shell(command: str, timeout_sec: int = 120) -> str:
        # [FUNCTION DOC] HELPER: Tool handler wrapping the provided shell callable: returns a "shell
        # closed" message once an artifact is committed, raises TimeoutError if the deadline passed,
        # and otherwise runs command with timeout min(timeout_sec, seconds remaining). Output: shell
        # output string.
        if accepted is not None:
            return "Stage output committed; shell closed."
        remaining = int(deadline - time.time())
        if remaining <= 0:
            raise TimeoutError("Candidate deadline exhausted")
        return await shell(command=command, timeout_sec=min(timeout_sec, remaining))

    handlers = {"submit_artifact": submit_artifact, "revise_artifact": revise_artifact}
    if shell:
        handlers["shell"] = remote_shell
    handlers.update(extra_handlers or {})
    try:
        async with asyncio.timeout(max(1, deadline - time.time())):
            used_turns = 0
            recovery = ""
            for provider_attempt in range(2):
                trace = root / ("trace.jsonl" if not provider_attempt else "trace-recovery1.jsonl")
                remaining_cost = max_cost - (budget.spent - operation["starting_spend"])
                if remaining_cost <= 0 or used_turns >= max_turns:
                    raise BudgetExceeded("Author recovery exhausted the original stage allowance")
                try:
                    await run_agent(
                        model=model,
                        system=system,
                        prompt=prompt + recovery,
                        budget=budget,
                        tools=tools,
                        handlers=handlers,
                        trace=trace,
                        max_turns=max_turns - used_turns,
                        max_cost=remaining_cost,
                        runtime=runtime,
                    )
                    break
                except ProviderOutputError as exc:
                    # Only complete, charged responses reach this path. Unknown
                    # transport outcomes retain their holds and require recovery.
                    if accepted is not None:
                        break
                    if provider_attempt:
                        raise
                    events = (
                        [json.loads(line) for line in trace.read_text().splitlines()]
                        if trace.exists()
                        else []
                    )
                    used_turns += max(
                        1, sum(event.get("kind") == "model_request" for event in events)
                    )
                    operation["provider_recovery"] = {
                        "reason": str(exc),
                        "trace": str(trace),
                        "used_turns": used_turns,
                    }
                    save_json(operation_path, operation)
                    recovery = (
                        "\nThe previous response completed and was charged but could not be used: "
                        + str(exc)
                        + ". Finish concisely within the remaining original allowance. "
                        "Use submit_artifact with a compact complete object; if a rejected draft "
                        "exists, use revise_artifact with only changed fields. Avoid long reasoning "
                        "and unrelated exploration. Partial tool arguments were not executed."
                    )
                    if draft is not None:
                        recovery += "\nRetained unvalidated draft: " + json.dumps(draft)
        if accepted is None:
            raise ValueError(f"{stage}: worker ended without a validated artifact")
        operation["status"] = "completed"
        return accepted
    except BaseException as exc:
        operation["status"] = "completed" if accepted is not None else "incomplete"
        operation["error"] = f"{type(exc).__name__}: {exc}"
        if accepted is not None and isinstance(exc, (BudgetExceeded, ValueError)):
            return accepted
        raise
    finally:
        operation["finished_at"] = time.time()
        operation["charged_or_reserved_usd"] = budget.spent - operation["starting_spend"]
        save_json(operation_path, operation)
