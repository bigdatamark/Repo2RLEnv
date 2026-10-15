# [MODULE DOC] ReAct-style bootstrap agent. The LLM replies in a "Thought / Action / Input" text
# format which is parsed leniently (_ACTION_RE stops the input at the next
# Action/Thought/Observation header; trailing tool-call XML tags in _TOOL_CALL_TAGS are stripped).
# Tools are BASH, READ_FILE and LIST_DIR, executed in a DockerSandbox; SAVE_SETUP (JSON with
# rebuild_cmds/test_cmds/summary) ends in success and GIVE_UP ends in failure. run_agent_loop
# enforces iteration, wall-clock and LLM spend budgets and accumulates cost from llm.complete
# responses. Logs per-step info via the module logger.
"""ReAct-style bootstrap agent loop. Original implementation, not from RepoLaunch."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from repo2rlenv.bootstrap.docker import DockerSandbox, ExecResult
from repo2rlenv.bootstrap.prompts import initial_user_prompt, system_prompt
from repo2rlenv.bootstrap.spec import LanguageHint
from repo2rlenv.llm import complete
from repo2rlenv.spec.input import LLMSpec

logger = logging.getLogger(__name__)


ActionName = Literal["BASH", "READ_FILE", "LIST_DIR", "SAVE_SETUP", "GIVE_UP", "INVALID"]


@dataclass(slots=True)
class AgentAction:
    # [CLASS DOC] Slotted dataclass for one parsed action: name (one of BASH, READ_FILE, LIST_DIR,
    # SAVE_SETUP, GIVE_UP or INVALID), input (sanitized argument string) and raw (the full LLM
    # text).
    name: ActionName
    input: str
    raw: str = ""


@dataclass(slots=True)
class AgentTurn:
    # [CLASS DOC] Slotted dataclass for one transcript entry: step index, thought, action,
    # observation text, plus per-turn LLM cost estimate, prompt/completion token counts and duration
    # in seconds (the latter four are only filled for executed tool turns; default 0).
    step: int
    thought: str
    action: AgentAction
    observation: str
    cost_estimate_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_sec: float = 0.0


@dataclass(slots=True)
class AgentOutcome:
    # [CLASS DOC] Slotted dataclass returned by run_agent_loop: success flag, human-readable reason,
    # rebuild_cmds and test_cmds (only on success), summary from SAVE_SETUP, iterations consumed,
    # transcript list and total_cost_estimate_usd accumulated over all LLM calls.
    success: bool
    reason: str
    rebuild_cmds: list[str] = field(default_factory=list)
    test_cmds: list[str] = field(default_factory=list)
    summary: str = ""
    iterations: int = 0
    transcript: list[AgentTurn] = field(default_factory=list)
    total_cost_estimate_usd: float = 0.0


# Loose regex parser. Stops capturing Input at the next Action:/Thought:/
# Observation: header so a runaway LLM that hallucinates its own observation
# inline doesn't get that fake text fed into bash as part of the command.
_ACTION_RE = re.compile(
    r"Thought:\s*(?P<thought>.*?)\n+Action:\s*(?P<action>\w+)\s*\n+Input:\s*"
    r"(?P<input>.*?)(?=\n+Action:|\n+Thought:|\n+Observation:|\Z)",
    re.DOTALL | re.IGNORECASE,
)

# Common XML / tool-call artifacts that some models (e.g. Sonnet) leak into
# action inputs when they slip back into native tool-call mode. Stripped
# from both ends BEFORE execution so they don't crash bash with syntax errors.
_TOOL_CALL_TAGS = (
    "</parameter>",
    "</invoke>",
    "</function_calls>",
    "</function>",
    "</tool_use>",
)


def _sanitize_input(inp: str) -> str:
    # [FUNCTION DOC] HELPER: Cleans an action input: strips whitespace, cuts everything from a
    # simulated "Observation:" header on its own line onward (case-insensitive), then repeatedly
    # removes known trailing tool-call closing tags (_TOOL_CALL_TAGS). Output: cleaned str. No side
    # effects.
    """Strip trailing tool-call XML artifacts + simulated Observation: tails."""
    s = inp.strip()
    # Drop anything after a simulated `Observation:` header inside the input —
    # models sometimes write the command AND simulate its output.
    obs_match = re.search(r"\n\s*Observation\s*:", s, flags=re.IGNORECASE)
    if obs_match:
        s = s[: obs_match.start()].rstrip()
    # Repeatedly strip known tag artifacts from the end.
    changed = True
    while changed:
        changed = False
        stripped = s.rstrip()
        for tag in _TOOL_CALL_TAGS:
            if stripped.endswith(tag):
                s = stripped[: -len(tag)].rstrip()
                changed = True
                break
    return s


def parse_action(text: str) -> tuple[str, AgentAction]:
    # [FUNCTION DOC] Parses an LLM response into (thought, AgentAction) using _ACTION_RE (first
    # match). No match yields ("", INVALID action whose input is the first 500 chars of the stripped
    # text). The action name is upper-cased; unknown names yield an INVALID action carrying the
    # thought and sanitized input. Output: tuple(str, AgentAction); raw always holds the full text.
    # Never raises.
    """Extract (thought, action) from an LLM response. Lenient."""
    m = _ACTION_RE.search(text)
    if not m:
        return ("", AgentAction(name="INVALID", input=text.strip()[:500], raw=text))
    thought = m.group("thought").strip()
    name = m.group("action").strip().upper()
    inp = _sanitize_input(m.group("input"))
    if name not in ("BASH", "READ_FILE", "LIST_DIR", "SAVE_SETUP", "GIVE_UP"):
        return (thought, AgentAction(name="INVALID", input=inp, raw=text))
    return (thought, AgentAction(name=name, input=inp, raw=text))  # type: ignore[arg-type]


def _execute(action: AgentAction, sandbox: DockerSandbox) -> str:
    # [FUNCTION DOC] HELPER: Runs a tool action in the sandbox and returns the observation string.
    # BASH: sandbox.exec, formatted as "exit=N" plus the truncated combined output. READ_FILE: file
    # contents, or "ERROR reading file: ..." on any exception. LIST_DIR: newline-joined entries
    # (input defaults to "."), or a placeholder when empty/missing. Other action names return an
    # empty string. Side effects: commands run inside the container.
    """Run a tool and return the observation string."""
    if action.name == "BASH":
        r: ExecResult = sandbox.exec(action.input)
        return f"exit={r.exit_code}\n{r.truncated()}"
    if action.name == "READ_FILE":
        try:
            return sandbox.read_file(action.input)
        except Exception as exc:
            return f"ERROR reading file: {exc}"
    if action.name == "LIST_DIR":
        entries = sandbox.list_dir(action.input or ".")
        return "\n".join(entries) if entries else "(empty or missing directory)"
    return ""  # SAVE_SETUP / GIVE_UP / INVALID handled by the loop


def _parse_save_setup(input_str: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Parses a SAVE_SETUP payload tolerantly: strips surrounding ``` /
    # ```json fences, starts at the first "{" and decodes only the first complete JSON value
    # (raw_decode), ignoring trailing text. Requires a dict with rebuild_cmds and test_cmds each a
    # list of strings; extra keys (e.g. summary) are kept. Output: dict. Raises: ValueError (no
    # object, not a dict, missing or mistyped key) or json.JSONDecodeError.
    """Tolerantly parse SAVE_SETUP JSON. Returns dict or raises.

    Uses `JSONDecoder.raw_decode()` so trailing text after the first complete
    JSON object is ignored — some models append a stray sentence, a second
    JSON object, or a literal `</invoke>` after the payload.
    """
    s = input_str.strip()
    # Strip code fences if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    # Find the first `{` and parse just the first complete object from there.
    brace = s.find("{")
    if brace == -1:
        raise ValueError("SAVE_SETUP input contains no JSON object")
    payload, _ = json.JSONDecoder().raw_decode(s[brace:])
    if not isinstance(payload, dict):
        raise ValueError("SAVE_SETUP input must be a JSON object")
    for key in ("rebuild_cmds", "test_cmds"):
        if key not in payload:
            raise ValueError(f"SAVE_SETUP missing required key: {key!r}")
        if not isinstance(payload[key], list) or not all(isinstance(c, str) for c in payload[key]):
            raise ValueError(f"SAVE_SETUP {key!r} must be a list[str]")
    return payload


def run_agent_loop(
    sandbox: DockerSandbox,
    *,
    repo: str,
    ref: str,
    language: LanguageHint,
    base_image: str,
    llm: LLMSpec,
    max_iterations: int = 20,
    max_seconds: int = 1800,
    max_spend_usd: float | None = None,
    platform: str = "linux/amd64",
    on_turn: Callable[[AgentTurn, float], None] | None = None,
    on_thinking: Callable[[int], None] | None = None,
    on_executing: Callable[[int, AgentAction], None] | None = None,
) -> AgentOutcome:
    # [FUNCTION DOC] Drives the bootstrap agent. Inputs: sandbox (running DockerSandbox); repo, ref
    # (for the initial prompt); language, base_image, platform (for the system prompt); llm
    # (LLMSpec); max_iterations (default 20); max_seconds (default 1800, checked at the start of
    # each step only); max_spend_usd (None = unlimited; stops once accumulated cost >= limit,
    # checked before each call); optional callbacks on_thinking(step), on_executing(step, action)
    # for executable tool actions, and on_turn(turn, total_cost) after each executed tool turn;
    # callback exceptions are logged at debug and swallowed. Each step concatenates the full history
    # (initial prompt plus every prior action/observation and correction message) into one user
    # message and calls llm.complete (max_tokens 2048, temperature 0.2). INVALID: appends a format-
    # reminder and continues. GIVE_UP: returns failure. SAVE_SETUP: returns success with the parsed
    # commands, or appends a rejection message and continues if parsing fails. Tool actions:
    # execute, append the observation to history, record the turn. Output: AgentOutcome (failure
    # reasons: timeout, cost budget exceeded, agent gave up, hit max_iterations). Raises: whatever
    # llm.complete raises. Side effects: LLM calls (cost), commands in the container, logging.
    # KNOWN ISSUE: turns recorded for INVALID and rejected SAVE_SETUP responses carry no cost/token
    # data and do not trigger on_turn, so per-turn costs in the transcript do not sum to the total.
    """Drive the bootstrap agent until it succeeds, gives up, or hits a budget."""
    system = system_prompt(language=language, base_image=base_image, platform=platform)
    history: list[str] = [initial_user_prompt(repo=repo, ref=ref)]
    transcript: list[AgentTurn] = []
    total_cost = 0.0
    start = time.monotonic()

    for step in range(max_iterations):
        if time.monotonic() - start > max_seconds:
            return AgentOutcome(
                success=False,
                reason=f"timeout after {max_seconds}s",
                iterations=step,
                transcript=transcript,
                total_cost_estimate_usd=total_cost,
            )
        if max_spend_usd is not None and total_cost >= max_spend_usd:
            return AgentOutcome(
                success=False,
                reason=f"cost budget exceeded: ${total_cost:.4f} ≥ ${max_spend_usd:.2f}",
                iterations=step,
                transcript=transcript,
                total_cost_estimate_usd=total_cost,
            )

        if on_thinking is not None:
            try:
                on_thinking(step)
            except Exception as exc:
                logger.debug("on_thinking callback failed: %s", exc)

        user_msg = "\n\n".join(history)
        turn_start = time.monotonic()
        response = complete(llm, system=system, user=user_msg, max_tokens=2048, temperature=0.2)
        total_cost += response.cost_usd
        thought, action = parse_action(response.content)
        logger.info(
            "step=%d action=%s cost=$%.4f thought=%.80s",
            step,
            action.name,
            response.cost_usd,
            thought,
        )

        if on_executing is not None and action.name not in ("INVALID", "GIVE_UP", "SAVE_SETUP"):
            try:
                on_executing(step, action)
            except Exception as exc:
                logger.debug("on_executing callback failed: %s", exc)

        if action.name == "INVALID":
            history.append(
                "Your previous response did not match the expected format.\n"
                "Re-emit using EXACTLY:\n  Thought: ...\n  Action: BASH|READ_FILE|LIST_DIR|SAVE_SETUP|GIVE_UP\n  Input: ..."
            )
            transcript.append(AgentTurn(step, thought, action, "format-error"))
            continue

        if action.name == "GIVE_UP":
            return AgentOutcome(
                success=False,
                reason=f"agent gave up: {action.input[:200]}",
                iterations=step + 1,
                transcript=transcript,
                total_cost_estimate_usd=total_cost,
            )

        if action.name == "SAVE_SETUP":
            try:
                payload = _parse_save_setup(action.input)
            except (json.JSONDecodeError, ValueError) as exc:
                history.append(
                    f"SAVE_SETUP rejected: {exc}. Re-emit with valid JSON containing rebuild_cmds and test_cmds."
                )
                transcript.append(AgentTurn(step, thought, action, f"save-rejected: {exc}"))
                continue
            return AgentOutcome(
                success=True,
                reason="agent declared success",
                rebuild_cmds=payload["rebuild_cmds"],
                test_cmds=payload["test_cmds"],
                summary=payload.get("summary", ""),
                iterations=step + 1,
                transcript=transcript,
                total_cost_estimate_usd=total_cost,
            )

        # BASH / READ_FILE / LIST_DIR
        observation = _execute(action, sandbox)
        history.append(
            f"Action: {action.name}\nInput: {action.input}\n\nObservation:\n{observation}\n"
        )
        turn = AgentTurn(
            step=step,
            thought=thought,
            action=action,
            observation=observation,
            cost_estimate_usd=response.cost_usd,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            duration_sec=time.monotonic() - turn_start,
        )
        transcript.append(turn)
        if on_turn is not None:
            try:
                on_turn(turn, total_cost)
            except Exception as exc:
                logger.debug("on_turn callback failed: %s", exc)

    return AgentOutcome(
        success=False,
        reason=f"hit max_iterations={max_iterations}",
        iterations=max_iterations,
        transcript=transcript,
        total_cost_estimate_usd=total_cost,
    )
