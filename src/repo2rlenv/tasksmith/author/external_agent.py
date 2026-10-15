# [MODULE DOC] Runs an external Node-based author agent runtime ("pi" or "opencode") as a
# subprocess, wired to an AgentBridge so the runtime only talks to a loopback, metered model/tool
# proxy and never sees provider keys.
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
from pathlib import Path

from repo2rlenv.tasksmith.author.bridge import AgentBridge
from repo2rlenv.tasksmith.author.inference import (
    MAX_OUTPUT_TOKENS,
    MODEL_TIMEOUT_SEC,
    anthropic_options,
    inference_settings,
)


def runtime_path(engine: str) -> Path:
    # [FUNCTION DOC] Returns the Path to the <engine>.mjs adapter inside the installed tasksmith
    # runtime directory. Raises ValueError for engines other than "pi"/"opencode", and RuntimeError
    # if node_modules is missing or `node` is not on PATH (asks the user to run install-runtime).
    if engine not in {"pi", "opencode"}:
        raise ValueError(f"Unknown external runtime: {engine}")
    from repo2rlenv.tasksmith.runtime import directory

    root = directory()
    if not (root / "node_modules").is_dir() or not shutil.which("node"):
        raise RuntimeError(
            f"Install Node >=22.19 and run `repo2rlenv tasksmith install-runtime` for the {engine} adapter"
        )
    return root / f"{engine}.mjs"


async def run_external_agent(
    *, engine, model, system, prompt, budget, tools, handlers, trace, max_turns, max_cost=8
):
    # [FUNCTION DOC] Runs one external author session to completion and returns {"messages",
    # "turns", "cost"}. Inputs (keyword-only): engine, model ("anthropic/..."), system and prompt
    # strings, budget (AuthorBudget), tools (schemas), handlers (tool name -> async callable), trace
    # (JSONL Path), max_turns, max_cost (USD, default 8). Creates a session dir next to the trace
    # ("<stem>-runtime"), starts an AgentBridge, records an "input" trace entry, writes runner-
    # config.json (mode 0600) containing model, prompts, bridge URL/token, tools, limits and
    # inference options, and launches `node <adapter> <config>` in a new process session with a
    # minimal environment (PATH/LANG/LC_ALL/TMPDIR, HOME set to a private dir) and stderr to
    # runtime-stderr.log. Waits up to 3600s for either process completion or a bridge failure.
    # Writes stdout to runtime-result.json and parses it as JSON. Raises: the bridge's stored
    # failure; TimeoutError after one hour; RuntimeError on nonzero exit or an "error" key in the
    # result; JSON errors on bad stdout. Cleanup always: stops the bridge (cancelling model calls
    # unless completed), SIGTERMs the process group then SIGKILL after 12s, cancels helper tasks and
    # deletes the config file. After the bridge exits (draining streams), re-raises any late bridge
    # failure.
    adapter = runtime_path(engine)
    trace.parent.mkdir(parents=True, exist_ok=True)
    session_dir = trace.with_suffix("").with_name(trace.stem + "-runtime").resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    config_path = session_dir / "runner-config.json"
    async with AgentBridge(
        model=model,
        budget=budget,
        tools=tools,
        handlers=handlers,
        trace=trace,
        max_turns=max_turns,
        max_cost=max_cost,
    ) as bridge:
        bridge.record(
            "input",
            model=model,
            system=system,
            prompt=prompt,
            runtime=engine,
            inference=inference_settings(model),
            timeout_sec=MODEL_TIMEOUT_SEC,
        )
        config = {
            "model": model.split("/", 1)[1],
            "system": system,
            "prompt": prompt,
            "bridge_url": bridge.url,
            "bridge_token": bridge.token,
            "tools": tools,
            "session_dir": str(session_dir),
            "max_turns": max_turns,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "inference_options": anthropic_options(model),
            "model_timeout_sec": MODEL_TIMEOUT_SEC,
        }
        config_path.write_text(json.dumps(config))
        config_path.chmod(0o600)
        env = {
            key: os.environ[key]
            for key in ("PATH", "LANG", "LC_ALL", "TMPDIR")
            if key in os.environ
        }
        env["HOME"] = str(session_dir / "home")
        Path(env["HOME"]).mkdir(exist_ok=True)
        process = None
        failed = None
        communication = None
        completed = False
        try:
            with (session_dir / "runtime-stderr.log").open("wb") as errors:
                process = await asyncio.create_subprocess_exec(
                    shutil.which("node"),
                    str(adapter),
                    str(config_path),
                    cwd=session_dir,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=errors,
                    start_new_session=True,
                )
                communication = asyncio.create_task(process.communicate())
                failed = asyncio.create_task(bridge.failed.wait())
                done, _ = await asyncio.wait(
                    [communication, failed], timeout=3600, return_when=asyncio.FIRST_COMPLETED
                )
                if bridge.failure:
                    raise bridge.failure
                if communication not in done:
                    raise TimeoutError(f"{engine} author runtime exceeded one hour")
                stdout, _ = communication.result()
                (session_dir / "runtime-result.json").write_bytes(stdout)
                if process.returncode:
                    raise RuntimeError(
                        f"{engine} runtime exited {process.returncode}: "
                        + stdout.decode(errors="replace")[-2000:]
                    )
                result = json.loads(stdout)
                if result.get("error"):
                    raise RuntimeError(result["error"])
                completed = True
        finally:
            bridge.stop(cancel_models=not completed)
            if process is not None and process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), timeout=12)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
            for task in (communication, failed):
                if task is not None and not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            config_path.unlink(missing_ok=True)
    # __aexit__ drains successful provider streams, including settlement or any
    # error that arrives after the runtime has already produced its final JSON.
    if bridge.failure:
        raise bridge.failure
    return {"messages": result["messages"], "turns": bridge.turns, "cost": bridge.cost}
