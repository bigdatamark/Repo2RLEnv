# [MODULE DOC] Controller-side helpers that launch a supervised remote job once (detached with
# nohup) and poll its durable job.json outcome, optionally streaming progress events.
"""Launch once and observe remote jobs without coupling them to an API stream."""

from __future__ import annotations

import json
import shlex
import time
from collections.abc import Callable

from repo2rlenv.campaigns.events import ProgressEvent
from repo2rlenv.execution.base import RemoteWorker
from repo2rlenv.execution.read_retry import retry_read


def launch_job(
    worker: RemoteWorker,
    directory: str,
    command: list[str],
    *,
    timeout_sec: int,
    env: dict[str, str] | None = None,
    python: str = "python",
) -> None:
    # [FUNCTION DOC] Starts `python -m repo2rlenv.execution.job --directory <dir> --timeout-sec <n>
    # -- <command>` detached in the background on worker via sh -c with nohup and null stdio (30s
    # exec timeout). Passes env plus REPO2RLENV_REMOTE_WORKER=1. Inputs: worker, directory (remote),
    # command, timeout_sec, env (optional), python (default "python"). Raises RemoteCommandError if
    # the launch shell fails. Returns None.
    supervisor = [
        python,
        "-m",
        "repo2rlenv.execution.job",
        "--directory",
        directory,
        "--timeout-sec",
        str(timeout_sec),
        "--",
        *command,
    ]
    # Arguments are individually shell-quoted. No credentials or source content
    # appears in this command; the child inherits the remote-only guard.
    shell = "nohup " + shlex.join(supervisor) + " >/dev/null 2>&1 </dev/null &"
    worker.exec(
        ["sh", "-c", shell],
        timeout=30,
        env={**(env or {}), "REPO2RLENV_REMOTE_WORKER": "1"},
    ).checked("Launch remote job")


def observe_job(
    worker: RemoteWorker,
    directory: str,
    *,
    event_file: str | None = None,
    timeout_sec: int,
    on_event: Callable[[ProgressEvent], None] | None = None,
) -> dict:
    # [FUNCTION DOC] Polls <directory>/job.json every 3s (reads retried on transient errors) until
    # state is completed or failed, then returns the record dict. When event_file and on_event are
    # given, reads that remote JSON-lines file each poll and delivers newly completed lines as
    # ProgressEvent objects. Raises TimeoutError after timeout_sec without a terminal state (job is
    # not relaunched).
    deadline = time.monotonic() + timeout_sec
    delivered = 0
    while time.monotonic() < deadline:
        result = retry_read(lambda: worker.exec(["cat", directory + "/job.json"], timeout=15))
        if result.returncode == 0:
            record = json.loads(result.stdout)
            if event_file is not None and on_event is not None:
                events = retry_read(lambda: worker.exec(["cat", event_file], timeout=15))
                if events.returncode == 0:
                    lines = events.stdout.split("\n")[:-1]
                    for line in lines[delivered:]:
                        on_event(ProgressEvent.model_validate_json(line))
                    delivered = len(lines)
            if record["state"] in {"completed", "failed"}:
                return record
        time.sleep(3)
    raise TimeoutError(
        "Remote job outcome is still uncertain; resume observation without relaunching"
    )
