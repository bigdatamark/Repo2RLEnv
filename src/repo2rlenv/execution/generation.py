# [MODULE DOC] Runs one remote generator module invocation exactly once, persisting a dispatch
# receipt before launch so an interrupted run is observed (never relaunched) on resume, and
# retrieves its evidence.
"""Persist one remote generator invocation and retrieve its evidence exactly once."""

from __future__ import annotations

import json
from pathlib import Path

from repo2rlenv.execution.artifacts import unpack_evidence
from repo2rlenv.execution.jobs import launch_job, observe_job
from repo2rlenv.execution.lifecycle import save_record


def run_generator(
    worker,
    *,
    python: str,
    module: str,
    config: dict,
    directory: Path,
    job_id: str,
    timeout_sec: int,
    resume: bool,
) -> Path | None:
    # [FUNCTION DOC] Dispatches `python -m <module> <remote>/config.json <output>` on worker as a
    # supervised job, or resumes observation of an earlier dispatch. Inputs: worker (RemoteWorker),
    # python (remote interpreter), module (str), config (dict uploaded as config.json), directory
    # (local receipt dir), job_id (names /work/generation/<id> and /evidence/generation/<id>),
    # timeout_sec, resume (bool). If directory/dispatch.json exists, requires resume=True and an
    # identical configuration (worker_id, python, module, config) else raises ValueError; otherwise
    # claims the remote directory with a non-recursive mkdir (fails if taken), saves config.json and
    # dispatch.json (state dispatched) and launches via launch_job. Returns directory/job_id
    # immediately if state is downloaded, None if failed. Otherwise observes the job (timeout_sec +
    # 15), saves remote-job.json, downloads stdout.txt and stderr.txt, and if the remote output dir
    # exists tars, downloads and unpacks it (unless already unpacked). Sets state downloaded when
    # the job completed with returncode 0, else failed, saves the receipt, and returns the evidence
    # path or None. Raises RemoteCommandError / TimeoutError from remote operations.
    """Return downloaded evidence, or None for a recorded failed generation.

    A dispatch claim is durable before launch. An interrupted or uncertain launch
    is observed on resume; it is never automatically repeated.
    """
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "dispatch.json"
    expected = {"worker_id": worker.id, "python": python, "module": module, "config": config}
    remote = "/work/generation/" + job_id
    output = "/evidence/generation/" + job_id
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if not resume or record["configuration"] != expected:
            raise ValueError("Generator exists; resume its identical configuration")
    else:
        worker.exec(
            ["mkdir", "-p", "/work/generation", "/evidence/generation"], timeout=30
        ).checked("Generation parents")
        worker.exec(["mkdir", remote], timeout=30).checked("Claim remote generator")
        save_record(directory / "config.json", config)
        worker.upload(directory / "config.json", remote + "/config.json")
        record = {
            "configuration": expected,
            "state": "dispatched",
            "remote": remote,
            "output": output,
        }
        save_record(receipt, record)
        launch_job(
            worker,
            remote,
            [python, "-m", module, remote + "/config.json", output],
            timeout_sec=timeout_sec,
            python=python,
        )
    if record["state"] == "downloaded":
        return directory / job_id
    if record["state"] == "failed":
        return None
    status = observe_job(worker, remote, timeout_sec=timeout_sec + 15)
    save_record(directory / "remote-job.json", status)
    for name in ("stdout.txt", "stderr.txt"):
        worker.download(remote + "/" + name, directory / name)
    if worker.exec(["test", "-d", output], timeout=15).returncode == 0:
        worker.exec(
            ["tar", "-czf", remote + "/evidence.tar.gz", "-C", "/evidence/generation", job_id],
            timeout=120,
        ).checked("Archive generation")
        worker.download(remote + "/evidence.tar.gz", directory / "evidence.tar.gz")
        if not (directory / job_id).exists():
            unpack_evidence(directory / "evidence.tar.gz", directory, root_name=job_id)
    record["state"] = (
        "downloaded"
        if status["state"] == "completed" and status.get("returncode") == 0
        else "failed"
    )
    save_record(receipt, record)
    return directory / job_id if record["state"] == "downloaded" else None
