# [MODULE DOC] Tests for repo2rlenv.execution.job, the remote-worker job supervisor: it must refuse
# to run outside a remote worker, kill the whole process group and owned containers on timeout, and
# clean up only containers labelled with its exact job id. All subprocess/os/signal calls are
# monkeypatched; nothing real is executed.
from __future__ import annotations

import json
import subprocess

import pytest

from repo2rlenv.execution import job


def test_supervisor_refuses_local_execution(tmp_path, monkeypatch):
    # [FUNCTION DOC] Unsets REPO2RLENV_REMOTE_WORKER and asserts job.execute raises RuntimeError
    # matching 'remote worker' and creates no job directory. Inputs: tmp_path, monkeypatch fixtures.
    monkeypatch.delenv("REPO2RLENV_REMOTE_WORKER", raising=False)
    with pytest.raises(RuntimeError, match="remote worker"):
        job.execute(tmp_path / "job", ["forbidden-command"], timeout_sec=1)
    assert not (tmp_path / "job").exists()


def test_timeout_terminates_process_group_and_its_owned_containers(tmp_path, monkeypatch):
    # [FUNCTION DOC] Sets REPO2RLENV_REMOTE_WORKER=1, fakes Popen with a process whose wait(timeout)
    # raises TimeoutExpired, fakes os.killpg/SIGKILL (raising=False for Windows) and
    # _cleanup_containers. Asserts execute re-raises TimeoutExpired, writes job.json with
    # state='failed' and the cleanup result, passes REPO2RLENV_JOB_ID equal to the receipt job_id in
    # the child env, and kills pid 12345's group with SIGKILL.
    monkeypatch.setenv("REPO2RLENV_REMOTE_WORKER", "1")
    calls = []

    class Process:
        # [CLASS DOC] HELPER: Fake Popen object with fixed pid 12345.
        pid = 12345

        def wait(self, timeout=None):
            # [FUNCTION DOC] HELPER: Fake wait. With a timeout raises subprocess.TimeoutExpired;
            # without one returns -9 (killed).
            if timeout is not None:
                raise subprocess.TimeoutExpired("command", timeout)
            return -9

    def start(command, **kwargs):
        # [FUNCTION DOC] HELPER: Fake Popen; records (command, kwargs) into the enclosing calls list
        # and returns a Process.
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(job.subprocess, "Popen", start)
    # raising=False: neither os.killpg nor signal.SIGKILL exists on Windows.
    # execute() only ever runs for real inside a Linux remote worker; both are
    # faked here so the supervisor's timeout/cleanup logic can be verified
    # from any host OS.
    monkeypatch.setattr(job.os, "killpg", lambda *args: calls.append(args), raising=False)
    monkeypatch.setattr(job.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(job, "_cleanup_containers", lambda token: {"passed": True, "removed": 1})
    with pytest.raises(subprocess.TimeoutExpired):
        job.execute(tmp_path, ["remote-test-fixture"], timeout_sec=1)
    receipt = json.loads((tmp_path / "job.json").read_text())
    assert receipt["state"] == "failed"
    assert receipt["cleanup"] == {"passed": True, "removed": 1}
    assert calls[0][1]["env"]["REPO2RLENV_JOB_ID"] == receipt["job_id"]
    assert calls[1] == (12345, job.signal.SIGKILL)


def test_cleanup_filters_by_exact_job_ownership(monkeypatch):
    # [FUNCTION DOC] Fakes subprocess.run to return two container ids and asserts
    # _cleanup_containers('owned-id') lists via 'docker ps -aq --filter label=repo2rlenv.job=owned-
    # id', then 'docker rm -f' both, returning {'passed': True, 'removed': 2}.
    calls = []

    def run(argv, **kwargs):
        # [FUNCTION DOC] HELPER: Fake subprocess.run; records argv and returns a CompletedProcess
        # with exit 0 and stdout 'container1\ncontainer2\n'.
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "container1\ncontainer2\n", "")

    monkeypatch.setattr(job.subprocess, "run", run)
    assert job._cleanup_containers("owned-id") == {"passed": True, "removed": 2}
    assert calls == [
        ["docker", "ps", "-aq", "--filter", "label=repo2rlenv.job=owned-id"],
        ["docker", "rm", "-f", "container1", "container2"],
    ]
