# [MODULE DOC] Contract test for execution/daytona.py:DaytonaWorker against the installed daytona
# SDK: verifies the sandbox-creation request uses the SDK's real parameter/resource/image model
# types without creating a cloud sandbox (the Daytona client is monkeypatched). Skipped when the
# daytona package is absent.
"""Use Daytona's real image and request models without creating a cloud sandbox."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from repo2rlenv.execution.base import WorkerSpec
from repo2rlenv.execution.daytona import DaytonaWorker


@pytest.mark.parametrize("image", [None, "ubuntu:24.04"])
def test_worker_request_matches_installed_sdk(monkeypatch, image):
    # [FUNCTION DOC] Parametrized over image None (expects an SDK Image object built by the worker)
    # and 'ubuntu:24.04' (passed through as a string). Replaces daytona.Daytona with a stub whose
    # create() validates params and returns a fake sandbox, then calls DaytonaWorker.create with
    # memory_mb=4097 and timeout_sec=600 and asserts the worker holds the stub client and sandbox
    # and that create was called once. Inputs: monkeypatch, image. Side effects: monkeypatches the
    # daytona module attribute.
    daytona = pytest.importorskip("daytona")
    sandbox = SimpleNamespace(id="sdk-contract")
    calls = []

    def create(params, *, timeout):
        # [FUNCTION DOC] HELPER: Fake Daytona.create. Records params and asserts they are
        # CreateSandboxFromImageParams with Resources memory 5 (GiB, rounded up from 4097 MB),
        # os_user root, ephemeral, auto_stop 10, auto_delete 0, timeout 180, and the expected
        # image. Returns the fake sandbox.
        calls.append(params)
        assert isinstance(params, daytona.CreateSandboxFromImageParams)
        assert isinstance(params.resources, daytona.Resources)
        assert params.resources.memory == 5
        assert params.os_user == "root"
        assert params.ephemeral is True
        assert params.auto_stop_interval == 10
        assert params.auto_delete_interval == 0
        assert timeout == 180
        if image is None:
            assert isinstance(params.image, daytona.Image)
        else:
            assert params.image == image
        return sandbox

    client = SimpleNamespace(create=create)
    monkeypatch.setattr(daytona, "Daytona", lambda: client)
    worker = DaytonaWorker.create(
        WorkerSpec(
            provider="daytona", name="sdk-contract", memory_mb=4097, timeout_sec=600, image=image
        )
    )
    assert worker.client is client and worker.sandbox is sandbox
    assert len(calls) == 1
