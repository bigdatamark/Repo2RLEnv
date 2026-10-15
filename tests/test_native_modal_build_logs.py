# [MODULE DOC] Tests for image-build failure diagnostics in MeteredModalEnvironment._create_sandbox:
# on ImageBuildError the allocation is settled first (state build_failed, reservation released),
# then failed-image logs are fetched with the same Modal client, redacted, bounded, persisted with
# sha256 receipts, and appended to the original error, which is always re-raised. Skipped unless
# harbor and modal are importable.
"""Failed image diagnostics must preserve settlement and the original SDK error."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")
pytest.importorskip("modal")

import modal

from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.execution import harbor_modal
from repo2rlenv.execution.harbor_modal import MeteredModalEnvironment, NativeAccounting
from repo2rlenv.quality.loop.client import RunBudget


@pytest.fixture
def failed_build(monkeypatch, tmp_path):
    # [FUNCTION DOC] Fixture: builds a MeteredModalEnvironment (bypassing __init__) with a 10 USD
    # ledger, patches Sandbox.create to raise an ImageBuildError and Client.from_env to assert
    # settlement happened first. Returns a namespace with env, ledger, error, calls, client and
    # install_logs(entries) which patches Image.from_id to serve the given async log generator.
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="10")
    accounting = NativeAccounting(RunBudget(ledger, "build-test", "10"), tmp_path / "receipts")
    env = object.__new__(MeteredModalEnvironment)
    env.accounting, env.receipt = accounting, accounting.directory / "allocation.json"
    env.session_id, env._app_name, env._app, env._image = "unique", "test", object(), object()
    env.task_env_config = SimpleNamespace(cpus=2, memory_mb=4096, gpus=1)
    env.override_gpus, env._sandbox_timeout = None, 900
    monkeypatch.setattr(env, "_gpu_config", lambda: "L4:1")
    monkeypatch.setattr(env, "_sandbox_labels", lambda: {})
    error = modal.exception.ImageBuildError("Image build failed", "im-failed-layer")
    calls = []
    client = object()

    async def create(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Fake Sandbox.create recording 'create' and raising the build error.
        calls.append("create")
        raise error

    async def from_env():
        # [FUNCTION DOC] HELPER: Fake Client.from_env asserting no reservation remains and the
        # receipt says build_failed, then recording 'client' and returning the shared client.
        assert ledger.status()["reserved_usd"] == "0.000000"
        assert json.loads(env.receipt.read_text())["state"] == "build_failed"
        calls.append("client")
        return client

    monkeypatch.setattr(modal.Sandbox, "create", SimpleNamespace(aio=create))
    monkeypatch.setattr(modal.Client, "from_env", SimpleNamespace(aio=from_env))

    def install_logs(entries):
        # [FUNCTION DOC] HELPER: Patches modal.Image.from_id with a fake that serves the given log-
        # entry generator.
        def from_id(image_id, *, client):
            # [FUNCTION DOC] HELPER: Fake Image.from_id asserting the failed image id and the same
            # client, recording 'logs', and returning an object whose logs.fetch.aio is the entries
            # generator.
            assert image_id == error.image_id
            assert client is fixture.client
            calls.append("logs")
            return SimpleNamespace(logs=SimpleNamespace(fetch=SimpleNamespace(aio=entries)))

        monkeypatch.setattr(modal.Image, "from_id", from_id)

    fixture = SimpleNamespace(
        env=env, ledger=ledger, error=error, calls=calls, client=client, install_logs=install_logs
    )
    return fixture


def entry(message, source="stderr"):
    # [FUNCTION DOC] HELPER: Builds a fake log entry with message and source (default 'stderr').
    return SimpleNamespace(message=message, source=source)


def fetch_receipt(env):
    # [FUNCTION DOC] HELPER: Verifies the allocation's build_logs path/sha256, the fetch receipt's
    # logs_receipt hash and each stream file hash; returns the parsed fetch.json dict. Raises
    # AssertionError on mismatch.
    allocation = json.loads(env.receipt.read_text())
    path = env.receipt.with_suffix(".image-build") / "fetch.json"
    assert allocation["build_logs"] == {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    data = json.loads(path.read_text())
    logs = path.parent / "build.logs.json"
    assert data["logs_receipt"] == str(logs.resolve())
    assert data["logs_receipt_sha256"] == hashlib.sha256(logs.read_bytes()).hexdigest()
    for stream in json.loads(logs.read_text())["streams"].values():
        content = (path.parent / stream["path"]).read_bytes()
        assert stream["sha256"] == hashlib.sha256(content).hexdigest()
    return data


@pytest.mark.asyncio
async def test_causal_logs_use_failed_image_and_same_client_after_settlement(
    failed_build, monkeypatch
):
    # [FUNCTION DOC] Asserts the original error is re-raised enriched with the causal line, length-
    # bounded, secrets/credentials redacted everywhere, call order create/client/logs, receipt
    # complete, allocation marked no_sandbox_dispatched, budget settled, and a retry raises 'already
    # claimed'.
    case = failed_build
    secret = "secret-token-for-build-diagnostics"
    monkeypatch.setenv("BUILD_TEST_TOKEN", secret)
    case.error.args = ("Image build failed: " + secret,)

    async def logs(*, layers):
        # [FUNCTION DOC] HELPER: Async log generator emitting split secrets, credentials, the causal
        # error and a large repeated block.
        assert layers is None
        yield entry("Preparing packages\n" + secret[:12], "stdout")
        yield entry(secret[12:] + "\nhttps://user:password@example.invalid/pkg\n", "stdout")
        yield entry("Authorization: Bearer confidential-value\n", "system")
        yield entry("ERROR: No matching distribution found for tokenizers==0.23.0\n")
        yield entry("RUN install repeated configuration\n" * 1000)

    case.install_logs(logs)
    with pytest.raises(modal.exception.ImageBuildError) as caught:
        await case.env._create_sandbox()
    assert caught.value is case.error
    assert caught.value.image_id == "im-failed-layer"
    assert "No matching distribution found for tokenizers==0.23.0" in str(caught.value)
    assert len(str(caught.value)) < 6500
    assert case.calls == ["create", "client", "logs"]
    data = fetch_receipt(case.env)
    assert data["status"] == "complete"
    assert data["image_id"] == case.error.image_id and data["layers"] is None
    retained = "\n".join(
        path.read_text() for path in case.env.receipt.with_suffix(".image-build").iterdir()
    ) + str(caught.value)
    assert secret not in retained
    assert "user:password" not in retained and "confidential-value" not in retained
    assert "[REDACTED]" in retained
    allocation = json.loads(case.env.receipt.read_text())
    assert allocation["no_sandbox_dispatched"] is True
    assert allocation["image_id"] == case.error.image_id
    assert case.ledger.status()["reserved_usd"] == "0.000000"
    assert float(case.ledger.status()["accounted_usd"]) >= 0.25
    with pytest.raises(ValueError, match="already claimed"):
        await case.env._create_sandbox()
    assert case.calls.count("create") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", ["bytes", "entries"])
async def test_log_limits_close_stream_and_mark_incomplete(failed_build, monkeypatch, bound):
    # [FUNCTION DOC] Parametrized over byte and entry limits: asserts the stream is closed when a
    # bound is hit, the receipt is 'truncated', and the cut entry (with its secret) is not
    # persisted.
    case = failed_build
    closed = []
    secret = "sensitive-credential-cut-at-log-limit"
    monkeypatch.setenv("BUILD_TEST_TOKEN", secret)
    monkeypatch.setattr(harbor_modal, "_IMAGE_LOG_MAX_BYTES", 64 if bound == "bytes" else 1024)
    monkeypatch.setattr(harbor_modal, "_IMAGE_LOG_MAX_ENTRIES", 10 if bound == "bytes" else 2)

    async def logs(*, layers):
        # [FUNCTION DOC] HELPER: Generator yielding two entries, failing if a third is requested,
        # and recording closure.
        try:
            yield entry("ERROR: package unavailable\n")
            yield entry("prefix " + secret + "x" * 200)
            pytest.fail("A bounded stream must not request another entry")
        finally:
            closed.append(True)

    case.install_logs(logs)
    with pytest.raises(modal.exception.ImageBuildError, match="package unavailable"):
        await case.env._create_sandbox()
    assert closed == [True]
    assert fetch_receipt(case.env)["status"] == "truncated"
    stderr = (case.env.receipt.with_suffix(".image-build") / "build.stderr").read_text()
    assert "prefix" not in stderr and "sensitive" not in stderr
    assert case.ledger.status()["reserved_usd"] == "0.000000"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "stream", "client"])
async def test_log_fetch_failure_preserves_original_error_and_settlement(
    failed_build, monkeypatch, failure
):
    # [FUNCTION DOC] Parametrized over timeout/stream error/client error: asserts the original error
    # is re-raised without private detail, receipt status timeout or failed, stream closure as
    # expected, and budget still settled.
    case = failed_build
    closed = []
    monkeypatch.setattr(harbor_modal, "_IMAGE_LOG_TIMEOUT_SEC", 0.01)

    async def logs(*, layers):
        # [FUNCTION DOC] HELPER: Generator yielding a causal line then hanging (timeout case) or
        # raising ConnectionError; records closure.
        try:
            yield entry("ERROR: retained causal line\n")
            if failure == "timeout":
                await asyncio.Event().wait()
            raise ConnectionError("private-error-detail-must-not-leak")
        finally:
            closed.append(True)

    async def denied_client():
        # [FUNCTION DOC] HELPER: Fake Client.from_env raising RuntimeError.
        raise RuntimeError("private-error-detail-must-not-leak")

    case.install_logs(logs)
    if failure == "client":
        monkeypatch.setattr(modal.Client, "from_env", SimpleNamespace(aio=denied_client))
    with pytest.raises(modal.exception.ImageBuildError) as caught:
        await case.env._create_sandbox()
    assert caught.value is case.error
    assert "private-error-detail" not in str(caught.value)
    data = fetch_receipt(case.env)
    assert data["status"] == ("timeout" if failure == "timeout" else "failed")
    assert closed == ([] if failure == "client" else [True])
    if failure != "client":
        assert "retained causal line" in str(caught.value)
    assert case.ledger.status()["reserved_usd"] == "0.000000"
    assert json.loads(case.env.receipt.read_text())["state"] == "build_failed"


@pytest.mark.asyncio
async def test_log_persistence_failure_does_not_reopen_settled_build(failed_build, monkeypatch):
    # [FUNCTION DOC] Asserts an OSError from save_build_logs adds 'diagnostics unavailable
    # (OSError)' without the private message and leaves the build settled.
    case = failed_build

    async def logs(*, layers):
        # [FUNCTION DOC] HELPER: Generator yielding one error entry.
        yield entry("ERROR: package unavailable\n")

    def disk_full(*args):
        # [FUNCTION DOC] HELPER: Fake save_build_logs raising OSError.
        raise OSError("private-disk-error")

    case.install_logs(logs)
    monkeypatch.setattr(harbor_modal, "save_build_logs", disk_full)
    with pytest.raises(modal.exception.ImageBuildError) as caught:
        await case.env._create_sandbox()
    assert caught.value is case.error
    assert "diagnostics unavailable (OSError)" in str(caught.value)
    assert "private-disk-error" not in str(caught.value)
    assert json.loads(case.env.receipt.read_text())["state"] == "build_failed"
    assert case.ledger.status()["reserved_usd"] == "0.000000"


@pytest.mark.asyncio
async def test_settlement_failure_is_not_hidden_by_diagnostics(failed_build, monkeypatch):
    # [FUNCTION DOC] Asserts a settle() OSError propagates, no logs are fetched, state is
    # build_failed and the 3.00 reservation remains.
    case = failed_build

    def settlement_failed(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Fake budget.settle raising OSError.
        raise OSError("ledger write failed")

    monkeypatch.setattr(case.env.accounting.budget, "settle", settlement_failed)
    with pytest.raises(OSError, match="ledger write failed"):
        await case.env._create_sandbox()
    assert case.calls == ["create"]
    assert json.loads(case.env.receipt.read_text())["state"] == "build_failed"
    assert case.ledger.status()["reserved_usd"] == "3.000000"


@pytest.mark.asyncio
async def test_cancellation_during_log_fetch_propagates_after_settlement(failed_build):
    # [FUNCTION DOC] Asserts CancelledError raised mid-fetch propagates, closes the stream, and
    # leaves the build settled.
    case = failed_build
    closed = []

    async def logs(*, layers):
        # [FUNCTION DOC] HELPER: Generator yielding one entry then raising CancelledError; records
        # closure.
        try:
            yield entry("ERROR: package unavailable\n")
            raise asyncio.CancelledError
        finally:
            closed.append(True)

    case.install_logs(logs)
    with pytest.raises(asyncio.CancelledError):
        await case.env._create_sandbox()
    assert closed == [True]
    assert json.loads(case.env.receipt.read_text())["state"] == "build_failed"
    assert case.ledger.status()["reserved_usd"] == "0.000000"
