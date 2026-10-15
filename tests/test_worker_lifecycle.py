# [MODULE DOC] Tests for execution/lifecycle worker provisioning and teardown: receipts
# survive lost creation responses, budget reservations are not silently zeroed on cleanup,
# and remote command failures do not leak output into exception messages.
from __future__ import annotations

import json

import pytest

from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.execution.base import CommandResult, WorkerSpec
from repo2rlenv.execution.lifecycle import provision_worker, stop_worker


class FakeWorker:
    # [CLASS DOC] Minimal stand-in for a provider worker: fixed id 'test-worker' and a no-op
    # terminate(), used to stub create_worker/connect_worker.
    id = "test-worker"

    def terminate(self):
        # [FUNCTION DOC] No-op terminate. Inputs: none. Output: None. Side effects: none.
        pass


def test_lost_creation_response_preserves_identity_and_reservation(tmp_path, monkeypatch):
    # [FUNCTION DOC] Checks that when worker creation raises (lost provider response),
    # provision_worker leaves a receipt workers/<name>.json in state 'creation_uncertain' with the
    # spec, keeps the $2 ledger reservation, re-raises TimeoutError, and refuses a second provision
    # of the same name with FileExistsError. Inputs: tmp_path, monkeypatch fixtures.
    ledger = BudgetLedger(tmp_path / "budget.db", limit_usd="5")

    def lost_response(spec):
        # [FUNCTION DOC] HELPER: Stub for create_worker that always raises TimeoutError.
        raise TimeoutError("provider response lost")

    monkeypatch.setattr("repo2rlenv.execution.lifecycle.create_worker", lost_response)
    spec = WorkerSpec(name="recoverable")
    with pytest.raises(TimeoutError):
        provision_worker(spec, tmp_path / "workers", ledger, reserve_usd="2")
    receipt = json.loads((tmp_path / "workers/recoverable.json").read_text())
    assert receipt["state"] == "creation_uncertain"
    assert receipt["spec"]["name"] == "recoverable"
    assert ledger.status()["reserved_usd"] == "2.000000"
    with pytest.raises(FileExistsError):
        provision_worker(spec, tmp_path / "workers", ledger, reserve_usd="2")


def test_cleanup_does_not_invent_zero_cost(tmp_path, monkeypatch):
    # [FUNCTION DOC] Checks stop_worker terminates the worker without settling cost: the
    # reservation stays and accounted cost is 0 until ledger.settle records 0.50, after which
    # remaining is 4.50; a second stop_worker is idempotent and returns the same receipt.
    worker = FakeWorker()
    monkeypatch.setattr("repo2rlenv.execution.lifecycle.create_worker", lambda spec: worker)
    monkeypatch.setattr("repo2rlenv.execution.lifecycle.connect_worker", lambda *args: worker)
    ledger = BudgetLedger(tmp_path / "budget.db", limit_usd="5")
    _, path = provision_worker(
        WorkerSpec(name="cleanup"), tmp_path / "workers", ledger, reserve_usd="2"
    )
    receipt = stop_worker(path, ledger)
    assert receipt["state"] == "terminated"
    assert ledger.status()["reserved_usd"] == "2.000000"
    assert ledger.status()["accounted_usd"] == "0.000000"
    ledger.settle(receipt["operation_id"], "0.50", evidence="recorded rate estimate")
    assert ledger.status()["remaining_usd"] == "4.500000"
    assert stop_worker(path, ledger) == receipt


def test_command_failure_does_not_leak_output():
    # [FUNCTION DOC] Checks CommandResult.checked raises RemoteCommandError on non-zero exit
    # without including command output in the message, while attaching the result object.
    result = CommandResult(1, "credential: example-sensitive-value")
    from repo2rlenv.execution.base import RemoteCommandError

    with pytest.raises(RemoteCommandError) as caught:
        result.checked("bootstrap")
    assert "sensitive" not in str(caught.value)
    assert caught.value.result is result
