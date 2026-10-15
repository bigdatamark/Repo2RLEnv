# [MODULE DOC] Tests for RunBudget.reserve_with_wait in repo2rlenv.quality.loop.client: waiting on a
# shared budget scope only for recoverable sibling reservation pressure, bounded by a deadline,
# never replaying an already-recorded operation, and never waiting for non-recoverable denials. Also
# checks the LoopOptions shared_reservation_wait_sec bounds. Uses a real SQLite BudgetLedger in
# tmp_path; time is faked.
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from repo2rlenv.campaigns.budget import BudgetExceeded, BudgetLedger, OperationAlreadyRecorded
from repo2rlenv.quality.loop import client
from repo2rlenv.quality.loop.client import RunBudget
from repo2rlenv.quality.loop.models import LoopOptions


def budgets(tmp_path, *, global_cap="100", candidate_cap="10", quality_cap="10", shared=True):
    # [FUNCTION DOC] HELPER: Creates a BudgetLedger (global_cap, default '100') and nested
    # RunBudgets: batch ('batch-', cap 5, shared flag), candidate ('batch-child-', candidate_cap)
    # and quality ('batch-child-quality-', quality_cap). Returns (ledger, batch, candidate,
    # quality).
    ledger = BudgetLedger(tmp_path / "budget.db", limit_usd=global_cap)
    batch = RunBudget(ledger, "batch-", "5", shared=shared)
    candidate = RunBudget(batch, "batch-child-", candidate_cap)
    quality = RunBudget(candidate, "batch-child-quality-", quality_cap)
    return ledger, batch, candidate, quality


def test_settlement_on_another_connection_unblocks_exactly_one_reservation(tmp_path):
    # [FUNCTION DOC] A sibling reserves 4 of 5; a thread waits for the on_wait callback then settles
    # it to 1 via a fresh ledger connection. The waiting reserve of 3 then succeeds; ledger shows 1
    # accounted and 3 reserved, and retrying the same operation id raises OperationAlreadyRecorded.
    ledger, batch, _, quality = budgets(tmp_path)
    batch.reserve("batch-sibling-worker", "4", "worker")
    waiting = Event()

    def settle():
        # [FUNCTION DOC] HELPER: Waits up to 3s for the waiting Event, then settles the sibling
        # operation at 1 USD on a new BudgetLedger connection.
        assert waiting.wait(3)
        BudgetLedger(ledger.path).settle("batch-sibling-worker", "1", evidence="usage")

    with ThreadPoolExecutor(max_workers=1) as pool:
        settlement = pool.submit(settle)
        quality.reserve_with_wait(
            "batch-child-quality-solver",
            "3",
            "solver",
            timeout_sec=3,
            on_wait=lambda exc: waiting.set(),
        )
        settlement.result(timeout=3)
    status = ledger.status()
    assert status["accounted_usd"] == "1.000000"
    assert status["reserved_usd"] == "3.000000"
    with pytest.raises(OperationAlreadyRecorded):
        quality.reserve_with_wait("batch-child-quality-solver", "3", "retry", timeout_sec=3)
    assert len(status["operations"]) == 2


def test_deadline_does_not_release_sibling_hold_or_allocate(tmp_path, monkeypatch):
    # [FUNCTION DOC] With fake monotonic/sleep, waiting for 2s then raises BudgetExceeded with a
    # 'reservation_pressure' denial on 'batch-' carrying exact amounts; the clock advanced exactly 2
    # and only the sibling's operation exists.
    ledger, batch, _, quality = budgets(tmp_path)
    batch.reserve("batch-sibling-worker", "4", "worker")
    clock = [0.0]
    monkeypatch.setattr(client.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        client.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    with pytest.raises(BudgetExceeded) as caught:
        quality.reserve_with_wait("batch-child-quality-solver", "3", "solver", timeout_sec=2)
    assert clock[0] == 2
    assert caught.value.denials[0].as_dict() == {
        "scope": "batch-",
        "reason": "reservation_pressure",
        "requested_usd": "3.000000",
        "limit_usd": "5.000000",
        "accounted_usd": "0.000000",
        "reserved_usd": "4.000000",
        "uncertain_usd": "0.000000",
        "remaining_usd": "1.000000",
    }
    assert len(ledger.status()["operations"]) == 1
    assert ledger.status()["reserved_usd"] == "4.000000"


@pytest.mark.parametrize(
    "kind",
    ["global", "candidate", "quality", "own_hold", "uncertain", "spent", "unmarked", "disabled"],
)
def test_nonrecoverable_or_disabled_limits_fail_without_sleep(tmp_path, monkeypatch, kind):
    # [FUNCTION DOC] Parametrized kinds (global/candidate/quality caps too small, own hold,
    # uncertain, spent, unshared scope, timeout 0) must raise BudgetExceeded without sleeping;
    # checks denial scopes for cap cases and that no new operation was recorded.
    ledger, batch, candidate, quality = budgets(
        tmp_path,
        global_cap="5" if kind == "global" else "100",
        candidate_cap="2" if kind == "candidate" else "10",
        quality_cap="2" if kind == "quality" else "10",
        shared=kind != "unmarked",
    )
    operation = "batch-child-worker" if kind == "own_hold" else "batch-sibling-worker"
    batch.reserve(operation, "4", "worker")
    if kind == "uncertain":
        ledger.mark_uncertain(operation, "uncertain effect")
    if kind == "spent":
        ledger.settle(operation, "4", evidence="usage")
    monkeypatch.setattr(client.time, "sleep", lambda _: pytest.fail("Must not wait on this denial"))
    with pytest.raises(BudgetExceeded) as caught:
        quality.reserve_with_wait(
            "batch-child-quality-solver",
            "3",
            "solver",
            timeout_sec=0 if kind == "disabled" else 2,
        )
    scopes = {denial.scope for denial in caught.value.denials}
    if kind == "candidate":
        assert scopes == {"batch-", candidate.prefix}
    if kind == "quality":
        assert scopes == {"batch-", quality.prefix}
    if kind == "global":
        assert scopes == {None, "batch-"}
    assert len(ledger.status()["operations"]) == 1


def test_same_operation_claimed_during_wait_is_never_replayed(tmp_path, monkeypatch):
    # [FUNCTION DOC] Fake sleep settles the sibling and lets another controller reserve the same
    # operation id; reserve_with_wait must raise OperationAlreadyRecorded and leave exactly two
    # operations.
    ledger, batch, _, quality = budgets(tmp_path)
    batch.reserve("batch-sibling-worker", "4", "worker")

    def contender(_):
        # [FUNCTION DOC] HELPER: Fake sleep that settles the sibling to 1 and reserves 'batch-child-
        # quality-solver' itself.
        ledger.settle("batch-sibling-worker", "1", evidence="usage")
        quality.reserve("batch-child-quality-solver", "3", "other controller admitted")

    monkeypatch.setattr(client.time, "sleep", contender)
    with pytest.raises(OperationAlreadyRecorded):
        quality.reserve_with_wait("batch-child-quality-solver", "3", "solver", timeout_sec=2)
    assert ledger.status()["reserved_usd"] == "3.000000"
    assert len(ledger.status()["operations"]) == 2


def test_legacy_exception_is_not_interpreted_as_retryable(tmp_path, monkeypatch):
    # [FUNCTION DOC] If reserve raises a BudgetExceeded without typed denials, reserve_with_wait re-
    # raises that same object without sleeping.
    _, _, _, quality = budgets(tmp_path)
    legacy = BudgetExceeded("legacy denial")

    def denied(*args):
        # [FUNCTION DOC] HELPER: Fake reserve that always raises the legacy BudgetExceeded instance.
        raise legacy

    monkeypatch.setattr(quality, "reserve", denied)
    monkeypatch.setattr(client.time, "sleep", lambda _: pytest.fail("No typed capacity evidence"))
    with pytest.raises(BudgetExceeded, match="legacy denial") as caught:
        quality.reserve_with_wait("batch-child-quality-solver", "3", "solver", timeout_sec=2)
    assert caught.value is legacy
    assert legacy.denials == ()


def test_ordinary_reserve_never_waits_even_for_a_shared_scope(tmp_path, monkeypatch):
    # [FUNCTION DOC] Plain reserve under pressure raises BudgetExceeded immediately without
    # sleeping.
    _, batch, _, quality = budgets(tmp_path)
    batch.reserve("batch-sibling-worker", "4", "worker")
    monkeypatch.setattr(
        client.time, "sleep", lambda _: pytest.fail("Only CPU solver opts into wait")
    )
    with pytest.raises(BudgetExceeded):
        quality.reserve("batch-child-quality-review", "3", "review model")


def test_insufficient_shared_limit_fails_without_wait(tmp_path, monkeypatch):
    # [FUNCTION DOC] A request larger than the scope limit raises with reason
    # 'request_exceeds_limit', no sleep, and no operations recorded.
    ledger, _, _, quality = budgets(tmp_path)
    monkeypatch.setattr(client.time, "sleep", lambda _: pytest.fail("Request can never fit"))
    with pytest.raises(BudgetExceeded) as caught:
        quality.reserve_with_wait("batch-child-quality-solver", "6", "solver", timeout_sec=2)
    assert caught.value.denials[0].reason == "request_exceeds_limit"
    assert not ledger.status()["operations"]


def test_capacity_arriving_after_deadline_cannot_admit_work(tmp_path, monkeypatch):
    # [FUNCTION DOC] Fake sleep advances past the deadline and then frees capacity; the reservation
    # must still fail and only the sibling operation exists.
    ledger, batch, _, quality = budgets(tmp_path)
    batch.reserve("batch-sibling-worker", "4", "worker")
    clock = [0.0]

    def late_settlement(seconds):
        # [FUNCTION DOC] HELPER: Fake sleep advancing the clock by seconds+0.1 and settling the
        # sibling to 1.
        clock[0] += seconds + 0.1
        ledger.settle("batch-sibling-worker", "1", evidence="late usage")

    monkeypatch.setattr(client.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(client.time, "sleep", late_settlement)
    with pytest.raises(BudgetExceeded):
        quality.reserve_with_wait("batch-child-quality-solver", "3", "solver", timeout_sec=1)
    assert len(ledger.status()["operations"]) == 1


def test_wait_configuration_is_bounded():
    # [FUNCTION DOC] LoopOptions default wait is 30, 0 is allowed, and -1, 301, True and 0.1 raise
    # ValueError.
    assert LoopOptions().shared_reservation_wait_sec == 30
    assert LoopOptions(shared_reservation_wait_sec=0).shared_reservation_wait_sec == 0
    for value in (-1, 301, True, 0.1):
        with pytest.raises(ValueError):
            LoopOptions(shared_reservation_wait_sec=value)
