# [MODULE DOC] Tests for tasksmith.batch campaign scheduling with fake controllers (no target code or providers
# run): target counting and no over-dispatch, prior verified inventory checks and admission
# rechecks before allocation, GPU parallelism caps, drain requests, retained unverified outputs,
# batch and global budget headroom, reconciliation stops after crashes or pending cleanup, frozen
# configuration/runtime/controller tamper detection, orphan import, budget scope sharing in child
# runs, and real subprocess isolation of frozen controllers.
"""Campaign scheduling uses fake controllers; target code and providers never run."""

from concurrent.futures import Future
from pathlib import Path

import pytest

from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.execution.lifecycle import save_record
from repo2rlenv.quality.loop.artifacts import digest
from repo2rlenv.tasksmith import batch
from repo2rlenv.tasksmith.models import Options


def candidate(number):
    # [FUNCTION DOC] HELPER: Returns a batch.Candidate for accelerate PR number `number` with
    # Options(worker_reservation_usd 3, max_spend_usd 10).
    return batch.Candidate(
        url=f"https://github.com/huggingface/accelerate/pull/{number}",
        options=Options(worker_reservation_usd="3", max_spend_usd="10"),
    )


class ImmediatePool:
    # [CLASS DOC] Fake ThreadPoolExecutor whose submit runs the function synchronously and returns an already-
    # completed Future, making admission deterministic.
    """Completed fake futures make admission behavior deterministic."""

    def __init__(self, **kwargs):
        # [FUNCTION DOC] HELPER: stores kwargs['max_workers'] as maximum (KeyError if absent).
        self.maximum = kwargs["max_workers"]

    def __enter__(self):
        # [FUNCTION DOC] HELPER: returns self for use as a context manager.
        return self

    def __exit__(self, *args):
        # [FUNCTION DOC] HELPER: returns False so exceptions propagate.
        return False

    def submit(self, fn, *args):
        # [FUNCTION DOC] HELPER: calls fn(*args) immediately; returns a Future holding its result or the raised
        # exception.
        result = Future()
        try:
            result.set_result(fn(*args))
        except Exception as exc:
            result.set_exception(exc)
        return result


@pytest.fixture
def local_batch(tmp_path, monkeypatch):
    # [FUNCTION DOC] Pytest fixture: writes a fake wheel, creates a 1000 USD campaign budget ledger, and patches
    # batch.check_runtime_wheel (to digest), _controller_identity ('controller-1'),
    # ThreadPoolExecutor (ImmediatePool) and _freeze_controller (no-op). Output: tuple (run
    # directory, campaign directory, wheel path).
    wheel = tmp_path / "owned.whl"
    wheel.write_text("frozen owned runtime")
    campaign = tmp_path / "campaign"
    BudgetLedger(campaign / "budget.sqlite3", limit_usd="1000")
    monkeypatch.setattr(batch, "check_runtime_wheel", digest)
    monkeypatch.setattr(batch, "_controller_identity", lambda: "controller-1")
    monkeypatch.setattr(batch, "ThreadPoolExecutor", ImmediatePool)
    monkeypatch.setattr(batch, "_freeze_controller", lambda *_: {})
    return tmp_path / "run", campaign, wheel


def success(url):
    # [FUNCTION DOC] HELPER: Returns a fake 'verified' candidate row for url, with a verified proof whose
    # bundle_hash and quality_result path derive from batch._key(url).
    return {
        "url": url,
        "status": "verified",
        "verified": {
            "url": url,
            "bundle_hash": "sha256:" + batch._key(url),
            "quality_result": str(Path("/") / batch._key(url) / "result.json"),
        },
    }


def test_target_counts_unique_prs_and_never_overdispatches(local_batch, monkeypatch):
    # [FUNCTION DOC] With target 2 and four candidates, asserts exactly two dispatches, verified/new_verified 2,
    # stop_reason target_reached, two pending, and an event mentioning '2 isolated PR workers'.
    calls, events = [], []

    def run(config, item):
        # [FUNCTION DOC] HELPER: fake _supervise_candidate recording the url and returning success.
        calls.append(item["url"])
        return success(item["url"])

    monkeypatch.setattr(batch, "_supervise_candidate", run)
    plan = batch.BatchPlan(
        name="test", candidates=[candidate(i) for i in range(1, 5)], target_verified=2
    )
    report = batch.run_batch(plan, *local_batch, on_event=events.append)
    assert report["verified"] == 2
    assert report["new_verified"] == 2
    assert report["stop_reason"] == "target_reached"
    assert len(report["pending"]) == 2
    assert len(calls) == 2
    assert any("2 isolated PR workers" in event for event in events)


@pytest.mark.parametrize("changed", [False, True])
def test_approved_prior_inventory_is_checked_once_before_allocation(
    local_batch, monkeypatch, changed
):
    # [FUNCTION DOC] Parametrized on whether the prior proof changed. Changed: run_batch raises ValueError
    # 'differs from the approved inventory' with no configuration, runtime or dispatch. Unchanged:
    # verified totals 2 with one dispatch. Either way the prior path is verified exactly once.
    directory, campaign, wheel = local_batch
    expected = success(candidate(1).url)["verified"]
    fresh = dict(expected, bundle_hash="sha256:changed") if changed else expected
    checked, dispatched = [], []

    def verify(path):
        # [FUNCTION DOC] HELPER: fake verified_result recording the path and returning the (possibly changed)
        # proof.
        checked.append(path)
        return fresh

    def run(config, item):
        # [FUNCTION DOC] HELPER: fake _supervise_candidate recording the url and returning success.
        dispatched.append(item["url"])
        return success(item["url"])

    monkeypatch.setattr(batch, "verified_result", verify)
    monkeypatch.setattr(batch, "_supervise_candidate", run)
    plan = batch.BatchPlan(
        name="approved",
        candidates=[candidate(2)],
        target_verified=2,
        prior_verified=[directory.parent / "prior.json"],
    )
    if changed:
        with pytest.raises(ValueError, match="differs from the approved inventory"):
            batch.run_batch(plan, directory, campaign, wheel, expected_prior_verified=[expected])
        assert not (directory / "configuration.json").exists()
        assert not (directory / "runtime").exists()
        assert dispatched == []
    else:
        result = batch.run_batch(
            plan,
            directory,
            campaign,
            wheel,
            expected_prior_verified=[expected],
            on_event=lambda _: None,
        )
        assert result["verified"] == 2
        assert dispatched == [candidate(2).url]
    assert checked == plan.prior_verified


@pytest.mark.parametrize("denied", [False, True])
def test_campaign_admission_recheck_follows_prior_proofs_and_precedes_dispatch(
    local_batch, monkeypatch, denied
):
    # [FUNCTION DOC] Parametrized on denial. Asserts ordering prior -> recheck -> dispatch; when
    # preallocation_check raises, run_batch propagates ValueError and nothing dispatches.
    events = []
    prior = success(candidate(1).url)["verified"]

    def verify(path):
        # [FUNCTION DOC] HELPER: records 'prior' and returns the prior proof.
        events.append("prior")
        return prior

    def recheck():
        # [FUNCTION DOC] HELPER: records 'recheck'; raises ValueError 'Peer capacity changed' when denied.
        events.append("recheck")
        if denied:
            raise ValueError("Peer capacity changed")

    def supervise(config, item):
        # [FUNCTION DOC] HELPER: records 'dispatch' and returns success.
        events.append("dispatch")
        return success(item["url"])

    monkeypatch.setattr(batch, "verified_result", verify)
    monkeypatch.setattr(batch, "_supervise_candidate", supervise)
    plan = batch.BatchPlan(
        name="peer",
        candidates=[candidate(2)],
        target_verified=2,
        prior_verified=[local_batch[0].parent / "prior.json"],
    )
    if denied:
        with pytest.raises(ValueError, match="Peer capacity changed"):
            batch.run_batch(plan, *local_batch, preallocation_check=recheck)
        assert events == ["prior", "recheck"]
    else:
        batch.run_batch(plan, *local_batch, preallocation_check=recheck, on_event=lambda _: None)
        assert events == ["prior", "recheck", "dispatch"]


def test_gpu_cap_allows_cpu_work_without_overlapping_more_gpu_jobs(local_batch, monkeypatch):
    # [FUNCTION DOC] Three 2-GPU and three CPU candidates with max_parallel 4 and max_gpu_parallel 2; patches
    # wait to complete all active jobs per wave. Asserts the first wave is GPU 0,1 plus CPU 3,4,
    # no wave has more than two GPU jobs, all six dispatched once and target reached.
    items = [candidate(i) for i in range(1, 7)]
    for item in items[:3]:
        item.options.gpus = 2
    admitted, waves = [], []

    def run(config, item):
        # [FUNCTION DOC] HELPER: fake _supervise_candidate recording the url and returning success.
        admitted.append(item["url"])
        return success(item["url"])

    def complete(active, **kwargs):
        # [FUNCTION DOC] HELPER: fake concurrent.futures.wait recording active URLs as a wave and reporting all
        # done, none pending.
        waves.append(list(active.values()))
        return set(active), set()

    monkeypatch.setattr(batch, "_supervise_candidate", run)
    monkeypatch.setattr(batch, "wait", complete)
    plan = batch.BatchPlan(
        name="test", candidates=items, target_verified=6, max_parallel=4, max_gpu_parallel=2
    )
    result = batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    assert waves[0] == [items[index].url for index in (0, 1, 3, 4)]
    assert all(sum(url in {item.url for item in items[:3]} for url in wave) <= 2 for wave in waves)
    assert len(admitted) == len(set(admitted)) == 6
    assert result["verified"] == 6 and result["stop_reason"] == "target_reached"


def test_drain_collects_active_children_and_never_admits_pending_work(local_batch, monkeypatch):
    # [FUNCTION DOC] Fake wait writes drain-request.json and completes only one future at a time. Asserts two
    # dispatches, wait sizes [2, 1], two generated_unverified, stop_reason drained and two
    # pending.
    output, _, _ = local_batch
    calls, waits = [], []

    def run(config, item):
        # [FUNCTION DOC] HELPER: records the url and returns a generated_unverified row with one task.
        calls.append(item["url"])
        return {"url": item["url"], "status": "generated_unverified", "tasks": ["saved/task"]}

    def complete(active, **kwargs):
        # [FUNCTION DOC] HELPER: fake wait recording active count, writing a drain request and returning the
        # first future as done and the rest pending.
        waits.append(len(active))
        save_record(output / "drain-request.json", {"reason": "Use the next frozen runtime"})
        # Keep one child active so the next loop must drain it without replacing
        # the completed child with a third PR.
        first = next(iter(active))
        return {first}, set(active) - {first}

    monkeypatch.setattr(batch, "_supervise_candidate", run)
    monkeypatch.setattr(batch, "wait", complete)
    plan = batch.BatchPlan(name="test", candidates=[candidate(i) for i in range(1, 5)])
    result = batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    assert len(calls) == 2 and waits == [2, 1]
    assert result["generated_unverified"] == 2
    assert result["stop_reason"] == "drained"
    assert len(result["pending"]) == 2


def test_existing_drain_request_prevents_any_dispatch(local_batch, monkeypatch):
    # [FUNCTION DOC] Pre-writes a drain request; asserts no dispatch, stop_reason drained and the candidate
    # remains pending.
    output, _, _ = local_batch
    save_record(output / "drain-request.json", {"reason": "Do not resume this batch"})
    monkeypatch.setattr(
        batch, "_supervise_candidate", lambda *_: pytest.fail("Unexpected dispatch")
    )
    plan = batch.BatchPlan(name="test", candidates=[candidate(1)])
    result = batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    assert result["stop_reason"] == "drained" and result["pending"] == [candidate(1).url]


def test_generated_failures_retained_while_next_pr_fills_target(local_batch, monkeypatch):
    # [FUNCTION DOC] First PR yields generated_unverified with a retained task; asserts three dispatches to reach
    # 2 new verified, one generated_unverified, and the retained task recorded.
    calls = []

    def run(config, item):
        # [FUNCTION DOC] HELPER: returns generated_unverified for URLs ending '/1', success otherwise; records
        # calls.
        calls.append(item["url"])
        if item["url"].endswith("/1"):
            return {
                "url": item["url"],
                "status": "generated_unverified",
                "tasks": ["retained/task"],
                "reason": "oracle failed",
            }
        return success(item["url"])

    monkeypatch.setattr(batch, "_supervise_candidate", run)
    plan = batch.BatchPlan(
        name="test", candidates=[candidate(i) for i in range(1, 5)], target_verified=2
    )
    report = batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    assert report["new_verified"] == 2
    assert report["generated_unverified"] == 1
    assert report["candidates"][candidate(1).url]["tasks"] == ["retained/task"]
    assert len(calls) == 3


def test_batch_headroom_stops_dispatch_without_deleting_pending(local_batch, monkeypatch):
    # [FUNCTION DOC] max_spend_usd 2 is below the 3 USD worker reservation; asserts budget_headroom stop,
    # candidate still pending, zero reserved.
    monkeypatch.setattr(
        batch, "_supervise_candidate", lambda *_: pytest.fail("No dispatch authorized")
    )
    plan = batch.BatchPlan(name="test", candidates=[candidate(1)], max_spend_usd="2")
    report = batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    assert report["stop_reason"] == "budget_headroom"
    assert report["pending"] == [candidate(1).url]
    assert report["budget"]["reserved_usd"] == "0"


def test_global_headroom_also_stops_dispatch(local_batch, monkeypatch):
    # [FUNCTION DOC] Reserves 999 USD as an uncertain prior operation in the shared ledger; asserts
    # budget_headroom with 1.000000 available and the reservation untouched.
    _, campaign, _ = local_batch
    ledger = BudgetLedger(campaign / "budget.sqlite3")
    ledger.reserve("previous-uncertain", "999", "existing operation")
    ledger.mark_uncertain("previous-uncertain", "provider outcome unknown")
    monkeypatch.setattr(
        batch, "_supervise_candidate", lambda *_: pytest.fail("No dispatch authorized")
    )
    plan = batch.BatchPlan(name="test", candidates=[candidate(1)])
    report = batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    assert report["stop_reason"] == "budget_headroom"
    assert report["available_usd"] == "1.000000"
    assert ledger.status()["reserved_usd"] == "999.000000"


def test_controller_crash_blocks_new_dispatch_and_preserves_children(local_batch, monkeypatch):
    # [FUNCTION DOC] Children raise RuntimeError; asserts reconciliation_required after two dispatches, and a
    # rerun dispatches nothing more.
    calls = []

    def run(config, item):
        # [FUNCTION DOC] HELPER: records the url and raises RuntimeError.
        calls.append(item["url"])
        raise RuntimeError("child crashed after provider dispatch")

    monkeypatch.setattr(batch, "_supervise_candidate", run)
    plan = batch.BatchPlan(name="test", candidates=[candidate(i) for i in range(1, 5)])
    report = batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    assert report["stop_reason"] == "reconciliation_required"
    assert len(calls) == 2
    report = batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    assert report["stop_reason"] == "reconciliation_required"
    assert len(calls) == 2


def test_unresolved_provider_cleanup_stops_new_work(local_batch, monkeypatch):
    # [FUNCTION DOC] Children return blocked with cleanup_pending; asserts reconciliation_required after two
    # dispatches.
    calls = []

    def run(config, item):
        # [FUNCTION DOC] HELPER: records the url and returns a blocked row with a pending worker cleanup.
        calls.append(item["url"])
        return {
            "url": item["url"],
            "status": "blocked",
            "cleanup_pending": ["workers/unknown.json"],
        }

    monkeypatch.setattr(batch, "_supervise_candidate", run)
    plan = batch.BatchPlan(name="test", candidates=[candidate(i) for i in range(1, 5)])
    report = batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    assert report["stop_reason"] == "reconciliation_required"
    assert len(calls) == 2


def test_completed_blocked_candidates_are_not_retried_implicitly(local_batch, monkeypatch):
    # [FUNCTION DOC] A blocked candidate is not redispatched on a second run; stop_reason panel_exhausted.
    calls = []

    def run(config, item):
        # [FUNCTION DOC] HELPER: records the url and returns a blocked row.
        calls.append(item["url"])
        return {"url": item["url"], "status": "blocked", "reason": "unsupported source language"}

    monkeypatch.setattr(batch, "_supervise_candidate", run)
    plan = batch.BatchPlan(name="test", candidates=[candidate(1)])
    batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    report = batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    assert calls == [candidate(1).url]
    assert report["stop_reason"] == "panel_exhausted"


def test_frozen_inputs_reject_silent_changes(local_batch, monkeypatch):
    # [FUNCTION DOC] After a first run, a changed max_spend_usd or a different controller identity makes
    # run_batch raise ValueError 'Frozen batch inputs changed'.
    plan = batch.BatchPlan(name="test", candidates=[candidate(1)], max_spend_usd="2")
    batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    changed = plan.model_copy(update={"max_spend_usd": "3"})
    with pytest.raises(ValueError, match="Frozen batch inputs changed"):
        batch.run_batch(changed, *local_batch)
    monkeypatch.setattr(batch, "_controller_identity", lambda: "controller-2")
    with pytest.raises(ValueError, match="Frozen batch inputs changed"):
        batch.run_batch(plan, *local_batch)


def test_frozen_wheel_rejects_tampering(local_batch):
    # [FUNCTION DOC] After a first run, modifying the frozen runtime wheel copy makes run_batch raise ValueError
    # 'Frozen runtime changed'.
    plan = batch.BatchPlan(name="test", candidates=[candidate(1)], max_spend_usd="2")
    batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    output, _, wheel = local_batch
    (output / "runtime" / wheel.name).write_text("changed")
    with pytest.raises(ValueError, match="Frozen runtime changed"):
        batch.run_batch(plan, *local_batch)


def test_finished_orphan_is_imported_without_redispatch(local_batch, monkeypatch):
    # [FUNCTION DOC] First run's child crashes; a batch-result.json then written for it is imported on rerun
    # (generated_unverified 1) without dispatching again.
    calls = []

    def run(config, item):
        # [FUNCTION DOC] HELPER: records the url and raises RuntimeError.
        calls.append(item["url"])
        raise RuntimeError("lost parent-child response")

    monkeypatch.setattr(batch, "_supervise_candidate", run)
    plan = batch.BatchPlan(name="test", candidates=[candidate(1)])
    batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    output, _, _ = local_batch
    save_record(
        output / "candidates" / batch._key(candidate(1).url) / "batch-result.json",
        {"url": candidate(1).url, "status": "generated_unverified", "tasks": ["retained/task"]},
    )
    report = batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    assert report["generated_unverified"] == 1
    assert calls == [candidate(1).url]


def test_retained_inventory_includes_incomplete_task_and_cleanup_receipt(tmp_path):
    # [FUNCTION DOC] Asserts batch._outcome lists a revision directory with task.toml as a retained task and a
    # creating worker receipt as cleanup_pending, status generated_unverified.
    task = tmp_path / "candidates" / "a" / "quality" / "revisions" / "r1"
    task.mkdir(parents=True)
    (task / "task.toml").write_text("incomplete but retained")
    save_record(tmp_path / "workers" / "worker.json", {"state": "creating", "operation_id": "test"})
    row = batch._outcome(tmp_path, candidate(1).url)
    assert row["status"] == "generated_unverified"
    assert row["tasks"] == [str(task)]
    assert row["cleanup_pending"] == [str(tmp_path / "workers" / "worker.json")]


def test_duplicate_and_nonfinite_plan_limits_are_rejected():
    # [FUNCTION DOC] Expects ValueError 'unique' for URLs differing only by a trailing slash and 'finite and
    # positive' for NaN/Infinity/0/-1 max_spend_usd.
    with pytest.raises(ValueError, match="unique"):
        batch.BatchPlan(
            name="test",
            candidates=[
                candidate(1),
                candidate(1).model_copy(update={"url": candidate(1).url + "/"}),
            ],
        )
    for value in ["NaN", "Infinity", "0", "-1"]:
        with pytest.raises(ValueError, match="finite and positive"):
            batch.BatchPlan(name="test", candidates=[candidate(1)], max_spend_usd=value)


def test_child_shares_scopes_and_reconciles_only_confirmed_worker(local_batch, monkeypatch):
    # [FUNCTION DOC] Runs batch._run_candidate with FakeTasksmith that spends through author, quality, native and
    # worker scopes. Asserts generated_unverified with no cleanup pending, source records
    # forwarded, four ledger operations all prefixed by the batch prefix, accounted 1.602111 USD
    # (worker charge computed from runtime/spec) and nothing reserved.
    from repo2rlenv.quality.loop.client import RunBudget
    from repo2rlenv.tasksmith import runner as runner_module
    from repo2rlenv.tasksmith.author.budget import AuthorBudget

    output, campaign, wheel = local_batch
    selected = candidate(1)
    selected.source_record = {"url": selected.url, "head": "frozen"}
    plan = batch.BatchPlan(name="test", candidates=[selected], max_spend_usd="6")
    seen = {}

    class FakeTasksmith:
        # [CLASS DOC] Fake Tasksmith that exercises budget scopes set up by _run_candidate.
        def __init__(self, directory, campaign, options, wheel):
            # [FUNCTION DOC] HELPER: stores directory and opens the campaign ledger; options and wheel ignored.
            self.directory = directory
            self.ledger = BudgetLedger(campaign / "budget.sqlite3")

        def run(self, panel, **kwargs):
            # [FUNCTION DOC] HELPER: records kwargs, asserts budget sharing flags, reserves/settles author,
            # review, native allocation charges, reserves a worker operation and writes a
            # terminated worker receipt, report.json and a retained task directory. Relies on
            # self.budget and self.prefix being injected by the caller.
            seen.update(kwargs)
            assert self.budget.shared is False
            assert self.budget.ledger.shared is True
            author = AuthorBudget(self.budget, self.directory / "author", "investigate")
            operation = author.reserve(1, "author")
            author.settle(operation, 0.1)
            quality = RunBudget(self.budget, self.prefix + "-q", "3")
            quality.reserve(quality.prefix + ":model:review", "1", "review")
            quality.settle(quality.prefix + ":model:review", "0.2", evidence="completed review")
            quality.reserve("native:" + quality.prefix + ":allocation", "1", "native")
            quality.settle(
                "native:" + quality.prefix + ":allocation", "0.3", evidence="terminated allocation"
            )
            worker_operation = "worker:modal:" + self.prefix + "-w0"
            self.budget.reserve(worker_operation, "3", "worker")
            save_record(
                self.directory / "workers" / "worker.json",
                {
                    "state": "terminated",
                    "operation_id": worker_operation,
                    "started_at": "2026-09-13T00:00:00+00:00",
                    "stopped_at": "2026-09-13T00:00:10+00:00",
                    "spec": {"cpus": 2, "memory_mb": 4096},
                },
            )
            save_record(self.directory / "report.json", {"candidates": [{"status": "generated"}]})
            task = self.directory / "retained-task"
            task.mkdir()
            (task / "task.toml").write_text("retained")

    monkeypatch.setattr(runner_module, "Tasksmith", FakeTasksmith)
    configuration = {
        "directory": str(output),
        "campaign": str(campaign),
        "wheel": str(wheel),
        "runtime": digest(wheel),
        "controller_sha256": "controller-1",
        "plan": plan.model_dump(mode="json"),
    }
    row = batch._run_candidate(configuration, selected.model_dump(mode="json"))
    assert row["status"] == "generated_unverified"
    assert not row["cleanup_pending"]
    assert seen["source_records"] == [selected.source_record]
    status = BudgetLedger(campaign / "budget.sqlite3").status()
    assert len(status["operations"]) == 4
    assert all(batch._prefix(output) in operation["id"] for operation in status["operations"])
    assert status["accounted_usd"] == "1.602111"
    assert status["reserved_usd"] == "0.000000"


def test_prior_verified_pr_is_skipped_not_counted_twice(local_batch, monkeypatch):
    # [FUNCTION DOC] A candidate already in prior_verified is skipped; asserts prior 1, new 1, verified 2 and
    # only the second PR dispatched.
    prior_path = local_batch[0].parent / "prior-quality.json"
    prior_proof = success(candidate(1).url)["verified"]
    monkeypatch.setattr(batch, "verified_result", lambda _: prior_proof)
    calls = []

    def run(config, item):
        # [FUNCTION DOC] HELPER: records the url and returns success.
        calls.append(item["url"])
        return success(item["url"])

    monkeypatch.setattr(batch, "_supervise_candidate", run)
    plan = batch.BatchPlan(
        name="test",
        candidates=[candidate(1), candidate(2)],
        prior_verified=[prior_path],
        target_verified=2,
    )
    report = batch.run_batch(plan, *local_batch, on_event=lambda _: None)
    assert report["prior_verified"] == 1
    assert report["new_verified"] == 1
    assert report["verified"] == 2
    assert calls == [candidate(2).url]


def frozen_test_controller(tmp_path, body):
    # [FUNCTION DOC] HELPER: Builds a fake controller wheel zip containing repo2rlenv/tasksmith/batch.py with the
    # given source body and a late.py MARKER module, freezes it via batch._freeze_controller into
    # tmp_path/frozen, and returns a configuration dict with directory, wheel and controller_root.
    import zipfile

    wheel = tmp_path / "controller.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("repo2rlenv/__init__.py", "")
        archive.writestr("repo2rlenv/tasksmith/__init__.py", "")
        archive.writestr("repo2rlenv/tasksmith/batch.py", body)
        archive.writestr("repo2rlenv/tasksmith/late.py", "MARKER = 'frozen'\n")
    controller = tmp_path / "frozen"
    batch._freeze_controller(wheel, controller)
    return {
        "directory": str(tmp_path / "run"),
        "wheel": str(wheel),
        "controller_root": str(controller),
    }


def test_fresh_subprocess_imports_frozen_controller_and_preserves_private_logs(
    tmp_path, monkeypatch
):
    # [FUNCTION DOC] Puts a poisoned live package on PYTHONPATH and runs _supervise_candidate. Asserts the child
    # imported the frozen controller in a different pid, inherited env vars unchanged, os.environ
    # unmodified, stdout captured to controller.stdout with mode 0600, and a second dispatch
    # raises ValueError 'dispatch already exists'.
    import os

    body = """
import json, os
from pathlib import Path

def _run_candidate(configuration, candidate):
    from .late import MARKER
    destination = Path(configuration['directory']) / 'candidates'
    import hashlib
    destination /= hashlib.sha256(candidate['url'].encode()).hexdigest()[:12]
    data = {'marker': MARKER, 'module': __file__, 'pid': os.getpid(),
            'sentinel': os.getenv('TASKSMITH_BATCH_TEST')}
    print('retained private output')
    (destination / 'batch-result.json').write_text(json.dumps(data))
"""
    configuration = frozen_test_controller(tmp_path, body)
    live = tmp_path / "live" / "repo2rlenv"
    live.mkdir(parents=True)
    (live / "__init__.py").write_text("raise RuntimeError('live package must not load')")
    monkeypatch.setenv("PYTHONPATH", str(live.parent))
    monkeypatch.setenv("TASKSMITH_BATCH_TEST", "inherited without mutation")
    before = dict(os.environ)
    result = batch._supervise_candidate(configuration, candidate(1).model_dump(mode="json"))
    assert result["marker"] == "frozen"
    assert result["pid"] != os.getpid()
    assert result["sentinel"] == "inherited without mutation"
    assert Path(result["module"]).is_relative_to(Path(configuration["controller_root"]))
    assert dict(os.environ) == before
    child = Path(configuration["directory"]) / "candidates" / batch._key(candidate(1).url)
    assert (child / "controller.stdout").read_text().strip() == "retained private output"
    assert (child / "controller.stdout").stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="dispatch already exists"):
        batch._supervise_candidate(configuration, candidate(1).model_dump(mode="json"))


def test_subprocess_failure_keeps_diagnostics_and_never_exposes_them(tmp_path):
    # [FUNCTION DOC] Child raises; expects RuntimeError 'retained private logs' without the diagnostic text,
    # which is kept in controller.stderr, and dispatch.json exists.
    configuration = frozen_test_controller(
        tmp_path,
        "def _run_candidate(*args):\n    raise RuntimeError('private diagnostic detail')\n",
    )
    with pytest.raises(RuntimeError, match="retained private logs") as error:
        batch._supervise_candidate(configuration, candidate(1).model_dump(mode="json"))
    assert "private diagnostic detail" not in str(error.value)
    child = Path(configuration["directory"]) / "candidates" / batch._key(candidate(1).url)
    assert "private diagnostic detail" in (child / "controller.stderr").read_text()
    assert (child / "dispatch.json").is_file()


def test_frozen_controller_rejects_changed_file_before_subprocess(tmp_path):
    # [FUNCTION DOC] Modifies a frozen controller file; expects ValueError 'Frozen controller files changed' and
    # no dispatch.json.
    configuration = frozen_test_controller(tmp_path, "def _run_candidate(*args):\n    pass\n")
    late = Path(configuration["controller_root"]) / "repo2rlenv/tasksmith/late.py"
    late.chmod(0o644)
    late.write_text("MARKER = 'changed'\n")
    with pytest.raises(ValueError, match="Frozen controller files changed"):
        batch._supervise_candidate(configuration, candidate(1).model_dump(mode="json"))
    child = Path(configuration["directory"]) / "candidates" / batch._key(candidate(1).url)
    assert not (child / "dispatch.json").exists()


def test_prepared_task_requires_source_and_forbids_generation_import(tmp_path):
    # [FUNCTION DOC] Expects Candidate validation ValueError 'Prepared tasks require' without a source_record and
    # 'cannot use generation_run' when combined with generation_run.
    selected = candidate(1).model_dump(mode="json")
    with pytest.raises(ValueError, match="Prepared tasks require"):
        batch.Candidate.model_validate({**selected, "prepared_task": str(tmp_path)})
    with pytest.raises(ValueError, match="cannot use generation_run"):
        batch.Candidate.model_validate(
            {
                **selected,
                "source_record": {"url": selected["url"]},
                "prepared_task": str(tmp_path),
                "generation_run": str(tmp_path),
            }
        )


def test_supervisors_run_two_real_isolated_controllers_concurrently(tmp_path):
    # [FUNCTION DOC] Runs two real _supervise_candidate subprocesses from a thread pool with a controller that
    # waits (5s deadline) until both are ready; asserts two distinct child pids.
    from concurrent.futures import ThreadPoolExecutor

    body = """
import hashlib, json, os, time
from pathlib import Path

def _run_candidate(configuration, candidate):
    root = Path(configuration['directory'])
    (root / ('ready-' + str(os.getpid()))).write_text('ready')
    deadline = time.monotonic() + 5
    while len(list(root.glob('ready-*'))) < 2:
        if time.monotonic() > deadline:
            raise RuntimeError('controllers did not run concurrently')
        time.sleep(.01)
    path = root / 'candidates' / hashlib.sha256(candidate['url'].encode()).hexdigest()[:12]
    (path / 'batch-result.json').write_text(json.dumps({'pid': os.getpid()}))
"""
    configuration = frozen_test_controller(tmp_path, body)
    with ThreadPoolExecutor(max_workers=2) as supervisors:
        jobs = [
            supervisors.submit(
                batch._supervise_candidate, configuration, candidate(index).model_dump(mode="json")
            )
            for index in (1, 2)
        ]
        results = [job.result() for job in jobs]
    assert len({row["pid"] for row in results}) == 2
