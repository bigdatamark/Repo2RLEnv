# [MODULE DOC] Tests for execution/harbor.py trial recovery: recover_trial reconciles an
# interrupted, already dispatched remote Harbor trial by collecting its evidence and settling the
# budget without redispatch; abandon_undispatched_trial releases holds only for trials proven never
# started. Workers are Mocks.
# KNOWN ISSUE: the module lacks 'from __future__ import annotations', violating the repo invariant
# that every module starts with it.
import json
import shutil
import tarfile
from unittest.mock import Mock

import pytest

from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.execution.base import CommandResult
from repo2rlenv.execution.harbor import abandon_undispatched_trial, recover_trial


@pytest.fixture
def interrupted(tmp_path):
    # [FUNCTION DOC] Fixture: writes an interrupted, dispatched trial.json under tmp_path/attempt;
    # creates a BudgetLedger with a $1 reservation marked uncertain; builds a remote evidence
    # tar.gz containing result.json with reward 1 and agent cost 0.05; and returns a Mock worker
    # whose exec returns the completed status JSON and whose download copies the archive for
    # evidence.tar.gz (else writes an empty file). Output: tuple (worker, output dir, ledger,
    # status dict; status is mutable by tests).
    output = tmp_path / "attempt"
    output.mkdir()
    record = {
        "trial_id": "existing-trial",
        "worker_id": "original-worker",
        "state": "interrupted",
        "trial_dispatched": True,
        "model": "openai/gpt-6-luna",
        "agent": "responses",
        "command": ["harbor", "run", "--job-name", "existing-trial"],
    }
    (output / "trial.json").write_text(json.dumps(record))
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="10")
    ledger.reserve("trial:existing-trial", "1", "original solve")
    ledger.mark_uncertain("trial:existing-trial", "connection lost")
    status = {
        "state": "completed",
        "returncode": 0,
        "command": record["command"],
        "cleanup": {"passed": True},
    }
    result = tmp_path / "remote" / "existing-trial" / "one" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "verifier_result": {"rewards": {"reward": 1}},
                "agent_result": {"cost_usd": 0.05},
            }
        )
    )
    archive = tmp_path / "remote.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(result.parent.parent, arcname="existing-trial")
    worker = Mock(id="original-worker")
    worker.exec.side_effect = lambda argv, **kw: CommandResult(0, json.dumps(status))

    def download(remote, local):
        # [FUNCTION DOC] HELPER: Fake worker.download(remote, local); copies the evidence archive
        # when remote ends with evidence.tar.gz, otherwise writes an empty local file.
        if remote.endswith("evidence.tar.gz"):
            shutil.copyfile(archive, local)
        else:
            local.write_text("")

    worker.download.side_effect = download
    return worker, output, ledger, status


def test_recovery_collects_original_attempt_and_settles_without_dispatch(interrupted):
    # [FUNCTION DOC] Asserts recover_trial returns reward 1, settles $0.05 with no reservation
    # left, marks trial.json completed, preserves interrupted-trial.json, writes recovery.json with
    # redispatched False, only runs cat then tar on the worker, and never uploads.
    worker, output, ledger, _ = interrupted
    assert recover_trial(worker, output, ledger=ledger).reward == 1
    assert ledger.status()["accounted_usd"] == "0.050000"
    assert ledger.status()["reserved_usd"] == "0.000000"
    assert json.loads((output / "trial.json").read_text())["state"] == "completed"
    assert json.loads((output / "interrupted-trial.json").read_text())["state"] == "interrupted"
    assert json.loads((output / "recovery.json").read_text())["redispatched"] is False
    assert [call.args[0][0] for call in worker.exec.call_args_list] == ["cat", "tar"]
    worker.upload.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"state": "running"},
        {"command": ["unrelated"]},
        {"returncode": 1},
        {"cleanup": {"passed": False}},
    ],
)
def test_uncertain_or_different_remote_job_cannot_be_reconciled(interrupted, change):
    # [FUNCTION DOC] Parametrized status changes (running, different command, nonzero rc, failed
    # cleanup); asserts recover_trial raises ValueError 'not confirmed', downloads nothing, keeps
    # the $1 reservation and the interrupted state.
    worker, output, ledger, status = interrupted
    status.update(change)
    with pytest.raises(ValueError, match="not confirmed"):
        recover_trial(worker, output, ledger=ledger)
    worker.download.assert_not_called()
    assert ledger.status()["reserved_usd"] == "1.000000"
    assert json.loads((output / "trial.json").read_text())["state"] == "interrupted"


def test_unstarted_trial_can_release_hold_without_erasing_failed_attempt(interrupted):
    # [FUNCTION DOC] Rewrites trial.json as undispatched without a command and makes worker 'test !
    # -e' checks succeed; asserts abandon_undispatched_trial is idempotent returning
    # not_dispatched, releases the reservation with no charge, archives the original record to
    # interrupted-trial.json, and never uploads.
    worker, output, ledger, _ = interrupted
    record = json.loads((output / "trial.json").read_text())
    record.update(trial_dispatched=False)
    record.pop("command")
    (output / "trial.json").write_text(json.dumps(record))
    worker.exec.side_effect = lambda *args, **kwargs: CommandResult(0, "")
    for _ in range(2):
        assert (
            abandon_undispatched_trial(worker, output, ledger=ledger)["state"] == "not_dispatched"
        )
    assert ledger.status()["accounted_usd"] == "0.000000"
    assert ledger.status()["reserved_usd"] == "0.000000"
    assert json.loads((output / "interrupted-trial.json").read_text()) == record
    assert all(call.args[0][:3] == ["test", "!", "-e"] for call in worker.exec.call_args_list)
    worker.upload.assert_not_called()


def test_dispatched_trial_or_existing_supervisor_keeps_its_hold(interrupted):
    # [FUNCTION DOC] Asserts abandoning a dispatched trial raises ValueError 'explicitly
    # undispatched', and that when the worker check fails (rc 1, e.g. an existing supervisor) it
    # raises RuntimeError; the $1 hold is kept.
    worker, output, ledger, _ = interrupted
    with pytest.raises(ValueError, match="explicitly undispatched"):
        abandon_undispatched_trial(worker, output, ledger=ledger)
    record = json.loads((output / "trial.json").read_text())
    record.update(trial_dispatched=False)
    record.pop("command")
    (output / "trial.json").write_text(json.dumps(record))
    worker.exec.side_effect = lambda *args, **kwargs: CommandResult(1, "")
    with pytest.raises(RuntimeError):
        abandon_undispatched_trial(worker, output, ledger=ledger)
    assert ledger.status()["reserved_usd"] == "1.000000"
