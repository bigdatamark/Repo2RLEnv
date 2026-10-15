# [MODULE DOC] Tests for Tasksmith.review_candidate's handling of required probe focus:
# when Options.required_probe_focus adds probe requirements, the runner must review an annotated
# copy
# of the task (new bundle hash, quality-input.json receipt) and drop imported probe/trial evidence,
# never mutating the original task. QualityLoop and native Modal trials are faked.
"""Probe requirements create a new review input without rebinding old evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.emitter.bundle import TaskBundle, TaskFile, write_bundle
from repo2rlenv.quality.loop.artifacts import task_identity
from repo2rlenv.quality.loop.requirements import with_probe_requirements
from repo2rlenv.tasksmith import runner as runner_module
from repo2rlenv.tasksmith.models import Options


@pytest.fixture
def review_setup(tmp_path, monkeypatch):
    # [FUNCTION DOC] Fixture: creates a budget ledger ($100), a GPU Tasksmith runner, and a
    # minimal written TaskBundle (Dockerfile, failing test.sh, passing solve.sh) under
    # tmp_path/constructed; patches runner_module.QualityLoop with a recording fake and
    # native.NativeModalTrials with a no-op. Output: (runner, task path, calls list) where each
    # QualityLoop
    # run appends a dict of the task plus keyword arguments.
    campaign = tmp_path / "campaign"
    BudgetLedger(campaign / "budget.sqlite3", limit_usd="100")
    runner = runner_module.Tasksmith(
        tmp_path / "run", campaign, Options(gpus=1), tmp_path / "runtime.whl"
    )
    task = write_bundle(
        TaskBundle(
            name="fixture",
            org="tests",
            instruction="Restore the specified public behavior.",
            metadata={
                "recipe": "tasksmith",
                "recipe_version": "1",
                "reward_kinds": ["test_execution"],
            },
            files={
                "environment/Dockerfile": TaskFile.text("FROM python:3.12-slim\n"),
                "tests/test.sh": TaskFile.text("#!/bin/sh\nexit 1\n", executable=True),
                "solution/solve.sh": TaskFile.text("#!/bin/sh\nexit 0\n", executable=True),
            },
        ),
        tmp_path / "constructed",
    )
    calls = []

    class Loop:
        # [CLASS DOC] HELPER: Fake QualityLoop that ignores constructor arguments and
        # records each run call.
        def __init__(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Accepts and ignores any arguments.
            pass

        def run(self, task, **kwargs):
            # [FUNCTION DOC] HELPER: Appends {"task": task, **kwargs} to calls and returns a
            # namespace with status "reviewed" and a model_dump returning {"status": "reviewed"}.
            calls.append({"task": task, **kwargs})
            return SimpleNamespace(
                status="reviewed", model_dump=lambda **kwargs: {"status": "reviewed"}
            )

    from repo2rlenv.quality.loop import native

    monkeypatch.setattr(runner_module, "QualityLoop", Loop)
    monkeypatch.setattr(native, "NativeModalTrials", lambda *args, **kwargs: None)
    return runner, task, calls


def review(runner, task):
    # [FUNCTION DOC] HELPER: Calls runner.review_candidate for candidate dir
    # runner.directory/candidate with a constructed record pointing at the task and imported_from
    # evidence (one prior probe and baseline/oracle/rollout trial result paths bound to the
    # configured
    # solver model). Output: the candidate root path.
    root = runner.directory / "candidate"
    constructed = {
        "local": str(task.parent),
        "value": {"task_relative": task.name},
        "imported_from": {
            "probes": [{"kind": "wrong_solution", "description": "Previously tested probe"}],
            "trials": {
                role: {
                    "result": str(root / "original-evidence" / f"{role}.json"),
                    "model": runner.options.quality.solver_model.qualified_name,
                }
                for role in ("baseline", "oracle", "rollout")
            },
        },
    }
    runner.review_candidate(root, {"id": "abcdefgh1234"}, constructed)
    return root


def test_default_requirements_keep_input_and_imported_evidence(review_setup):
    # [FUNCTION DOC] Without required focus,
    # the original task is reviewed, imported probes are written to imported-probes.json, all three
    # trial
    # roles are passed as Paths, and no quality-input.json is written.
    runner, task, calls = review_setup
    root = review(runner, task)
    assert calls[0]["task"] == task
    assert calls[0]["probes"] == root / "imported-probes.json"
    assert all(isinstance(calls[0][role], Path) for role in ("baseline", "oracle", "rollout"))
    assert not (root / "quality-input.json").exists()


def test_new_requirements_copy_input_and_discard_prior_proofs(review_setup):
    # [FUNCTION DOC] With required_probe_focus
    # set, review uses an annotated copy under quality-input/, passes no probes or trial evidence,
    # leaves
    # the original task bytes and hash unchanged, writes a quality-input.json receipt with source
    # and new
    # hashes, and a repeated review yields identical arguments.
    runner, task, calls = review_setup
    runner.options.required_probe_focus = ["compiled_execution"]
    before = {
        str(path.relative_to(task)): path.read_bytes() for path in task.rglob("*") if path.is_file()
    }
    old_hash = task_identity(task)
    root = review(runner, task)
    annotated = root / "quality-input" / task.name
    assert calls[0]["task"] == annotated and calls[0]["probes"] is None
    assert all(role not in calls[0] for role in ("baseline", "oracle", "rollout"))
    assert not (root / "imported-probes.json").exists()
    assert task_identity(task) == old_hash != task_identity(annotated)
    assert before == {
        str(path.relative_to(task)): path.read_bytes() for path in task.rglob("*") if path.is_file()
    }
    receipt = json.loads((root / "quality-input.json").read_text())
    assert receipt == {
        "source_task": str(task.resolve()),
        "source_bundle_hash": old_hash,
        "task_path": str(annotated.resolve()),
        "bundle_hash": task_identity(annotated),
        "required_probe_focus": ["compiled_execution"],
        "imported_evidence_reused": False,
    }
    review(runner, task)
    assert calls[1] == calls[0]


def test_existing_requirements_keep_exact_identity_and_imported_evidence(review_setup):
    # [FUNCTION DOC] If the task
    # already carries the required probe requirements (via with_probe_requirements), it is reviewed
    # as
    # is, with imported probes and trial evidence retained and no new receipt.
    runner, task, calls = review_setup
    runner.options.required_probe_focus = ["compiled_execution"]
    annotated = with_probe_requirements(
        task, task.parent / "previous-input", ["compiled_execution"]
    )
    root = review(runner, annotated)
    assert calls[0]["task"] == annotated
    assert calls[0]["probes"] == root / "imported-probes.json"
    assert all(role in calls[0] for role in ("baseline", "oracle", "rollout"))
    assert not (root / "quality-input.json").exists()


def test_invalid_required_input_prevents_quality_dispatch(review_setup):
    # [FUNCTION DOC] Modifying instruction.md after
    # generation makes review raise ValueError ("recorded bundle hash") and QualityLoop never runs.
    runner, task, calls = review_setup
    runner.options.required_probe_focus = ["compiled_execution"]
    (task / "instruction.md").write_text("Changed after generation.")
    with pytest.raises(ValueError, match="recorded bundle hash"):
        review(runner, task)
    assert calls == []
