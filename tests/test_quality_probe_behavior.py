# [MODULE DOC] Tests that required behavioral probe controls (focus model_behavior / compiled_execution) are
# counted only when a probe produced a genuine assertion failure, not syntax/import/collection
# errors or tampered evidence. Builds fixture task bundles and fake trial receipts on disk; never
# runs target code. Also covers reviewer protocol (severity downgrade refusal) and publication of
# usable labels.
"""Required behavioral controls must reject behavior, not broken instrumentation."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.emitter.bundle import TaskBundle, TaskFile, write_bundle
from repo2rlenv.quality.labels import _verified_result
from repo2rlenv.quality.loop.artifacts import digest, probe_variant, task_identity
from repo2rlenv.quality.loop.models import (
    Assessment,
    Citation,
    Issue,
    LoopOptions,
    LoopResult,
    Review,
    SemanticProbe,
    TrialRecord,
)
from repo2rlenv.quality.loop.probe_audit import changed_files, submission_files
from repo2rlenv.quality.loop.probe_behavior import behavioral_failure
from repo2rlenv.quality.loop.probe_recovery import record_attempt
from repo2rlenv.quality.loop.runner import QualityLoop, _reusable_probe_trial, probe_failures


@pytest.fixture
def task(tmp_path):
    # [FUNCTION DOC] Pytest fixture: writes a minimal Harbor task bundle 'behavior-probe' (Dockerfile, solve.sh,
    # test.sh, tests/contract.json listing src/model.py) under tmp_path/tasks via write_bundle.
    # Output: Path of the written task directory.
    return write_bundle(
        TaskBundle(
            name="behavior-probe",
            org="tests",
            instruction="Compute the promised model outputs.\n",
            metadata={"recipe": "test", "recipe_version": "1", "reward_kinds": ["test_execution"]},
            files={
                "environment/Dockerfile": TaskFile.text("FROM python:3.12-slim\n"),
                "solution/solve.sh": TaskFile.text("#!/bin/sh\nexit 0\n", executable=True),
                "tests/test.sh": TaskFile.text("#!/bin/sh\nexit 0\n", executable=True),
                "tests/contract.json": TaskFile.text(
                    json.dumps({"submitted_files": ["src/model.py"]})
                ),
            },
        ),
        tmp_path / "tasks",
    )


def materialize(
    task,
    root,
    *,
    focus="model_behavior",
    syntax=True,
    message="AssertionError: values differ",
    collection=False,
):
    # [FUNCTION DOC] HELPER: Fabricates on-disk evidence of one completed probe trial without executing code.
    # Builds a wrong_solution SemanticProbe with the given focus, materializes its variant via
    # probe_variant, writes result.json (reward 0), agent exit-code and oracle.txt containing the
    # changed-files marker (after syntax hash None when syntax=False), verifier result.json
    # (returncode 2 if collection else 1) and a JUnit results.xml with a failure or error element
    # carrying message, plus a trial.json receipt binding the hashes. Records the attempt with
    # record_attempt. Inputs: task path, root quality dir, focus, syntax, message, collection.
    # Output: TrialRecord (role probe, binding receipt). Side effects: writes many files under root.
    """Create owned fixture receipts and audit logs without running target code."""
    probe = SemanticProbe(
        name="wrong-no-reparameterize",
        kind="wrong_solution",
        focus=focus,
        rationale="Remove sampling while preserving model shapes.",
        script="# fixture mutation",
        evidence=[Citation(path="instruction.md", quote="Compute the promised model outputs.")],
    )
    variant = probe_variant(task, probe, root / "probes/r0-probe0" / task.name)
    result = root / "trials/r0-probe0/test-r0-probe0" / task.name / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "config": {
                    "task": {"path": str(variant)},
                    "agent": {"name": "oracle", "model_name": None},
                },
                "verifier_result": {"rewards": {"reward": 0.0}},
                "exception_info": None,
            }
        )
    )
    agent = result.parent / "agent"
    agent.mkdir()
    (agent / "exit-code.txt").write_text("0")
    changes = {
        "src/model.py": {
            "before": {"sha256": "a" * 64, "syntax_sha256": "b" * 64},
            "after": {"sha256": "c" * 64, "syntax_sha256": "d" * 64 if syntax else None},
        }
    }
    (agent / "oracle.txt").write_text(
        "__QUALITY_PROBE_CHANGED_FILES__ " + json.dumps(changes) + "\n__QUALITY_PROBE_COMPLETED__\n"
    )
    verifier = result.parent / "verifier"
    verifier.mkdir()
    (verifier / "result.json").write_text(
        json.dumps(
            {
                "passed": False,
                "returncode": 2 if collection else 1,
                "statuses": {"test_model": "failed"},
            }
        )
    )
    xml = ElementTree.Element("testsuites")
    suite = ElementTree.SubElement(xml, "testsuite")
    case = ElementTree.SubElement(suite, "testcase", name="test_model", classname="tests")
    failure = ElementTree.SubElement(case, "error" if collection else "failure", message=message)
    failure.text = message
    ElementTree.ElementTree(xml).write(verifier / "results.xml", encoding="unicode")
    (result.parents[2] / "trial.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "trial_id": "test-r0-probe0",
                "bundle_hash": task_identity(variant),
                "result_sha256": digest(result),
            }
        )
    )
    trial = TrialRecord(
        role="probe",
        bundle_hash=task_identity(variant),
        result=str(result),
        result_sha256=digest(result),
        agent="oracle",
        model=None,
        reward=0.0,
        exception_type=None,
        binding="receipt",
        agent_exit_code=0,
        probe=probe,
        probe_installed=True,
    )
    record_attempt(root, "r0-probe0", task_identity(task), trial)
    return trial


def test_peft_syntax_failure_is_not_behavioral_rejection(task, tmp_path):
    # [FUNCTION DOC] Verifies a probe whose mutation left unparsable Python is reported by behavioral_failure and
    # probe_failures and is not reusable via _reusable_probe_trial.
    trial = materialize(
        task,
        tmp_path / "quality",
        syntax=False,
        message="IndentationError: unindent does not match any outer indentation level",
    )
    assert "unparseable Python in src/model.py" in behavioral_failure(trial)
    assert "unparseable Python" in probe_failures([trial], 1.0)[0]
    assert not _reusable_probe_trial(trial, trial.probe, task_identity(task), 1.0)


@pytest.mark.parametrize("focus", ["model_behavior", "compiled_execution"])
def test_real_assertion_failure_satisfies_execution_floor(task, tmp_path, focus):
    # [FUNCTION DOC] Parametrized over focus; verifies a real AssertionError failure satisfies the behavioral floor
    # (no failure, no probe_failures).
    trial = materialize(task, tmp_path / "quality", focus=focus)
    assert behavioral_failure(trial) is None
    assert probe_failures([trial], 1.0) == []


@pytest.mark.parametrize(
    "message",
    [
        "ModuleNotFoundError: No module named 'missing'",
        "ImportError: cannot import name 'Linear'",
        "E   SyntaxError: invalid syntax",
    ],
)
def test_import_failure_inside_test_is_not_runtime_behavior(task, tmp_path, message):
    # [FUNCTION DOC] Parametrized over import/syntax error messages inside a test; verifies behavioral_failure
    # reports 'no behavioral rejection'.
    trial = materialize(task, tmp_path / "quality", message=message)
    assert "no behavioral rejection" in behavioral_failure(trial)


def test_collection_failure_is_not_runtime_behavior(task, tmp_path):
    # [FUNCTION DOC] Verifies a collection error (returncode 2 / error element) is reported as lacking a completed
    # failing test execution.
    trial = materialize(task, tmp_path / "quality", collection=True)
    assert "completed failing test execution" in behavioral_failure(trial)


def test_error_mentioned_in_assertion_is_not_an_import_failure(task, tmp_path):
    # [FUNCTION DOC] Verifies an assertion message that merely mentions ImportError is still treated as a
    # behavioral rejection (no failure).
    trial = materialize(
        task,
        tmp_path / "quality",
        message="AssertionError: expected no ImportError in returned text",
    )
    assert behavioral_failure(trial) is None


@pytest.mark.parametrize(
    "relative", ["agent/oracle.txt", "verifier/results.xml", "verifier/result.json"]
)
def test_modified_execution_evidence_cannot_count(task, tmp_path, relative):
    # [FUNCTION DOC] Parametrized over evidence files (oracle log, results.xml, verifier result.json); appending to
    # any makes behavioral_failure report 'needs diagnosis'. Side effects: modifies fixture files.
    trial = materialize(task, tmp_path / "quality")
    path = Path(trial.result).parent / relative
    path.write_text(path.read_text() + "\n")
    assert "needs diagnosis" in behavioral_failure(trial)


def test_missing_old_xml_binding_is_not_retroactively_sealed(task, tmp_path):
    # [FUNCTION DOC] Verifies a journal lacking the behavior_files binding is returned unchanged by record_attempt
    # (not retroactively sealed, bytes unchanged) and the trial still needs diagnosis.
    root = tmp_path / "quality"
    trial = materialize(task, root)
    journal = root / "probe-attempts/r0-probe0.json"
    saved = json.loads(journal.read_text())
    saved.pop("behavior_files")
    journal.write_text(json.dumps(saved))
    before = journal.read_bytes()
    assert record_attempt(root, "r0-probe0", task_identity(task), trial) == saved
    assert journal.read_bytes() == before
    assert "needs diagnosis" in behavioral_failure(trial)


def test_generic_wrong_control_policy_is_unchanged(task, tmp_path):
    # [FUNCTION DOC] Verifies a general-focus wrong-solution probe keeps the old policy: no behavioral failure even
    # with invalid syntax, and the probe contract has require_valid_python false.
    trial = materialize(task, tmp_path / "quality", focus="general", syntax=False)
    assert behavioral_failure(trial) is None
    assert probe_failures([trial], 1.0) == []
    assert not json.loads(
        (
            tmp_path
            / "quality/probes/r0-probe0"
            / task.name
            / "solution/quality-probe-contract.json"
        ).read_text()
    )["require_valid_python"]


def test_wrong_solution_that_passes_remains_a_verifier_gap(task, tmp_path):
    # [FUNCTION DOC] Verifies a wrong solution that earned reward 1.0 is not a behavioral failure but is reported
    # by probe_failures as a verifier gap.
    trial = materialize(task, tmp_path / "quality")
    accepted_wrong = trial.model_copy(update={"reward": 1.0})
    assert behavioral_failure(accepted_wrong) is None
    assert probe_failures([accepted_wrong], 1.0) == [
        "Probe wrong-no-reparameterize: wrong_solution earned 1.0"
    ]


def test_custom_reward_threshold_does_not_reclassify_verifier_gap(task, tmp_path):
    # [FUNCTION DOC] Verifies a custom success threshold 0.5 still classifies reward 0.5 as a verifier gap in
    # probe_failures rather than a behavioral failure.
    trial = materialize(task, tmp_path / "quality").model_copy(update={"reward": 0.5})
    assert behavioral_failure(trial, 0.5) is None
    assert "earned 0.5" in probe_failures([trial], 0.5)[0]


def test_future_behavioral_mutation_fails_before_completion(task, tmp_path):
    # [FUNCTION DOC] Verifies the behavioral probe contract sets require_valid_python and that changed_files with
    # require_valid_python=True raises 'introduced unparseable Python' while the default byte-level
    # policy still reports a change.
    trial = materialize(task, tmp_path / "quality")
    contract = (
        tmp_path / "quality/probes/r0-probe0" / task.name / "solution/quality-probe-contract.json"
    )
    assert json.loads(contract.read_text())["require_valid_python"]
    root = tmp_path / "submission"
    root.mkdir()
    path = root / "model.py"
    path.write_text("def forward(): return 1\n")
    before = submission_files(root, {"submitted_files": ["model.py"]})
    path.write_text("def forward():\nreturn 0\n")
    after = submission_files(root, {"submitted_files": ["model.py"]})
    with pytest.raises(ValueError, match="introduced unparseable Python"):
        changed_files(before, after, require_valid_python=True)
    assert changed_files(before, after)  # Generic byte-level controls retain their policy.
    assert trial.probe.focus == "model_behavior"


def assessment():
    # [FUNCTION DOC] HELPER: Returns a passing Assessment (score 3) citing instruction.md, used to build fake
    # Review objects.
    return Assessment(
        status="pass",
        score=3,
        explanation="Coherent behavior.",
        evidence=[Citation(path="instruction.md", quote="Compute the promised model outputs.")],
    )


class Reviewer:
    # [CLASS DOC] Fake reviewer model client. Yields scripted issue severities per call and records each request
    # payload.
    def __init__(self, severities):
        # [FUNCTION DOC] Inputs: severities (iterable of severity strings). Stores an iterator over them and an
        # empty calls list.
        self.severities, self.calls = iter(severities), []

    def ask(self, schema, model, system, user, key):
        # [FUNCTION DOC] Fake ask: asserts schema is Review, appends the JSON-decoded user payload to calls, and
        # returns a Review with one 'probe' Issue whose severity is the next scripted value. Raises
        # StopIteration if severities run out. Other params (model, system, key) ignored.
        assert schema is Review
        self.calls.append(json.loads(user))
        return Review(
            summary="Check the actual rejection cause.",
            task=assessment(),
            verifier=assessment(),
            leakage=assessment(),
            rollout="not_run",
            probes=[],
            read_requests=[],
            issues=[
                Issue(
                    category="probe",
                    severity=next(self.severities),
                    problem="Mutation broke Python syntax.",
                    repair="Restore parseable mutation before testing behavior.",
                    evidence=[Citation(path="evidence/0-probe/result.json", quote='"reward": 0.0')],
                )
            ],
        )


def test_reviewer_cannot_downgrade_broken_behavioral_probe(task, tmp_path):
    # [FUNCTION DOC] Verifies QualityLoop._review rejects a reviewer downgrading a broken behavioral probe issue to
    # 'improvement' and accepts the corrected 'blocking' review after a second call with 'cannot be
    # downgraded' feedback. Side effects: creates a sqlite budget ledger in tmp_path.
    root = tmp_path / "quality"
    trial = materialize(task, root, syntax=False)
    model = Reviewer(["improvement", "blocking"])
    loop = QualityLoop(
        LoopOptions(max_read_rounds=1),
        root,
        BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="1"),
        model_client=model,
    )
    review, _ = loop._review(task, [trial], "r0-after", probe_limit=0)
    assert review.issues[0].severity == "blocking"
    assert len(model.calls) == 2
    assert "cannot be downgraded" in model.calls[-1]["protocol_feedback"][0]


def test_incorrect_review_exhausts_existing_limit(task, tmp_path):
    # [FUNCTION DOC] Verifies two consecutive downgraded reviews exhaust the call limit, raising ValueError('within
    # the call limit') after exactly two calls.
    root = tmp_path / "quality"
    trial = materialize(task, root, syntax=False)
    model = Reviewer(["improvement", "improvement"])
    loop = QualityLoop(
        LoopOptions(max_read_rounds=1),
        root,
        BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="1"),
        model_client=model,
    )
    with pytest.raises(ValueError, match="within the call limit"):
        loop._review(task, [trial], "r0-after", probe_limit=0)
    assert len(model.calls) == 2


def test_publication_rejects_usable_claim_despite_sound_review(task, tmp_path):
    # [FUNCTION DOC] Verifies _verified_result refuses a 'usable' LoopResult whose required behavioral probe was
    # syntactically broken, raising ValueError('required reviewed controls'), even with a sound
    # review and passing controls.
    trial = materialize(task, tmp_path / "quality", syntax=False)
    review = Reviewer(["improvement"]).ask(Review, None, None, "{}", "review")
    review = review.model_copy(update={"rollout": "legitimate_failure"})
    controls = [
        trial.model_copy(
            update={
                "role": role,
                "probe": None,
                "probe_installed": None,
                "reward": reward,
                "bundle_hash": task_identity(task),
            }
        )
        for role, reward in [("baseline", 0.0), ("oracle", 1.0), ("rollout", 0.0)]
    ]
    alternative = trial.model_copy(
        update={
            "probe": trial.probe.model_copy(update={"kind": "valid_alternative"}),
            "reward": 1.0,
        }
    )
    result = LoopResult(
        status="usable",
        source_hash=task_identity(task),
        bundle_hash=task_identity(task),
        task_path=str(task),
        repairs=2,
        review=review,
        trials=[*controls, trial, alternative],
        reasons=[],
        accounted_usd="0",
        reserved_usd="0",
    )
    with pytest.raises(ValueError, match="required reviewed controls"):
        _verified_result(result, task)
