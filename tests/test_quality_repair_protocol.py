# [MODULE DOC] Tests for QualityLoop._repair protocol: correction calls receive the rejected draft and actionable
# feedback, contract list changes use append_expected_passes, protected paths are honored, wrong-
# solution probes stay immutable, and provider errors are not retried. Uses a scripted fake Model;
# no network.
"""Repair feedback stays actionable, bounded, and faithful to retained probes."""

from __future__ import annotations

import json

import pytest

from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.emitter.bundle import TaskBundle, TaskFile, write_bundle
from repo2rlenv.quality.loop.artifacts import task_identity
from repo2rlenv.quality.loop.client import ModelRequestError
from repo2rlenv.quality.loop.models import (
    Assessment,
    Citation,
    Edit,
    Issue,
    LoopOptions,
    Repair,
    Review,
    SemanticProbe,
)
from repo2rlenv.quality.loop.runner import QualityLoop


@pytest.fixture
def task(tmp_path):
    # [FUNCTION DOC] Pytest fixture: writes 'repair-protocol' task bundle with tests/test.sh '# weak check' and
    # tests/contract.json expected_passes ['z','a']. Output: task Path.
    return write_bundle(
        TaskBundle(
            name="repair-protocol",
            org="tests",
            instruction="Compute the sum of two integers.\n",
            metadata={"recipe": "test", "recipe_version": "1", "reward_kinds": ["test_execution"]},
            files={
                "environment/Dockerfile": TaskFile.text("FROM python:3.12-slim\n"),
                "solution/solve.sh": TaskFile.text("#!/bin/sh\nexit 0\n", executable=True),
                "tests/test.sh": TaskFile.text("#!/bin/sh\n# weak check\n", executable=True),
                "tests/contract.json": TaskFile.text('{"expected_passes": ["z", "a"]}\n'),
            },
        ),
        tmp_path / "tasks",
    )


def probes():
    # [FUNCTION DOC] HELPER: Returns wrong-sum (wrong_solution) and valid-format (valid_alternative)
    # SemanticProbes.
    return [
        SemanticProbe(
            name=name,
            kind=kind,
            rationale="Exercise a concrete arithmetic behavior.",
            evidence=[Citation(path="instruction.md", quote="sum of two integers")],
            script=script,
        )
        for name, kind, script in [
            ("wrong-sum", "wrong_solution", "printf 7 > /workspace/answer.txt"),
            ("valid-format", "valid_alternative", "printf '4\\n' > /workspace/answer.txt"),
        ]
    ]


def review(category="verifier"):
    # [FUNCTION DOC] HELPER: Returns a Review with passing assessments and one blocking Issue of the given category
    # (default 'verifier') citing tests/test.sh.
    assessment = Assessment(
        status="pass",
        score=3,
        explanation="The task requests useful arithmetic behavior.",
        evidence=[Citation(path="instruction.md", quote="sum of two integers")],
    )
    return Review(
        summary="Repair a grounded verification issue.",
        task=assessment,
        verifier=assessment,
        leakage=assessment,
        rollout="not_run",
        probes=[],
        read_requests=[],
        issues=[
            Issue(
                category=category,
                severity="blocking",
                problem="The existing check rejects a valid representation.",
                repair="Check the requested observable behavior.",
                evidence=[Citation(path="tests/test.sh", quote="weak check")],
            )
        ],
    )


def patch(*, old="weak check", replacements=()):
    # [FUNCTION DOC] HELPER: Returns a Repair editing tests/test.sh old text (default 'weak check') to 'strong
    # check', with optional probe_replacements.
    return Repair(
        explanation="Correct the check without weakening the arithmetic contract.",
        addressed_issues=["The existing check rejects a valid representation."],
        edits=[Edit(path="tests/test.sh", old=old, new="strong check", executable=True)],
        probe_replacements=list(replacements),
    )


class Model:
    # [CLASS DOC] Scripted fake model returning queued Repair responses.
    def __init__(self, responses):
        # [FUNCTION DOC] Inputs: responses (iterable of Repair, dict, or Exception). Initializes calls.
        self.responses, self.calls = iter(responses), []

    def ask(self, schema, model, system, user, key):
        # [FUNCTION DOC] Asserts schema is Repair, records key plus JSON payload, then takes the next response:
        # raises it if an Exception, else returns schema.model_validate of it (dumping Repair
        # objects first).
        assert schema is Repair
        self.calls.append({"key": key, **json.loads(user)})
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return schema.model_validate(
            response.model_dump() if isinstance(response, Repair) else response
        )


def loop(tmp_path, model):
    # [FUNCTION DOC] HELPER: Builds a QualityLoop with repair=True, quality dir tmp_path/quality, $10 BudgetLedger,
    # given model.
    return QualityLoop(
        LoopOptions(repair=True),
        tmp_path / "quality",
        BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="10"),
        model_client=model,
    )


def test_contract_list_correction_uses_append_without_reconstructing_existing_order(task, tmp_path):
    # [FUNCTION DOC] Verifies an edit that reconstructs the contract list is rejected and the correction using
    # append_expected_passes appends 'new' while preserving existing order; checks correction
    # payload fields.
    rejected = Repair(
        explanation="Register a new test.",
        addressed_issues=["Missing case"],
        edits=[
            Edit(
                path="tests/contract.json",
                old='["a", "z"]',
                new='["a", "z", "new"]',
                executable=False,
            )
        ],
    )
    corrected = rejected.model_copy(update={"edits": [], "append_expected_passes": ["new"]})
    model = Model([rejected, corrected])
    runner = loop(tmp_path, model)
    destination, retained = runner._repair(
        task, review(), runner._context(task, []), probes(), 0, []
    )
    assert json.loads((destination / "tests/contract.json").read_text())["expected_passes"] == [
        "z",
        "a",
        "new",
    ]
    assert retained == probes()
    assert len(model.calls) == 2
    assert model.calls[1]["previous_repair"] == rejected.model_dump()
    assert any("append_expected_passes" in value for value in model.calls[1]["patch_feedback"])
    assert model.calls[1]["correction_calls_remaining"] == 0


@pytest.mark.parametrize("protected", ["tests/contract.json", "tests"])
def test_contract_append_obeys_protected_inputs(task, tmp_path, protected):
    # [FUNCTION DOC] Parametrized over protected paths; append_expected_passes against a protected contract raises
    # 'immutable source or oracle' after two calls and writes no r1 revision.
    from pathlib import Path

    proposal = Repair(
        explanation="Register cases.",
        addressed_issues=["Missing case"],
        edits=[],
        append_expected_passes=["new"],
    )
    model = Model([proposal, proposal])
    runner = loop(tmp_path, model)
    runner.protected_paths = (Path(protected),)
    with pytest.raises(ValueError, match="immutable source or oracle"):
        runner._repair(task, review(), runner._context(task, []), probes(), 0, [])
    assert len(model.calls) == 2
    assert not (runner.directory / "revisions/r1").exists()


def test_legacy_saved_repair_replay_accepts_absent_empty_append_field(task, tmp_path):
    # [FUNCTION DOC] Verifies replaying a saved repair.json lacking append_expected_passes reuses the same
    # destination without a new model call.
    model = Model([patch(), patch()])
    runner = loop(tmp_path, model)
    destination, _ = runner._repair(task, review(), runner._context(task, []), probes(), 0, [])
    path = destination.parent / "repair.json"
    saved = json.loads(path.read_text())
    del saved["repair"]["append_expected_passes"]
    path.write_text(json.dumps(saved))
    replay, _ = runner._repair(task, review(), runner._context(task, []), probes(), 0, [])
    assert replay == destination
    assert len(model.calls) == 2


def test_verifier_repair_correction_retains_probes_and_receives_rejected_draft(task, tmp_path):
    # [FUNCTION DOC] Verifies a verifier repair proposing a probe replacement is corrected: correction receives the
    # rejected draft, empty allowed replacements, immutable wrong-sum; retained probes unchanged;
    # only the destination copy is edited.
    retained = probes()
    rejected = patch(replacements=[retained[1]])
    corrected = patch()
    model = Model([rejected, corrected])
    runner = loop(tmp_path, model)
    original_hash = task_identity(task)
    destination, updated = runner._repair(
        task, review(), runner._context(task, []), retained, 0, []
    )
    correction = model.calls[1]
    assert correction["previous_repair"] == rejected.model_dump()
    assert correction["probe_replacement_policy"]["allowed_replacements"] == []
    assert correction["probe_replacement_policy"]["immutable_wrong_solution_probes"] == [
        "wrong-sum"
    ]
    assert "Set probe_replacements=[]" in correction["patch_feedback"][0]
    assert correction["correction_calls_remaining"] == 0
    assert updated == retained
    assert task_identity(task) == original_hash != task_identity(destination)
    assert (task / "tests/test.sh").read_text().endswith("# weak check\n")
    assert (destination / "tests/test.sh").read_text().endswith("# strong check\n")
    assert [call["key"] for call in model.calls] == ["r0-repair", "r0-repair-correction1"]


def test_wrong_probe_replacement_stays_forbidden_even_with_probe_diagnosis(task, tmp_path):
    # [FUNCTION DOC] Verifies wrong-solution probe replacement stays forbidden even with a probe diagnosis, raising
    # after two calls; policy lists only valid-format and mentions no-op probe; no r1 written.
    retained = probes()
    invalid = patch().model_dump()
    invalid["probe_replacements"] = [retained[0].model_dump()]
    model = Model([invalid, invalid])
    runner = loop(tmp_path, model)
    original_hash = task_identity(task)
    with pytest.raises(ValueError, match="wrong-solution probes cannot be replaced"):
        runner._repair(task, review("probe"), runner._context(task, []), retained, 0, [])
    assert len(model.calls) == 2
    assert model.calls[1]["previous_repair"] == invalid
    policy = model.calls[1]["probe_replacement_policy"]
    assert policy["allowed_replacements"] == [
        {"name": "valid-format", "kind": "valid_alternative", "focus": "general"}
    ]
    assert "no-op probe" in policy["rule"]
    assert task_identity(task) == original_hash
    assert not (runner.directory / "revisions/r1").exists()


@pytest.mark.parametrize("corrected", [True, False])
def test_excerpt_budget_failure_uses_existing_correction_without_inventing_source(
    task, tmp_path, monkeypatch, corrected
):
    # [FUNCTION DOC] Parametrized: when reading excerpts fails (monkeypatched read_more raising), the correction
    # call gets the feedback and same documents; a corrected patch applies, else 'old text must
    # occur exactly once' is raised. Side effects: monkeypatches context.read_more.
    rejected = patch(old="missing source text")
    model = Model([rejected, patch() if corrected else rejected])
    runner = loop(tmp_path, model)
    context = runner._context(task, [])
    documents = dict(context.documents)

    def unavailable(requests):
        # [FUNCTION DOC] HELPER: Stand-in for read_more that always raises ValueError('Requested excerpts exceed
        # context limit').
        raise ValueError("Requested excerpts exceed context limit")

    monkeypatch.setattr(context, "read_more", unavailable)
    if corrected:
        destination, _ = runner._repair(task, review(), context, probes(), 0, [])
        assert (destination / "tests/test.sh").read_text().endswith("# strong check\n")
    else:
        with pytest.raises(ValueError, match="old text must occur exactly once"):
            runner._repair(task, review(), context, probes(), 0, [])
        assert not (runner.directory / "revisions/r1" / task.name).exists()
    assert len(model.calls) == 2
    correction = model.calls[1]
    assert correction["documents"] == documents
    assert correction["previous_repair"] == rejected.model_dump()
    assert any(
        "Repair-source excerpt unavailable for tests/test.sh" in feedback
        and "Requested excerpts exceed context limit" in feedback
        for feedback in correction["patch_feedback"]
    )


def test_provider_uncertainty_never_triggers_protocol_redispatch(task, tmp_path):
    # [FUNCTION DOC] Verifies a ModelRequestError from the provider propagates after one call without protocol re-
    # dispatch and no revision is written.
    model = Model([ModelRequestError("Provider response uncertain")])
    runner = loop(tmp_path, model)
    with pytest.raises(ModelRequestError, match="uncertain"):
        runner._repair(task, review(), runner._context(task, []), probes(), 0, [])
    assert len(model.calls) == 1
    assert not (runner.directory / "revisions/r1").exists()
