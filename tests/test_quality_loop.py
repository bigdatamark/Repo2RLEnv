# [MODULE DOC] End-to-end and unit tests for the evidence-based quality loop
# (repo2rlenv.quality.loop): QualityLoop review -> controls (baseline/oracle) -> semantic probes
# (wrong_solution / valid_alternative) -> optional blind rollout -> bounded repair, plus artifacts
# (apply_repair, probe_variant, import_trial, task identity), protocol citation/path resolution,
# EvidenceContext budgeting, JsonModel/RunBudget paid-call accounting and probe requirement
# metadata. Uses a scripted fake model (Model) and fake trial runner (Trials) writing local
# receipts, so no LLM or Docker is needed. Skips entirely without `harbor`. Side effects: files,
# sqlite budget ledgers and a few bash subprocesses under tmp_path; monkeypatches LLM key resolution
# and campaigns.llm.complete in client tests.
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from repo2rlenv.campaigns.budget import BudgetExceeded, BudgetLedger
from repo2rlenv.emitter.bundle import TaskBundle, TaskFile, write_bundle
from repo2rlenv.quality.loop.artifacts import (
    apply_repair,
    digest,
    import_trial,
    probe_variant,
    refresh_identity,
    task_identity,
)
from repo2rlenv.quality.loop.client import JsonModel, ModelRequestError, RunBudget, response_schema
from repo2rlenv.quality.loop.context import EvidenceContext
from repo2rlenv.quality.loop.models import (
    Assessment,
    Citation,
    Edit,
    Issue,
    LoopOptions,
    ReadRequest,
    Repair,
    Review,
    SemanticProbe,
    TrialRecord,
)
from repo2rlenv.quality.loop.protocol import (
    distinct_probes,
    resolve_json_citations,
    resolve_markdown_citations,
    resolve_verifier_paths,
)
from repo2rlenv.quality.loop.requirements import with_probe_requirements
from repo2rlenv.quality.loop.runner import (
    QualityLoop,
    _reusable_probe_trial,
    control_failures,
    probe_failures,
    required_probe_focus,
)


@pytest.fixture
def task(tmp_path):
    # [FUNCTION DOC] Fixture: writes a minimal 'example' Harbor task bundle (sum-of-two-integers instruction,
    # solve.sh printing 4 to /workspace/answer.txt, a deliberately weak tests/test.sh containing
    # 'weak check') under tmp_path/tasks and returns its directory.
    return write_bundle(
        TaskBundle(
            name="example",
            org="tests",
            instruction="Write the sum of two integers to answer.txt.\n",
            files={
                "environment/Dockerfile": TaskFile.text(
                    "FROM python:3.12-slim\nWORKDIR /workspace\n"
                ),
                "solution/solve.sh": TaskFile.text(
                    "#!/bin/sh\nprintf '4' > /workspace/answer.txt\n", executable=True
                ),
                "tests/test.sh": TaskFile.text("#!/bin/sh\n# weak check\n", executable=True),
            },
            metadata={"recipe": "test", "recipe_version": "1", "reward_kinds": ["test_execution"]},
        ),
        tmp_path / "tasks",
    )


def probes():
    # [FUNCTION DOC] HELPER: Returns two SemanticProbes citing instruction.md: 'wrong-sum' (wrong_solution, writes
    # 7) and 'valid-format' (valid_alternative, writes '4' plus newline).
    evidence = [Citation(path="instruction.md", quote="sum of two integers")]
    return [
        SemanticProbe(
            name="wrong-sum",
            kind="wrong_solution",
            rationale="Wrong arithmetic",
            evidence=evidence,
            script="printf '7' > /workspace/answer.txt",
        ),
        SemanticProbe(
            name="valid-format",
            kind="valid_alternative",
            rationale="Trailing newline is valid",
            evidence=evidence,
            script="printf '4\\n' > /workspace/answer.txt",
        ),
    ]


def review(*, broken=False, propose=False, rollout="not_run"):
    # [FUNCTION DOC] HELPER: Builds a Review with passing task/leakage assessments; keyword-only broken=False makes
    # the verifier fail and adds one blocking verifier Issue citing 'weak check'; propose=False adds
    # probes() when True; rollout sets the rollout verdict (default 'not_run'). Output: Review.
    assessment = Assessment(
        status="pass",
        score=3,
        explanation="Useful behavioral task",
        evidence=[Citation(path="instruction.md", quote="sum of two integers")],
    )
    return Review(
        summary="Check an arithmetic task",
        task=assessment,
        leakage=assessment,
        verifier=assessment.model_copy(update={"status": "fail" if broken else "pass"}),
        rollout=rollout,
        issues=[
            Issue(
                category="verifier",
                severity="blocking",
                problem="Weak arithmetic check",
                repair="Exercise the sum",
                evidence=[Citation(path="tests/test.sh", quote="weak check")],
            )
        ]
        if broken
        else [],
        probes=probes() if propose else [],
        read_requests=[],
    )


class Model:
    # [CLASS DOC] Fake JSON model client used by QualityLoop. Records each request key in self.calls and returns
    # scripted Repair or Review objects.
    def __init__(self, *, honest=True, solver_failure=False):
        # [FUNCTION DOC] Inputs: keyword-only honest=True (report the weak check as broken) and
        # solver_failure=False (classify rollouts as legitimate failures). Initializes calls to an
        # empty list.
        self.calls, self.honest, self.solver_failure = [], honest, solver_failure

    def ask(self, schema, model, system, user, key):
        # [FUNCTION DOC] Appends key to calls. For schema Repair returns an edit replacing 'weak check' with
        # 'strong check' in tests/test.sh. Otherwise parses the JSON user payload and returns
        # review(): broken when honest and tests/test.sh still contains 'weak check', probes
        # proposed when probe_limit > 0, and a rollout verdict when a rollout result document is
        # present.
        self.calls.append(key)
        if schema is Repair:
            return Repair(
                explanation="Repair arithmetic coverage",
                addressed_issues=["Weak check"],
                edits=[
                    Edit(
                        path="tests/test.sh", old="weak check", new="strong check", executable=True
                    )
                ],
            )
        payload = json.loads(user)
        broken = self.honest and "weak check" in payload["documents"]["tests/test.sh"]
        rolled = any("rollout/result.json" in path for path in payload["documents"])
        outcome = "legitimate_failure" if self.solver_failure else "legitimate_success"
        return review(
            broken=broken,
            propose=payload["probe_limit"] > 0,
            rollout=outcome if rolled else "not_run",
        )


class Trials:
    # [CLASS DOC] Fake trial runner: writes an empty result.json per key under root and returns TrialRecords
    # bound by receipt.
    def __init__(self, root, *, solver_failure=False):
        # [FUNCTION DOC] Inputs: root (Path for results), keyword-only solver_failure=False. Initializes calls list
        # and closed=False.
        self.root, self.calls, self.closed = root, [], False
        self.solver_failure = solver_failure

    def run(self, task, role, key):
        # [FUNCTION DOC] Records (role, key, task identity), writes root/key/result.json, and returns a TrialRecord
        # with reward 0 for baseline (and for rollout when solver_failure), 0 for probe0 once
        # tests/test.sh contains 'strong check', else 1. probe_installed is True for probes. Side
        # effect: writes files.
        self.calls.append((role, key, task_identity(task)))
        path = self.root / key / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        reward = 0.0 if role == "baseline" or (role == "rollout" and self.solver_failure) else 1.0
        if (
            role == "probe"
            and "probe0" in key
            and "strong check" in (task / "tests/test.sh").read_text()
        ):
            reward = 0.0
        return TrialRecord(
            role=role,
            bundle_hash=task_identity(task),
            result=str(path),
            result_sha256=digest(path),
            agent=role,
            model=None,
            reward=reward,
            exception_type=None,
            binding="receipt",
            probe_installed=role == "probe",
        )

    def close(self):
        # [FUNCTION DOC] Sets closed=True so tests can assert the loop closed the runner.
        self.closed = True


def make_loop(tmp_path, *, model=None, remote=None, **options):
    # [FUNCTION DOC] HELPER: Constructs a QualityLoop with LoopOptions(**options), output tmp_path/quality, a $20
    # BudgetLedger at tmp_path/budget.sqlite3, the given model (default Model()) and trial runner
    # `remote` (None means review only).
    return QualityLoop(
        LoopOptions(**options),
        tmp_path / "quality",
        BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="20"),
        model_client=model or Model(),
        trial_runner=remote,
    )


def test_repairs_copy_and_revalidate_everything(task, tmp_path):
    # [FUNCTION DOC] Asserts a repair run ends usable with one repair, the original task untouched and a new bundle
    # hash, 9 trials total, no after-rollout model call, the r1 revision re-running
    # baseline/oracle/two probes/rollout, and the runner closed.
    before = task_identity(task)
    remote = Trials(tmp_path / "evidence")
    loop = make_loop(tmp_path, remote=remote, repair=True)
    result = loop.run(task)
    assert result.status == "usable", result.reasons
    assert result.repairs == 1
    assert task_identity(task) == before != result.bundle_hash
    assert len(remote.calls) == 9
    assert not any("after-rollout" in key for key in loop.model.calls)
    assert [role for role, key, _ in remote.calls if key.startswith("r1")] == [
        "baseline",
        "oracle",
        "probe",
        "probe",
        "rollout",
    ]
    assert remote.closed


def test_append_only_registration_revalidates_new_task_and_preserves_existing_probes(
    task, tmp_path
):
    # [FUNCTION DOC] Uses a model whose repair only appends 'new-case' to contract.json expected_passes and trials
    # where probe0 fails once it is registered; asserts usable after one repair, original contract
    # untouched, full revalidation of r1, 9 trials, and the probe definitions preserved.
    contract_path = task / "tests/contract.json"
    contract_path.write_text('{"expected_passes": ["old-case"]}')
    original_hash = refresh_identity(task)

    class AppendModel(Model):
        # [CLASS DOC] Model variant whose Repair appends an expected pass instead of editing text.
        def ask(self, schema, model, system, user, key):
            # [FUNCTION DOC] Records key; returns an append-only Repair for schema Repair, else a review that is
            # broken until 'new-case' appears in the contract document.
            self.calls.append(key)
            if schema is Repair:
                return Repair(
                    explanation="Register the existing behavioral case in the reward manifest.",
                    addressed_issues=["Weak arithmetic check"],
                    edits=[],
                    append_expected_passes=["new-case"],
                )
            payload = json.loads(user)
            broken = "new-case" not in payload["documents"]["tests/contract.json"]
            rolled = any("rollout/result.json" in path for path in payload["documents"])
            return review(
                broken=broken,
                propose=payload["probe_limit"] > 0,
                rollout="legitimate_success" if rolled else "not_run",
            )

    class AppendTrials(Trials):
        # [CLASS DOC] Trials variant where probe0's reward depends on whether 'new-case' is registered in the
        # contract.
        def run(self, current, role, key):
            # [FUNCTION DOC] Delegates to Trials.run, then for probe0 sets reward 0 when 'new-case' is in
            # expected_passes, else 1.
            trial = super().run(current, role, key)
            if role == "probe" and "probe0" in key:
                registered = json.loads((current / "tests/contract.json").read_text())[
                    "expected_passes"
                ]
                trial.reward = 0.0 if "new-case" in registered else 1.0
            return trial

    remote = AppendTrials(tmp_path / "evidence")
    runner = make_loop(tmp_path, model=AppendModel(), remote=remote, repair=True)
    result = runner.run(task)
    assert result.status == "usable", result.reasons
    assert result.repairs == 1
    assert task_identity(task) == original_hash != result.bundle_hash
    assert json.loads(contract_path.read_text())["expected_passes"] == ["old-case"]
    assert [role for role, key, _ in remote.calls if key.startswith("r1")] == [
        "baseline",
        "oracle",
        "probe",
        "probe",
        "rollout",
    ]
    assert len(remote.calls) == 9
    assert [trial.probe for trial in result.trials if trial.role == "probe"] == probes()
    assert remote.closed


def test_false_positive_cannot_be_outvoted_by_model(task, tmp_path):
    # [FUNCTION DOC] With a dishonest model, asserts a wrong_solution probe earning reward 1 forces status
    # needs_repair regardless of the model's pass verdict.
    result = make_loop(
        tmp_path, remote=Trials(tmp_path / "evidence"), model=Model(honest=False), run_rollout=True
    ).run(task)
    assert result.status == "needs_repair"
    assert any("wrong_solution earned 1" in reason for reason in result.reasons)


def test_budget_denied_control_does_not_trigger_paid_diagnosis(task, tmp_path):
    # [FUNCTION DOC] Asserts a BudgetExceeded control trial yields status budget_exhausted after only the baseline
    # trial, with no model calls and the runner closed.
    class DeniedTrials(Trials):
        # [CLASS DOC] Trials variant whose every trial reports exception_type BudgetExceeded with no reward.
        def run(self, task, role, key):
            # [FUNCTION DOC] Returns a copy of the base trial with exception_type 'BudgetExceeded' and reward None.
            trial = super().run(task, role, key)
            return trial.model_copy(update={"exception_type": "BudgetExceeded", "reward": None})

    class NoModel:
        # [CLASS DOC] Model stub that must never be called.
        def ask(self, *args):
            # [FUNCTION DOC] Fails the test via pytest.fail if invoked.
            pytest.fail("A budget denial does not need model diagnosis or repair")

    remote = DeniedTrials(tmp_path / "evidence")
    result = make_loop(tmp_path, remote=remote, model=NoModel(), run_rollout=True).run(task)
    assert result.status == "budget_exhausted"
    assert len(result.trials) == 1 and result.trials[0].exception_type == "BudgetExceeded"
    assert [role for role, *_ in remote.calls] == ["baseline"]
    assert remote.closed


def test_legitimate_solver_failure_keeps_a_sound_task(task, tmp_path):
    # [FUNCTION DOC] Asserts a rollout with reward 0 judged a legitimate solver failure still leaves the task
    # usable.
    remote = Trials(tmp_path / "evidence", solver_failure=True)
    result = make_loop(tmp_path, remote=remote, model=Model(solver_failure=True), repair=True).run(
        task
    )
    assert result.status == "usable"
    assert next(trial for trial in result.trials if trial.role == "rollout").reward == 0


@pytest.mark.parametrize("solver_failure", [False, True])
def test_post_probe_review_can_unlock_one_rollout_on_same_revision(task, tmp_path, solver_failure):
    # [FUNCTION DOC] Parametrized on solver_failure: a model that flags the task before probes but clears it after
    # passing probes; asserts usable with no repair, trials baseline/oracle/probe/probe/rollout, and
    # exactly three model calls (before, after, after-rollout).
    class PassingProbes(Trials):
        # [CLASS DOC] Trials variant where probe0 (the wrong solution) is rejected with reward 0.
        def run(self, task, role, key):
            # [FUNCTION DOC] Delegates to Trials.run and sets probe0's reward to 0.
            trial = super().run(task, role, key)
            if role == "probe" and "probe0" in key:
                trial.reward = 0.0
            return trial

    class ClearedAfterProbes(Model):
        # [CLASS DOC] Model variant scripted by request key: broken and proposing probes on 'r0-before', clean
        # afterwards, rollout verdict on 'r0-after-rollout'.
        def ask(self, schema, model, system, user, key):
            # [FUNCTION DOC] Records key, asserts the schema is Review, and returns review() configured from the
            # key prefix and the enclosing solver_failure parameter.
            self.calls.append(key)
            assert schema is Review
            outcome = "legitimate_failure" if solver_failure else "legitimate_success"
            return review(
                broken=key.startswith("r0-before"),
                propose=key.startswith("r0-before"),
                rollout=outcome if key.startswith("r0-after-rollout") else "not_run",
            )

    remote = PassingProbes(tmp_path / "evidence", solver_failure=solver_failure)
    model = ClearedAfterProbes()
    result = make_loop(tmp_path, remote=remote, model=model, run_rollout=True).run(task)
    assert result.status == "usable", result.reasons
    assert result.repairs == 0 and result.bundle_hash == task_identity(task)
    assert [role for role, _, _ in remote.calls] == [
        "baseline",
        "oracle",
        "probe",
        "probe",
        "rollout",
    ]
    assert model.calls == ["r0-before-0", "r0-after-0", "r0-after-rollout-0"]
    assert remote.closed


@pytest.mark.parametrize("cpu_denial", [False, True])
def test_post_probe_rollout_budget_denial_skips_further_model_calls(task, tmp_path, cpu_denial):
    # [FUNCTION DOC] Parametrized on cpu_denial: the rollout either raises BudgetExceeded or returns a
    # BudgetExceeded trial; asserts budget_exhausted (with the raised reason surfaced when raised)
    # and no model call after 'r0-after-0'.
    reason = "Budget scope batch-child-quality: requested $3, remaining $1 (reservation_pressure)"

    class DeniedRollout(Trials):
        # [CLASS DOC] Trials variant that denies the rollout for budget reasons and rejects probe0.
        def run(self, task, role, key):
            # [FUNCTION DOC] Raises BudgetExceeded(reason) for rollout when cpu_denial; otherwise sets probe0
            # reward 0 and marks rollout trials as BudgetExceeded with no reward.
            if role == "rollout" and cpu_denial:
                raise BudgetExceeded(reason)
            trial = super().run(task, role, key)
            if role == "probe" and "probe0" in key:
                trial.reward = 0.0
            if role == "rollout":
                trial.reward, trial.exception_type = None, "BudgetExceeded"
            return trial

    class ClearedAfterProbes(Model):
        # [CLASS DOC] Model variant that only expects the before/after review keys.
        def ask(self, schema, model, system, user, key):
            # [FUNCTION DOC] Records key, asserts it is 'r0-before-0' or 'r0-after-0', and returns a broken-with-
            # probes review before probes and a clean one after.
            self.calls.append(key)
            assert key in {"r0-before-0", "r0-after-0"}
            return review(broken=key.startswith("r0-before"), propose=key.startswith("r0-before"))

    remote = DeniedRollout(tmp_path / "evidence")
    model = ClearedAfterProbes()
    result = make_loop(tmp_path, remote=remote, model=model, run_rollout=True).run(task)
    assert result.status == "budget_exhausted"
    if cpu_denial:
        assert result.reasons == [reason]
    assert model.calls == ["r0-before-0", "r0-after-0"]
    assert remote.closed


def test_review_only_is_never_validated(task, tmp_path):
    # [FUNCTION DOC] Asserts a loop without a trial runner ends 'reviewed' with no trials and reasons mentioning
    # missing evidence.
    result = make_loop(tmp_path, model=Model(honest=False)).run(task)
    assert result.status == "reviewed"
    assert result.trials == []
    assert any("Missing" in reason for reason in result.reasons)


def test_max_repairs_keeps_bad_task_unselected(task, tmp_path):
    # [FUNCTION DOC] Asserts max_repairs=0 leaves a broken task at needs_repair with zero repairs.
    result = make_loop(
        tmp_path, remote=Trials(tmp_path / "evidence"), repair=True, max_repairs=0
    ).run(task)
    assert result.status == "needs_repair"
    assert result.repairs == 0


def test_invalid_patch_gets_one_correction_without_an_extra_execution_revision(task, tmp_path):
    # [FUNCTION DOC] Asserts a Repair whose edit 'old' text does not match gets one correction round with 'exactly
    # once' feedback, then yields usable with one repair, two repair calls and 9 trials.
    class Correcting(Model):
        # [CLASS DOC] Model variant that corrupts the first Repair edit and checks the correction feedback.
        def ask(self, schema, model, system, user, key):
            # [FUNCTION DOC] Calls Model.ask; for Repair with empty patch_feedback replaces the edit's old text
            # with invented text, otherwise asserts the feedback mentions 'exactly once'.
            result = super().ask(schema, model, system, user, key)
            if schema is Repair:
                payload = json.loads(user)
                if not payload["patch_feedback"]:
                    result.edits[0].old = "weak check with invented surrounding code"
                else:
                    assert "exactly once" in payload["patch_feedback"][0]
            return result

    model = Correcting()
    remote = Trials(tmp_path / "results")
    result = make_loop(tmp_path, model=model, remote=remote, repair=True).run(task)
    assert result.status == "usable"
    assert result.repairs == 1
    assert len([key for key in model.calls if "repair" in key]) == 2
    assert len(remote.calls) == 9


def test_repeated_invalid_patch_stops_without_changing_task(task, tmp_path):
    # [FUNCTION DOC] Asserts a Repair that never matches stops after two repair calls with needs_evidence, no
    # repairs and an unchanged bundle hash.
    class Invalid(Model):
        # [CLASS DOC] Model variant whose every Repair edit has non-matching old text.
        def ask(self, schema, *args):
            # [FUNCTION DOC] Calls Model.ask and, for Repair, sets the first edit's old text to 'invented text'.
            result = super().ask(schema, *args)
            if schema is Repair:
                result.edits[0].old = "invented text"
            return result

    model = Invalid()
    result = make_loop(tmp_path, model=model, repair=True).run(task)
    assert result.status == "needs_evidence"
    assert result.repairs == 0
    assert len([key for key in model.calls if "repair" in key]) == 2
    assert result.bundle_hash == task_identity(task)


def test_probe_collisions_preserve_distinct_scripts_and_retained_controls():
    # [FUNCTION DOC] Asserts distinct_probes drops proposals duplicating a retained probe's script, renames a name-
    # colliding new script, dedupes repeats, is deterministic, and does not mutate the retained
    # list.
    retained = probes()
    collision = retained[0].model_copy(update={"script": "printf '8' > /workspace/answer.txt"})
    repeated = retained[1].model_copy(update={"name": "renamed-repeat"})
    before = [p.model_dump() for p in retained]
    normalized = distinct_probes([collision, repeated, collision], retained)
    assert len(normalized) == 1
    assert normalized[0].name not in {p.name for p in retained}
    assert normalized[0].model_dump(exclude={"name"}) == collision.model_dump(exclude={"name"})
    assert distinct_probes([collision], retained) == normalized
    assert [p.model_dump() for p in retained] == before


@pytest.mark.parametrize(
    ("quote", "data"),
    [
        ('"test_result": "failed"', {"statuses": {"tests.behavior::test_result": "failed"}}),
        (
            '"probe_failures": ["wrong solution passed"]',
            {"probe_failures": ["wrong solution passed"]},
        ),
    ],
)
def test_json_citation_resolution_keeps_decisions_and_uses_actual_bytes(quote, data):
    # [FUNCTION DOC] Asserts resolve_json_citations rewrites a JSON-ish quote to text actually present in the
    # pretty-printed document (one change), preserving status/explanation and not mutating the input
    # review.
    decision = review()
    decision.verifier = decision.verifier.model_copy(deep=True)
    decision.verifier.evidence = [Citation(path="evidence/checks.json", quote=quote)]
    before = decision.model_dump()
    document = json.dumps(data, indent=2)
    normalized, changes = resolve_json_citations(decision, {"evidence/checks.json": document})
    assert len(changes) == 1
    assert normalized.verifier.evidence[0].quote in document
    assert normalized.verifier.status == decision.verifier.status
    assert normalized.verifier.explanation == decision.verifier.explanation
    assert decision.model_dump() == before


@pytest.mark.parametrize(
    "data",
    [
        {"a::test_result": "failed", "b::test_result": "failed"},
        {"a::test_result": "passed"},
        {"a::another_test": "failed"},
        {"message": 'the text says "test_result": "failed"'},
    ],
)
def test_json_citations_do_not_resolve_ambiguous_or_unsupported_claims(data):
    # [FUNCTION DOC] Asserts resolve_json_citations makes no change for ambiguous matches, contradicting values,
    # different keys, or text only inside a string value.
    decision = review()
    decision.verifier.evidence = [
        Citation(path="evidence/checks.json", quote='"test_result": "failed"')
    ]
    normalized, changes = resolve_json_citations(
        decision, {"evidence/checks.json": json.dumps(data)}
    )
    assert not changes and normalized == decision


def test_markdown_citation_restores_only_missing_inline_formatting():
    # [FUNCTION DOC] Asserts resolve_markdown_citations restores missing backticks so the quote matches
    # instruction.md, changing only task evidence and not mutating the input.
    quote = "MetaClip2TextModel – text encoder; `pooler_output` selects the first EOS token."
    document = (
        "- `MetaClip2TextModel` – text encoder; `pooler_output` selects the first EOS token.\n"
    )
    decision = review()
    decision.task = decision.task.model_copy(deep=True)
    decision.task.evidence = [Citation(path="instruction.md", quote=quote)]
    before = decision.model_dump()
    normalized, changes = resolve_markdown_citations(decision, {"instruction.md": document})
    assert len(changes) == 1
    assert normalized.task.evidence[0].quote in document
    assert normalized.model_dump(exclude={"task"}) == decision.model_dump(exclude={"task"})
    assert normalized.task.model_dump(exclude={"evidence"}) == decision.task.model_dump(
        exclude={"evidence"}
    )
    assert decision.model_dump() == before


@pytest.mark.parametrize(
    ("path", "document"),
    [
        ("instruction.md", "`First` selects 2 tokens from the incoming sequence.\n"),
        ("instruction.md", "`First` selects 1 token from the incoming sequence.\n" * 2),
        ("instruction.md", "```python\n`First` selects 1 token from the incoming sequence.\n```"),
        ("instruction.md", "~~~text\n`First` selects 1 token from the incoming sequence.\n~~~"),
        ("instruction.md", "    `First` selects 1 token from the incoming sequence.\n"),
        ("instruction.md", "\\`First\\` selects 1 token from the incoming sequence.\n"),
        ("instruction.md", "``First`` selects 1 token from the incoming sequence.\n"),
        ("source.py", "`First` selects 1 token from the incoming sequence.\n"),
    ],
)
def test_markdown_citations_keep_unsupported_or_ambiguous_quotes(path, document):
    # [FUNCTION DOC] Asserts no Markdown citation rewrite for differing text, duplicates, fenced/indented code,
    # escaped or double backticks, or non-Markdown files.
    decision = review()
    decision.task.evidence = [
        Citation(path=path, quote="First selects 1 token from the incoming sequence.")
    ]
    normalized, changes = resolve_markdown_citations(decision, {path: document})
    assert not changes and normalized == decision


def test_large_private_pr_context_is_bounded_and_remains_searchable(task, tmp_path):
    # [FUNCTION DOC] With a huge merged_pr source_diff, asserts the task-context excerpt is <= 4000 chars ending
    # with 'Excerpt ends', omits the tail marker, is in the inventory, and that a search read
    # retrieves the marker; payload has documents.
    context_data = {
        "kind": "merged_pr",
        "source_diff": "unchanged line\n" * 30000 + "UNIQUE_CONTRACT: correct behavior",
    }
    loop = QualityLoop(
        LoopOptions(context_chars=16000),
        tmp_path / "bounded-context",
        BudgetLedger(tmp_path / "bounded.sqlite3", limit_usd="1"),
        task_context=context_data,
    )
    context = loop._context(task, [])
    excerpt = context.documents["evidence/task-context.json"]
    assert len(excerpt) <= 4000 and "Excerpt ends" in excerpt
    assert "UNIQUE_CONTRACT" not in excerpt
    assert any(item["path"] == "evidence/task-context.json" for item in context.inventory)
    context.read_more(
        [
            ReadRequest(
                path="evidence/task-context.json", query="UNIQUE_CONTRACT", start_line=1, end_line=1
            )
        ]
    )
    assert (
        "UNIQUE_CONTRACT: correct behavior"
        in context.documents["evidence/task-context.json:search=UNIQUE_CONTRACT"]
    )
    assert "documents" in json.loads(context.payload())


def test_saved_repair_findings_precede_long_pr_patch_without_more_context(task, tmp_path):
    # [FUNCTION DOC] Asserts campaign_design_guidance diagnosis is placed in the bounded task-context excerpt ahead
    # of the long diff, total context <= 16000, the input dict is not mutated, and the diff tail is
    # still searchable.
    context_data = {
        "kind": "merged_pr",
        "source_diff": "unchanged line\n" * 30000 + "UNIQUE_SOURCE: retained patch",
        "campaign_design_guidance": {
            "current_diagnosis": "Distinct pixel values must reject the known naive reshape.",
        },
    }
    original = json.dumps(context_data)
    loop = QualityLoop(
        LoopOptions(context_chars=16000),
        tmp_path / "diagnosis-context",
        BudgetLedger(tmp_path / "diagnosis.sqlite3", limit_usd="1"),
        task_context=context_data,
    )
    context = loop._context(task, [])
    excerpt = context.documents["evidence/task-context.json"]
    assert "Distinct pixel values must reject the known naive reshape." in excerpt
    assert "UNIQUE_SOURCE" not in excerpt
    assert len(excerpt) <= 4000 and "Excerpt ends" in excerpt
    assert sum(map(len, context.documents.values())) <= 16000
    assert json.dumps(context_data) == original
    context.read_more(
        [
            ReadRequest(
                path="evidence/task-context.json", query="UNIQUE_SOURCE", start_line=1, end_line=1
            )
        ]
    )
    assert (
        "UNIQUE_SOURCE: retained patch"
        in context.documents["evidence/task-context.json:search=UNIQUE_SOURCE"]
    )


def test_short_verifier_path_requires_unique_exact_match(task):
    # [FUNCTION DOC] Asserts resolve_verifier_paths maps a short tests/ path to the unique tests/source/tests file
    # containing the old text exactly, without mutating the input or the file, and leaves the repair
    # unchanged when ambiguous, unmatched, or old is empty.
    path = task / "tests/source/tests/contract.py"
    path.parent.mkdir(parents=True)
    path.write_text("assert actual == expected\n")
    repair = Repair(
        explanation="Fix fixture",
        addressed_issues=["fixture"],
        edits=[
            Edit(
                path="tests/contract.py",
                old="assert actual == expected",
                new="assert result == expected",
                executable=False,
            )
        ],
    )
    resolved = resolve_verifier_paths(task, repair)
    assert resolved.edits[0].path == "tests/source/tests/contract.py"
    assert repair.edits[0].path == "tests/contract.py"
    assert path.read_text() == "assert actual == expected\n"
    other = task / "tests/other/contract.py"
    other.parent.mkdir()
    other.write_text(path.read_text())
    assert resolve_verifier_paths(task, repair) == repair
    other.unlink()
    path.write_text("assert actual == unexpected\n")
    assert resolve_verifier_paths(task, repair) == repair
    repair.edits[0].old = ""
    assert resolve_verifier_paths(task, repair) == repair


def test_path_resolution_never_redirects_to_source_or_outside_task(task, tmp_path):
    # [FUNCTION DOC] Asserts resolution does not follow a symlink outside the task or redirect a missing path, and
    # raises ValueError for a path traversing into solution/.
    outside = tmp_path / "contract.py"
    outside.write_text("original")
    (task / "tests/contract.py").symlink_to(outside)
    repair = Repair(
        explanation="Fixture",
        addressed_issues=["fixture"],
        edits=[
            Edit(
                path="tests/missing/contract.py",
                old="original",
                new="replacement",
                executable=False,
            )
        ],
    )
    assert resolve_verifier_paths(task, repair) == repair
    repair.edits[0].path = "tests/../solution/solve.sh"
    with pytest.raises(ValueError):
        resolve_verifier_paths(task, repair)


@pytest.mark.parametrize(
    "path", ["../escape", "/tmp/escape", "environment/../../escape", "task.toml"]
)
def test_repair_cannot_escape_or_rewrite_execution_policy(task, tmp_path, path):
    # [FUNCTION DOC] Asserts apply_repair raises ValueError for paths escaping the task or targeting task.toml and
    # creates no revision directory.
    patch = Repair(
        explanation="bad edit",
        addressed_issues=["test"],
        edits=[Edit(path=path, old="", new="x", executable=False)],
    )
    with pytest.raises(ValueError):
        apply_repair(task, patch, tmp_path / "revision")
    assert not (tmp_path / "revision").exists()


def test_ambiguous_edit_is_atomic(task, tmp_path):
    # [FUNCTION DOC] Asserts an edit whose old text is not found raises ValueError 'exactly once' and creates no
    # revision.
    patch = Repair(
        explanation="bad match",
        addressed_issues=["test"],
        edits=[Edit(path="instruction.md", old="not found", new="changed", executable=False)],
    )
    with pytest.raises(ValueError, match="exactly once"):
        apply_repair(task, patch, tmp_path / "revision")
    assert not (tmp_path / "revision").exists()


def test_probe_keeps_task_and_verifier_bytes(task, tmp_path):
    # [FUNCTION DOC] Asserts probe_variant keeps instruction, tests and Dockerfile bytes, wraps solve.sh with the
    # completion marker, and saves the original as quality-original-solve.sh.
    revised = probe_variant(task, probes()[0], tmp_path / "probe" / task.name)
    for file in ["instruction.md", "tests/test.sh", "environment/Dockerfile"]:
        assert (task / file).read_bytes() == (revised / file).read_bytes()
    assert "__QUALITY_PROBE_COMPLETED__" in (revised / "solution/solve.sh").read_text()
    assert (revised / "solution/quality-original-solve.sh").read_bytes() == (
        task / "solution/solve.sh"
    ).read_bytes()


def test_repository_probe_audits_the_collected_submission(task, tmp_path):
    # [FUNCTION DOC] With submitted_files in the contract, asserts the probe wrapper snapshots before the script
    # and after it before printing the marker, ships quality-probe-audit.py and a boundary contract
    # with only submitted_files, and keeps tests/contract.json bytes.
    contract = task / "tests/contract.json"
    contract.write_text(
        json.dumps({"submitted_files": ["answer.txt"], "expected_passes": ["private"]})
    )
    refresh_identity(task)
    revised = probe_variant(task, probes()[0], tmp_path / "probe" / task.name)
    wrapper = (revised / "solution/solve.sh").read_text()
    assert wrapper.index("before\n") < wrapper.index(shlex.quote(probes()[0].script))
    assert wrapper.index("after\n") < wrapper.index("__QUALITY_PROBE_COMPLETED__")
    assert (revised / "solution/quality-probe-audit.py").is_file()
    boundary = json.loads((revised / "solution/quality-probe-contract.json").read_text())
    assert boundary["submitted_files"] == ["answer.txt"]
    assert "expected_passes" not in boundary
    assert (revised / "tests/contract.json").read_bytes() == contract.read_bytes()


@pytest.mark.parametrize(
    "ending,change,success",
    [
        ("exit 0", True, True),
        ("exit 7", True, False),
        ("false", True, False),
        ("exit 0", False, False),
    ],
)
def test_probe_shell_exit_cannot_bypass_the_change_audit(task, tmp_path, ending, change, success):
    # Execute only this owned wrapper and audit against a tiny text fixture.
    # Target repositories and their tests are never imported locally.
    # [FUNCTION DOC] Executes the generated probe wrapper locally via bash with paths rewritten into tmp_path;
    # asserts success (exit 0, completed and changed-files markers, before/after JSON) only when the
    # script exits 0 and actually changed answer.txt. Side effect: subprocess with PATH prefixed by
    # the current interpreter dir.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    contract = task / "tests/contract.json"
    contract.write_text(json.dumps({"submitted_files": ["answer.txt"]}))
    refresh_identity(task)
    script = (probes()[0].script + "\n" if change else "") + ending
    probe = probes()[0].model_copy(update={"script": script})
    revised = probe_variant(task, probe, tmp_path / "probe" / task.name)
    substitutions = {
        "/solution/": str(revised / "solution") + "/",
        "/workspace": str(workspace),
        "/tmp/quality-probe-before.json": str(tmp_path / "before.json"),
    }
    for name in ["solve.sh", "quality-original-solve.sh", "quality-probe-audit.py"]:
        path = revised / "solution" / name
        text = path.read_text()
        for old, new in substitutions.items():
            text = text.replace(old, new)
        path.write_text(text)
    result = subprocess.run(
        ["bash", str(revised / "solution/solve.sh")],
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        },
    )
    assert (workspace / "answer.txt").read_text() == ("7" if change else "4")
    assert (result.returncode == 0) is success
    assert ("__QUALITY_PROBE_COMPLETED__" in result.stdout) is success
    assert ("__QUALITY_PROBE_CHANGED_FILES__" in result.stdout) is success
    if success:
        changes = json.loads(result.stdout.splitlines()[0].split(" ", 1)[1])
        assert set(changes) == {"answer.txt"}
        assert changes["answer.txt"]["before"] != changes["answer.txt"]["after"]


def test_fabricated_citation_is_rejected(task):
    # [FUNCTION DOC] Asserts EvidenceContext.validate_review raises 'not grounded' for an invented quote.
    context = EvidenceContext(task, [], limit=16000)
    value = review()
    value.task.evidence[0].quote = "invented evidence"
    with pytest.raises(ValueError, match="not grounded"):
        context.validate_review(value)


def raw_trial(task, tmp_path, *, agent="terminus-2"):
    # [FUNCTION DOC] HELPER: Writes tmp_path/raw/result.json mimicking a native Harbor trial (task checksum from
    # harbor Task, agent name, reward 1) and returns its path. Input: keyword-only agent (default
    # 'terminus-2').
    from harbor.models.task.task import Task

    path = tmp_path / "raw" / "result.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "task_checksum": Task(task).checksum,
                "config": {"agent": {"name": agent, "model_name": "test"}},
                "verifier_result": {"rewards": {"reward": 1}},
            }
        )
    )
    return path


def test_native_harbor_rollout_is_bound_by_checksum(task, tmp_path):
    # [FUNCTION DOC] Asserts import_trial binds a native result by harbor_checksum and raises 'checksum' once the
    # checksum is altered.
    path = raw_trial(task, tmp_path)
    evidence = import_trial(path, task, "rollout")
    assert evidence.binding == "harbor_checksum"
    data = json.loads(path.read_text())
    data["task_checksum"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        import_trial(path, task, "rollout")


def test_oracle_cannot_be_imported_as_blind_solver(task, tmp_path):
    # [FUNCTION DOC] Asserts importing an oracle-agent result as a rollout raises ValueError mentioning solver.
    path = raw_trial(task, tmp_path, agent="oracle")
    with pytest.raises(ValueError, match="solver"):
        import_trial(path, task, "rollout")


def test_receipt_cannot_redirect_to_another_directory(task, tmp_path):
    # [FUNCTION DOC] Asserts a receipt whose trial_id contains '..' raises 'invalid trial ID'.
    receipt = tmp_path / "trial.json"
    receipt.write_text(
        json.dumps(
            {
                "state": "completed",
                "bundle_hash": task_identity(task),
                "trial_id": "../outside",
            }
        )
    )
    with pytest.raises(ValueError, match="invalid trial ID"):
        import_trial(receipt, task, "rollout")


def test_oracle_exit_failure_and_probe_install_failure_are_not_valid_controls(task, tmp_path):
    # [FUNCTION DOC] Asserts control_failures flags an oracle with nonzero agent exit code and probe_failures flags
    # a probe that was not installed.
    item = Trials(tmp_path / "results").run(task, "oracle", "one")
    assert "oracle did not execute cleanly" in control_failures(
        [item.model_copy(update={"agent_exit_code": 1})], 1
    )
    item = item.model_copy(
        update={"role": "probe", "probe": probes()[0], "reward": 0, "probe_installed": False}
    )
    assert "install/execute cleanly" in probe_failures([item], 1)[0]


def test_run_budget_preserves_campaign_and_uncertain_holds(tmp_path):
    # [FUNCTION DOC] Asserts an uncertain reservation stays held so a further reservation beyond the $2 run cap
    # raises BudgetExceeded, with $1.50 still reserved.
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="10")
    budget = RunBudget(ledger, "quality-example", "2")
    budget.reserve("quality-example:one", "1.50", "review")
    budget.mark_uncertain("quality-example:one", "provider timed out")
    with pytest.raises(BudgetExceeded):
        budget.reserve("quality-example:two", "1", "review")
    assert ledger.status()["reserved_usd"] == "1.500000"


def test_changed_options_cannot_resume(task, tmp_path):
    # [FUNCTION DOC] Asserts resuming a run with different LoopOptions raises ValueError 'identical'.
    make_loop(tmp_path).run(task)
    with pytest.raises(ValueError, match="identical"):
        make_loop(tmp_path, max_repairs=1).run(task, resume=True)


def test_boolean_reward_is_invalid(task, tmp_path):
    # [FUNCTION DOC] Asserts import_trial rejects a boolean reward value with ValueError.
    path = raw_trial(task, tmp_path)
    data = json.loads(path.read_text())
    data["verifier_result"]["rewards"]["reward"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        import_trial(path, task, "rollout")


def test_completed_model_response_is_reused_without_another_paid_call(tmp_path, monkeypatch):
    # [FUNCTION DOC] Monkeypatches key resolution and llm.complete; asserts JsonModel caches a completed response
    # by key (one call, $0.01 accounted) and that reusing the key with a different prompt raises
    # ModelRequestError without calling again.
    from repo2rlenv.llm import LLMResponse

    calls = []
    monkeypatch.setattr("repo2rlenv.quality.loop.client.resolve_llm_api_key", lambda *args: "test")

    def complete(*args, **kwargs):
        # [FUNCTION DOC] Fake LLM completion: records kwargs and returns an LLMResponse with a serialized review
        # and cost 0.01.
        calls.append(kwargs)
        return LLMResponse(
            content=review().model_dump_json(), usage={"prompt_tokens": 10}, cost_usd=0.01
        )

    monkeypatch.setattr("repo2rlenv.campaigns.llm.complete", complete)
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="10")
    client = JsonModel(
        tmp_path / "calls", RunBudget(ledger, "quality-test", "2"), reservation="1", max_tokens=2000
    )
    args = (Review, LoopOptions().review_model, "system", "user", "review")
    assert client.ask(*args) == client.ask(*args)
    assert len(calls) == 1
    assert ledger.status()["accounted_usd"] == "0.010000"
    with pytest.raises(ModelRequestError, match="request failed"):
        client.ask(Review, LoopOptions().review_model, "system", "changed user", "review")
    assert len(calls) == 1


def test_uncertain_model_request_is_never_dispatched_again(tmp_path, monkeypatch):
    # [FUNCTION DOC] Asserts a request that timed out is marked uncertain: both attempts raise ModelRequestError,
    # only one dispatch happens, and $1 stays reserved.
    monkeypatch.setattr("repo2rlenv.quality.loop.client.resolve_llm_api_key", lambda *args: "test")
    calls = []

    def complete(*args, **kwargs):
        # [FUNCTION DOC] Fake LLM completion that records kwargs and raises TimeoutError.
        calls.append(kwargs)
        raise TimeoutError("uncertain provider response")

    monkeypatch.setattr("repo2rlenv.campaigns.llm.complete", complete)
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="10")
    client = JsonModel(
        tmp_path / "calls", RunBudget(ledger, "quality-test", "2"), reservation="1", max_tokens=2000
    )
    for _ in range(2):
        with pytest.raises(ModelRequestError):
            client.ask(Review, LoopOptions().review_model, "system", "user", "review")
    assert len(calls) == 1
    assert ledger.status()["reserved_usd"] == "1.000000"


def test_missing_credentials_do_not_reserve_money(tmp_path, monkeypatch):
    # [FUNCTION DOC] Asserts a missing API key raises ValueError 'No API key' before any budget operation is
    # recorded.
    monkeypatch.setattr("repo2rlenv.quality.loop.client.resolve_llm_api_key", lambda *args: None)
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="10")
    client = JsonModel(
        tmp_path / "calls", RunBudget(ledger, "quality-test", "2"), reservation="1", max_tokens=2000
    )
    with pytest.raises(ValueError, match="No API key"):
        client.ask(Review, LoopOptions().review_model, "system", "user", "review")
    assert ledger.status()["operations"] == []


def test_fake_citations_cannot_trigger_a_repair(task, tmp_path):
    # [FUNCTION DOC] With zero read rounds, asserts a review citing invented text ends needs_evidence with no
    # repair after one model call.
    class Ungrounded(Model):
        # [CLASS DOC] Model variant that replaces the task evidence quote with invented text.
        def ask(self, *args):
            # [FUNCTION DOC] Calls Model.ask and overwrites the first task evidence quote.
            value = super().ask(*args)
            value.task.evidence[0].quote = "invented requirement"
            return value

    model = Ungrounded()
    result = make_loop(tmp_path, model=model, max_read_rounds=0, repair=True).run(task)
    assert result.status == "needs_evidence"
    assert result.repairs == 0
    assert len(model.calls) == 1


def test_trial_artifact_symlink_cannot_escape_evidence_root(task, tmp_path):
    # [FUNCTION DOC] Asserts EvidenceContext raises 'symlinks' when a trial artifact directory is a symlink
    # pointing outside.
    item = Trials(tmp_path / "results").run(task, "rollout", "one")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "trajectory.json").write_text("private material")
    (Path(item.result).parent / "agent").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        EvidenceContext(task, [item], limit=16000)


def test_bounded_search_finds_relevant_code_without_loading_whole_repository(task):
    # [FUNCTION DOC] Asserts a search read finds 'printf' in solve.sh and a path outside the inventory raises
    # 'inventory'.
    context = EvidenceContext(task, [], limit=16000)
    context.read_more(
        [ReadRequest(path="solution/solve.sh", query="printf", start_line=1, end_line=1)]
    )
    assert "printf" in context.documents["solution/solve.sh:search=printf"]
    with pytest.raises(ValueError, match="inventory"):
        context.read_more([ReadRequest(path="../../secret", query=None, start_line=1, end_line=2)])


def test_prior_revision_controls_cannot_validate_current_task(task, tmp_path):
    # [FUNCTION DOC] Asserts _context rejects a trial bound to another bundle hash with 'another task revision'.
    evidence = Trials(tmp_path / "results").run(task, "oracle", "one")
    evidence.bundle_hash = "sha256:" + "0" * 64
    loop = make_loop(tmp_path)
    with pytest.raises(ValueError, match="another task revision"):
        loop._context(task, [evidence])


def test_generic_probes_cannot_cover_explicit_laziness(task, tmp_path):
    # [FUNCTION DOC] After adding 'Yield lazily' to the instruction, asserts required_probe_focus is
    # {'lazy_output'} and generic probes leave the task needs_evidence without running probes or
    # rollout.
    patch = Repair(
        explanation="Add an output contract",
        addressed_issues=["contract"],
        edits=[
            Edit(
                path="instruction.md",
                old="Write the sum",
                new="Yield lazily. Write the sum",
                executable=False,
            )
        ],
    )
    revised = apply_repair(task, patch, tmp_path / "input" / task.name)
    assert required_probe_focus(revised) == {"lazy_output"}
    result = make_loop(
        tmp_path,
        remote=Trials(tmp_path / "results"),
        model=Model(honest=False),
        run_rollout=True,
        max_read_rounds=0,
    ).run(revised)
    assert result.status == "needs_evidence"
    assert not any(item.role in {"rollout", "probe"} for item in result.trials)


def test_generator_input_does_not_imply_lazy_output(task, tmp_path):
    # [FUNCTION DOC] Asserts mentioning generator inputs does not create a lazy_output probe requirement.
    patch = Repair(
        explanation="Document inputs",
        addressed_issues=["contract"],
        edits=[
            Edit(
                path="instruction.md",
                old="two integers",
                new="integers supplied by generators",
                executable=False,
            )
        ],
    )
    revised = apply_repair(task, patch, tmp_path / "input" / task.name)
    assert required_probe_focus(revised) == set()


@pytest.mark.parametrize("focus", ["compiled_execution", "model_behavior"])
def test_explicit_requirement_rejects_generic_probes(task, tmp_path, focus):
    # [FUNCTION DOC] For each focus, asserts with_probe_requirements adds the requirement only to the copy, and
    # generic probes yield needs_evidence naming the focus with no probe/rollout trials.
    revised = with_probe_requirements(task, tmp_path / "input" / task.name, [focus])
    assert required_probe_focus(task) == set()
    assert required_probe_focus(revised) == {focus}
    result = make_loop(
        tmp_path,
        remote=Trials(tmp_path / "results"),
        model=Model(honest=False),
        run_rollout=True,
        max_read_rounds=0,
    ).run(revised)
    assert result.status == "needs_evidence"
    assert focus in result.reasons[0]
    assert not any(item.role in {"rollout", "probe"} for item in result.trials)


@pytest.mark.parametrize("focus", ["compiled_execution", "model_behavior"])
def test_explicit_probe_requirement_survives_task_repair(task, tmp_path, focus):
    # [FUNCTION DOC] Asserts QualityLoop._repair produces a new revision that still requires the focus and returns
    # the retained probes unchanged.
    revised = with_probe_requirements(task, tmp_path / "input" / task.name, [focus])
    loop = make_loop(tmp_path, repair=True)
    retained = probes()
    retained[0].focus = focus
    repaired, resulting_probes = loop._repair(
        revised, review(broken=True), loop._context(revised, []), retained, 0, ["Weak check"]
    )
    assert task_identity(repaired) != task_identity(revised)
    assert required_probe_focus(repaired) == {focus}
    assert resulting_probes == retained


@pytest.mark.parametrize("focus", ["compiled_execution", "model_behavior"])
def test_repair_cannot_remove_explicit_probe_metadata(task, tmp_path, focus):
    # [FUNCTION DOC] Asserts editing task.toml to drop the focus raises ValueError, keeps the requirement and
    # creates no revision.
    revised = with_probe_requirements(task, tmp_path / "input" / task.name, [focus])
    repair = Repair(
        explanation="Try to bypass the requirement",
        addressed_issues=["required control"],
        edits=[Edit(path="task.toml", old=focus, new="general", executable=False)],
    )
    with pytest.raises(ValueError):
        apply_repair(revised, repair, tmp_path / "revision" / task.name)
    assert required_probe_focus(revised) == {focus}
    assert not (tmp_path / "revision" / task.name).exists()


def test_known_counterexamples_are_bound_and_replayed(task, tmp_path):
    # [FUNCTION DOC] Asserts a probes.json bound to the task hash is replayed (usable after one repair) and one
    # bound to another hash raises 'match the task'.
    path = tmp_path / "probes.json"
    path.write_text(
        json.dumps(
            {"bundle_hash": task_identity(task), "probes": [p.model_dump() for p in probes()]}
        )
    )
    result = make_loop(tmp_path, remote=Trials(tmp_path / "results"), repair=True).run(
        task, probes=path
    )
    assert result.status == "usable"
    assert result.repairs == 1
    path.write_text(
        json.dumps(
            {"bundle_hash": "sha256:" + "0" * 64, "probes": [p.model_dump() for p in probes()]}
        )
    )
    with pytest.raises(ValueError, match="match the task"):
        make_loop(tmp_path).run(task, probes=path)


def test_openai_schema_requires_focus_even_for_legacy_read_defaults():
    # [FUNCTION DOC] Asserts response_schema(Review) marks SemanticProbe.focus required and without a default
    # (strict structured outputs).
    schema = response_schema(Review)
    assert "focus" in schema["$defs"]["SemanticProbe"]["required"]
    assert "default" not in schema["$defs"]["SemanticProbe"]["properties"]["focus"]


def test_wrong_solution_proposal_requires_runtime_evidence_authorization():
    # The schema accepts a proposal; only the loop can inspect its execution
    # history. Missing proof and installed controls are rejected in protocol tests.
    # [FUNCTION DOC] Asserts the Repair schema itself accepts a wrong_solution probe replacement (authorization is
    # enforced by the loop elsewhere).
    proposal = Repair(
        explanation="Correct a control whose installation failed",
        addressed_issues=["probe"],
        edits=[],
        probe_replacements=[probes()[0]],
    )
    assert proposal.probe_replacements[0].kind == "wrong_solution"


class AlternativeRepairModel(Model):
    # [CLASS DOC] Model variant whose repair replaces only the valid_alternative probe (adds a '# corrected'
    # script) and whose post-probe review raises a blocking issue in a configurable category.
    def __init__(self, *, category="probe", replacement=None):
        # [FUNCTION DOC] Inputs: keyword-only category='probe' (issue category after probes) and replacement
        # (SemanticProbe; default a corrected copy of probes()[1]).
        super().__init__()
        self.category = category
        self.replacement = replacement or probes()[1].model_copy(
            update={"script": "printf '4\\n' > /workspace/answer.txt # corrected"}
        )

    def ask(self, schema, model, system, user, key):
        # [FUNCTION DOC] Records key; returns a probe-replacement-only Repair for schema Repair, else a review
        # proposing probes when allowed, with a blocking issue on 'r0-after' keys.
        self.calls.append(key)
        if schema is Repair:
            return Repair(
                explanation="Correct the alternative's diagnosed implementation bug",
                addressed_issues=["Invalid alternative implementation"],
                edits=[],
                probe_replacements=[self.replacement],
            )
        payload = json.loads(user)
        rolled = any("rollout/result.json" in path for path in payload["documents"])
        result = review(
            propose=payload["probe_limit"] > 0,
            rollout="legitimate_success" if rolled else "not_run",
        )
        if key.startswith("r0-after"):
            result.issues = [
                Issue(
                    category=self.category,
                    severity="blocking",
                    problem="Invalid alternative implementation",
                    repair="Correct the probe while preserving the grading contract",
                    evidence=result.task.evidence,
                )
            ]
        return result


class AlternativeTrials(Trials):
    # [CLASS DOC] Trials variant where only a corrected valid_alternative earns reward and results are written
    # in the real adapters' receipt layout.
    def run(self, task, role, key):
        # [FUNCTION DOC] Delegates to Trials.run; for probes sets reward 1.0 only for probe1 whose solve.sh
        # contains 'corrected', else 0; then bind_result.
        result = super().run(task, role, key)
        if role == "probe":
            correct_alternative = "corrected" in (task / "solution/solve.sh").read_text()
            result.reward = float("probe1" in key and correct_alternative)
        return self.bind_result(task, result, key)

    def bind_result(self, task, result, key):
        # [FUNCTION DOC] HELPER: Rewrites the trial into the real receipt layout: sets agent name and exit code 0,
        # writes root/key/test-<key>/<task>/result.json, agent/exit-code.txt, the probe completion
        # marker for probes, and root/key/trial.json receipt; updates result path and sha256.
        # Returns the mutated TrialRecord.
        """Preserve the same local receipt layout as both real trial adapters."""
        role = result.role
        result.agent = {"baseline": "nop", "oracle": "oracle", "probe": "oracle"}.get(
            role, "terminus-2"
        )
        result.agent_exit_code = 0
        trial_id = "test-" + key
        path = self.root / key / trial_id / task.name / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "config": {
                        "task": {"path": str(task)},
                        "agent": {"name": result.agent, "model_name": result.model},
                    },
                    "verifier_result": {"rewards": {"reward": result.reward}},
                    "exception_info": {"exception_type": result.exception_type}
                    if result.exception_type
                    else None,
                }
            )
        )
        logs = path.parent / "agent"
        logs.mkdir(exist_ok=True)
        (logs / "exit-code.txt").write_text("0")
        if role == "probe":
            (logs / "oracle.txt").write_text("__QUALITY_PROBE_COMPLETED__\n")
        result.result = str(path)
        result.result_sha256 = digest(path)
        (self.root / key / "trial.json").write_text(
            json.dumps(
                {
                    "state": "completed",
                    "trial_id": trial_id,
                    "bundle_hash": result.bundle_hash,
                    "result_sha256": result.result_sha256,
                }
            )
        )
        return result


def test_probe_only_repair_reuses_controls_and_preserves_counterexample(task, tmp_path):
    # [FUNCTION DOC] Asserts a probe-only repair keeps the bundle hash, reruns only probe1 and rollout in r1,
    # reuses the r0 wrong-solution trial (reward 0), and writes revisions/r1/repair.json without an
    # r1-probe0 directory.
    remote = AlternativeTrials(tmp_path / "results")
    result = make_loop(tmp_path, remote=remote, model=AlternativeRepairModel(), repair=True).run(
        task
    )
    assert result.status == "usable", result.reasons
    assert result.repairs == 1
    assert result.bundle_hash == task_identity(task)
    assert [role for role, key, _ in remote.calls if key.startswith("r1")] == [
        "probe",
        "rollout",
    ]
    wrong = next(
        item for item in result.trials if item.probe and item.probe.kind == "wrong_solution"
    )
    assert wrong.probe == probes()[0]
    assert wrong.reward == 0
    assert "/r0-probe0/" in wrong.result
    assert len([item for item in result.trials if item.role == "probe"]) == 2
    assert not (tmp_path / "quality/probes/r1-probe0").exists()
    assert (tmp_path / "quality/revisions/r1/repair.json").is_file()


def test_task_repair_invalidates_previously_successful_probes(task, tmp_path):
    # [FUNCTION DOC] Asserts that when the repair also edits the task, all r1 controls and probes rerun and every
    # trial result is from r1.
    class TaskRepairModel(AlternativeRepairModel):
        # [CLASS DOC] AlternativeRepairModel variant whose Repair also edits tests/test.sh.
        def ask(self, schema, *args):
            # [FUNCTION DOC] Calls the parent and, for Repair, sets edits to replace 'weak check' with 'strong
            # check'.
            response = super().ask(schema, *args)
            if schema is Repair:
                response.edits = [
                    Edit(
                        path="tests/test.sh",
                        old="weak check",
                        new="strong check",
                        executable=True,
                    )
                ]
            return response

    remote = AlternativeTrials(tmp_path / "results")
    result = make_loop(tmp_path, remote=remote, model=TaskRepairModel(), repair=True).run(task)
    assert result.status == "usable", result.reasons
    assert result.bundle_hash != task_identity(task)
    assert [key for _, key, _ in remote.calls if key.startswith("r1")] == [
        "r1-baseline",
        "r1-oracle",
        "r1-probe0",
        "r1-probe1",
        "r1-rollout",
    ]
    assert all("/r1-" in trial.result for trial in result.trials)


@pytest.mark.parametrize("failure", ["reward", "infrastructure", "uninstalled", "exit"])
def test_probe_only_repair_reruns_unchanged_failed_probes(task, tmp_path, failure):
    # [FUNCTION DOC] For each failure mode of the original probe0 (reward 1, infrastructure error, not installed,
    # nonzero exit), asserts a probe-only repair reruns probe0 as well as probe1 and rollout.
    class RecoveringTrials(AlternativeTrials):
        # [CLASS DOC] AlternativeTrials variant that damages the r0-probe0 trial according to `failure`.
        def run(self, task, role, key):
            # [FUNCTION DOC] For key r0-probe0 applies the failure (reward 1, exception type, not installed, or
            # exit code 1 written to exit-code.txt) and rebinds the result.
            result = super().run(task, role, key)
            if key == "r0-probe0":
                if failure == "reward":
                    result.reward = 1.0
                elif failure == "infrastructure":
                    result.exception_type = "AgentExecutionError"
                elif failure == "uninstalled":
                    result.probe_installed = False
                result = self.bind_result(task, result, key)
                if failure == "exit":
                    result.agent_exit_code = 1
                    (Path(result.result).parent / "agent/exit-code.txt").write_text("1")
            return result

    remote = RecoveringTrials(tmp_path / "results")
    result = make_loop(tmp_path, remote=remote, model=AlternativeRepairModel(), repair=True).run(
        task
    )
    assert result.status == "usable", result.reasons
    assert result.bundle_hash == task_identity(task)
    assert [key for _, key, _ in remote.calls if key.startswith("r1")] == [
        "r1-probe0",
        "r1-probe1",
        "r1-rollout",
    ]
    assert all("/r1-" in trial.result for trial in result.trials if trial.role == "probe")


@pytest.mark.parametrize("kind", ["wrong_solution", "valid_alternative"])
def test_probe_reuse_requires_the_entire_unchanged_definition(task, tmp_path, kind):
    # [FUNCTION DOC] Asserts _reusable_probe_trial accepts an unchanged probe trial and rejects any change to
    # script, rationale, focus or evidence.
    probe = next(item for item in probes() if item.kind == kind)
    if kind == "valid_alternative":
        probe = probe.model_copy(update={"script": probe.script + " # corrected"})
    variant = probe_variant(task, probe, tmp_path / "variants" / task.name)
    key = "r0-probe0" if kind == "wrong_solution" else "r0-probe1"
    trial = AlternativeTrials(tmp_path / "results").run(variant, "probe", key)
    trial.probe = probe
    assert _reusable_probe_trial(trial, probe, task_identity(task), 1.0)
    for change in (
        {"script": probe.script + "\nprintf done"},
        {"rationale": "Changed diagnosis"},
        {"focus": "numeric_tolerance"},
        {"evidence": [Citation(path="instruction.md", quote="Write the sum")]},
    ):
        assert not _reusable_probe_trial(
            trial, probe.model_copy(update=change), task_identity(task), 1.0
        )


@pytest.mark.parametrize(
    "tamper", ["result", "parent", "definition", "variant", "marker", "receipt", "symlink"]
)
def test_successful_probe_reuse_checks_original_bound_evidence(task, tmp_path, tamper):
    # [FUNCTION DOC] For each tampering (result bytes, probe.json parent hash or definition, variant verifier,
    # completion marker, receipt state, symlinked result), asserts _reusable_probe_trial raises
    # ValueError.
    probe = probes()[0]
    variant = probe_variant(task, probe, tmp_path / "variants" / task.name)
    trial = AlternativeTrials(tmp_path / "results").run(variant, "probe", "r0-probe0")
    trial.probe = probe
    result = Path(trial.result)
    if tamper == "result":
        result.write_text(result.read_text() + " ")
    elif tamper in {"parent", "definition"}:
        manifest = variant.parent / "probe.json"
        data = json.loads(manifest.read_text())
        if tamper == "parent":
            data["parent_hash"] = "sha256:" + "0" * 64
        else:
            data["probe"]["script"] += "\nprintf changed"
        manifest.write_text(json.dumps(data))
    elif tamper == "variant":
        (variant / "tests/test.sh").write_text("changed verifier")
    elif tamper == "marker":
        (result.parent / "agent/oracle.txt").write_text("not completed")
    elif tamper == "receipt":
        receipt = result.parents[2] / "trial.json"
        data = json.loads(receipt.read_text())
        data["state"] = "running"
        receipt.write_text(json.dumps(data))
    else:
        moved = result.with_name("moved.json")
        result.rename(moved)
        result.symlink_to(moved)
    with pytest.raises(ValueError):
        _reusable_probe_trial(trial, probe, task_identity(task), 1.0)


@pytest.mark.parametrize("defect", ["diagnosis", "name", "focus", "evidence"])
def test_alternative_repair_requires_grounding_and_stable_identity(task, tmp_path, defect):
    # [FUNCTION DOC] Asserts a replacement probe is rejected (needs_evidence, no revision) when the diagnosis is
    # not a probe issue or the name, focus or evidence changed; for evidence the reason names the
    # invalid citation.
    replacement = probes()[1]
    category = "verifier" if defect == "diagnosis" else "probe"
    updates = {
        "name": {"name": "unknown-probe"},
        "focus": {"focus": "lazy_output"},
        "evidence": {"evidence": [Citation(path="instruction.md", quote="invented contract")]},
    }
    replacement = replacement.model_copy(update=updates.get(defect, {}))
    result = make_loop(
        tmp_path,
        remote=AlternativeTrials(tmp_path / "results"),
        model=AlternativeRepairModel(category=category, replacement=replacement),
        repair=True,
    ).run(task)
    assert result.status == "needs_evidence"
    assert result.repairs == 0
    assert not (tmp_path / "quality/revisions/r1").exists()
    if defect == "evidence":
        assert "probe_replacements[0] (valid-format).evidence[0]" in result.reasons[0]
        assert "path='instruction.md'" in result.reasons[0]
        assert "quote='invented contract'" in result.reasons[0]


@pytest.mark.parametrize("defect", ["unknown_path", "traceback"])
def test_alternative_repair_corrects_citation_once_with_source_context(task, tmp_path, defect):
    # [FUNCTION DOC] Asserts a replacement probe with a bad citation (unknown path, or traceback quote missing the
    # 'E   ' prefix) gets one correction round with precise feedback and the source excerpt, then
    # succeeds.
    actual = "assert False is True\nE   AssertionError: assert False is True\n"
    bad_quote = actual.replace("E   ", "")
    citation_path = "evidence/3-probe/verifier/stdout.txt"
    bad_path = "evidence/3-probe/not-an-artifact.txt" if defect == "unknown_path" else citation_path

    class LoggedTrials(AlternativeTrials):
        # [CLASS DOC] AlternativeTrials variant that writes a verifier stdout.txt with a pytest traceback for
        # r0-probe1.
        def run(self, *args):
            # [FUNCTION DOC] Delegates to the parent and, for key r0-probe1, writes verifier/stdout.txt beside the
            # result.
            result = super().run(*args)
            if args[2] == "r0-probe1":
                directory = Path(result.result).parent / "verifier"
                directory.mkdir()
                (directory / "stdout.txt").write_text(actual)
            return result

    class CorrectingModel(AlternativeRepairModel):
        # [CLASS DOC] AlternativeRepairModel variant that first returns a badly cited replacement and checks the
        # correction feedback.
        def ask(self, schema, model, system, user, key):
            # [FUNCTION DOC] For Repair: on the correction key asserts feedback names the evidence slot, bad path
            # and quote (and the traceback excerpt is supplied); otherwise replaces the evidence
            # with the bad citation.
            response = super().ask(schema, model, system, user, key)
            if schema is Repair:
                if key.endswith("-correction1"):
                    payload = json.loads(user)
                    feedback = payload["patch_feedback"][0]
                    assert "probe_replacements[0] (valid-format).evidence[0]" in feedback
                    assert f"path={bad_path!r}" in feedback
                    assert f"quote={bad_quote!r}" in feedback
                    if defect == "traceback":
                        excerpt = citation_path + ":search=assert False is True"
                        assert actual in payload["documents"][excerpt]
                else:
                    response.probe_replacements = [
                        response.probe_replacements[0].model_copy(
                            update={"evidence": [Citation(path=bad_path, quote=bad_quote)]}
                        )
                    ]
            return response

    model = CorrectingModel()
    result = make_loop(
        tmp_path, remote=LoggedTrials(tmp_path / "results"), model=model, repair=True
    ).run(task)
    assert result.status == "usable", result.reasons
    assert [key for key in model.calls if "repair" in key] == [
        "r0-repair",
        "r0-repair-correction1",
    ]


def test_review_contains_actual_repository_test_failure(task, tmp_path):
    # [FUNCTION DOC] Asserts verifier stdout from a trial appears in the context under
    # evidence/0-oracle/verifier/stdout.txt.
    item = Trials(tmp_path / "results").run(task, "oracle", "one")
    verifier = Path(item.result).parent / "verifier"
    verifier.mkdir()
    (verifier / "stdout.txt").write_text("FAILED test_root_atomicity: actual != expected")
    context = EvidenceContext(task, [item], limit=16000)
    assert (
        "FAILED test_root_atomicity" in context.documents["evidence/0-oracle/verifier/stdout.txt"]
    )


def test_large_baseline_inventory_cannot_hide_later_probe_failure(task, tmp_path):
    # [FUNCTION DOC] Asserts huge baseline artifacts do not crowd out a later probe's failure output, and the probe
    # script is readable as evidence/1-probe/probe-script.sh.
    remote = Trials(tmp_path / "results")
    baseline = remote.run(task, "baseline", "baseline")
    alternative = remote.run(task, "probe", "alternative").model_copy(
        update={"probe": probes()[1], "reward": 0.0}
    )
    for item in (baseline, alternative):
        (Path(item.result).parent / "verifier").mkdir()
    (Path(baseline.result).parent / "verifier/result.json").write_text("x" * 100000)
    (Path(baseline.result).parent / "verifier/stdout.txt").write_text("y" * 100000)
    (Path(alternative.result).parent / "verifier/stdout.txt").write_text(
        "FAILED test_root_atomicity: actual != expected"
    )
    context = EvidenceContext(task, [baseline, alternative], limit=16000)
    assert "FAILED test_root_atomicity" in context.documents["evidence/1-probe/verifier/stdout.txt"]
    script_path = "evidence/1-probe/probe-script.sh"
    context.read_more([ReadRequest(path=script_path, query=None, start_line=1, end_line=2)])
    assert context.documents[script_path + ":L1-L2"] == probes()[1].script


def test_immutable_oracle_cannot_be_changed_by_component_repair(task, tmp_path):
    # [FUNCTION DOC] Asserts a repair editing solution/solve.sh with protected_paths set ends needs_evidence citing
    # the immutable oracle, with the task unchanged and no revision.
    class OracleEditingModel(Model):
        # [CLASS DOC] Model variant whose Repair edits the protected oracle script.
        def ask(self, schema, model, system, user, key):
            # [FUNCTION DOC] Returns a Repair editing solution/solve.sh for schema Repair, else delegates to
            # Model.ask.
            if schema is Repair:
                return Repair(
                    explanation="Attempt to alter the oracle",
                    addressed_issues=["Weak check"],
                    edits=[
                        Edit(
                            path="solution/solve.sh",
                            old="printf '4'",
                            new="printf '5'",
                            executable=True,
                        )
                    ],
                )
            return super().ask(schema, model, system, user, key)

    before = task_identity(task)
    loop = QualityLoop(
        LoopOptions(repair=True),
        tmp_path / "quality",
        BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="20"),
        model_client=OracleEditingModel(),
        trial_runner=Trials(tmp_path / "evidence"),
        protected_paths=("solution", "environment/source"),
    )
    result = loop.run(task)
    assert result.status == "needs_evidence"
    assert "immutable source or oracle" in result.reasons[0]
    assert task_identity(task) == before
    assert not (tmp_path / "quality/revisions/r1").exists()


def test_selected_unittest_methods_are_in_initial_review_context(task):
    # [FUNCTION DOC] Asserts an expected-pass test method id pulls the private test file containing it into the
    # initial context.
    path = task / "tests/source/tests/test_large.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "class InterleaveEvenlyTests:\n    def test_no_iterables(self):\n        assert result == []\n"
    )
    (task / "tests/contract.json").write_text(
        json.dumps(
            {"expected_passes": ["tests.test_large.InterleaveEvenlyTests::test_no_iterables"]}
        )
    )
    context = EvidenceContext(task, [], limit=100000)
    assert "test_no_iterables" in context.documents["tests/source/tests/test_large.py"]


def test_real_build_failure_text_is_available_without_guessing(task, tmp_path):
    # [FUNCTION DOC] Asserts a trial result.json with an exception message is included verbatim under
    # evidence/0-oracle/result.json.
    path = tmp_path / "evidence/result.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "exception_info": {
                    "exception_message": "ConfigError: Description file README.rst does not exist"
                }
            }
        )
    )
    trial = TrialRecord(
        role="oracle",
        bundle_hash=task_identity(task),
        result=str(path),
        result_sha256=digest(path),
        agent="oracle",
        model=None,
        reward=None,
        exception_type="RuntimeError",
        binding="receipt",
    )
    context = EvidenceContext(task, [trial], limit=100000)
    assert (
        "Description file README.rst does not exist"
        in context.documents["evidence/0-oracle/result.json"]
    )


def test_selected_class_wins_over_generic_method_names_and_source_copies(task, tmp_path):
    # [FUNCTION DOC] Asserts a class selector in test_paths brings the selected class's assertions into context
    # despite 250 same-named methods, and excludes unrelated library source.
    private = task / "tests/source/tests/test_large.py"
    private.parent.mkdir(parents=True)
    private.write_text(
        "\n".join(
            f"class Unrelated{i}:\n    def test_empty(self):\n        assert unrelated == []\n"
            for i in range(250)
        )
        + "\nclass ValueChainTests:\n    def test_empty(self):\n        assert selected_regression == []\n"
    )
    unrelated = task / "tests/source/library/more.py"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("def more():\n    pass\n" * 200)
    (task / "tests/contract.json").write_text(
        json.dumps(
            {
                "expected_passes": ["tests.test_large.ValueChainTests::test_empty"],
                "test_paths": ["tests/test_large.py::ValueChainTests"],
            }
        )
    )
    refresh_identity(task)
    evidence = Trials(tmp_path / "results").run(task, "baseline", "baseline")
    context = EvidenceContext(task, [evidence], limit=100000)
    assert "selected_regression" in context.documents["tests/source/tests/test_large.py"]
    assert "tests/source/library/more.py" not in context.documents


def test_review_reads_missing_evidence_before_demanding_probe_scripts(task, tmp_path):
    # [FUNCTION DOC] Asserts _review honors a model read request (one round) and the second call sees the search
    # excerpt; the final review has no pending read requests.
    class Reader(Model):
        # [CLASS DOC] Model variant that first requests a read, then reviews.
        def ask(self, schema, model, system, user, key):
            # [FUNCTION DOC] Appends the parsed payload to calls; first call returns a review with a ReadRequest,
            # later calls assert the excerpt exists and return review(propose=True).
            payload = json.loads(user)
            self.calls.append(payload)
            if len(self.calls) == 1:
                return review().model_copy(
                    update={
                        "read_requests": [
                            ReadRequest(
                                path="tests/test.sh", query="weak check", start_line=1, end_line=1
                            )
                        ]
                    }
                )
            assert "tests/test.sh:search=weak check" in payload["documents"]
            return review(propose=True)

    model = Reader()
    loop = make_loop(tmp_path, model=model, max_read_rounds=1)
    result, _ = loop._review(task, [], "read", probe_limit=2)
    assert not result.read_requests
    assert len(model.calls) == 2


def test_review_reserves_a_slot_for_valid_alternative(task, tmp_path):
    # [FUNCTION DOC] Asserts that when the model proposes only wrong_solution probes, protocol feedback demands the
    # missing kind and the second answer covers both kinds.
    class WrongOnly(Model):
        # [CLASS DOC] Model variant that first proposes two wrong_solution probes.
        def ask(self, schema, model, system, user, key):
            # [FUNCTION DOC] Asserts required_probe_kinds; first call returns two wrong-solution probes, later
            # calls assert 'missing control kinds' feedback and return both kinds.
            payload = json.loads(user)
            self.calls.append(payload)
            assert payload["required_probe_kinds"] == ["valid_alternative", "wrong_solution"]
            value = review(propose=True)
            if len(self.calls) == 1:
                value.probes = [probes()[0], probes()[0].model_copy(update={"name": "other-wrong"})]
            else:
                assert "missing control kinds" in payload["protocol_feedback"][-1]
            return value

    model = WrongOnly()
    value, _ = make_loop(tmp_path, model=model)._review(task, [], "kinds", probe_limit=2)
    assert {p.kind for p in value.probes} == {"wrong_solution", "valid_alternative"}
    assert len(model.calls) == 2


def test_blocking_task_defect_can_be_repaired_before_authoring_probes(task, tmp_path):
    # [FUNCTION DOC] Asserts a review with a blocking issue and no probes is accepted by _review (repair may
    # precede probe authoring).
    class Defect(Model):
        # [CLASS DOC] Model variant that always reports a broken task without probes.
        def ask(self, *args):
            # [FUNCTION DOC] Returns review(broken=True, propose=False).
            return review(broken=True, propose=False)

    value, _ = make_loop(tmp_path, model=Defect())._review(task, [], "defect", probe_limit=2)
    assert value.issues and not value.probes


def test_review_inventory_preserves_readable_paths_without_duplicate_hash_tokens(task):
    # [FUNCTION DOC] Asserts a 600-entry inventory is grouped by directory losslessly, budget omissions are
    # counted, and sha256 hashes are kept in the inventory but not emitted in the payload.
    context = EvidenceContext(task, [], limit=100000)
    for index in range(600):
        key = f"evidence/3-probe/artifacts/workspace/large_package/module_{index:04}.py"
        context.inventory.append({"path": key, "bytes": 200, "sha256": "a" * 64})
        context.omitted.append(key + ": context budget")
    payload = json.loads(context.payload())
    entries = {
        "/".join(filter(None, [prefix, name])): size
        for prefix, files in payload["inventory_by_directory"].items()
        for name, size in files.items()
    }
    assert entries == {item["path"]: item["bytes"] for item in context.inventory}
    assert entries[key] == 200
    assert payload["budget_omissions"]["count"] >= 600
    assert "a" * 64 not in json.dumps(payload)
    assert all("sha256" in item for item in context.inventory)


def test_original_task_intent_and_protected_paths_reach_reviewer(task, tmp_path):
    # [FUNCTION DOC] Asserts the reviewer payload includes protected_paths and the original task_context as
    # evidence/task-context.json.
    class IntentModel(Model):
        # [CLASS DOC] Model variant asserting protected paths and task context reach the reviewer.
        def ask(self, schema, model, system, user, key):
            # [FUNCTION DOC] Asserts payload protected_paths and the merged_pr task context, then returns review().
            payload = json.loads(user)
            assert payload["protected_paths"] == ["solution"]
            context = json.loads(payload["documents"]["evidence/task-context.json"])
            assert context["kind"] == "merged_pr"
            assert context["source_diff"] == "fixed original change"
            return review()

    loop = QualityLoop(
        LoopOptions(),
        tmp_path / "review",
        BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="5"),
        protected_paths=("solution",),
        task_context={"kind": "merged_pr", "source_diff": "fixed original change"},
        model_client=IntentModel(),
    )
    loop._review(task, [], "intent", probe_limit=0)
