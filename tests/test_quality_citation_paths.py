# [MODULE DOC] Tests for citation-path validation in the quality loop:
# EvidenceContext.validate_review and protocol.citation_path_error must reject a Review whose
# Citation quote does not appear verbatim in the cited document, report which supplied documents
# contain the exact quote (bounded list and quote length), never relocate a citation automatically,
# and feed that message back so QualityLoop._review can obtain a corrected review from the model.
# Uses tmp_path; the last test needs `harbor` installed.
from __future__ import annotations

import json

import pytest

from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.emitter.bundle import TaskBundle, TaskFile, write_bundle
from repo2rlenv.quality.loop.context import EvidenceContext
from repo2rlenv.quality.loop.models import Assessment, Citation, LoopOptions, Review
from repo2rlenv.quality.loop.protocol import citation_path_error
from repo2rlenv.quality.loop.runner import QualityLoop


def decision(citation):
    # [FUNCTION DOC] HELPER: Builds a passing Review whose task/verifier/leakage Assessments (status pass, score 3)
    # all cite the given Citation, with rollout 'not_run' and empty issues/probes/read_requests.
    # Input: citation. Output: Review.
    assessment = Assessment(
        status="pass", score=3, explanation="Check state restoration.", evidence=[citation]
    )
    return Review(
        summary="Review the context-manager contract.",
        task=assessment.model_copy(deep=True),
        verifier=assessment.model_copy(deep=True),
        leakage=assessment.model_copy(deep=True),
        rollout="not_run",
        issues=[],
        probes=[],
        read_requests=[],
    )


@pytest.fixture
def context(tmp_path):
    # [FUNCTION DOC] Fixture: writes instruction.md ('Restore the original state.') into tmp_path and returns an
    # EvidenceContext over tmp_path with no extra paths and a 16000-char limit.
    (tmp_path / "instruction.md").write_text("Restore the original state.\n")
    return EvidenceContext(tmp_path, [], limit=16000)


@pytest.mark.parametrize(
    "quote,other_text",
    [
        ('"status": "needs_repair"', 'status = "needs_repair"'),
        ('"status": "needs_repair"', 'status = "generated"'),
        ("expected 1.0", "expected 1.1"),
        ("Restore the original State.", "Restore the original state."),
        ("assert False is True\nAssertionError", "assert False is True\nE AssertionError"),
    ],
)
def test_absent_citations_report_exact_source_miss_without_inference(context, quote, other_text):
    # [FUNCTION DOC] For quotes absent from every document (only near-misses exist in other.txt), asserts
    # validate_review raises ValueError naming the cited path, the repr of the invalid quote and '0
    # supplied document(s): []', and leaves the Review unmodified.
    context.documents["other.txt"] = other_text
    value = decision(Citation(path="evidence/task-context.json", quote=quote))
    before = value.model_dump()
    with pytest.raises(ValueError) as error:
        context.validate_review(value)
    feedback = str(error.value)
    assert "evidence/task-context.json" in feedback
    assert f"invalid quote={quote!r}" in feedback
    assert "Exact quote matches 0 supplied document(s): []" in feedback
    assert value.model_dump() == before


def test_ambiguous_path_feedback_lists_bounded_exact_sources(context):
    # [FUNCTION DOC] With the quote present in 12 documents and a wrong cited path, asserts the error lists only
    # the first 8 sources plus '4 additional paths omitted' and the citation path is unchanged.
    citation = Citation(path="wrong.json", quote="a specific unchanged claim")
    context.documents.update({f"source-{index:02}.txt": citation.quote for index in range(12)})
    with pytest.raises(ValueError) as error:
        context.validate_review(decision(citation))
    feedback = str(error.value)
    assert "wrong.json" in feedback
    assert "Exact quote matches 12 supplied document(s)" in feedback
    assert all(f"source-{index:02}.txt" in feedback for index in range(8))
    assert "source-08.txt" not in feedback
    assert "4 additional paths omitted" in feedback
    assert citation.path == "wrong.json"


def test_unique_wrong_path_is_reported_without_automatic_relocation(context):
    # [FUNCTION DOC] Asserts a quote found only in instruction.md but cited from wrong.md raises naming
    # instruction.md, does not rewrite the path, and that a corrected copy validates.
    citation = Citation(path="wrong.md", quote="Restore the original state.")
    with pytest.raises(ValueError) as error:
        context.validate_review(decision(citation))
    assert "Exact quote matches 1 supplied document(s): ['instruction.md']" in str(error.value)
    assert citation.path == "wrong.md"
    corrected = citation.model_copy(update={"path": "instruction.md"})
    context.validate_review(decision(corrected))


def test_supplied_matching_path_remains_valid_with_duplicate_text(context):
    # [FUNCTION DOC] Asserts a correct citation stays valid even when the same text also exists in another
    # document.
    context.documents["duplicate.md"] = "Restore the original state."
    context.validate_review(
        decision(Citation(path="instruction.md", quote="Restore the original state."))
    )


def test_feedback_bounds_the_repeated_quote(context):
    # [FUNCTION DOC] Asserts citation_path_error truncates the echoed quote to 250 characters.
    citation = Citation(path="wrong.md", quote="x" * 1000)
    feedback = citation_path_error(citation, context.documents)
    assert repr("x" * 250) in feedback
    assert "x" * 251 not in feedback


def test_precise_citation_feedback_uses_existing_review_correction(tmp_path):
    # [FUNCTION DOC] Skips without harbor. Writes a minimal task bundle, runs QualityLoop._review with a scripted
    # model that first returns a bad citation then (after checking protocol_feedback in the JSON
    # user prompt) a correct one; asserts the final review is sound, the model was called twice and
    # the evidence path is instruction.md. Side effects: files and a sqlite budget ledger in
    # tmp_path.
    pytest.importorskip("harbor")
    task = write_bundle(
        TaskBundle(
            name="citation-review",
            org="tests",
            instruction="Restore the original state.\n",
            files={
                "environment/Dockerfile": TaskFile.text("FROM python:3.12-slim\n"),
                "solution/solve.sh": TaskFile.text("#!/bin/sh\ntrue\n", executable=True),
                "tests/test.sh": TaskFile.text("#!/bin/sh\ntrue\n", executable=True),
            },
            metadata={"recipe": "test", "recipe_version": "1", "reward_kinds": ["test_execution"]},
        ),
        tmp_path / "tasks",
    )

    class Model:
        # [CLASS DOC] Fake model client for QualityLoop; class attribute `calls` counts ask() invocations.
        calls = 0

        def ask(self, schema, model, system, user, key):
            # [FUNCTION DOC] Increments calls; on the first call returns a Review with an invalid citation,
            # otherwise asserts the last protocol_feedback entry in the JSON `user` message
            # describes the miss and returns a Review citing instruction.md. Other params (schema,
            # model, system, key) are ignored.
            self.calls += 1
            if self.calls == 1:
                return decision(
                    Citation(path="evidence/task-context.json", quote='"status": "needs_repair"')
                )
            feedback = json.loads(user)["protocol_feedback"][-1]
            assert "evidence/task-context.json" in feedback
            assert "Exact quote matches 0 supplied document(s): []" in feedback
            return decision(Citation(path="instruction.md", quote="Restore the original state."))

    model = Model()
    loop = QualityLoop(
        LoopOptions(max_read_rounds=1),
        tmp_path / "quality",
        BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="1"),
        model_client=model,
    )
    review, _ = loop._review(task, [], "citation", probe_limit=0)
    assert review.sound
    assert model.calls == 2
    assert review.task.evidence[0].path == "instruction.md"
