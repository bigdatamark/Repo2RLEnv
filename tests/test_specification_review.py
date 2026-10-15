# [MODULE DOC] Tests for SpecificationReview.passed in repo2rlenv.quality.specification.
from __future__ import annotations

from repo2rlenv.quality.specification import SpecificationReview


def test_reviewer_cannot_pass_while_still_requesting_repairs():
    # [FUNCTION DOC] All four findings pass but actionable_repairs is non-empty, so passed is False.
    finding = {"status": "pass", "explanation": "Looks supported", "evidence": "an excerpt"}
    review = SpecificationReview(
        **{
            name: finding
            for name in ("clarity", "verifier_alignment", "answer_leakage", "reference_validity")
        },
        actionable_repairs=["Disclose the numeric tolerance required by the verifier."],
    )
    assert not review.passed


def test_unknown_reference_evidence_blocks_specification_acceptance():
    # [FUNCTION DOC] reference_validity status 'unknown' makes passed False even with no repairs.
    finding = {"status": "pass", "explanation": "Looks supported", "evidence": "an excerpt"}
    review = SpecificationReview(
        **{name: finding for name in ("clarity", "verifier_alignment", "answer_leakage")},
        reference_validity={**finding, "status": "unknown"},
        actionable_repairs=[],
    )
    assert not review.passed
