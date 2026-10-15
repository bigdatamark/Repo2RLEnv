# [MODULE DOC] Pydantic schema for a task's quality acceptance report. Acceptance requires every
# check in REQUIRED_CHECKS to pass with artifact evidence (regression may be not_applicable for the
# reasoning profile); review scores are informational and never averaged into acceptance. CheckName
# enumerates the nine allowed check names.
"""Acceptance requires all declared hard checks, never an average review score."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CheckName = Literal[
    "build",
    "contrast",
    "oracle_repeat",
    "regression",
    "isolation",
    "verifier_attacks",
    "specification",
    "blind_sonnet",
    "blind_opus",
]
REQUIRED_CHECKS = frozenset(
    {
        "build",
        "contrast",
        "oracle_repeat",
        "regression",
        "isolation",
        "verifier_attacks",
        "specification",
        "blind_sonnet",
        "blind_opus",
    }
)


class Evidence(BaseModel):
    # [CLASS DOC] Immutable pointer to an evidence artifact: path (non-empty) and sha256 (64
    # lowercase hex chars). Extra fields are forbidden.
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CheckResult(BaseModel):
    # [CLASS DOC] Outcome of one hard check: status (pass/fail/unknown/not_applicable), non-empty
    # explanation and a list of Evidence. Extra fields forbidden; completed checks need evidence.
    model_config = ConfigDict(extra="forbid")
    status: Literal["pass", "fail", "unknown", "not_applicable"]
    explanation: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_evidence(self) -> CheckResult:
        # [FUNCTION DOC] After-validator: raises ValueError when status is pass or fail but evidence
        # is empty. Returns self otherwise.
        if self.status in {"pass", "fail"} and not self.evidence:
            raise ValueError("A completed check requires artifact evidence")
        return self


class ReviewScores(BaseModel):
    # [CLASS DOC] Optional 0-4 reviewer scores for clarity, realism, coverage and robustness (each
    # None by default). Corpus diversity is deliberately not scored per task. Extra fields
    # forbidden.
    model_config = ConfigDict(extra="forbid")
    clarity: int | None = Field(default=None, ge=0, le=4)
    realism: int | None = Field(default=None, ge=0, le=4)
    coverage: int | None = Field(default=None, ge=0, le=4)
    robustness: int | None = Field(default=None, ge=0, le=4)
    # Corpus diversity is measured separately; a single-task reviewer cannot
    # establish population diversity from one instruction.


class QualityReport(BaseModel):
    # [CLASS DOC] The full report (schema_version '1'): bundle_hash ('sha256:' + 64 hex) it applies
    # to, recipe and recipe_version, profile ('coding' default or 'reasoning'), checks keyed by
    # CheckName, scores and per-solver success flags. Extra fields forbidden.
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1"] = "1"
    bundle_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    recipe: str
    recipe_version: str
    profile: Literal["coding", "reasoning"] = "coding"
    checks: dict[CheckName, CheckResult] = Field(default_factory=dict)
    scores: ReviewScores = Field(default_factory=ReviewScores)
    solver_success: dict[str, bool | None] = Field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        # [FUNCTION DOC] Property: True only if the checks keys equal REQUIRED_CHECKS exactly and
        # every check passed with evidence, except that 'regression' may be not_applicable under the
        # reasoning profile. Otherwise False.
        if self.checks.keys() != REQUIRED_CHECKS:
            return False
        for name, result in self.checks.items():
            if result.status == "pass" and result.evidence:
                continue
            # Reasoning instances do not have pre-existing repository regressions.
            if (
                name == "regression"
                and self.profile == "reasoning"
                and result.status == "not_applicable"
            ):
                continue
            return False
        return True

    def current_for(self, bundle_hash: str) -> bool:
        # [FUNCTION DOC] Returns True if this report's bundle_hash equals the given bundle_hash
        # (i.e. the report is still valid for that task bundle).
        return self.bundle_hash == bundle_hash
