# [MODULE DOC] Pydantic schemas (extra fields forbidden) for CodeMidas LLM artifacts: the selected
# feature with its behavioral requirements and implementation symbols, the generated verifier with
# its test-to-requirement map, and the independent review verdict.
"""Artifacts bind behavioral requirements, implementation boundaries and tests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Artifact(BaseModel):
    # [CLASS DOC] Base model for all CodeMidas artifacts; only sets extra="forbid".
    model_config = ConfigDict(extra="forbid")


class Symbol(Artifact):
    # [CLASS DOC] An implementation boundary: path (repo-relative file) and qualified_name
    # (Class.method or function name).
    path: str
    qualified_name: str


class Requirement(Artifact):
    # [CLASS DOC] One behavioral requirement: id matching R<n> (n >= 1, no leading zero) and
    # behavior text (15-2000 chars).
    id: str = Field(pattern=r"^R[1-9][0-9]*$")
    behavior: str = Field(min_length=15, max_length=2000)


class Feature(Artifact):
    # [CLASS DOC] The selected feature: title (10-140), instruction (200-12000), symbols (1-12
    # Symbols whose bodies are removed), requirements (3-20), rationale (30-3000).
    title: str = Field(min_length=10, max_length=140)
    instruction: str = Field(min_length=200, max_length=12000)
    symbols: list[Symbol] = Field(min_length=1, max_length=12)
    requirements: list[Requirement] = Field(min_length=3, max_length=20)
    rationale: str = Field(min_length=30, max_length=3000)

    def task_instruction(self) -> str:
        # [FUNCTION DOC] Returns the learner-facing instruction: the instruction text (trailing
        # whitespace stripped) followed by a "Required behavior:" bullet list of every requirement's
        # behavior, so the verifier's requirement map is always public. Output: str.
        """The verifier's requirement map must never be a private task contract."""
        return (
            self.instruction.rstrip()
            + "\n\nRequired behavior:\n\n"
            + "\n".join("- " + requirement.behavior for requirement in self.requirements)
        )

    @model_validator(mode="after")
    def unique(self):
        # [FUNCTION DOC] After-validator: raises ValueError if requirement ids or (path,
        # qualified_name) symbol pairs are not unique. Returns self.
        ids = [requirement.id for requirement in self.requirements]
        symbols = [(symbol.path, symbol.qualified_name) for symbol in self.symbols]
        if len(ids) != len(set(ids)) or len(symbols) != len(set(symbols)):
            raise ValueError("Requirement IDs and selected symbols must be unique")
        return self


class AssertionMap(Artifact):
    # [CLASS DOC] Maps one top-level pytest test (name test_[A-Za-z0-9_]+) to the requirement ids it
    # checks (at least one) with an observation (15-1500 chars).
    test: str = Field(pattern=r"^test_[A-Za-z0-9_]+$")
    requirements: list[str] = Field(min_length=1)
    observation: str = Field(min_length=15, max_length=1500)


class Verifier(Artifact):
    # [CLASS DOC] Generated pytest source test_code (200-32000 chars) plus assertions (3-40
    # AssertionMap entries).
    test_code: str = Field(min_length=200, max_length=32000)
    assertions: list[AssertionMap] = Field(min_length=3, max_length=40)


class Review(Artifact):
    # [CLASS DOC] Independent reviewer verdict: approved, issues (list), explanation (30-5000) and
    # repair_target ("tests" default, or "contract" to revise the feature).
    approved: bool
    issues: list[str]
    explanation: str = Field(min_length=30, max_length=5000)
    repair_target: Literal["tests", "contract"] = "tests"

    @model_validator(mode="after")
    def consistent(self):
        # [FUNCTION DOC] After-validator: raises ValueError if approved is True while issues is
        # non-empty. Returns self.
        if self.approved and self.issues:
            raise ValueError("An approved review must have no unresolved issues")
        return self
