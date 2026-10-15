# [MODULE DOC] FrontierSmith stage contracts: pydantic models for each LLM-authored artifact (seed,
# design, review, program, infrastructure, divergence). Validators only parse code with ast; no
# generated code is executed here.
"""Stage contracts for the FrontierSmith adaptation; no generated code runs here."""

from __future__ import annotations

import ast
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Artifact(BaseModel):
    # [CLASS DOC] Base model for all stage artifacts; forbids extra fields.
    model_config = ConfigDict(extra="forbid")


class Seed(Artifact):
    # [CLASS DOC] Source optimization problem: id and family are lowercase slugs (max 61 chars), title,
    # problem text (40-12000 chars), source and license strings; family defaults to "unspecified".
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,60}$")
    title: str
    problem: str = Field(min_length=40, max_length=12000)
    source: str
    license: str
    family: str = Field(default="unspecified", pattern=r"^[a-z][a-z0-9-]{0,60}$")


class Design(Artifact):
    # [CLASS DOC] Mutated task design: title, mutation kind (objective, output_constraints or
    # input_constraints), instruction (200-12000 chars), objective, feasibility, score formula,
    # baseline strategy and an open-endedness rationale.
    title: str
    mutation: Literal["objective", "output_constraints", "input_constraints"]
    instruction: str = Field(min_length=200, max_length=12000)
    objective: str
    feasibility: str
    score_formula: str
    baseline_strategy: str
    why_open_ended: str


class Review(Artifact):
    # [CLASS DOC] Reviewer verdict: approved flag, list of issues, rationale; must be consistent.
    approved: bool
    issues: list[str]
    rationale: str

    @model_validator(mode="after")
    def consistent(self):
        # [FUNCTION DOC] After-validator: raises ValueError unless approved iff issues is empty. Returns self.
        if self.approved == bool(self.issues):
            raise ValueError("Approval requires no issues; rejection requires concrete issues")
        return self


class Program(Artifact):
    # [CLASS DOC] Authored solution program: strategy description and code (30-30000 chars).
    strategy: str
    code: str = Field(min_length=30, max_length=30000)

    @model_validator(mode="after")
    def syntax(self):
        # [FUNCTION DOC] After-validator: ast-parses code (SyntaxError, surfaced by pydantic as a
        # validation error, if invalid). Returns self.
        ast.parse(self.code)
        return self


class Infrastructure(Artifact):
    # [CLASS DOC] Private grading infrastructure sources: generator and scorer (60-24000 chars each)
    # and optional feasibility validator (default empty, max 24000 chars).
    generator: str = Field(min_length=60, max_length=24000)
    scorer: str = Field(min_length=60, max_length=24000)
    feasibility: str = Field(default="", max_length=24000)

    @model_validator(mode="after")
    def syntax(self):
        # [FUNCTION DOC] After-validator: parses each source and requires top-level functions generate
        # (generator), score (scorer) and is_feasible (feasibility, if non-empty). Raises ValueError (or
        # SyntaxError) otherwise. Returns self.
        for source, entry in ((self.generator, "generate"), (self.scorer, "score")):
            tree = ast.parse(source)
            if entry not in {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}:
                raise ValueError(f"Infrastructure must define {entry}")
        if self.feasibility:
            tree = ast.parse(self.feasibility)
            if "is_feasible" not in {
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            }:
                raise ValueError("Feasibility validator must define is_feasible")
        return self


class Divergence(Artifact):
    # [CLASS DOC] Divergence judgement: per-pair booleans saying whether pairs are distinct, plus a
    # rationale.
    distinct_pairs: list[bool]
    rationale: str
