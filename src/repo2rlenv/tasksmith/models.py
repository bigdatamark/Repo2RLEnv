# [MODULE DOC] Pydantic contracts exchanged between Tasksmith stages: the repository Profile, the
# task Design and its Requirements, the PR Panel, BootstrapHints, prepared (reusable) profiles and
# designs bound to an exact PR source, and the run-wide Options with spending limits.
"""Small, explicit contracts between Tasksmith stages."""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from repo2rlenv.quality.loop.models import LoopOptions, ProbeFocus
from repo2rlenv.quality.loop.verifier_policy import check_behavioral_verifier
from repo2rlenv.spec.recipe_options import PythonRepositoryProfile


class Record(BaseModel):
    # [CLASS DOC] Base Pydantic model for all Tasksmith records; forbids unknown fields.
    model_config = ConfigDict(extra="forbid")


class Profile(Record):
    # [CLASS DOC] Author-produced build/test profile for a PR: reasoning, resource (cpu/gpu), a
    # PythonRepositoryProfile (source/test/exclude paths and pytest selectors), repository-relative
    # dependency_inputs (>=1) and an upstream_test_rationale.
    reasoning: str = Field(min_length=20)
    resource: Literal["cpu", "gpu"]
    options: PythonRepositoryProfile
    dependency_inputs: list[str] = Field(min_length=1)
    upstream_test_rationale: str = Field(min_length=20)

    @model_validator(mode="after")
    def offline_profile(self):
        # [FUNCTION DOC] After-validator. Raises ValueError unless: test_selectors is non-empty;
        # source roots are pairwise disjoint and not nested with any private test path or
        # public_exclude path; every selector's file part lies inside a private test path;
        # dependency_inputs are relative without "..". Returns self.
        if not self.options.test_selectors:
            raise ValueError("Select explicit offline pytest files or node IDs")
        hidden = [PurePosixPath(value) for value in self.options.test_paths]
        source = [PurePosixPath(value) for value in self.options.source_paths]
        excluded = hidden + [PurePosixPath(value) for value in self.options.public_exclude]
        for index, root in enumerate(source):
            if any(
                root == other or root in other.parents or other in root.parents
                for other in source[index + 1 :] + excluded
            ):
                raise ValueError("Source roots must be disjoint and outside private/excluded paths")
        for selector in self.options.test_selectors:
            path = PurePosixPath(selector.split("::", 1)[0])
            if not any(path == root or root in path.parents for root in hidden):
                raise ValueError("Every test selector must be inside a private test path")
        for item in self.dependency_inputs:
            if PurePosixPath(item).is_absolute() or ".." in PurePosixPath(item).parts:
                raise ValueError("Dependency inputs must be repository-relative paths")
        return self


class Requirement(Record):
    # [CLASS DOC] One behavioral requirement of a task design: behavior, how it is verified, and the
    # source evidence supporting it (minimum lengths enforced).
    behavior: str = Field(min_length=10)
    verification: str = Field(min_length=10)
    source_evidence: str = Field(min_length=5)


class Design(Record):
    # [CLASS DOC] Author-produced task design: instruction (100-12000 chars), 1-10 Requirements,
    # strategy (only "pr_regression"), verifier_rationale, upstream_test_policy retain/replace,
    # optional private additional_tests (<=24000 chars), and 1-4 wrong-solution and valid-
    # alternative ideas.
    instruction: str = Field(min_length=100, max_length=12000)
    requirements: list[Requirement] = Field(min_length=1, max_length=10)
    strategy: Literal["pr_regression"] = "pr_regression"
    verifier_rationale: str = Field(min_length=20)
    upstream_test_policy: Literal["retain", "replace"] = "retain"
    # The private file supplements or replaces the selected grading suite. Its
    # fixed path cannot overwrite upstream files; bootstrap readiness is unchanged.
    additional_tests: str = Field(default="", max_length=24000)
    wrong_solution_ideas: list[str] = Field(min_length=1, max_length=4)
    valid_alternative_ideas: list[str] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def replacement_requires_tests(self):
        # [FUNCTION DOC] After-validator: raises ValueError when policy is "replace" but
        # additional_tests is blank; runs check_behavioral_verifier on additional_tests (may raise).
        # Returns self.
        if self.upstream_test_policy == "replace" and not self.additional_tests.strip():
            raise ValueError("Replacing the upstream grading selection requires private tests")
        check_behavioral_verifier(self.additional_tests)
        return self


class Panel(Record):
    # [CLASS DOC] A named fixed set of 1-100 PR inputs (name is a lowercase slug of <=31 chars).
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,30}$")
    prs: list[str] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_inputs(self):
        # [FUNCTION DOC] After-validator: raises ValueError on duplicate PR entries. Returns self.
        if len(set(self.prs)) != len(self.prs):
            raise ValueError("Panel inputs must be unique")
        return self


class BootstrapHint(Record):
    # [CLASS DOC] Dependency hint derived from a bootstrap run: commit ref (40 hex), base_image,
    # pinned dependency strings and a human-readable scope.
    ref: str = Field(pattern=r"^[0-9a-f]{40}$")
    base_image: str
    dependencies: list[str]
    scope: str

    @model_validator(mode="after")
    def build_fields(self):
        # [FUNCTION DOC] After-validator: raises ValueError if base_image or any dependency is blank
        # or contains a newline/carriage return. Returns self.
        for value in [self.base_image, *self.dependencies]:
            if not value.strip() or "\n" in value or "\r" in value:
                raise ValueError("Bootstrap build hints must contain nonempty single-line values")
        return self


class PreparedSource(Record):
    # [CLASS DOC] Immutable PR identity for reusable preparation inputs: canonical PR URL, head and
    # base commit shas, sha256 of the source diff and the workspace strategy
    # ("head_minus_source_patch").
    """Immutable PR identity required by reusable preparation inputs."""

    url: str = Field(pattern=r"^https://github.com/[\w.-]+/[\w.-]+/pull/[1-9][0-9]*$")
    head: str = Field(pattern=r"^[0-9a-f]{40}$")
    base: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_diff_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    workspace_strategy: Literal["head_minus_source_patch"] = "head_minus_source_patch"


class PreparedProfile(PreparedSource):
    # [CLASS DOC] PreparedSource plus a Profile: an exact reusable build recipe for that PR (no
    # prior execution results).
    """An exact PR build recipe; prior execution results are never imported."""

    profile: Profile


class PreparedDesign(PreparedSource):
    # [CLASS DOC] PreparedSource plus a Design and the sha256 of the canonical Profile JSON it was
    # authored against.
    """A complete design bound to the exact source and canonical Profile JSON."""

    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    design: Design


class Options(Record):
    # [CLASS DOC] Run-wide Tasksmith options: provider (modal/daytona), gpus (0-2), author runtime
    # (pi/opencode) and model, author turns and per-stage USD, total max spend, worker reservation
    # and sizing (cpus, memory, timeout), optional Modal worker snapshot id, bootstrap hints keyed
    # by repo URL, prepared profiles/designs keyed by 12-hex PR source id, max stage attempts,
    # required probe focus list, and quality LoopOptions (defaults: repair, rollout, 3 repairs, 4
    # probes, 20 turns, $20).
    provider: Literal["modal", "daytona"] = "modal"
    gpus: int = Field(default=0, ge=0, le=2)
    author_runtime: Literal["pi", "opencode"] = "pi"
    author_model: str = "anthropic/claude-sonnet-4-6"
    author_turns: int = Field(default=18, ge=2, le=50)
    author_stage_usd: str = "4.00"
    max_spend_usd: str = "100.00"
    worker_reservation_usd: str = "12.00"
    worker_cpus: int = Field(default=2, ge=1, le=16)
    worker_memory_mb: int = Field(default=4096, ge=1024, le=65536)
    worker_timeout_sec: int = Field(default=14400, ge=300, le=14400)
    worker_snapshot: str | None = Field(default=None, pattern=r"^im-[A-Za-z0-9]+$")
    bootstrap_hints: dict[str, BootstrapHint] = Field(default_factory=dict)
    prepared_profiles: dict[str, PreparedProfile] = Field(default_factory=dict)
    prepared_designs: dict[str, PreparedDesign] = Field(default_factory=dict)
    max_stage_attempts: int = Field(default=3, ge=1, le=5)
    required_probe_focus: list[ProbeFocus] = Field(default_factory=list, max_length=4)
    quality: LoopOptions = Field(
        default_factory=lambda: LoopOptions(
            repair=True,
            run_rollout=True,
            max_repairs=3,
            max_probes=4,
            max_turns=20,
            max_spend_usd="20.00",
        )
    )

    @model_validator(mode="after")
    def limits(self):
        # [FUNCTION DOC] After-validator. Raises ValueError if prepared profile/design keys are not
        # 12 lowercase hex chars, probe focus entries repeat, GPUs or a worker snapshot are
        # requested with a non-modal provider, author_model is not "anthropic/...", or any USD limit
        # is non-finite or <= 0. Sorts required_probe_focus in place. Returns self.
        if any(not re.fullmatch(r"[0-9a-f]{12}", key) for key in self.prepared_profiles):
            raise ValueError("Prepared profiles must be keyed by the exact PR source ID")
        if any(not re.fullmatch(r"[0-9a-f]{12}", key) for key in self.prepared_designs):
            raise ValueError("Prepared designs must be keyed by the exact PR source ID")
        if len(set(self.required_probe_focus)) != len(self.required_probe_focus):
            raise ValueError("Required probe focus entries must be unique")
        self.required_probe_focus = sorted(self.required_probe_focus)
        if self.gpus and self.provider != "modal":
            raise ValueError(
                "GPU task execution currently requires the validated native Modal profile"
            )
        if self.worker_snapshot and self.provider != "modal":
            raise ValueError("Worker snapshots currently require Modal")
        if not self.author_model.startswith("anthropic/"):
            raise ValueError(
                "Pi/OpenCode author bridge currently supports Anthropic; quality models may use OpenAI or Anthropic"
            )
        for value in (self.author_stage_usd, self.max_spend_usd, self.worker_reservation_usd):
            if not Decimal(value).is_finite() or Decimal(value) <= 0:
                raise ValueError("Spending limits must be finite and positive")
        return self
