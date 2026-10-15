# [MODULE DOC] Tests for Tasksmith exact prepared-profile recovery
# (Options.prepared_profiles): a frozen, previously successful build Profile for a PR may skip the
# dependency-investigation stage and bootstrap hints, but the runner must still inspect, bootstrap,
# design, construct and quality-review the current PR freshly. All remote stages, LLM authoring and
# quality review are replaced by monkeypatched fakes; no network, Docker or LLM calls happen.
"""Exact-profile recovery still builds and validates the current PR remotely."""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy

import pytest

from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.tasksmith import runner as runner_module
from repo2rlenv.tasksmith.models import (
    BootstrapHint,
    Design,
    Options,
    Panel,
    PreparedProfile,
    Profile,
)


@pytest.fixture
def source():
    # [FUNCTION DOC] Fixture: builds a frozen PR source record dict (url, repo, head/base SHAs, a
    # one-line source_diff on src/core.py, source_files/operations, workspace_strategy) and sets its
    # "id" to the first 12 hex chars of the SHA-256 of the sorted-JSON of url/head/source_diff,
    # mirroring how Tasksmith identifies sources. Output: the dict. Side effects: none.
    record = {
        "url": "https://github.com/example/project/pull/7",
        "repo": "https://github.com/example/project",
        "head": "b" * 40,
        "base": "a" * 40,
        "source_diff": "diff --git a/src/core.py b/src/core.py\n--- a/src/core.py\n+++ b/src/core.py\n@@ -1 +1 @@\n-value=1\n+value=2\n",
        "source_files": ["src/core.py"],
        "source_operations": [{"path": "src/core.py", "operation": "modified"}],
        "workspace_strategy": "head_minus_source_patch",
    }
    identity = {key: record[key] for key in ("url", "head", "source_diff")}
    record["id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return record


@pytest.fixture
def profile():
    # [FUNCTION DOC] Fixture: returns a CPU Profile with python options (source_paths src,
    # test_paths tests, one test selector, pinned pytest dependency), dependency_inputs
    # pyproject.toml
    # and an upstream test rationale. Side effects: none.
    return Profile(
        reasoning="Preserve a complete, previously successful offline build profile.",
        resource="cpu",
        options={
            "source_paths": ["src"],
            "test_paths": ["tests"],
            "test_selectors": ["tests/test_core.py"],
            "dependencies": ["pytest==8.4.2"],
        },
        dependency_inputs=["pyproject.toml"],
        upstream_test_rationale="Offline regression tests exercise the selected public behavior.",
    )


@pytest.fixture
def prepared(source, profile):
    # [FUNCTION DOC] Fixture: wraps the profile in a PreparedProfile bound to the source's url,
    # head, base, workspace_strategy and the SHA-256 of its source_diff. Output: PreparedProfile.
    return PreparedProfile(
        **{key: source[key] for key in ("url", "head", "base", "workspace_strategy")},
        source_diff_sha256=hashlib.sha256(source["source_diff"].encode()).hexdigest(),
        profile=profile,
    )


@pytest.fixture
def runner(monkeypatch, tmp_path, source, prepared):
    # [FUNCTION DOC] Fixture: creates a BudgetLedger (limit $20) under tmp_path/campaign and a
    # Tasksmith runner whose Options carry the prepared profile keyed by source id plus an unrelated
    # BootstrapHint for the repo. Monkeypatches the runner module's LLM key resolution and runtime
    # wheel
    # check, and the external agent runtime_path, so no credentials or wheels are needed. Output:
    # the
    # Tasksmith instance. Side effects: writes the ledger sqlite file under tmp_path.
    campaign = tmp_path / "campaign"
    BudgetLedger(campaign / "budget.sqlite3", limit_usd="20")
    options = Options(
        prepared_profiles={source["id"]: prepared},
        bootstrap_hints={
            source["repo"]: BootstrapHint(
                ref="c" * 40,
                base_image="python:3.12-slim",
                dependencies=["unrelated-hint==1.0"],
                scope="Repository cache hint, not the selected PR recipe.",
            )
        },
    )
    value = runner_module.Tasksmith(tmp_path / "run", campaign, options, tmp_path / "runtime.whl")
    monkeypatch.setattr(runner_module, "resolve_llm_api_key", lambda *args: "test-only")
    monkeypatch.setattr(runner_module, "check_runtime_wheel", lambda *args: "frozen-test-runtime")
    monkeypatch.setattr(
        "repo2rlenv.tasksmith.author.external_agent.runtime_path", lambda *args: tmp_path
    )
    return value


def install_stages(monkeypatch, runner, profile, *, fail_bootstraps=0, links=()):
    # [FUNCTION DOC] HELPER: Replaces runner.remote, runner.author and
    # runner.review_candidate with recording fakes and returns the shared call list.
    # Inputs: monkeypatch; runner; profile (returned by the fake Profile author); fail_bootstraps
    # (int, default 0: the first N bootstrap stages return a failed status); links (snapshot
    # document
    # links reported by the inspect stage, default empty). Output: list of (kind, stage, deep-copied
    # payload/inputs) tuples appended as stages run. Side effects: monkeypatches the runner
    # instance.
    calls = []
    bootstraps = 0

    def remote(root, stage, payload):
        # [FUNCTION DOC] HELPER: Fake remote stage executor. Records the call; for
        # "inspect" returns a completed checkout with the configured snapshot links; for "bootstrap"
        # increments a counter and returns failed while within fail_bootstraps, otherwise a
        # completed
        # readiness result with image "fresh-image"; for "construct" asserts the ready image is the
        # fresh
        # one. Default return: completed with local=root and task_relative "task".
        nonlocal bootstraps
        calls.append(("remote", stage, deepcopy(payload)))
        if payload["stage"] == "inspect":
            return {
                "status": "completed",
                "value": {"checkout": "/remote/checkout", "snapshot_links": list(links)},
            }
        if payload["stage"] == "bootstrap":
            bootstraps += 1
            if bootstraps <= fail_bootstraps:
                return {"status": "failed", "error": "Fresh bootstrap dependency failure"}
            return {
                "status": "completed",
                "value": {
                    "readiness": {"tests.test_core::test_value": "passed"},
                    "dependency_cache": {"cache_hit": False},
                    "image": "fresh-image",
                },
            }
        if payload["stage"] == "construct":
            assert payload["ready"]["image"] == "fresh-image"
        return {"status": "completed", "local": str(root), "value": {"task_relative": "task"}}

    def author(root, stage, schema, inputs, checkout, *, validate=None):
        # [FUNCTION DOC] HELPER: Fake LLM author. Records the call; for the Profile
        # schema returns a copy of the fixture profile after running the supplied async validate
        # callback
        # via asyncio.run (so runner validation still applies); otherwise asserts the schema is
        # Design and
        # returns a fixed valid Design. Raises whatever validate raises.
        calls.append(("author", stage, deepcopy(inputs)))
        if schema is Profile:
            result = Profile.model_validate(profile.model_dump())
            asyncio.run(validate(result))
            return result
        assert schema is Design
        return Design(
            instruction="Restore the documented public value returned by the core module. " * 3,
            requirements=[
                {
                    "behavior": "Return the correct documented public value.",
                    "source_evidence": "src/core.py",
                    "verification": "Compare the public value against an independent expected value.",
                }
            ],
            verifier_rationale="Use the independently specified expected value from the contract.",
            wrong_solution_ideas=["Return a constant zero."],
            valid_alternative_ideas=["Compute the same value through an equivalent expression."],
        )

    def review(root, current_source, constructed):
        # [FUNCTION DOC] HELPER: Fake quality review; records ("quality", "fresh",
        # None) and returns a usable status dict.
        calls.append(("quality", "fresh", None))
        return {"status": "usable", "quality": {"fresh": True}}

    monkeypatch.setattr(runner, "remote", remote)
    monkeypatch.setattr(runner, "author", author)
    monkeypatch.setattr(runner, "review_candidate", review)
    return calls


def execute(runner, source):
    # [FUNCTION DOC] HELPER: Runs the runner on a single-PR Panel named "prepared" with the given
    # source record supplied as source_records. Output: the runner's result dict.
    return runner.run(Panel(name="prepared", prs=[source["url"]]), source_records=[source])


def test_exact_profile_skips_investigation_and_hint_but_runs_fresh_stages(
    monkeypatch, runner, source, profile
):
    # [FUNCTION DOC] Asserts that
    # with a matching prepared profile the stage sequence is inspect, bootstrap-1, design-0,
    # construct-1, fresh quality (no investigate and no dependency hint), that bootstrap receives
    # the
    # exact profile dump, and that the candidate records profile_attempt 1 and the fresh image.
    calls = install_stages(monkeypatch, runner, profile)
    result = execute(runner, source)
    assert [(kind, stage) for kind, stage, _ in calls] == [
        ("remote", "inspect"),
        ("remote", "bootstrap-1"),
        ("author", "design-0"),
        ("remote", "construct-1"),
        ("quality", "fresh"),
    ]
    assert calls[1][2]["profile"] == profile.model_dump()
    assert result["candidates"][0]["profile_attempt"] == 1
    assert result["candidates"][0]["ready"]["image"] == "fresh-image"


@pytest.mark.parametrize("field", ["url", "head", "base", "source_diff_sha256"])
def test_source_mismatch_stops_before_any_remote_dependency_work(
    monkeypatch, runner, source, profile, field
):
    # [FUNCTION DOC] Parametrized over url,
    # head, base, source_diff_sha256: mutating that field of the prepared profile must raise
    # ValueError
    # ("differs from the frozen PR") before any remote/author call is made.
    prepared = runner.options.prepared_profiles[source["id"]]
    replacement = (
        "https://github.com/example/project/pull/8"
        if field == "url"
        else "d" * (64 if field == "source_diff_sha256" else 40)
    )
    setattr(prepared, field, replacement)
    calls = install_stages(monkeypatch, runner, profile)
    with pytest.raises(ValueError, match="differs from the frozen PR"):
        execute(runner, source)
    assert calls == []


@pytest.mark.parametrize("change", ["gpu", "uncovered", "hidden"])
def test_invalid_resource_or_source_coverage_cannot_allocate(
    monkeypatch, runner, source, profile, change
):
    # [FUNCTION DOC] Parametrized: a GPU
    # resource, source_paths not covering the PR change, or public_exclude hiding the changed file
    # must
    # each raise ValueError with no stage calls recorded.
    if change == "gpu":
        runner.options.prepared_profiles[source["id"]].profile.resource = "gpu"
    elif change == "uncovered":
        profile.options.source_paths = ["other"]
    else:
        profile.options.public_exclude = ["src/core.py"]
    calls = install_stages(monkeypatch, runner, profile)
    with pytest.raises(ValueError):
        execute(runner, source)
    assert calls == []


def test_unknown_profile_id_is_rejected_before_remote_work(monkeypatch, runner, source, profile):
    # [FUNCTION DOC] Re-keys the prepared profile
    # to an id not among the selected sources and asserts ValueError ("outside the selected
    # sources")
    # with no stage calls.
    runner.options.prepared_profiles["c" * 12] = runner.options.prepared_profiles.pop(source["id"])
    calls = install_stages(monkeypatch, runner, profile)
    with pytest.raises(ValueError, match="outside the selected sources"):
        execute(runner, source)
    assert calls == []


def test_imported_task_and_prepared_profile_are_not_combined(monkeypatch, runner, source, profile):
    # [FUNCTION DOC] Sets runner.imported for the
    # same source id and asserts ValueError ("prepared profile or generated task") before any stage.
    runner.imported = {source["id"]: {"task": "unused"}}
    calls = install_stages(monkeypatch, runner, profile)
    with pytest.raises(ValueError, match="prepared profile or generated task"):
        execute(runner, source)
    assert calls == []


@pytest.mark.parametrize("field", ["test_cpus", "test_memory_mb"])
def test_prepared_resource_overflow_stops_before_remote_work(
    monkeypatch, runner, source, profile, field
):
    # [FUNCTION DOC] Parametrized over test_cpus
    # (3) and test_memory_mb (8192): values above the worker allocation must raise ValueError naming
    # the option and "exceeds worker allocation", with no stage calls.
    setattr(profile.options, field, {"test_cpus": 3, "test_memory_mb": 8192}[field])
    calls = install_stages(monkeypatch, runner, profile)
    with pytest.raises(ValueError, match=f"options.{field}.*exceeds worker allocation"):
        execute(runner, source)
    assert calls == []


def test_investigator_resource_overflow_cannot_build(monkeypatch, runner, source, profile):
    # [FUNCTION DOC] Without prepared profiles or hints,
    # an investigator-authored profile requesting 8192 MB records a candidate error "exceeds worker
    # allocation" after only inspect and investigate-0, and the investigator was told the worker has
    # 4096 MB.
    runner.options.prepared_profiles = {}
    runner.options.bootstrap_hints = {}
    profile.options.test_memory_mb = 8192
    calls = install_stages(monkeypatch, runner, profile)
    result = execute(runner, source)
    assert "exceeds worker allocation" in result["candidates"][0]["error"]
    assert [(kind, stage) for kind, stage, _ in calls] == [
        ("remote", "inspect"),
        ("author", "investigate-0"),
    ]
    assert calls[1][2]["requested_resources"]["worker_memory_mb"] == 4096


def test_document_links_are_still_validated_before_build(monkeypatch, runner, source, profile):
    # [FUNCTION DOC] When inspect reports a snapshot
    # document link, the candidate becomes "incomplete" with an "identified document links" error
    # and
    # only the inspect stage runs.
    calls = install_stages(monkeypatch, runner, profile, links=[{"path": "CONTRIBUTING.md"}])
    result = execute(runner, source)
    assert result["candidates"][0]["status"] == "incomplete"
    assert "identified document links" in result["candidates"][0]["error"]
    assert [(kind, stage) for kind, stage, _ in calls] == [("remote", "inspect")]


def test_fresh_bootstrap_failure_can_repair_within_original_attempt_limit(
    monkeypatch, runner, source, profile
):
    # [FUNCTION DOC] With
    # max_stage_attempts=2 and two failing bootstraps, the runner calls the investigator once to
    # repair
    # (receiving the previous profile and failure), retries bootstrap once, then marks the candidate
    # "bootstrap_failed".
    runner.options.max_stage_attempts = 2
    calls = install_stages(monkeypatch, runner, profile, fail_bootstraps=2)
    result = execute(runner, source)
    assert [(kind, stage) for kind, stage, _ in calls] == [
        ("remote", "inspect"),
        ("remote", "bootstrap-1"),
        ("author", "investigate-1"),
        ("remote", "bootstrap-2"),
    ]
    assert calls[2][2]["previous_profile"] == profile.model_dump()
    assert calls[2][2]["previous_failure"]["error"] == "Fresh bootstrap dependency failure"
    assert result["candidates"][0]["status"] == "bootstrap_failed"


def test_changed_inline_profile_invalidates_frozen_resume(monkeypatch, runner, source, profile):
    # [FUNCTION DOC] After a successful run,
    # changing the prepared profile's dependencies and re-running must raise ValueError
    # ("configuration
    # changed") without issuing new stage calls.
    calls = install_stages(monkeypatch, runner, profile)
    execute(runner, source)
    before = len(calls)
    runner.options.prepared_profiles[source["id"]].profile.options.dependencies = ["pytest==8.4.1"]
    with pytest.raises(ValueError, match="configuration changed"):
        execute(runner, source)
    assert len(calls) == before


def test_default_pipeline_still_uses_hint_and_investigator(monkeypatch, runner, source, profile):
    # [FUNCTION DOC] Without prepared profiles the
    # first remote stage is "dependencies" (the hint), authoring runs investigate-0 then design-0,
    # and
    # the result reports one usable candidate.
    runner.options.prepared_profiles = {}
    calls = install_stages(monkeypatch, runner, profile)
    result = execute(runner, source)
    assert calls[0][2]["stage"] == "dependencies"
    assert [stage for kind, stage, _ in calls if kind == "author"] == [
        "investigate-0",
        "design-0",
    ]
    assert result["usable"] == 1
