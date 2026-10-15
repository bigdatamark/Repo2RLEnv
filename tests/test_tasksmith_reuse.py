# [MODULE DOC] Tests for tasksmith.reuse.load_generation: reusing a prior generation run
# must verify that each reused task belongs to the same frozen PR sources (metadata and order), lies
# inside the run, matches its bundle hash, and that optional execution evidence is still bound to
# unchanged result files.
"""A hash-valid task must still belong to the requested frozen PR and run."""

from __future__ import annotations

import hashlib
import json

import pytest

from repo2rlenv.emitter.bundle import TaskBundle, TaskFile, write_bundle
from repo2rlenv.quality.loop.artifacts import task_identity
from repo2rlenv.tasksmith.models import Panel
from repo2rlenv.tasksmith.reuse import load_generation


@pytest.fixture
def generation(tmp_path):
    # [FUNCTION DOC] Fixture: writes a prior run under tmp_path/prior: a TaskBundle whose
    # metadata pins the source url/head/base/strategy/diff hash, a panel.json manifest with the
    # panel and
    # source, and candidates/one/result.json with the task path and bundle hash. Output: tuple (run
    # dir,
    # panel, source, task path, receipt path, record dict, manifest dict).
    run = tmp_path / "prior"
    panel = Panel(name="fixed", prs=["https://github.com/example/lib/pull/1"])
    source = {
        "id": "one",
        "url": panel.prs[0],
        "head": "a" * 40,
        "base": "b" * 40,
        "source_diff": "pinned source diff",
        "workspace_strategy": "head_minus_source_patch",
    }
    task = write_bundle(
        TaskBundle(
            name="tasksmith-one",
            org="tests",
            instruction="Repair the library behavior.",
            files={
                "environment/Dockerfile": TaskFile.text("FROM python:3.12-slim\n"),
                "solution/solve.sh": TaskFile.text("#!/bin/sh\ntrue\n", executable=True),
                "tests/test.sh": TaskFile.text("#!/bin/sh\ntrue\n", executable=True),
            },
            metadata={
                "recipe": "tasksmith",
                "recipe_version": "1",
                "source_url": source["url"],
                "source_head": source["head"],
                "source_base": source["base"],
                "workspace_strategy": source["workspace_strategy"],
                "source_diff_sha256": hashlib.sha256(source["source_diff"].encode()).hexdigest(),
                "reward_kinds": ["test_execution"],
            },
        ),
        run / "tasks",
    )
    manifest = {"configuration": {"panel": panel.model_dump()}, "sources": [source]}
    (run / "panel.json").write_text(json.dumps(manifest))
    receipt = run / "candidates/one/result.json"
    receipt.parent.mkdir(parents=True)
    record = {"id": "one", "quality": {"task_path": str(task), "bundle_hash": task_identity(task)}}
    receipt.write_text(json.dumps(record))
    return run, panel, source, task, receipt, record, manifest


def test_generation_reuse_preserves_task_and_frozen_inputs(generation):
    # [FUNCTION DOC] load_generation returns the
    # frozen sources and imports the task with resolved path and current bundle hash.
    run, panel, source, task, *_ = generation
    sources, imported = load_generation(run, panel)
    assert sources == [source]
    assert imported["one"]["task"] == str(task.resolve())
    assert imported["one"]["bundle_hash"] == task_identity(task)


def test_generation_reuse_rejects_outside_path_even_with_valid_hash(generation, tmp_path):
    # [FUNCTION DOC] Moving the task outside
    # the run and repointing the receipt raises ValueError ("outside its run").
    run, panel, _, task, receipt, record, _ = generation
    outside = tmp_path / "outside"
    task.rename(outside)
    record["quality"]["task_path"] = str(outside)
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="outside its run"):
        load_generation(run, panel)


def test_generation_reuse_rejects_hash_valid_task_for_another_pr(generation):
    # [FUNCTION DOC] Changing the manifest
    # source head makes task metadata disagree, raising ValueError ("metadata differs").
    run, panel, _, _, _, _, manifest = generation
    manifest["sources"][0]["head"] = "c" * 40
    (run / "panel.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metadata differs"):
        load_generation(run, panel)


def test_generation_reuse_rejects_reordered_or_substituted_sources(generation):
    # [FUNCTION DOC] Substituting the source
    # URL raises ValueError ("source identities").
    run, panel, _, _, _, _, manifest = generation
    manifest["sources"][0]["url"] = "https://github.com/example/lib/pull/2"
    (run / "panel.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source identities"):
        load_generation(run, panel)


def test_generation_reuse_retains_previous_semantic_controls(generation):
    # [FUNCTION DOC] Probe trials in the receipt
    # are imported as the task's probes list.
    run, panel, _, _, receipt, record, _ = generation
    probe = {
        "name": "always-empty",
        "kind": "wrong_solution",
        "focus": "general",
        "rationale": "Nonempty input must retain its values",
        "evidence": [{"path": "instruction.md", "quote": "library behavior"}],
        "script": "printf 'def function(): return []\\n' > /workspace/lib.py",
    }
    record["quality"]["trials"] = [{"role": "probe", "probe": probe}]
    receipt.write_text(json.dumps(record))
    _, imported = load_generation(run, panel)
    assert imported["one"]["probes"] == [probe]


def test_reference_conflict_requires_fresh_design_without_relabeling_probes(generation):
    # [FUNCTION DOC] A blocking
    # "reference" review issue causes the source to be returned without importing the task, and the
    # receipt file is left unmodified.
    run, panel, source, _, receipt, record, _ = generation
    record["quality"]["review"] = {
        "issues": [
            {"category": "reference", "severity": "blocking", "problem": "Draft exceeds PR scope"}
        ]
    }
    receipt.write_text(json.dumps(record))
    sources, imported = load_generation(run, panel)
    assert sources == [source]
    assert imported == {}
    assert json.loads(receipt.read_text()) == record


@pytest.mark.parametrize("change", [None, "result", "task", "outside", "infrastructure"])
def test_execution_reuse_requires_unchanged_bound_results(generation, tmp_path, change):
    # [FUNCTION DOC] Parametrized over no change and
    # changes to the trial result bytes, bundle hash, an outside result path, or an infrastructure
    # exception: tampered evidence raises ValueError with reuse_evidence=True; untampered evidence
    # is
    # imported (infrastructure failures are dropped), and without reuse_evidence no trials are
    # imported.
    # Skipped if harbor is not installed.
    Task = pytest.importorskip("harbor.models.task.task").Task

    from repo2rlenv.quality.loop.artifacts import import_trial

    run, panel, _, task, receipt, record, _ = generation
    trial = run / "trials/baseline/result.json"
    trial.parent.mkdir(parents=True)
    native = {
        "task_checksum": Task(task).checksum,
        "config": {"agent": {"name": "nop"}},
        "verifier_result": {"rewards": {"reward": 0.0}},
    }
    if change == "infrastructure":
        native["exception_info"] = {"exception_type": "TimeoutError"}
    trial.write_text(json.dumps(native))
    evidence = import_trial(trial, task, "baseline").model_dump(mode="json")
    record["quality"]["trials"] = [evidence]
    if change == "result":
        trial.write_text(trial.read_text() + "\n")
    elif change == "task":
        evidence["bundle_hash"] = "sha256:" + "0" * 64
    elif change == "outside":
        outside = tmp_path / "outside-result.json"
        outside.write_bytes(trial.read_bytes())
        evidence["result"] = str(outside)
    receipt.write_text(json.dumps(record))
    if change in {"result", "task", "outside"}:
        with pytest.raises(ValueError, match="Reusable execution evidence"):
            load_generation(run, panel, reuse_evidence=True)
    else:
        _, imported = load_generation(run, panel, reuse_evidence=True)
        expected = {} if change == "infrastructure" else {"baseline": evidence}
        assert imported["one"]["trials"] == expected
        _, fresh_review = load_generation(run, panel)
        assert "trials" not in fresh_review["one"]


def test_execution_reuse_requires_generation_input(generation):
    # [FUNCTION DOC] Tasksmith.run with reuse_evidence=True
    # and no generation run raises ValueError ("requires --generation-run"); the instance is created
    # via
    # __new__ to bypass initialization.
    from repo2rlenv.tasksmith.runner import Tasksmith

    _, panel, *_ = generation
    with pytest.raises(ValueError, match="requires --generation-run"):
        Tasksmith.__new__(Tasksmith).run(panel, reuse_evidence=True)
