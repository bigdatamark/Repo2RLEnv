# [MODULE DOC] Behavioral contract tests for Tasksmith (PR source intake, shared budgets, private
# test export, profile validation, remote-only worker, artifact authoring stage resume/revision, and
# the runner graph). All agents, remote calls and GitHub access are faked; no target code or cloud
# execution runs.
"""Behavioral contracts for Tasksmith; no target code or cloud execution locally."""

from __future__ import annotations

import json
import time

import pytest
from pydantic import BaseModel

from repo2rlenv.campaigns.budget import BudgetExceeded, BudgetLedger
from repo2rlenv.pipelines.recipes.repository.export import export_repository_task
from repo2rlenv.quality.loop.client import RunBudget
from repo2rlenv.spec.recipe_options import PythonRepositoryProfile
from repo2rlenv.tasksmith.author.budget import AuthorBudget
from repo2rlenv.tasksmith.models import Panel, Profile


def test_frozen_pr_records_bind_source_inventory_before_dispatch():
    # [FUNCTION DOC] Builds a source record with id derived from sha256 of url/head/diff;
    # validate_source_records accepts it and raises ValueError for changed head, repo, operations,
    # traversal paths, or a URL not in the panel.
    import hashlib

    from repo2rlenv.tasksmith.source import validate_source_records

    source = {
        "url": "https://github.com/example/project/pull/7",
        "repo": "https://github.com/example/project",
        "base": "a" * 40,
        "head": "b" * 40,
        "source_diff": "diff --git a/src/core.py b/src/core.py\n--- a/src/core.py\n+++ b/src/core.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n",
        "source_files": ["src/core.py"],
        "source_operations": [{"path": "src/core.py", "operation": "modified"}],
        "workspace_strategy": "head_minus_source_patch",
    }
    identity = {key: source[key] for key in ("url", "head", "source_diff")}
    source["id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    assert validate_source_records([source], [source["url"]]) == [source]
    for changed in (
        {**source, "head": "c" * 40},
        {**source, "repo": "https://github.com/other/project"},
        {**source, "source_operations": [{"path": "src/core.py", "operation": "removed"}]},
        {**source, "source_files": ["../outside.py"]},
    ):
        with pytest.raises(ValueError):
            validate_source_records([changed], [source["url"]])
    with pytest.raises(ValueError, match="panel URLs"):
        validate_source_records([source], ["https://github.com/example/project/pull/8"])


def test_component_spend_and_uncertainty_share_one_campaign(tmp_path):
    # [FUNCTION DOC] AuthorBudget and a quality RunBudget share a pilot cap of 5: a third
    # reservation fails, settlement updates totals, and marking uncertain keeps the reservation.
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="10")
    pilot = RunBudget(ledger, "ts-one", "5")
    quality = RunBudget(pilot, "ts-one-q", "3")
    author = AuthorBudget(pilot, tmp_path / "charges", "investigate")
    operation = author.reserve(2, "author call")
    quality.reserve("trial:ts-one-q-oracle", "2", "quality call")
    with pytest.raises(BudgetExceeded):
        author.reserve(2, "must not dispatch")
    author.settle(operation, 0.5)
    assert pilot.totals() == {"accounted_usd": "0.5", "reserved_usd": "2"}
    ledger.mark_uncertain("trial:ts-one-q-oracle", "lost provider response")
    assert pilot.totals()["reserved_usd"] == "2"
    assert len(ledger.status()["operations"]) == 2


def test_selected_tests_and_release_notes_stay_private(tmp_path):
    # [FUNCTION DOC] Export hides tests and public_exclude files from the workspace, keeps private
    # tests in the verifier context, records selectors in contract.json, and Harbor config has a
    # separate verifier and no-network agent.
    base = tmp_path / "snapshot"
    for name, content in {
        "lib/core.py": "value = 1\n",
        "tests/test_core.py": "private test\n",
        "tests/other.py": "also private\n",
        "CHANGES.md": "the exact fix\n",
        "README.md": "package install text\n",
    }.items():
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    options = PythonRepositoryProfile(
        source_paths=["lib"],
        test_paths=["tests"],
        test_selectors=["tests/test_core.py::test_value"],
        public_exclude=["CHANGES.md"],
    )
    task = export_repository_task(
        base=base,
        defective={"lib/core.py": b"value = 0\n"},
        reference={"lib/core.py": b"value = 1\n"},
        options=options,
        instruction="Correct the public behavior.",
        destination=tmp_path / "output",
        name="tasksmith-fixture",
        org="test",
        contrast={"FAIL_TO_PASS": ["tests.test_core::test_value"], "PASS_TO_PASS": []},
        metadata={"recipe": "tasksmith", "recipe_version": "1"},
    )
    assert not (task / "environment/source/tests").exists()
    assert not (task / "environment/source/CHANGES.md").exists()
    assert (task / "tests/source/tests/other.py").is_file()
    assert (task / "environment/source/README.md").is_file()
    assert (
        json.loads((task / "tests/contract.json").read_text())["test_paths"]
        == options.test_selectors
    )
    assert (task / "environment/source/lib/core.py").read_bytes() == b"value = 0\n"
    harbor = pytest.importorskip("harbor.models.task.task")
    config = harbor.Task(task).config
    assert config.verifier.environment_mode.value == "separate"
    assert config.agent.network_mode.value == "no-network"


def test_profile_requires_private_selectors_and_fixed_panel():
    # [FUNCTION DOC] Profile rejects selectors outside test paths; Panel rejects duplicate PRs.
    with pytest.raises(ValueError, match="inside a private"):
        Profile(
            reasoning="Inspected dependency metadata.",
            resource="cpu",
            options=PythonRepositoryProfile(
                source_paths=["src"], test_paths=["tests"], test_selectors=["public/test_a.py"]
            ),
            dependency_inputs=["pyproject.toml"],
            upstream_test_rationale="Offline unit tests of the changed behavior.",
        )
    with pytest.raises(ValueError, match="unique"):
        Panel(name="fixed", prs=["one", "one"])


def test_remote_worker_refuses_controller_execution(monkeypatch, tmp_path):
    # [FUNCTION DOC] worker.main without REPO2RLENV_REMOTE_WORKER raises 'remote only' and creates
    # no output.
    from repo2rlenv.tasksmith import worker

    monkeypatch.delenv("REPO2RLENV_REMOTE_WORKER", raising=False)
    monkeypatch.setattr("sys.argv", ["worker", str(tmp_path / "config"), str(tmp_path / "out")])
    with pytest.raises(RuntimeError, match="remote only"):
        worker.main()
    assert not (tmp_path / "out").exists()


@pytest.mark.asyncio
async def test_author_revises_only_rejected_fields_and_reuses_completed_artifact(
    monkeypatch, tmp_path
):
    # [FUNCTION DOC] Fake agent submits an invalid artifact (rejected), then revises only 'count';
    # artifact_stage returns the merged artifact, reuses it on rerun without another agent call, and
    # raises 'different inputs' when inputs change.
    from repo2rlenv.tasksmith.author import artifact

    class Artifact(BaseModel):
        # [CLASS DOC] HELPER: Pydantic schema with title (str) and count (int).
        title: str
        count: int

    calls = 0

    async def agent(**kwargs):
        # [FUNCTION DOC] HELPER: Fake run_agent that counts calls, submits count=0 expecting
        # rejection, then revises count to 2 expecting commit.
        nonlocal calls
        calls += 1
        assert (
            "rejected"
            in (await kwargs["handlers"]["submit_artifact"](title="retain", count=0)).lower()
        )
        assert "committed" in await kwargs["handlers"]["revise_artifact"](patch={"count": 2})

    async def validate(value):
        # [FUNCTION DOC] HELPER: Async validator raising ValueError when count < 1.
        if value.count < 1:
            raise ValueError("count must be positive")

    monkeypatch.setattr(artifact, "run_agent", agent)
    budget = RunBudget(BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="10"), "ts-test", "5")
    kwargs = dict(
        schema=Artifact,
        stage="design",
        inputs={"source": "pinned"},
        system="test",
        prompt="test",
        root=tmp_path / "artifact",
        budget=AuthorBudget(budget, tmp_path / "charges", "design"),
        model="anthropic/claude-sonnet-4-6",
        runtime="pi",
        max_cost=1,
        max_turns=4,
        deadline=time.time() + 30,
        validate=validate,
    )
    result = await artifact.artifact_stage(**kwargs)
    assert result.title == "retain" and result.count == 2
    assert await artifact.artifact_stage(**kwargs) == result
    assert calls == 1
    with pytest.raises(ValueError, match="different inputs"):
        await artifact.artifact_stage(**{**kwargs, "inputs": {"source": "changed"}})


@pytest.mark.asyncio
async def test_incomplete_author_never_blindly_replays(monkeypatch, tmp_path):
    # [FUNCTION DOC] After the agent raises, a second artifact_stage call raises 'reconciliation'
    # instead of rerunning; agent called once.
    from repo2rlenv.tasksmith.author import artifact

    class Artifact(BaseModel):
        # [CLASS DOC] HELPER: Pydantic schema with a title field.
        title: str

    calls = 0

    async def agent(**kwargs):
        # [FUNCTION DOC] HELPER: Fake run_agent counting calls and raising RuntimeError('provider
        # disconnected').
        nonlocal calls
        calls += 1
        raise RuntimeError("provider disconnected")

    monkeypatch.setattr(artifact, "run_agent", agent)
    budget = RunBudget(BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="10"), "ts-test", "5")
    kwargs = dict(
        schema=Artifact,
        stage="design",
        inputs={},
        system="test",
        prompt="test",
        root=tmp_path / "artifact",
        budget=AuthorBudget(budget, tmp_path / "charges", "design"),
        model="anthropic/claude-sonnet-4-6",
        runtime="pi",
        max_cost=1,
        max_turns=4,
        deadline=time.time() + 30,
    )
    with pytest.raises(RuntimeError, match="disconnected"):
        await artifact.artifact_stage(**kwargs)
    with pytest.raises(RuntimeError, match="reconciliation"):
        await artifact.artifact_stage(**kwargs)
    assert calls == 1


@pytest.mark.asyncio
async def test_completed_truncation_recovers_within_original_allowance(monkeypatch, tmp_path):
    # [FUNCTION DOC] First fake run charges 0.2 and raises ProviderOutputError after 2 requests; the
    # retry gets the remaining 2 turns and 0.8 cost with a recovery note and succeeds; accounted
    # spend is 0.2.
    from repo2rlenv.tasksmith.author import artifact
    from repo2rlenv.tasksmith.author.bridge import ProviderOutputError

    class Artifact(BaseModel):
        # [CLASS DOC] HELPER: Pydantic schema with title.
        title: str

    calls = []

    async def agent(**kwargs):
        # [FUNCTION DOC] HELPER: Fake run_agent: first call reserves/settles 0.2, writes a two-
        # request trace and raises truncation; second asserts reduced allowance and submits the
        # artifact.
        calls.append(kwargs)
        if len(calls) == 1:
            charge = kwargs["budget"].reserve(0.2, "completed response")
            kwargs["budget"].settle(charge, 0.2)
            kwargs["trace"].write_text('{"kind":"model_request"}\n' * 2)
            raise ProviderOutputError("Provider response truncated")
        assert kwargs["max_turns"] == 2
        assert kwargs["max_cost"] == pytest.approx(0.8)
        assert "Partial tool arguments were not executed" in kwargs["prompt"]
        await kwargs["handlers"]["submit_artifact"](title="complete")

    monkeypatch.setattr(artifact, "run_agent", agent)
    budget = RunBudget(BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="10"), "ts-test", "5")
    result = await artifact.artifact_stage(
        schema=Artifact,
        stage="design",
        inputs={},
        system="test",
        prompt="test",
        root=tmp_path / "artifact",
        budget=AuthorBudget(budget, tmp_path / "charges", "design"),
        model="anthropic/claude-sonnet-4-6",
        runtime="pi",
        max_cost=1,
        max_turns=4,
        deadline=time.time() + 30,
    )
    assert result.title == "complete" and len(calls) == 2
    assert budget.totals()["accounted_usd"] == "0.2"


def test_graph_repairs_bootstrap_before_design_without_replacing_input(monkeypatch, tmp_path):
    # [FUNCTION DOC] Requires langgraph sqlite. Drives Tasksmith.candidate with faked
    # remote/author/QualityLoop: inspect, investigate, a failing bootstrap, re-investigation given
    # the failure, successful bootstrap, design and construct, ending usable; a repeated call
    # returns the cached result without new calls.
    pytest.importorskip("langgraph.checkpoint.sqlite")
    from types import SimpleNamespace

    from repo2rlenv.tasksmith import runner as module
    from repo2rlenv.tasksmith.models import Design, Options

    BudgetLedger(tmp_path / "campaign/budget.sqlite3", limit_usd="100")
    runner = module.Tasksmith(
        tmp_path / "run", tmp_path / "campaign", Options(), tmp_path / "wheel.whl"
    )
    runner.receipt, runner.worker, runner.python = (
        tmp_path / "worker.json",
        object(),
        "remote-python",
    )
    calls = []
    profile = Profile(
        reasoning="Inspected Python metadata and tests.",
        resource="cpu",
        options=PythonRepositoryProfile(
            source_paths=["lib"],
            test_paths=["tests"],
            test_selectors=["tests/test_lib.py"],
            materialize_document_links=["CONTRIBUTING.md"],
        ),
        dependency_inputs=["pyproject.toml"],
        upstream_test_rationale="Small offline regression suite.",
    )
    design = Design(
        instruction="Repair the public behavior while keeping all existing cases and errors compatible with the original library API.",
        requirements=[
            {
                "behavior": "Handle an empty input correctly",
                "verification": "Assert the resulting iterator is empty",
                "source_evidence": "tests/test_lib.py",
            }
        ],
        verifier_rationale="Existing regression tests cover the change.",
        wrong_solution_ideas=["Always return empty"],
        valid_alternative_ideas=["Use a different early guard"],
    )

    def remote(root, key, data):
        # [FUNCTION DOC] HELPER: Fake remote step recording the key and returning canned inspect,
        # failing bootstrap-1, successful bootstrap-2 and construct-1 results; asserts any other key
        # is construct-1.
        calls.append(key)
        if key == "inspect":
            return {
                "status": "completed",
                "value": {
                    "checkout": "/remote/checkout",
                    "snapshot_links": [{"path": "CONTRIBUTING.md", "target": "docs/guide.md"}],
                },
            }
        if key == "bootstrap-1":
            return {"status": "failed", "error": "README required by installer"}
        if key == "bootstrap-2":
            return {
                "status": "completed",
                "value": {"readiness": {"test": "passed"}, "dependency_cache": {"cache_hit": True}},
            }
        assert key == "construct-1"
        return {
            "status": "completed",
            "local": "/local/evidence",
            "value": {"task_relative": "task/generated"},
        }

    def author(root, stage, schema, inputs, checkout, **kwargs):
        # [FUNCTION DOC] HELPER: Fake author recording the stage; on investigate-0 checks links are
        # passed and that validation rejects a profile missing document links; on investigate-1
        # checks the previous failure/profile; returns profile or design by schema.
        import asyncio

        calls.append(stage)
        if stage == "investigate-0":
            assert inputs["snapshot_links"][0]["path"] == "CONTRIBUTING.md"
            incomplete = profile.model_copy(deep=True)
            incomplete.options.materialize_document_links = []
            with pytest.raises(ValueError, match="all identified document links"):
                asyncio.run(kwargs["validate"](incomplete))
            assert calls == ["inspect", "investigate-0"]
            asyncio.run(kwargs["validate"](profile))
        if stage == "investigate-1":
            assert "README" in inputs["previous_failure"]["error"]
            assert inputs["previous_profile"] == profile.model_dump()
        return profile if schema is Profile else design

    class Loop:
        # [CLASS DOC] HELPER: Fake QualityLoop.
        def __init__(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Asserts protected_paths is ('solution', 'environment/source').
            assert kwargs["protected_paths"] == ("solution", "environment/source")

        def run(self, task, **kwargs):
            # [FUNCTION DOC] HELPER: Asserts the local task path and returns a usable result object
            # with model_dump.
            assert str(task) == "/local/evidence/task/generated"
            return SimpleNamespace(
                status="usable",
                model_dump=lambda **kwargs: {
                    "status": "usable",
                    "task_path": "/local/evidence/task/generated",
                    "bundle_hash": "sha256:test",
                },
            )

    monkeypatch.setattr(module, "task_identity", lambda path: "sha256:test", raising=False)
    monkeypatch.setattr(runner, "remote", remote)
    monkeypatch.setattr(runner, "author", author)
    monkeypatch.setattr(module, "QualityLoop", Loop)
    monkeypatch.setattr(module, "RemoteTrials", lambda *args, **kwargs: SimpleNamespace())
    result = runner.candidate(
        {
            "id": "pinned",
            "url": "https://github.com/example/lib/pull/1",
            "source_files": ["lib/core.py"],
        }
    )
    assert result["status"] == "usable"
    assert calls == [
        "inspect",
        "investigate-0",
        "bootstrap-1",
        "investigate-1",
        "bootstrap-2",
        "design-0",
        "construct-1",
    ]
    assert runner.candidate(result["source"]) == result
    assert len(calls) == 7


def test_source_intake_rejects_a_moving_pr_and_malformed_url(monkeypatch):
    # [FUNCTION DOC] Fake gh returns a different head on the second read so resolve_pr raises
    # 'changed during intake'; a URL with a query raises 'public GitHub'.
    from repo2rlenv.tasksmith import source

    head_reads = 0

    def api(args):
        # [FUNCTION DOC] HELPER: Fake gh returning the file list for '/files?' and PR metadata whose
        # head sha changes after the first read.
        nonlocal head_reads
        if "/files?" in args[1]:
            return json.dumps(
                [
                    [
                        {
                            "filename": "lib/core.py",
                            "status": "modified",
                            "additions": 1,
                            "deletions": 1,
                        }
                    ]
                ]
            )
        head_reads += 1
        return json.dumps(
            {
                "merged_at": "2026-01-01",
                "base": {"sha": "a" * 40, "repo": {"private": False}},
                "head": {"sha": ("b" if head_reads == 1 else "c") * 40},
                "title": "fix",
                "body": "",
            }
        )

    monkeypatch.setattr(source, "_run_gh", api)
    monkeypatch.setattr(
        source,
        "fetch_pr_diff",
        lambda *args: (
            "diff --git a/lib/core.py b/lib/core.py\n--- a/lib/core.py\n+++ b/lib/core.py\n@@ -1 +1 @@\n-old\n+new\n"
        ),
    )
    with pytest.raises(ValueError, match="changed during intake"):
        source.resolve_pr("https://github.com/example/library/pull/1")
    with pytest.raises(ValueError, match="public GitHub"):
        source.resolve_pr("https://github.com/example/library/pull/1?redirect=other")


def test_supplemental_only_fail_to_pass_has_static_evidence(tmp_path):
    # [FUNCTION DOC] test_excerpts finds a test that exists only in additional_sources and does not
    # write it to disk.
    from repo2rlenv.quality.python_evidence import test_excerpts

    snippet = "def test_empty():\n    assert list(public_api([])) == []\n"
    result = test_excerpts(
        tmp_path,
        ["tests.tasksmith_behavior::test_empty"],
        additional_sources={"tests/tasksmith_behavior.py": snippet},
    )
    assert "test_empty" in result["tests/tasksmith_behavior.py"]
    assert not (tmp_path / "tests/tasksmith_behavior.py").exists()
