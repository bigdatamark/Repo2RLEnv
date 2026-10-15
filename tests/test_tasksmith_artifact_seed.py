# [MODULE DOC] Tests for seeded author revisions in tasksmith.author.artifact.artifact_stage: when
# initial_draft_key names an existing stage input, that value is written as an unvalidated
# initial-draft.json seed which the agent revises; nothing commits without operator validation,
# invalid/unknown seeds fail before any provider call, completed stages reject changed inputs, and
# orphaned seeds force reconciliation. run_agent is monkeypatched, so no paid LLM calls happen.
# Also checks the Tasksmith runner only seeds design-N stages (N>0) from previous_design.
"""Seeded author revisions use existing validated stage input without paid calls."""

from __future__ import annotations

import json
import time
from copy import deepcopy

import pytest
from pydantic import BaseModel, ConfigDict

from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.quality.loop.client import RunBudget
from repo2rlenv.tasksmith.author import artifact
from repo2rlenv.tasksmith.author.budget import AuthorBudget


class Artifact(BaseModel):
    # [CLASS DOC] Pydantic test schema for the authored artifact: extra fields forbidden; fields title (str),
    # count (int), tests (list of str).
    model_config = ConfigDict(extra="forbid")

    title: str
    count: int
    tests: list[str]


@pytest.fixture
def options(tmp_path):
    # [FUNCTION DOC] Pytest fixture building the keyword arguments for artifact_stage: Artifact schema, stage
    # 'design-1', inputs with source, previous_design seed and an equal other_design copy,
    # system/prompt text, root at tmp_path/artifact, an AuthorBudget over a RunBudget backed by a
    # SQLite BudgetLedger (limit 10 USD, run cap 5), model/runtime names, max_cost 1, max_turns 4,
    # deadline now+30s and initial_draft_key 'previous_design'. Input: tmp_path. Output: dict.
    # Side effects: creates the ledger database under tmp_path.
    seed = {"title": "preserve", "count": 0, "tests": ["large unchanged verifier"]}
    budget = RunBudget(BudgetLedger(tmp_path / "budget.sqlite3", limit_usd="10"), "seed", "5")
    return {
        "schema": Artifact,
        "stage": "design-1",
        "inputs": {"source": "fixed", "previous_design": seed, "other_design": deepcopy(seed)},
        "system": "Author the revised design.",
        "prompt": json.dumps({"previous_design": seed}),
        "root": tmp_path / "artifact",
        "budget": AuthorBudget(budget, tmp_path / "charges", "design-1"),
        "model": "anthropic/claude-sonnet-4-6",
        "runtime": "pi",
        "max_cost": 1,
        "max_turns": 4,
        "deadline": time.time() + 30,
        "initial_draft_key": "previous_design",
    }


@pytest.mark.asyncio
async def test_seeded_minimal_revision_preserves_original_and_requires_validation(
    monkeypatch, options
):
    # [FUNCTION DOC] Runs artifact_stage with a fake agent that verifies the seed file (status unvalidated,
    # number 0, input_key, artifact equal to the original) and prompt wording, then revises count
    # to 2. Asserts the result, that inputs were not mutated, validate called once with 2,
    # draft/committed/seed share input_digest, and that a second call replays the committed result
    # without calling the agent again or rewriting the seed bytes.
    original = deepcopy(options["inputs"])
    validated = []
    calls = 0
    root = options["root"]

    async def validate(value):
        # [FUNCTION DOC] HELPER: async validator recording value.count; raises ValueError when count < 1.
        validated.append(value.count)
        if value.count < 1:
            raise ValueError("count must be positive")

    async def agent(**kwargs):
        # [FUNCTION DOC] HELPER: fake async run_agent. Increments the outer call counter, asserts nothing
        # validated or committed yet, checks the initial-draft.json contents and prompt text, then
        # calls handlers['revise_artifact'] with patch count=2 and asserts the response says
        # committed.
        nonlocal calls
        calls += 1
        assert not validated
        assert not (root / "artifact.json").exists()
        seed = json.loads((root / "initial-draft.json").read_text())
        assert seed["status"] == "unvalidated" and seed["number"] == 0
        assert seed["input_key"] == "previous_design"
        assert seed["artifact"] == original["previous_design"]
        assert "inputs['previous_design']" in kwargs["prompt"]
        assert "Nothing is committed yet" in kwargs["prompt"]
        assert "committed" in await kwargs["handlers"]["revise_artifact"](patch={"count": 2})

    monkeypatch.setattr(artifact, "run_agent", agent)
    result = await artifact.artifact_stage(**options, validate=validate)
    assert result == Artifact(**{**original["previous_design"], "count": 2})
    assert options["inputs"] == original
    assert validated == [2]
    seed_bytes = (root / "initial-draft.json").read_bytes()
    draft = json.loads((root / "draft.json").read_text())
    committed = json.loads((root / "artifact.json").read_text())
    seed = json.loads(seed_bytes)
    assert draft["origin"] == "revise_artifact" and draft["number"] == 1
    assert seed["input_digest"] == draft["input_digest"] == committed["input_digest"]
    assert seed["artifact"] == original["previous_design"]
    assert await artifact.artifact_stage(**options, validate=validate) == result
    assert (root / "initial-draft.json").read_bytes() == seed_bytes
    assert calls == 1 and validated == [2]


@pytest.mark.asyncio
async def test_seed_revision_cannot_bypass_operator_validation(monkeypatch, options):
    # [FUNCTION DOC] Verifies an empty patch on the seed (count 0) is rejected by validation and not committed,
    # while a patch to count 1 commits. Asserts the result count is 1 and validation saw [0, 1].
    checked = []

    async def validate(value):
        # [FUNCTION DOC] HELPER: async validator recording value.count; raises ValueError when count == 0.
        checked.append(value.count)
        if value.count == 0:
            raise ValueError("Seed still fails construction requirements")

    async def agent(**kwargs):
        # [FUNCTION DOC] HELPER: fake run_agent that first revises with an empty patch (expects 'rejected' and no
        # artifact.json), then with count=1 (expects 'committed').
        response = await kwargs["handlers"]["revise_artifact"](patch={})
        assert "rejected" in response
        assert not (options["root"] / "artifact.json").exists()
        assert "committed" in await kwargs["handlers"]["revise_artifact"](patch={"count": 1})

    monkeypatch.setattr(artifact, "run_agent", agent)
    result = await artifact.artifact_stage(**options, validate=validate)
    assert result.count == 1
    assert checked == [0, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [None, [], {"title": "incomplete"}, {"title": "x" * 512001}])
async def test_invalid_seed_fails_before_model_or_operation(monkeypatch, options, seed):
    # [FUNCTION DOC] Parametrized over invalid seeds (None, empty list, incomplete dict, oversized title).
    # Expects ValueError 'invalid initial draft', zero budget spent and no root directory created;
    # the agent must never be called.
    async def forbidden(**kwargs):
        # [FUNCTION DOC] HELPER: fake run_agent that fails the test if invoked.
        pytest.fail("Invalid seed must not reach the provider")

    monkeypatch.setattr(artifact, "run_agent", forbidden)
    options["inputs"]["previous_design"] = seed
    with pytest.raises(ValueError, match="invalid initial draft"):
        await artifact.artifact_stage(**options)
    assert options["budget"].spent == 0
    assert not options["root"].exists()


@pytest.mark.asyncio
async def test_seed_key_must_be_an_existing_controller_input(monkeypatch, options):
    # [FUNCTION DOC] Passes initial_draft_key 'unknown' and expects ValueError 'existing input' with no root
    # created and no agent call.
    async def forbidden(**kwargs):
        # [FUNCTION DOC] HELPER: fake run_agent that fails the test if invoked.
        pytest.fail("Unknown seed input must not reach the provider")

    monkeypatch.setattr(artifact, "run_agent", forbidden)
    with pytest.raises(ValueError, match="existing input"):
        await artifact.artifact_stage(**{**options, "initial_draft_key": "unknown"})
    assert not options["root"].exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["seed", "source", "seed_key", "disable_seed"])
async def test_completed_seeded_stage_rejects_changed_inputs(monkeypatch, options, change):
    # [FUNCTION DOC] Completes a stage once, then re-runs with one change (seed value, source input, different
    # seed key, or seed disabled). Expects ValueError 'different inputs', the agent called only
    # once and artifact.json/initial-draft.json bytes unchanged.
    calls = 0

    async def agent(**kwargs):
        # [FUNCTION DOC] HELPER: fake run_agent counting calls and revising count to 2.
        nonlocal calls
        calls += 1
        await kwargs["handlers"]["revise_artifact"](patch={"count": 2})

    monkeypatch.setattr(artifact, "run_agent", agent)
    await artifact.artifact_stage(**options)
    root = options["root"]
    saved = {name: (root / name).read_bytes() for name in ("artifact.json", "initial-draft.json")}
    changed = {**options, "inputs": deepcopy(options["inputs"])}
    if change == "seed":
        changed["inputs"]["previous_design"]["count"] = 3
    elif change == "source":
        changed["inputs"]["source"] = "another source"
    else:
        changed["initial_draft_key"] = "other_design" if change == "seed_key" else None
    with pytest.raises(ValueError, match="different inputs"):
        await artifact.artifact_stage(**changed)
    assert calls == 1
    assert {name: (root / name).read_bytes() for name in saved} == saved


@pytest.mark.asyncio
async def test_seed_alone_is_not_accepted_and_incomplete_stage_cannot_replay(monkeypatch, options):
    # [FUNCTION DOC] Agent does nothing, so the stage ends with ValueError 'without a validated artifact'; the
    # seed exists but artifact.json does not. A second call raises RuntimeError 'reconciliation'
    # without rewriting the seed or calling the agent again.
    calls = 0

    async def agent(**kwargs):
        # [FUNCTION DOC] HELPER: fake run_agent that only counts calls.
        nonlocal calls
        calls += 1

    monkeypatch.setattr(artifact, "run_agent", agent)
    with pytest.raises(ValueError, match="without a validated artifact"):
        await artifact.artifact_stage(**options)
    seed = (options["root"] / "initial-draft.json").read_bytes()
    assert not (options["root"] / "artifact.json").exists()
    with pytest.raises(RuntimeError, match="reconciliation"):
        await artifact.artifact_stage(**options)
    assert (options["root"] / "initial-draft.json").read_bytes() == seed
    assert calls == 1


@pytest.mark.asyncio
async def test_orphaned_seed_is_not_overwritten(monkeypatch, options):
    # [FUNCTION DOC] Pre-creates root/initial-draft.json with arbitrary text and expects RuntimeError
    # 'reconciliation', the orphan left intact and no operation.json written; the agent must not
    # run.
    async def forbidden(**kwargs):
        # [FUNCTION DOC] HELPER: fake run_agent that fails the test if invoked.
        pytest.fail("Unreconciled seed must not reach the provider")

    monkeypatch.setattr(artifact, "run_agent", forbidden)
    root = options["root"]
    root.mkdir()
    (root / "initial-draft.json").write_text("preserved orphan")
    with pytest.raises(RuntimeError, match="reconciliation"):
        await artifact.artifact_stage(**options)
    assert (root / "initial-draft.json").read_text() == "preserved orphan"
    assert not (root / "operation.json").exists()


@pytest.mark.parametrize(
    "stage,previous,expected",
    [
        ("design-0", None, None),
        ("design-1", {"count": 1}, "previous_design"),
        ("investigate-1", {"count": 1}, None),
    ],
)
def test_runner_seeds_only_existing_design_input(monkeypatch, tmp_path, stage, previous, expected):
    # [FUNCTION DOC] Parametrized (stage, previous, expected key). Builds a Tasksmith runner (with a budget
    # ledger under the campaign dir) and patches runner.artifact_stage with a capture that asserts
    # initial_draft_key equals expected (previous_design only for design-1 with a previous design)
    # and that inputs pass previous_design through by identity. Asserts Tasksmith.author returns
    # the capture's sentinel.
    from repo2rlenv.tasksmith import runner as module
    from repo2rlenv.tasksmith.models import Options

    BudgetLedger(tmp_path / "campaign/budget.sqlite3", limit_usd="10")
    tasksmith = module.Tasksmith(
        tmp_path / "run", tmp_path / "campaign", Options(), tmp_path / "wheel.whl"
    )
    sentinel = object()

    async def capture(**kwargs):
        # [FUNCTION DOC] HELPER: fake async artifact_stage asserting initial_draft_key and
        # inputs['previous_design'] identity; returns the sentinel object.
        assert kwargs["initial_draft_key"] == expected
        assert kwargs["inputs"]["previous_design"] is previous
        return sentinel

    monkeypatch.setattr(module, "artifact_stage", capture)
    assert (
        tasksmith.author(
            tmp_path / "candidate", stage, Artifact, {"previous_design": previous}, "/workspace"
        )
        is sentinel
    )
