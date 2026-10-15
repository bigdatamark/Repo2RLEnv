# [MODULE DOC] Tests for the Endless Terminals owned recipe: deterministic category/complexity
# sampling (sampler.sample_inputs) and the staged design flow where separate author calls receive
# prior-stage evidence. Model calls are monkeypatched.
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from repo2rlenv.pipelines.recipes.endless_terminals import recipe
from repo2rlenv.pipelines.recipes.endless_terminals.sampler import sample_inputs
from repo2rlenv.pipelines.recipes.terminal import templates


def test_native_category_complexity_context_sampling(tmp_path):
    # [FUNCTION DOC] Writes sampler JSON configs to tmp_path and asserts sample_inputs is
    # deterministic for a seed, yields 20 uniquely titled rows spanning >5 categories and >1
    # complexity, honours an explicit categories filter, and raises ValueError 'Unknown' for an
    # unknown category. Side effects: writes tmp_path file.
    source = tmp_path / "sampler.json"
    source.write_text(json.dumps({"seed": 24, "count": 20}))
    seeds = sample_inputs(source)
    assert seeds == sample_inputs(source)
    assert len(seeds) == len({item["title"] for item in seeds}) == 20
    assert len({item["category"] for item in seeds}) > 5
    assert len({item["task_complexity"] for item in seeds}) > 1
    source.write_text(
        json.dumps({"count": 2, "categories": ["text diffing and patch application"]})
    )
    assert {row["category"] for row in sample_inputs(source)} == {
        "text diffing and patch application"
    }
    source.write_text('{"categories":["unknown"]}')
    with pytest.raises(ValueError, match="Unknown"):
        sample_inputs(source)


def test_separate_authors_receive_prior_stage_evidence(tmp_path, monkeypatch):
    # [FUNCTION DOC] Patches templates.metered_complete and campaigns.structured.metered_complete
    # with a fake that returns a description/truth JSON on the first call and a test program
    # afterwards, then calls recipe.design and asserts three calls with operation ids design,
    # :initial, :final; that the final author receives the initial tests; stage-specific system
    # prompts; equal initial/final tests; and core_capabilities equal to the category. Inputs:
    # tmp_path, monkeypatch. Side effects: receipt under tmp_path.
    calls = []
    program = "\n".join(f"def test_case_{i}():\n    assert {i} == {i}\n" for i in range(5))

    def complete(model, **kwargs):
        # [FUNCTION DOC] HELPER: Fake metered completion. Records kwargs; returns a SimpleNamespace
        # whose content is JSON with description/truth on the first call, else {'code': program}.
        calls.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "description": "A concrete required outcome. " * 8,
                    "truth": "Private initial data. " * 8,
                }
                if len(calls) == 1
                else {"code": program}
            )
        )

    monkeypatch.setattr(templates, "metered_complete", complete)
    monkeypatch.setattr("repo2rlenv.campaigns.structured.metered_complete", complete)
    design = recipe.design(
        {"category": "text processing", "request": "Prepare the described report."},
        model=None,
        ledger=None,
        receipt=tmp_path / "design.json",
        operation_id="design:fixture",
        resume=False,
    )
    assert [call["operation_id"] for call in calls] == [
        "design:fixture",
        "design:fixture:initial",
        "design:fixture:final",
    ]
    assert json.loads(calls[2]["user"])["initial_tests"] == program
    assert "before" in calls[1]["system"]
    assert "FINAL" in calls[2]["system"]
    assert design.initial_tests == design.final_tests == program
    assert design.core_capabilities == ["text processing"]
