# [MODULE DOC] Tests for the shared history recipe infrastructure (recipes/history): entity-level
# change detection and per-recipe selection policies (check_entities, entities), cached merged-PR
# input (merged_pulls), and recipe-specific option parsing for swe_next (SWENextOptions) and
# r2e_gym (R2EGymOptions).
from __future__ import annotations

import json

import pytest

from repo2rlenv.pipelines.recipes.history.selection import check_entities, entities
from repo2rlenv.pipelines.recipes.history.source import merged_pulls
from repo2rlenv.spec.options import parse_options
from repo2rlenv.spec.recipe_options import R2EGymOptions, SWENextOptions


def options(cls):
    # [FUNCTION DOC] HELPER: Instantiates the given options class with source_paths ['pkg'] and
    # test_paths ['tests']. Input: an options model class. Output: its instance.
    return cls(source_paths=["pkg"], test_paths=["tests"])


def test_distinct_historical_selection_policies():
    # [FUNCTION DOC] Asserts check_entities reports the edited function and a test match for
    # R2E-Gym; that R2E-Gym rejects tests not naming the entity ('does not name') and pure
    # additions ('bug-edit') while SWE-Next accepts both (test_match False, added non-empty).
    source = {"pkg/core.py": ("def square(x):\n return x\n", "def square(x):\n return x*x\n")}
    tests = {"tests/test_core.py": ("", "def test_square():\n assert square(3) == 9\n")}
    result = check_entities(source, tests, options(R2EGymOptions))
    assert result["edited"] == ["pkg/core.py:square"]
    assert result["test_match"]
    unrelated = {"tests/test_core.py": ("", "def test_behavior():\n assert api() == 9\n")}
    with pytest.raises(ValueError, match="does not name"):
        check_entities(source, unrelated, options(R2EGymOptions))
    assert not check_entities(source, unrelated, options(SWENextOptions))["test_match"]
    added = {"pkg/core.py": ("", "def square(x):\n return x*x\n")}
    with pytest.raises(ValueError, match="bug-edit"):
        check_entities(added, tests, options(R2EGymOptions))
    assert check_entities(added, tests, options(SWENextOptions))["added"]


def test_docstrings_and_comments_do_not_create_behavior_changes():
    # [FUNCTION DOC] Asserts entities() ignores docstring and comment changes and that
    # check_entities raises 'No non-docstring' when only those change.
    before = 'class Number:\n "old"\n def value(self):\n  "old"\n  return 1\n'
    after = before.replace('"old"', '"new"') + "# a comment\n"
    assert entities(before) == entities(after)
    with pytest.raises(ValueError, match="No non-docstring"):
        check_entities({"pkg/x.py": (before, after)}, {}, options(SWENextOptions))


def test_pr_input_records_merge_revision_and_caches_api_result(tmp_path, monkeypatch):
    # [FUNCTION DOC] Monkeypatches source._run_gh with a fake returning PR JSON; asserts
    # merged_pulls records the merge commit SHA as head and a second call reads the receipt without
    # another gh call. Side effects: writes tmp_path/pulls.json.
    calls = []

    def gh(args):
        # [FUNCTION DOC] HELPER: Fake _run_gh(args); records args and returns a merged PR #17 JSON
        # payload.
        calls.append(args)
        return json.dumps(
            {
                "number": 17,
                "merge_commit_sha": "a" * 40,
                "head": {"sha": "b" * 40},
                "title": "Fix square",
                "body": "Expected nine.",
                "html_url": "https://github.com/a/b/pull/17",
                "merged_at": "2026-09-01",
            }
        )

    monkeypatch.setattr("repo2rlenv.pipelines.recipes.history.source._run_gh", gh)
    settings = options(SWENextOptions).model_copy(update={"pr_numbers": [17]})
    receipt = tmp_path / "pulls.json"
    rows = merged_pulls("https://github.com/a/b", settings, receipt)
    assert rows[0]["head"] == "a" * 40
    assert merged_pulls("https://github.com/a/b", settings, receipt) == rows
    assert len(calls) == 1


@pytest.mark.parametrize(
    "recipe,family,cls",
    [("swe_next", "pr_runtime", SWENextOptions), ("r2e_gym", "commit_runtime", R2EGymOptions)],
)
def test_owned_history_options_do_not_change_native_defaults(recipe, family, cls):
    # [FUNCTION DOC] Parametrized (swe_next/pr_runtime, r2e_gym/commit_runtime); asserts
    # parse_options returns the recipe options class for that recipe but not for recipe 'native'.
    assert isinstance(
        parse_options(family, {"source_paths": ["pkg"], "test_paths": ["tests"]}, recipe=recipe),
        cls,
    )
    assert not isinstance(parse_options(family, {}, recipe="native"), cls)
