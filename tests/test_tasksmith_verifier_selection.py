# [MODULE DOC] Tests for Tasksmith private verifier selection in worker.construct and the
# Design upstream_test_policy ("retain" vs "replace"), plus the strict submission contract exported
# for modified-only CPU tasks. Repository test execution and export are mocked except the final
# real export test.
"""Private verifier selection contracts; repository execution is mocked."""

import hashlib
import json

import pytest
from pydantic import ValidationError

from repo2rlenv.quality.test_results import TestResults
from repo2rlenv.spec.recipe_options import PythonRepositoryProfile
from repo2rlenv.tasksmith import worker
from repo2rlenv.tasksmith.models import Design, Profile


def design(**overrides):
    # [FUNCTION DOC] HELPER: Builds a valid Design with fixed instruction, one requirement,
    # rationale
    # and solution ideas, updated with keyword overrides. Output: Design; raises ValidationError on
    # bad
    # overrides.
    return Design(
        **{
            "instruction": "Implement the requested public feature while preserving existing behavior. "
            * 2,
            "requirements": [
                {
                    "behavior": "The new public API computes the requested value.",
                    "verification": "Call the API inside a test and compare an independent expected value.",
                    "source_evidence": "The pinned PR introduces this public API.",
                }
            ],
            "verifier_rationale": "Private tests preserve the upstream assertions without collection-time imports.",
            "wrong_solution_ideas": ["Return the wrong value."],
            "valid_alternative_ideas": ["Compute the same value another way."],
            **overrides,
        }
    )


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_replacement_cannot_remove_every_grading_test(text):
    # [FUNCTION DOC] Parametrized blank texts: policy
    # "replace" with empty additional_tests raises ValidationError ("requires private tests"); the
    # default
    # policy is "retain".
    with pytest.raises(ValidationError, match="requires private tests"):
        design(upstream_test_policy="replace", additional_tests=text)
    assert design().upstream_test_policy == "retain"


@pytest.mark.parametrize("policy", ["retain", "replace"])
def test_selection_preserves_readiness_and_routes_both_controls(monkeypatch, tmp_path, policy):
    # [FUNCTION DOC] Parametrized policy:
    # construct runs the healthy and defective controls on the verified image with selectors equal
    # to
    # the upstream ones (retain) or none (replace) plus the private behavior file, does not mutate
    # the
    # profile, records the policy and verifier source in exported metadata, and derives F2P/P2P
    # contrast
    # from the two runs.
    base = tmp_path / "base"
    (base / "lib").mkdir(parents=True)
    (base / "tests").mkdir()
    (base / "lib/core.py").write_text("value = 1\n")
    original = ["checks/test_upstream.py::test_feature"]
    profile = Profile(
        reasoning="The actual upstream tests passed on the merged revision.",
        resource="cpu",
        options=PythonRepositoryProfile(
            source_paths=["lib"],
            test_paths=["checks"],
            test_selectors=original,
        ),
        dependency_inputs=["pyproject.toml"],
        upstream_test_rationale="The upstream cases establish offline merged-head readiness.",
    )
    before = profile.model_dump()
    expected = ([] if policy == "replace" else original) + ["tests/tasksmith_behavior.py"]
    observed = []

    def test_image(image, options, output, *, replacements, removals=()):
        # [FUNCTION DOC] HELPER: Fake
        # test runner asserting image, selectors, test paths and replacements, recording the output
        # name and
        # returning TestResults where "feature" fails only on the defective control.
        assert image == "the-verified-merged-image"
        assert options.test_selectors == expected
        assert options.test_paths == ["checks", "tests"]
        assert "tests/tasksmith_behavior.py" in replacements
        assert not removals
        observed.append(output.name)
        broken = output.name == "defective"
        assert ("lib/core.py" in replacements) == broken
        return TestResults(
            {"feature": "failed" if broken else "passed", "adjacent": "passed"},
            int(broken),
        )

    exported = {}

    def export(**kwargs):
        # [FUNCTION DOC] HELPER: Fake
        # export_repository_task recording its kwargs and returning destination/name.
        exported.update(kwargs)
        return kwargs["destination"] / kwargs["name"]

    monkeypatch.setattr(worker, "run", lambda *args, **kwargs: "")
    monkeypatch.setattr(worker, "test_image", test_image)
    monkeypatch.setattr(worker, "export_repository_task", export)
    monkeypatch.setattr(worker, "test_excerpts", lambda *args, **kwargs: [])
    output = tmp_path / "construct"
    output.mkdir()
    source = {
        "source_files": ["lib/core.py"],
        "changed_files": [{"filename": "lib/core.py", "status": "modified"}],
        "source_diff": "a pinned source patch",
        "id": "fixture",
        "url": "https://example.org/repo/pull/1",
        "head": "h",
        "base": "b",
        "workspace_strategy": "head_minus_source_patch",
    }
    text = "def test_feature():\n    from lib.core import value\n    assert value == 1\n"
    result = worker.construct(
        source,
        profile,
        design(upstream_test_policy=policy, additional_tests=text),
        {"base": str(base), "image": "the-verified-merged-image"},
        output,
    )
    assert observed == ["healthy", "defective"]
    assert profile.model_dump() == before
    assert result["options"]["test_selectors"] == expected
    assert exported["metadata"]["upstream_test_policy"] == policy
    assert exported["verifier_source"] == {"tests/tasksmith_behavior.py": text.encode()}
    assert exported["contrast"] == {"FAIL_TO_PASS": ["feature"], "PASS_TO_PASS": ["adjacent"]}


def test_modified_only_cpu_task_collects_directories_with_strict_contract(monkeypatch, tmp_path):
    # [FUNCTION DOC] Runs real export
    # for a modification-only source: the Harbor task collects /workspace/lib as one artifact with
    # cache
    # excludes; tests/contract.json lists submitted roots/files, no optional files and the hash of
    # the
    # non-Python asset; the workspace holds the defective source without tests; validate_submission
    # accepts new Python helpers but rejects a changed immutable asset. Requires harbor.
    from harbor.models.task.task import Task

    from repo2rlenv.pipelines.recipes.swe_smith.grade import validate_submission

    base = tmp_path / "base"
    for name, text in {
        "lib/core.py": "value = 1\n",
        "lib/adjacent.py": "other = 2\n",
        "lib/schema.json": '{"version": 1}\n',
        "tests/test_core.py": "private readiness fixture\n",
    }.items():
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    defective = tmp_path / "defective"
    (defective / "lib").mkdir(parents=True)
    (defective / "lib/core.py").write_text("value = 0\n")
    profile = Profile(
        reasoning="The changed behavior has offline CPU unit tests.",
        resource="cpu",
        options=PythonRepositoryProfile(
            source_paths=["lib"], test_paths=["tests"], test_selectors=["tests/test_core.py"]
        ),
        dependency_inputs=["pyproject.toml"],
        upstream_test_rationale="An existing feature check and adjacent behavior both passed.",
    )
    source = {
        "source_files": ["lib/core.py"],
        "changed_files": [{"filename": "lib/core.py", "status": "modified"}],
        "source_diff": "pinned modification-only fixture",
        "id": "fixture",
        "url": "https://example.org/repo/pull/1",
        "head": "h",
        "base": "b",
        "workspace_strategy": "head_minus_source_patch",
    }

    def test_image(image, options, output, **kwargs):
        # [FUNCTION DOC] HELPER:
        # Fake test runner where "feature" fails only for the defective control.
        broken = output.name == "defective"
        return TestResults(
            {"feature": "failed" if broken else "passed", "adjacent": "passed"},
            int(broken),
        )

    monkeypatch.setattr(worker, "reverse_source", lambda *args: (defective, []))
    monkeypatch.setattr(worker, "test_image", test_image)
    monkeypatch.setattr(worker, "test_excerpts", lambda *args, **kwargs: [])
    # raising=False: os.chown doesn't exist on Windows.
    monkeypatch.setattr("os.chown", lambda *args: None, raising=False)
    output = tmp_path / "construct"
    output.mkdir()
    result = worker.construct(
        source, profile, design(), {"base": str(base), "image": "verified-fixture"}, output
    )
    task = output / result["task_relative"]
    artifacts = Task(task).config.artifacts
    assert len(artifacts) == 1
    assert artifacts[0].source == "/workspace/lib"
    assert artifacts[0].exclude == ["__pycache__", "*.pyc", ".pytest_cache"]

    contract = json.loads((task / "tests/contract.json").read_text())
    assert contract["submitted_roots"] == ["lib"]
    assert contract["submitted_files"] == ["lib/adjacent.py", "lib/core.py"]
    assert contract["optional_files"] == []
    assert contract["immutable_assets"] == {
        "lib/schema.json": hashlib.sha256((base / "lib/schema.json").read_bytes()).hexdigest()
    }
    workspace = task / "environment/source"
    assert not (workspace / "tests").exists()
    assert (task / "tests/source/tests/test_core.py").is_file()
    assert (workspace / "lib/core.py").read_text() == "value = 0\n"
    validate_submission(workspace, contract)
    (workspace / "lib/helper.py").write_text("helper = 3\n")
    validate_submission(workspace, contract)
    (workspace / "lib/schema.json").write_text("modified immutable asset\n")
    with pytest.raises(ValueError, match="Non-Python source assets must remain unchanged"):
        validate_submission(workspace, contract)
