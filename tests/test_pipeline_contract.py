# [MODULE DOC] Contract tests over the pipeline registry (repo2rlenv.pipelines.PIPELINES): every
# registered class declares a matching `name`, a PipelineName enum value, an OPTIONS_REGISTRY
# entry, a boolean requires_bootstrap and a boolean experimental flag (only
# pr_diff/pr_runtime/commit_runtime stable); Python-only pipelines restrict supported_languages to
# {PYTHON} and language-agnostic ones leave it None; PipelineResult has its documented fields;
# Pipeline/PipelineResult are exported. Must be run after touching pipelines/. Side effects: none.
"""Every registered pipeline must satisfy the Pipeline Protocol."""

from __future__ import annotations

from repo2rlenv.pipelines import PIPELINES, PipelineResult
from repo2rlenv.spec.input import PipelineName
from repo2rlenv.spec.options import OPTIONS_REGISTRY


def test_pipelines_implement_protocol():
    # [FUNCTION DOC] Asserts the registry is non-empty and each entry is a class with a `name`
    # attribute equal to its registry key.
    """Each PIPELINES entry duck-conforms to Pipeline."""
    assert PIPELINES, "PIPELINES registry is empty"
    for name, cls in PIPELINES.items():
        # runtime_checkable Protocol — works on the class itself
        assert isinstance(cls, type), f"{name}: not a class"
        # Each implementation must declare its name
        assert hasattr(cls, "name"), f"{name}: missing class attribute `name`"
        # And the declared name must be the registered name
        declared = cls.name
        assert declared == name, f"{name}: class declares name={declared!r}, registered as {name!r}"


def test_pipeline_names_are_in_enum():
    # [FUNCTION DOC] Asserts every registry key converts to PipelineName (ValueError would fail the
    # test).
    """Every registered pipeline name is a valid PipelineName enum value."""
    for name in PIPELINES:
        # Will raise ValueError if not in the enum
        PipelineName(name)


def test_options_registry_aligns_with_pipelines():
    # [FUNCTION DOC] Asserts every registered pipeline has an Options class in OPTIONS_REGISTRY.
    """Every registered pipeline has a corresponding Options class."""
    for name in PIPELINES:
        assert name in OPTIONS_REGISTRY, f"pipeline {name!r} has no Options class registered"


def test_pipelines_declare_requires_bootstrap():
    # [FUNCTION DOC] Asserts every pipeline class has a bool requires_bootstrap attribute
    # (cmd_generate dispatches on it).
    """Each pipeline must declare requires_bootstrap (True/False) — cmd_generate dispatches on it."""
    for name, cls in PIPELINES.items():
        assert hasattr(cls, "requires_bootstrap"), (
            f"{name}: missing class attribute `requires_bootstrap`"
        )
        assert isinstance(cls.requires_bootstrap, bool)


def test_pipeline_stability_classification():
    # [FUNCTION DOC] Asserts each class's `experimental` (default False if missing) is a bool and
    # is True exactly for pipelines outside {pr_diff, pr_runtime, commit_runtime}.
    """pr_diff, pr_runtime, commit_runtime are stable; the rest are experimental.

    cmd_generate warns before running any pipeline whose `experimental` is True.
    """
    stable = {"pr_diff", "pr_runtime", "commit_runtime"}
    for name, cls in PIPELINES.items():
        flag = getattr(cls, "experimental", False)
        assert isinstance(flag, bool)
        expected_experimental = name not in stable
        assert flag is expected_experimental, (
            f"{name}: experimental should be {expected_experimental}"
        )


def test_python_only_pipelines_declare_supported_languages():
    # [FUNCTION DOC] Asserts code_instruct and equivalence_tests set supported_languages to
    # frozenset({LanguageHint.PYTHON}).
    """The Python-only pipelines must restrict supported_languages to {PYTHON}.

    Anything else risks the silent-late-failure mode the pre-flight check
    in cmd_generate is designed to prevent.
    """
    from repo2rlenv.bootstrap.spec import LanguageHint

    python_only = {
        "code_instruct",
        "equivalence_tests",
    }
    for name in python_only:
        cls = PIPELINES[name]
        supported = getattr(cls, "supported_languages", None)
        assert supported is not None, f"{name}: expected supported_languages={{PYTHON}}"
        assert supported == frozenset({LanguageHint.PYTHON}), (
            f"{name}: expected frozenset({{PYTHON}}), got {supported}"
        )


def test_language_agnostic_pipelines_have_no_restriction():
    # [FUNCTION DOC] Asserts pr_diff, pr_runtime, commit_runtime and cve_patches have
    # supported_languages None (absent or None).
    """Lite + language-agnostic full pipelines must NOT restrict languages."""
    agnostic = {"pr_diff", "pr_runtime", "commit_runtime", "cve_patches"}
    for name in agnostic:
        cls = PIPELINES[name]
        supported = getattr(cls, "supported_languages", None)
        assert supported is None, (
            f"{name}: expected supported_languages=None (any language ok), got {supported}"
        )


def test_pipeline_result_shape():
    # [FUNCTION DOC] Constructs a PipelineResult and asserts fields candidates, emitted, skipped,
    # out_dir and skip_reasons exist.
    """PipelineResult has the documented fields."""
    r = PipelineResult(
        candidates=0,
        emitted=0,
        skipped=0,
        out_dir=__import__("pathlib").Path("."),
        skip_reasons={},
    )
    # Slots dataclass — these field names must exist
    for attr in ("candidates", "emitted", "skipped", "out_dir", "skip_reasons"):
        assert hasattr(r, attr), f"PipelineResult missing field {attr}"


def test_protocol_export():
    # [FUNCTION DOC] Asserts the repo2rlenv.pipelines package exports Pipeline and PipelineResult.
    """Pipeline + PipelineResult are public exports of the package."""
    from repo2rlenv import pipelines

    assert hasattr(pipelines, "Pipeline")
    assert hasattr(pipelines, "PipelineResult")
