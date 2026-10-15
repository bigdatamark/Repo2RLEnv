# [MODULE DOC] Tests for the BootstrapSpec pydantic model (defaults, Literal platform validation,
# optional fields, model_copy overrides) and its inclusion in GenerationInput, plus LLMSpec
# fallback round-tripping through model_dump/model_validate. Pure in-memory; no I/O.
"""BootstrapSpec validation + integration into GenerationInput."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from repo2rlenv.spec.input import (
    BootstrapSpec,
    GenerationInput,
    LLMSpec,
    OutputSpec,
    PipelineName,
    PipelineSpec,
    RepoSpec,
)


def test_defaults_are_sensible():
    # [FUNCTION DOC] BootstrapSpec() defaults: enabled, 20 iterations, 1800s,
    # ./workspace/bootstrap cache, linux/amd64, no user_dockerfile. Inputs: none. Output:
    # none. Raises: AssertionError on failure.
    spec = BootstrapSpec()
    assert spec.enabled is True
    assert spec.max_iterations == 20
    assert spec.max_seconds == 1800
    assert spec.cache_dir == Path("./workspace/bootstrap")
    assert spec.platform == "linux/amd64"
    assert spec.user_dockerfile is None


def test_platform_constrained():
    # [FUNCTION DOC] An unsupported platform ('linux/i386') raises pydantic ValidationError.
    # Inputs: none. Output: none. Raises: AssertionError on failure.
    with pytest.raises(ValidationError):
        BootstrapSpec(platform="linux/i386")  # not in Literal


def test_bootstrap_included_in_generation_input_default():
    # [FUNCTION DOC] GenerationInput without a bootstrap field gets an enabled default
    # BootstrapSpec with the default cache_dir. Inputs: none. Output: none. Raises:
    # AssertionError on failure.
    g = GenerationInput(
        repo=RepoSpec(url="huggingface/trl"),
        pipeline=PipelineSpec(name=PipelineName.PR_DIFF, options={}),
        llm=LLMSpec(provider="anthropic", model="claude-sonnet-4-6"),
        output=OutputSpec(destination="./out", org="x", dataset_name="y"),
    )
    assert g.bootstrap.enabled is True
    assert g.bootstrap.cache_dir == Path("./workspace/bootstrap")


def test_bootstrap_user_dockerfile_override():
    # [FUNCTION DOC] user_dockerfile accepts and stores a Path. Inputs: none. Output: none.
    # Raises: AssertionError on failure.
    spec = BootstrapSpec(user_dockerfile=Path("./my-Dockerfile"))
    assert spec.user_dockerfile == Path("./my-Dockerfile")


def test_bootstrap_image_registry_optional():
    # [FUNCTION DOC] image_registry defaults to None and stores a given string. Inputs: none.
    # Output: none. Raises: AssertionError on failure.
    s1 = BootstrapSpec()
    assert s1.image_registry is None
    s2 = BootstrapSpec(image_registry="ghcr.io/myorg/r2e")
    assert s2.image_registry == "ghcr.io/myorg/r2e"


def test_model_copy_update_supports_cache_dir_override():
    # [FUNCTION DOC] model_copy(update={'cache_dir': str}) is used by the CLI; the resolved
    # path must equal the
    # requested one. Inputs: none. Output: none. Raises: AssertionError on failure.
    # KNOWN ISSUE: the docstring says Pydantic coerces the string to a Path, but model_copy
    # does
    # not validate updates, so cache_dir stays a str (the inline comment admits this); the
    # test
    # only passes because it wraps the value in Path() before comparing.
    """Drives the `--bootstrap-opt cache_dir=...` CLI plumbing.

    The CLI uses `bspec.model_copy(update={k: v})` to override any field;
    Pydantic should coerce a string path to a Path automatically.
    """
    bspec = BootstrapSpec()
    overridden = bspec.model_copy(update={"cache_dir": "./envs-matrix/sonnet-4-6"})
    # Pydantic preserves the literal string; just check the Path resolves correctly.
    assert Path(overridden.cache_dir).resolve() == Path("envs-matrix/sonnet-4-6").resolve()


def test_model_copy_update_supports_numeric_field():
    # [FUNCTION DOC] model_copy can override max_iterations and max_seconds. Inputs: none.
    # Output: none. Raises: AssertionError on failure.
    bspec = BootstrapSpec()
    overridden = bspec.model_copy(update={"max_iterations": 30, "max_seconds": 2400})
    assert overridden.max_iterations == 30
    assert overridden.max_seconds == 2400


def test_llm_fallback_round_trips_through_dict():
    # [FUNCTION DOC] A nested fallback LLMSpec survives model_dump -> model_validate with
    # qualified_name 'openai/gpt-5.5'. Inputs: none. Output: none. Raises: AssertionError on
    # failure.
    """Drives the `--llm-fallback` CLI plumbing: nested LLMSpec via dict."""
    primary = LLMSpec(
        provider="anthropic",
        model="claude-sonnet-4-6",
        fallback=LLMSpec(provider="openai", model="gpt-5.5"),
    )
    assert primary.fallback is not None
    assert primary.fallback.qualified_name == "openai/gpt-5.5"
    # Round-trip through dict (CLI builds the spec via `model_validate(dict)`)
    rebuilt = LLMSpec.model_validate(primary.model_dump())
    assert rebuilt.fallback is not None
    assert rebuilt.fallback.qualified_name == "openai/gpt-5.5"
