# [MODULE DOC] Pure-logic tests for pipeline/repo language compatibility:
# pipelines.base.check_language_compatibility (raises LanguageMismatchError unless the pipeline's
# supported_languages admits the detected LanguageHint, or force=True which only logs a warning)
# and bootstrap.language.language_from_github_name (GitHub Linguist name to LanguageHint). No
# network, Docker or LLM.
"""Pipeline ↔ repo-language compatibility: helper + GitHub-name mapper.

Pure-logic tests; no network, no Docker, no LLM.
"""

from __future__ import annotations

import logging

import pytest

from repo2rlenv.bootstrap.language import language_from_github_name
from repo2rlenv.bootstrap.spec import LanguageHint
from repo2rlenv.pipelines.base import (
    LanguageMismatchError,
    check_language_compatibility,
)

# ----------------------------------------------------------------------------
# check_language_compatibility — the core dispatcher helper
# ----------------------------------------------------------------------------


class _LangAgnostic:
    # [CLASS DOC] HELPER: Fake pipeline class with supported_languages None, meaning it accepts
    # every language.
    name = "pr_runtime"
    supported_languages = None


class _PythonOnly:
    # [CLASS DOC] HELPER: Fake pipeline class named 'equivalence_tests' restricted to
    # LanguageHint.PYTHON.
    name = "equivalence_tests"
    supported_languages = frozenset({LanguageHint.PYTHON})


class _DualLang:
    # [CLASS DOC] HELPER: Fake pipeline class supporting PYTHON and NODE, to exercise multi-
    # language sets.
    """Hypothetical pipeline supporting both Python and Node — exercise multi-lang."""

    name = "hypothetical_dual"
    supported_languages = frozenset({LanguageHint.PYTHON, LanguageHint.NODE})


def test_agnostic_pipeline_accepts_any_language():
    # No exception for any LanguageHint when supported_languages is None
    # [FUNCTION DOC] Asserts no exception for every LanguageHint member when supported_languages is
    # None.
    for lang in LanguageHint:
        check_language_compatibility(_LangAgnostic, lang)  # must not raise


def test_python_only_accepts_python():
    # [FUNCTION DOC] Asserts a Python-only pipeline accepts PYTHON without raising.
    check_language_compatibility(_PythonOnly, LanguageHint.PYTHON)  # must not raise


def test_python_only_rejects_go_without_force():
    # [FUNCTION DOC] Asserts GO raises LanguageMismatchError whose message names the pipeline, both
    # languages and the --force-language escape hatch.
    with pytest.raises(LanguageMismatchError) as exc:
        check_language_compatibility(_PythonOnly, LanguageHint.GO)
    msg = str(exc.value)
    assert "equivalence_tests" in msg
    assert "python" in msg.lower()
    assert "go" in msg.lower()
    assert "--force-language" in msg


def test_python_only_rejects_rust_without_force():
    # [FUNCTION DOC] Asserts RUST raises LanguageMismatchError for a Python-only pipeline.
    with pytest.raises(LanguageMismatchError):
        check_language_compatibility(_PythonOnly, LanguageHint.RUST)


def test_python_only_rejects_node_without_force():
    # [FUNCTION DOC] Asserts NODE raises LanguageMismatchError for a Python-only pipeline.
    with pytest.raises(LanguageMismatchError):
        check_language_compatibility(_PythonOnly, LanguageHint.NODE)


def test_python_only_with_force_warns_then_returns(caplog):
    # [FUNCTION DOC] Asserts force=True bypasses the mismatch but logs a WARNING record containing
    # 'requires' (captured via caplog).
    caplog.set_level(logging.WARNING)
    # Force overrides the strict check
    check_language_compatibility(_PythonOnly, LanguageHint.GO, force=True)
    assert any("requires" in rec.message.lower() for rec in caplog.records), (
        "force=True must emit a warning so the user knows the check was bypassed"
    )


def test_dual_lang_accepts_either():
    # [FUNCTION DOC] Asserts the dual-language pipeline accepts PYTHON and NODE but rejects RUST.
    check_language_compatibility(_DualLang, LanguageHint.PYTHON)
    check_language_compatibility(_DualLang, LanguageHint.NODE)
    with pytest.raises(LanguageMismatchError):
        check_language_compatibility(_DualLang, LanguageHint.RUST)


def test_unknown_language_rejected_by_restricted_pipeline():
    # [FUNCTION DOC] Asserts UNKNOWN is treated as a mismatch for a restricted pipeline (raises
    # LanguageMismatchError).
    """If GitHub returns no primary language, we shouldn't silently let a
    Python-only pipeline through. UNKNOWN is treated as "doesn't match"."""
    with pytest.raises(LanguageMismatchError):
        check_language_compatibility(_PythonOnly, LanguageHint.UNKNOWN)


def test_unknown_language_with_force_allowed():
    # [FUNCTION DOC] Asserts force=True lets UNKNOWN through without raising.
    """Force overrides everything — including unknown."""
    check_language_compatibility(_PythonOnly, LanguageHint.UNKNOWN, force=True)


# ----------------------------------------------------------------------------
# language_from_github_name — GitHub Linguist name → LanguageHint
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "github_name,expected",
    [
        ("Python", LanguageHint.PYTHON),
        ("python", LanguageHint.PYTHON),
        ("PYTHON", LanguageHint.PYTHON),
        ("JavaScript", LanguageHint.NODE),
        ("TypeScript", LanguageHint.NODE),
        ("Go", LanguageHint.GO),
        ("Rust", LanguageHint.RUST),
        ("Java", LanguageHint.JAVA),
        ("Kotlin", LanguageHint.JAVA),
        ("Scala", LanguageHint.JAVA),
        ("C", LanguageHint.C_CPP),
        ("C++", LanguageHint.C_CPP),
        ("Objective-C", LanguageHint.C_CPP),
        ("Haskell", LanguageHint.UNKNOWN),  # not in the map
        ("", LanguageHint.UNKNOWN),
        (None, LanguageHint.UNKNOWN),
    ],
)
def test_github_name_to_language_hint(github_name, expected):
    # [FUNCTION DOC] Parametrized: asserts language_from_github_name maps case-insensitive Linguist
    # names (JS/TS to NODE, Kotlin/Scala to JAVA, C/C++/Objective-C to C_CPP) and unmapped, empty
    # or None names to UNKNOWN.
    assert language_from_github_name(github_name) == expected


def test_github_name_handles_whitespace():
    # [FUNCTION DOC] Asserts surrounding whitespace is stripped before mapping ('  Python  ' gives
    # PYTHON).
    assert language_from_github_name("  Python  ") == LanguageHint.PYTHON
