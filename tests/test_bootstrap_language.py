# [MODULE DOC] Tests for bootstrap.language: detect_language from marker files (pyproject.toml,
# package.json, go.mod, Cargo.toml, with Rust beating a Makefile, UNKNOWN for empty dirs) and
# base_image_for's per-language Docker base images. Uses tmp_path.
"""Language detection from repo files."""

from __future__ import annotations

from pathlib import Path

from repo2rlenv.bootstrap.language import base_image_for, detect_language
from repo2rlenv.bootstrap.spec import LanguageHint


def test_detects_python(tmp_path: Path):
    # [FUNCTION DOC] pyproject.toml -> PYTHON.
    (tmp_path / "pyproject.toml").write_text("")
    assert detect_language(tmp_path) == LanguageHint.PYTHON


def test_detects_node(tmp_path: Path):
    # [FUNCTION DOC] package.json -> NODE.
    (tmp_path / "package.json").write_text("{}")
    assert detect_language(tmp_path) == LanguageHint.NODE


def test_detects_go(tmp_path: Path):
    # [FUNCTION DOC] go.mod -> GO.
    (tmp_path / "go.mod").write_text("module x")
    assert detect_language(tmp_path) == LanguageHint.GO


def test_detects_rust(tmp_path: Path):
    # [FUNCTION DOC] Cargo.toml -> RUST.
    (tmp_path / "Cargo.toml").write_text("[package]")
    assert detect_language(tmp_path) == LanguageHint.RUST


def test_priority_more_specific_wins(tmp_path: Path):
    # [FUNCTION DOC] Cargo.toml plus Makefile resolves to RUST, not C/C++.
    """A repo with both Cargo.toml and Makefile should resolve to Rust, not C/C++."""
    (tmp_path / "Cargo.toml").write_text("")
    (tmp_path / "Makefile").write_text("")
    assert detect_language(tmp_path) == LanguageHint.RUST


def test_unknown_for_empty_dir(tmp_path: Path):
    # [FUNCTION DOC] Empty directory -> UNKNOWN.
    assert detect_language(tmp_path) == LanguageHint.UNKNOWN


def test_base_image_per_language():
    # [FUNCTION DOC] Base images start with python:, node:, golang:, rust: and ubuntu: (UNKNOWN)
    # respectively.
    assert base_image_for(LanguageHint.PYTHON).startswith("python:")
    assert base_image_for(LanguageHint.NODE).startswith("node:")
    assert base_image_for(LanguageHint.GO).startswith("golang:")
    assert base_image_for(LanguageHint.RUST).startswith("rust:")
    assert base_image_for(LanguageHint.UNKNOWN).startswith("ubuntu:")
