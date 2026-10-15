# [MODULE DOC] Tests for code_instruct quality gates in pipelines._oss_instruct: repo package
# detection, top-level symbol listing, repo-anchoring (solution must import and use the repo
# package), symbol-collision, test-strength heuristics, and task fingerprinting for dedup.
"""Quality gates on LLM-synthesized code_instruct candidates."""

from __future__ import annotations

from pathlib import Path

from repo2rlenv.pipelines._oss_instruct import (
    check_repo_anchoring,
    check_symbol_collision,
    check_test_strength,
    detect_repo_package,
    list_repo_top_level_symbols,
    task_fingerprint,
    task_fingerprints,
)

# ---------------------------------------------------------------------------
# detect_repo_package
# ---------------------------------------------------------------------------


def test_detect_repo_package_from_src_layout(tmp_path: Path):
    # [FUNCTION DOC] Asserts a src/<name>/__init__.py package is detected.
    (tmp_path / "src" / "foo").mkdir(parents=True)
    (tmp_path / "src" / "foo" / "__init__.py").write_text("")
    assert detect_repo_package(tmp_path, "foo") == "foo"


def test_detect_repo_package_from_flat_layout(tmp_path: Path):
    # [FUNCTION DOC] Asserts a top-level <name>/__init__.py package is detected.
    (tmp_path / "bar").mkdir()
    (tmp_path / "bar" / "__init__.py").write_text("")
    assert detect_repo_package(tmp_path, "bar") == "bar"


def test_detect_repo_package_dash_to_underscore(tmp_path: Path):
    # [FUNCTION DOC] Asserts dashes in the repo name map to underscores.
    (tmp_path / "flask_restful").mkdir()
    (tmp_path / "flask_restful" / "__init__.py").write_text("")
    assert detect_repo_package(tmp_path, "flask-restful") == "flask_restful"


def test_detect_repo_package_pyproject(tmp_path: Path):
    # [FUNCTION DOC] Asserts the [project] name from pyproject.toml takes precedence over the repo
    # name.
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "coolpkg"\n')
    (tmp_path / "src" / "coolpkg").mkdir(parents=True)
    (tmp_path / "src" / "coolpkg" / "__init__.py").write_text("")
    assert detect_repo_package(tmp_path, "some-other-name") == "coolpkg"


def test_detect_repo_package_attrs_special_case(tmp_path: Path):
    # [FUNCTION DOC] Asserts the 'attrs' repo resolves to the 'attr' package.
    (tmp_path / "src" / "attr").mkdir(parents=True)
    (tmp_path / "src" / "attr" / "__init__.py").write_text("")
    assert detect_repo_package(tmp_path, "attrs") == "attr"


def test_detect_repo_package_missing_returns_none(tmp_path: Path):
    # [FUNCTION DOC] Asserts None when no package is found.
    assert detect_repo_package(tmp_path, "nonexistent") is None


# ---------------------------------------------------------------------------
# list_repo_top_level_symbols
# ---------------------------------------------------------------------------


def test_list_repo_top_level_symbols_collects(tmp_path: Path):
    # [FUNCTION DOC] Asserts public top-level classes/functions across package modules are collected
    # and underscore names excluded.
    pkg = tmp_path / "foo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("class Public: ...\ndef helper(): ...\n")
    (pkg / "sub.py").write_text("class Nested: ...\n_priv = 1\n")
    names = list_repo_top_level_symbols(tmp_path, "foo")
    assert "Public" in names
    assert "helper" in names
    assert "Nested" in names
    assert "_priv" not in names


def test_list_repo_top_level_symbols_missing_pkg(tmp_path: Path):
    # [FUNCTION DOC] Asserts an empty set for a missing package.
    assert list_repo_top_level_symbols(tmp_path, "nope") == set()


def test_list_repo_top_level_symbols_swallows_corrupt(tmp_path: Path):
    # [FUNCTION DOC] Asserts unparsable files are skipped without failing.
    pkg = tmp_path / "foo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("class OK: ...\n")
    (pkg / "broken.py").write_text("def x( ; syntax error\n")
    names = list_repo_top_level_symbols(tmp_path, "foo")
    assert "OK" in names  # corrupt file is skipped, not fatal


# ---------------------------------------------------------------------------
# check_repo_anchoring
# ---------------------------------------------------------------------------


def test_check_repo_anchoring_accepts_used_import():
    # [FUNCTION DOC] Asserts a used 'from pkg import X' passes.
    code = "from click import ParamType\n\nclass MyParam(ParamType):\n    pass\n"
    ok, reason = check_repo_anchoring(code, "click")
    assert ok, reason


def test_check_repo_anchoring_accepts_submodule_import():
    # [FUNCTION DOC] Asserts a used submodule import passes.
    code = "from click.core import Context\n\ndef helper(ctx: Context) -> None:\n    pass\n"
    ok, reason = check_repo_anchoring(code, "click")
    assert ok, reason


def test_check_repo_anchoring_accepts_bare_import():
    # [FUNCTION DOC] Asserts a used bare 'import pkg' passes.
    code = "import click\n\nclass X:\n    ctx = click.get_current_context\n"
    ok, reason = check_repo_anchoring(code, "click")
    assert ok, reason


def test_check_repo_anchoring_rejects_missing_import():
    # [FUNCTION DOC] Asserts code without a repo import fails with 'no_repo_import'.
    code = "class MyClass:\n    def foo(self):\n        return 42\n"
    ok, reason = check_repo_anchoring(code, "click")
    assert not ok
    assert reason == "no_repo_import"


def test_check_repo_anchoring_rejects_unused_import():
    # [FUNCTION DOC] Asserts an imported but unused repo name fails with 'repo_import_unused'.
    code = "from click import ParamType\n\nclass MyClass:\n    def foo(self):\n        return 42\n"
    ok, reason = check_repo_anchoring(code, "click")
    assert not ok
    assert reason == "repo_import_unused"


def test_check_repo_anchoring_rejects_wrong_package():
    # [FUNCTION DOC] Asserts importing a different package fails with 'no_repo_import'.
    code = "from flask import Flask\n\nclass MyClass:\n    app = Flask\n"
    ok, reason = check_repo_anchoring(code, "click")
    assert not ok
    assert reason == "no_repo_import"


def test_check_repo_anchoring_reports_syntax_error():
    # [FUNCTION DOC] Asserts unparsable code fails with a 'solution_syntax_error:' reason.
    code = "def broken( :\n"
    ok, reason = check_repo_anchoring(code, "click")
    assert not ok
    assert reason.startswith("solution_syntax_error:")


# ---------------------------------------------------------------------------
# check_symbol_collision
# ---------------------------------------------------------------------------


def test_check_symbol_collision_accepts_novel():
    # [FUNCTION DOC] Asserts definitions not in the repo symbol set pass.
    code = "class TaskThing:\n    pass\n\ndef helper_fn():\n    pass\n"
    ok, _ = check_symbol_collision(code, {"ParamType", "get_app_dir"})
    assert ok


def test_check_symbol_collision_rejects_class():
    # [FUNCTION DOC] Asserts a top-level class shadowing a repo symbol fails with
    # 'symbol_collides_with_repo:<name>'.
    code = "class ParamType:\n    pass\n"
    ok, reason = check_symbol_collision(code, {"ParamType"})
    assert not ok
    assert reason == "symbol_collides_with_repo:ParamType"


def test_check_symbol_collision_rejects_function():
    # [FUNCTION DOC] Asserts a top-level function shadowing a repo symbol fails naming it.
    code = "def get_app_dir(name):\n    return name\n"
    ok, reason = check_symbol_collision(code, {"get_app_dir"})
    assert not ok
    assert "get_app_dir" in reason


def test_check_symbol_collision_ignores_nested():
    # [FUNCTION DOC] Asserts nested definitions do not count as collisions.
    code = "class Outer:\n    class ParamType:\n        pass\n"
    ok, _ = check_symbol_collision(code, {"ParamType"})
    assert ok  # only top-level definitions collide


# ---------------------------------------------------------------------------
# check_test_strength
# ---------------------------------------------------------------------------


def test_check_test_strength_accepts_rigorous():
    # [FUNCTION DOC] Asserts several tests with enough asserts plus pytest.raises for an error-
    # raising instruction pass.
    code = """\
from task_module import compute
import pytest

def test_positive():
    assert compute(2, 3) == 5
    assert compute(0, 0) == 0

def test_negative():
    assert compute(-1, 1) == 0

def test_raises():
    with pytest.raises(ValueError):
        compute(None, 1)
"""
    ok, reason = check_test_strength(code, "Function must raise ValueError on None input.")
    assert ok, reason


def test_check_test_strength_rejects_trivial_assert():
    # [FUNCTION DOC] Asserts 'assert True' fails with 'trivial_assert_present'.
    code = """\
from task_module import compute

def test_a():
    assert True
    assert compute(1) == 1

def test_b():
    assert compute(2) == 2
    assert compute(3) == 3
"""
    ok, reason = check_test_strength(code, "Compute stuff.")
    assert not ok
    assert reason == "trivial_assert_present"


def test_check_test_strength_rejects_too_few_asserts():
    # [FUNCTION DOC] Asserts a single assert fails with 'too_few_asserts'.
    code = "from task_module import x\n\ndef test_a():\n    assert x() == 1\n"
    ok, reason = check_test_strength(code, "Return 1.")
    assert not ok
    assert reason == "too_few_asserts"


def test_check_test_strength_rejects_missing_raises_for_error_instruction():
    # [FUNCTION DOC] Asserts an instruction mentioning raising without pytest.raises in tests fails
    # with 'missing_pytest_raises'.
    code = """\
from task_module import compute

def test_a():
    assert compute(1) == 1
    assert compute(2) == 2
    assert compute(3) == 3
"""
    ok, reason = check_test_strength(code, "Function must raise ValueError on invalid input.")
    assert not ok
    assert reason == "missing_pytest_raises"


def test_check_test_strength_syntax_error():
    # [FUNCTION DOC] Asserts unparsable tests fail with 'test_syntax_error:'.
    ok, reason = check_test_strength("def test_x( ; broken\n", "hi")
    assert not ok
    assert reason.startswith("test_syntax_error:")


def test_check_test_strength_no_test_functions():
    # [FUNCTION DOC] Asserts code with no test_ functions fails with 'no_test_functions'.
    ok, reason = check_test_strength(
        "from task_module import x\n\nresult = x()\nassert result == 1\n", "hi"
    )
    assert not ok
    assert reason == "no_test_functions"


# ---------------------------------------------------------------------------
# task_fingerprint
# ---------------------------------------------------------------------------


def test_task_fingerprint_stable_across_whitespace():
    # [FUNCTION DOC] Asserts whitespace normalization makes fingerprints equal.
    a = "Implement a function `frobnicate` that returns 42."
    b = "  Implement    a   function `frobnicate` that returns 42.   "
    assert task_fingerprint(a) == task_fingerprint(b)


def test_task_fingerprint_case_insensitive():
    # [FUNCTION DOC] Asserts fingerprints ignore case.
    a = "IMPLEMENT A FUNCTION"
    b = "implement a function"
    assert task_fingerprint(a) == task_fingerprint(b)


def test_task_fingerprint_differs_on_content():
    # [FUNCTION DOC] Asserts different content gives different fingerprints.
    a = "Implement a function called frobnicate."
    b = "Implement a class called Widget."
    assert task_fingerprint(a) != task_fingerprint(b)


def test_task_fingerprints_catches_reworded_but_same_class():
    # Two reworded problem statements that both produce the SAME class.
    # Iter1 audit surfaced this exact failure mode (RangedFloatType ×2).
    # [FUNCTION DOC] Asserts reworded problems with the same solution class share at least one
    # fingerprint (symbol-set signal).
    p1 = "Implement a click ParamType `RangedFloatType` accepting a float in [min, max]. Wire it into a click.Command with --value."
    p2 = "Design a click.Option accepting a float in [lo, hi] using a custom `RangedFloatType`."
    sol = "from click import ParamType\n\nclass RangedFloatType(ParamType):\n    def convert(self, value, param, ctx):\n        return float(value)\n"
    fps1 = task_fingerprints(p1, sol)
    fps2 = task_fingerprints(p2, sol)
    assert fps1 & fps2  # overlap on the symbol-set signal


def test_task_fingerprints_ignores_private_helpers():
    # [FUNCTION DOC] Asserts differing private helper names do not change the fingerprint set.
    p = "Same problem statement here."
    sol_a = "class X:\n    pass\n\ndef _helper_a():\n    pass\n"
    sol_b = "class X:\n    pass\n\ndef _helper_b():\n    pass\n"
    # Private helpers ignored → symbol signals match; problem signals also match
    assert task_fingerprints(p, sol_a) == task_fingerprints(p, sol_b)


def test_task_fingerprints_distinct_when_all_signals_differ():
    # [FUNCTION DOC] Asserts fully different problems and solutions share no fingerprints.
    fps1 = task_fingerprints("Do foo.", "class Foo:\n    pass\n")
    fps2 = task_fingerprints("Do bar.", "class Bar:\n    pass\n")
    assert not (fps1 & fps2)
