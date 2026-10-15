# [MODULE DOC] Tests for log_parsers.parse_pytest, which maps pytest output (short summary lines
# and -v progress lines) to {test_id: PASSED|FAILED|SKIPPED}; used by the validation harness to
# compute F2P/P2P sets.
"""Pytest output parser — what the validation harness depends on."""

from __future__ import annotations

from repo2rlenv.log_parsers import parse_pytest

_PYTEST_SAMPLE = """\
============================= test session starts ==============================
platform darwin -- Python 3.12.1, pytest-8.3.0
collected 4 items

tests/test_a.py::test_one PASSED                                         [ 25%]
tests/test_a.py::test_two FAILED                                         [ 50%]
tests/test_b.py::test_three SKIPPED                                      [ 75%]
tests/test_b.py::test_four PASSED                                        [100%]

=================================== FAILURES ===================================
______________________________ test_two ______________________________
    >       assert add(2, 3) == 6
E       AssertionError
=========================== short test summary info ============================
PASSED tests/test_a.py::test_one
FAILED tests/test_a.py::test_two - AssertionError: assert 5 == 6
SKIPPED tests/test_b.py::test_three
PASSED tests/test_b.py::test_four
=================== 2 passed, 1 failed, 1 skipped in 0.04s ====================
"""


def test_parses_canonical_pytest_summary():
    # [FUNCTION DOC] Asserts every test in a full -v run with summary is mapped to its status.
    status = parse_pytest(_PYTEST_SAMPLE)
    assert status["tests/test_a.py::test_one"] == "PASSED"
    assert status["tests/test_a.py::test_two"] == "FAILED"
    assert status["tests/test_b.py::test_three"] == "SKIPPED"
    assert status["tests/test_b.py::test_four"] == "PASSED"


def test_failed_line_strips_message_after_dash():
    # [FUNCTION DOC] Asserts the ' - message' suffix on a FAILED summary line is dropped from the
    # test id.
    """`FAILED tests/foo.py::test_x - AssertionError: ...` should yield just the test name."""
    status = parse_pytest("FAILED tests/foo.py::test_x - AssertionError: nope\n")
    assert "tests/foo.py::test_x" in status
    assert status["tests/foo.py::test_x"] == "FAILED"


def test_empty_log_returns_empty_dict():
    # [FUNCTION DOC] Asserts empty string and None both yield {}.
    assert parse_pytest("") == {}
    assert parse_pytest(None) == {}  # type: ignore[arg-type]


def test_parametrized_test_names_preserved():
    # [FUNCTION DOC] Asserts bracketed parametrize ids stay part of the test id.
    """pytest -v parametrized tests use bracket notation that mustn't be split."""
    log = "PASSED tests/test_p.py::test_x[case-1]\n"
    status = parse_pytest(log)
    assert status == {"tests/test_p.py::test_x[case-1]": "PASSED"}


def test_skipped_with_count_prefix_skips_bracket_token():
    # [FUNCTION DOC] Asserts 'SKIPPED [1] path:line' records the path:line token as SKIPPED.
    """Lines like 'SKIPPED [1] tests/foo.py:42' should still record the file token."""
    log = "SKIPPED [1] tests/foo.py:42\n"
    status = parse_pytest(log)
    # The third token is the actual file location
    assert "tests/foo.py:42" in status
    assert status["tests/foo.py:42"] == "SKIPPED"


def test_last_write_wins_for_same_test():
    # [FUNCTION DOC] Asserts a later status line for the same id overwrites the earlier one.
    log = "SKIPPED tests/foo.py::test_x\nPASSED tests/foo.py::test_x\n"
    status = parse_pytest(log)
    assert status["tests/foo.py::test_x"] == "PASSED"


def test_parses_verbose_progress_lines():
    # [FUNCTION DOC] Asserts '<id> STATUS [N%]' progress lines are parsed.
    """pytest -v emits `test_name STATUS [N%]` — must be handled (P1 fix)."""
    log = """\
tests/test_a.py::test_one PASSED                                         [ 25%]
tests/test_a.py::test_two FAILED                                         [ 50%]
tests/test_b.py::test_three SKIPPED                                      [ 75%]
"""
    status = parse_pytest(log)
    assert status == {
        "tests/test_a.py::test_one": "PASSED",
        "tests/test_a.py::test_two": "FAILED",
        "tests/test_b.py::test_three": "SKIPPED",
    }


def test_ignores_random_lines_with_PASSED_in_them():
    # [FUNCTION DOC] Asserts lines containing PASSED where the name does not look like a test id
    # are ignored.
    """Lines like 'Some thing PASSED something' shouldn't match — name must look like a test id."""
    log = """\
INFO     setup PASSED initialization
some random output
checking PASSED status
"""
    status = parse_pytest(log)
    assert status == {}


def test_handles_mixed_verbose_and_summary():
    # [FUNCTION DOC] Asserts progress lines followed by a summary section parse consistently.
    """Real pytest -v output has both formats — progress THEN summary."""
    log = """\
tests/test_a.py::test_one PASSED                                         [ 50%]
tests/test_a.py::test_two FAILED                                         [100%]

=================================== FAILURES ===================================
______________________________ test_two ______________________________
    assert 1 == 2
=========================== short test summary info ============================
FAILED tests/test_a.py::test_two - AssertionError
"""
    status = parse_pytest(log)
    assert status["tests/test_a.py::test_one"] == "PASSED"
    assert status["tests/test_a.py::test_two"] == "FAILED"
