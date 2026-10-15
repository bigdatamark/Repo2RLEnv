# [MODULE DOC] Unit tests for repo2rlenv.pipelines._pr_runtime_verifier, the stdlib-only in-
# container graded F2P/P2P verifier. Covers the per-runner log parsers (pytest, go test, cargo test,
# jest) and the parse_logs dispatcher, grade() scoring (reward = f2p_rate * p2p_rate, strict
# `resolved`, P2P regression list, untracked failures, Go subtest collapsing), and main() CLI I/O:
# it writes reward.txt (6-decimal float) and reward-details.json into --out-dir, and falls back to
# the exit code when the log cannot be parsed. Tests use pytest tmp_path only; no network or Docker.
"""Unit tests for the in-container graded F2P/P2P verifier.

Covers the 4 per-runner parsers (kept in lockstep with
repo2rlenv.log_parsers.*), the graded scoring + strict `resolved` bool,
P2P-regression penalty, the oracle=1.0 invariant, and the
exit-code fallback on unparseable logs.
"""

from __future__ import annotations

import json
from pathlib import Path

from repo2rlenv.pipelines._pr_runtime_verifier import (
    grade,
    main,
    parse_cargo_test,
    parse_go_test,
    parse_jest,
    parse_logs,
    parse_pytest,
)

# --- parsers -----------------------------------------------------------------


def test_parse_pytest_verbose_and_summary():
    # [FUNCTION DOC] Feeds a verbose pytest log (PASSED/FAILED lines plus a 'FAILED id - msg' summary) to
    # parse_pytest and asserts each node id maps to PASSED/FAILED. Inputs: none. Output: none;
    # asserts.
    log = (
        "tests/test_a.py::test_x PASSED  [ 50%]\n"
        "tests/test_a.py::test_y FAILED  [100%]\n"
        "FAILED tests/test_a.py::test_y - AssertionError: nope\n"
    )
    out = parse_pytest(log)
    assert out["tests/test_a.py::test_x"] == "PASSED"
    assert out["tests/test_a.py::test_y"] == "FAILED"


def test_parse_pytest_error_status():
    # [FUNCTION DOC] Asserts parse_pytest maps an 'ERROR <nodeid>' summary line to status ERROR.
    out = parse_pytest("ERROR tests/test_a.py::test_setup\n")
    assert out["tests/test_a.py::test_setup"] == "ERROR"


def test_parse_go_test():
    # [FUNCTION DOC] Asserts parse_go_test turns '--- PASS:'/'--- FAIL:' lines into {name: PASSED/FAILED}, ignoring
    # '=== RUN' lines.
    log = "=== RUN   TestA\n--- PASS: TestA (0.00s)\n--- FAIL: TestB (0.01s)\n"
    out = parse_go_test(log)
    assert out == {"TestA": "PASSED", "TestB": "FAILED"}


def test_parse_cargo_test():
    # [FUNCTION DOC] Asserts parse_cargo_test maps cargo 'test X ... ok/FAILED/ignored' lines to
    # PASSED/FAILED/SKIPPED.
    log = "test tests::a ... ok\ntest tests::b ... FAILED\ntest tests::c ... ignored\n"
    out = parse_cargo_test(log)
    assert out == {"tests::a": "PASSED", "tests::b": "FAILED", "tests::c": "SKIPPED"}


def test_parse_jest_qualified_names():
    # [FUNCTION DOC] Asserts parse_jest qualifies test names as 'file > describe > title' and maps check/cross
    # marks to PASSED/FAILED.
    log = "PASS  src/foo.test.ts\n  Foo\n    ✓ returns 200 (4 ms)\n    ✕ returns 500 (1 ms)\n"
    out = parse_jest(log)
    assert out["src/foo.test.ts > Foo > returns 200"] == "PASSED"
    assert out["src/foo.test.ts > Foo > returns 500"] == "FAILED"


def test_parse_logs_dispatch_by_runner():
    # [FUNCTION DOC] Asserts parse_logs dispatches by runner name ('go' uses the Go parser) and returns an empty
    # dict for an unknown runner.
    assert parse_logs("go", "--- PASS: T (0s)\n") == {"T": "PASSED"}
    assert parse_logs("unknown", "anything") == {}


# --- grading -----------------------------------------------------------------


def test_grade_oracle_full_resolution_is_one():
    # [FUNCTION DOC] Invariant check: all F2P and P2P tests passing yields reward 1.0 and resolved True.
    """The invariant: all F2P pass + all P2P pass -> reward 1.0, resolved."""
    status = {"t_fix": "PASSED", "t_keep": "PASSED"}
    r = grade(["t_fix"], ["t_keep"], status)
    assert r["reward"] == 1.0
    assert r["resolved"] is True


def test_grade_partial_f2p_is_graded():
    # [FUNCTION DOC] Asserts half the F2P passing with all P2P passing gives f2p_rate 0.5, reward 0.5, resolved
    # False.
    status = {"f1": "PASSED", "f2": "FAILED", "keep": "PASSED"}
    r = grade(["f1", "f2"], ["keep"], status)
    assert r["f2p_rate"] == 0.5
    assert r["reward"] == 0.5  # p2p_rate == 1.0
    assert r["resolved"] is False


def test_grade_p2p_regression_penalizes():
    # [FUNCTION DOC] Asserts a failing P2P test halves the reward (p2p_rate 0.5), unsets resolved, and is listed in
    # 'regressions'.
    """Breaking a previously-passing test scales the reward down."""
    status = {"f1": "PASSED", "keep1": "PASSED", "keep2": "FAILED"}
    r = grade(["f1"], ["keep1", "keep2"], status)
    assert r["f2p_rate"] == 1.0
    assert r["p2p_rate"] == 0.5
    assert r["reward"] == 0.5
    assert r["resolved"] is False
    assert r["regressions"] == ["keep2"]


def test_grade_no_p2p_means_factor_one():
    # [FUNCTION DOC] Asserts an empty P2P list counts as p2p_rate 1.0 so full F2P passing gives reward 1.0 and
    # resolved.
    r = grade(["f1"], [], {"f1": "PASSED"})
    assert r["p2p_rate"] == 1.0
    assert r["reward"] == 1.0
    assert r["resolved"] is True


def test_grade_missing_test_counts_as_not_passed():
    # [FUNCTION DOC] Asserts an F2P test absent from the status map is not credited (f2p_passed 1 of 2, rate 0.5).
    """An F2P test that didn't run at all is not credited."""
    r = grade(["f1", "f2"], [], {"f1": "PASSED"})  # f2 absent
    assert r["f2p_passed"] == 1
    assert r["f2p_rate"] == 0.5


def test_grade_zero_fix_zero_reward():
    # [FUNCTION DOC] Asserts a failing sole F2P test yields reward 0.0 and resolved False even when P2P passes.
    r = grade(["f1"], ["keep"], {"f1": "FAILED", "keep": "PASSED"})
    assert r["reward"] == 0.0
    assert r["resolved"] is False


def test_grade_untracked_failure_keeps_tracked_resolved():
    # [FUNCTION DOC] Asserts a failure outside both F2P/P2P sets leaves reward 1.0 and resolved True but is
    # reported in untracked_failed / untracked_failed_count (so main() can block command_resolved).
    """All F2P + P2P pass, but a test outside both sets FAILED. Tracked
    `resolved` stays True (gold-patch oracle invariant), reward stays 1.0, and
    the untracked failure is recorded so main() can block `command_resolved`.
    Audit P0 #1 (httpx-3412 case)."""
    r = grade(
        ["f1"],
        ["keep"],
        {"f1": "PASSED", "keep": "PASSED", "tests/other::cp1252": "FAILED"},
    )
    assert r["reward"] == 1.0  # tracked tests all green
    assert r["resolved"] is True  # tracked resolution preserved
    assert r["untracked_failed_count"] == 1
    assert r["untracked_failed"] == ["tests/other::cp1252"]


def test_grade_go_subtests_remain_failures_when_parent_passes():
    # [FUNCTION DOC] With runner='go', asserts failing subtests under a passing parent are each counted as
    # untracked failures (sorted list of 2).
    """Go subtest failures under a passing parent remain untracked."""
    r = grade(
        [],
        [],
        {
            "TestExample": "PASSED",
            "TestExample/subtest1": "FAILED",
            "TestExample/subtest2": "FAILED",
        },
        runner="go",
    )
    assert r["untracked_failed_count"] == 2
    assert r["untracked_failed"] == [
        "TestExample/subtest1",
        "TestExample/subtest2",
    ]


def test_grade_go_subtests_count_parent_failure_once():
    # [FUNCTION DOC] With runner='go', asserts subtests under a failing parent collapse into the single parent
    # failure (count 1).
    """Go subtests under a failed parent are represented by the parent failure."""
    r = grade(
        [],
        [],
        {
            "TestExample": "FAILED",
            "TestExample/subtest1": "FAILED",
            "TestExample/subtest2": "FAILED",
        },
        runner="go",
    )

    assert r["untracked_failed_count"] == 1
    assert r["untracked_failed"] == ["TestExample"]


def test_grade_no_untracked_failure_resolves():
    # [FUNCTION DOC] Asserts a clean tracked run with no extra failures is resolved with untracked_failed_count 0.
    r = grade(["f1"], ["keep"], {"f1": "PASSED", "keep": "PASSED"})
    assert r["resolved"] is True
    assert r["untracked_failed_count"] == 0


# --- main() / IO -------------------------------------------------------------


def _write(p: Path, content: str) -> str:
    # [FUNCTION DOC] HELPER: Writes `content` to path `p` as UTF-8 and returns the path as a string. Side effect:
    # creates/overwrites the file.
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_main_writes_graded_reward(tmp_path: Path):
    # [FUNCTION DOC] Runs main() with a passing pytest log, F2P/P2P json files and exit code 0; asserts return 0,
    # reward.txt '1.000000', no reward.json, and reward-details.json with resolved, command_resolved
    # True, parse_status 'ok', exit_code 0. Writes under tmp_path.
    log = _write(
        tmp_path / "out.log",
        "tests/t.py::t_fix PASSED\ntests/t.py::t_keep PASSED\n",
    )
    f2p = _write(tmp_path / "f2p.json", json.dumps(["tests/t.py::t_fix"]))
    p2p = _write(tmp_path / "p2p.json", json.dumps(["tests/t.py::t_keep"]))
    out_dir = tmp_path / "verifier"
    rc = main(
        [
            "--log",
            log,
            "--f2p",
            f2p,
            "--p2p",
            p2p,
            "--runner",
            "pytest",
            "--exit-code",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "reward.txt").read_text().strip() == "1.000000"
    assert not (out_dir / "reward.json").exists()
    breakdown = json.loads((out_dir / "reward-details.json").read_text())
    assert breakdown["resolved"] is True
    assert breakdown["command_resolved"] is True  # clean command, exit 0
    assert breakdown["parse_status"] == "ok"
    assert breakdown["exit_code"] == 0  # always recorded, not just in fallback


def test_main_go_subtest_failure_under_passing_parent_is_untracked(tmp_path: Path):
    # [FUNCTION DOC] Runs main() with a Go log where nested subtests fail under a passing parent; asserts F2P
    # credited, 2 untracked failures recorded, and command_resolved False.
    """Go subtest failures under a passing parent remain untracked."""
    log = _write(
        tmp_path / "out.log",
        """--- PASS: TestTracked (0.01s)
    --- PASS: TestExample (0.01s)
        --- FAIL: TestExample/subtest1 (0.00s)
        --- FAIL: TestExample/subtest2 (0.00s)
    """,
    )
    f2p = _write(tmp_path / "f2p.json", json.dumps(["TestTracked"]))
    p2p = _write(tmp_path / "p2p.json", json.dumps([]))
    out_dir = tmp_path / "verifier"

    main(
        [
            "--log",
            log,
            "--f2p",
            f2p,
            "--p2p",
            p2p,
            "--runner",
            "go",
            "--exit-code",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )

    breakdown = json.loads((out_dir / "reward-details.json").read_text())
    assert breakdown["f2p_passed"] == 1
    assert breakdown["untracked_failed_count"] == 2
    assert breakdown["untracked_failed"] == [
        "TestExample/subtest1",
        "TestExample/subtest2",
    ]
    assert breakdown["command_resolved"] is False


def test_main_command_resolved_false_on_untracked_failure(tmp_path: Path):
    # [FUNCTION DOC] Runs main() where tracked tests pass but an untracked test fails with exit code 1; asserts
    # reward 1.0, resolved True, command_resolved False, untracked count 1.
    """All tracked tests pass (resolved True) but an untracked test failed and
    the command exited nonzero -> resolved True, command_resolved False."""
    log = _write(
        tmp_path / "out.log",
        "tests/t.py::t_fix PASSED\ntests/t.py::t_keep PASSED\ntests/t.py::t_flaky FAILED\n",
    )
    f2p = _write(tmp_path / "f2p.json", json.dumps(["tests/t.py::t_fix"]))
    p2p = _write(tmp_path / "p2p.json", json.dumps(["tests/t.py::t_keep"]))
    out_dir = tmp_path / "verifier"
    main(
        [
            "--log",
            log,
            "--f2p",
            f2p,
            "--p2p",
            p2p,
            "--runner",
            "pytest",
            "--exit-code",
            "1",
            "--out-dir",
            str(out_dir),
        ]
    )
    b = json.loads((out_dir / "reward-details.json").read_text())
    assert b["reward"] == 1.0  # tracked subset clean -> training reward 1.0
    assert b["resolved"] is True  # tracked resolution preserved
    assert b["command_resolved"] is False  # untracked failure + nonzero exit
    assert b["untracked_failed_count"] == 1


def test_main_falls_back_to_exit_code_on_unparseable_log(tmp_path: Path):
    # [FUNCTION DOC] Runs main() on an unparseable log with exit code 0; asserts parse_status 'fallback_exitcode'
    # and reward.txt '1.000000'.
    log = _write(tmp_path / "out.log", "garbage that no parser understands\n")
    f2p = _write(tmp_path / "f2p.json", json.dumps(["t_fix"]))
    p2p = _write(tmp_path / "p2p.json", json.dumps([]))
    out_dir = tmp_path / "verifier"
    main(
        [
            "--log",
            log,
            "--f2p",
            f2p,
            "--p2p",
            p2p,
            "--runner",
            "pytest",
            "--exit-code",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    breakdown = json.loads((out_dir / "reward-details.json").read_text())
    assert breakdown["parse_status"] == "fallback_exitcode"
    assert (out_dir / "reward.txt").read_text().strip() == "1.000000"


def test_main_fallback_not_resolved_when_oracle_declared(tmp_path: Path):
    # [FUNCTION DOC] Runs main() on an unparseable log with a declared F2P list (default runner); asserts fallback
    # parse_status, resolved False and eval_trustworthy False.
    """Unparseable log + declared F2P → reward may be exit-code-based, but
    `resolved` must be False (no per-test evidence). Audit P1."""
    log = _write(tmp_path / "out.log", "garbage no parser understands\n")
    f2p = _write(tmp_path / "f2p.json", json.dumps(["t_fix"]))  # declared oracle
    p2p = _write(tmp_path / "p2p.json", json.dumps([]))
    out_dir = tmp_path / "verifier"
    main(["--log", log, "--f2p", f2p, "--p2p", p2p, "--exit-code", "0", "--out-dir", str(out_dir)])
    b = json.loads((out_dir / "reward-details.json").read_text())
    assert b["parse_status"] == "fallback_exitcode"
    assert b["resolved"] is False  # cannot confirm F2P passed
    assert b["eval_trustworthy"] is False


def test_main_fallback_exit_nonzero_is_zero(tmp_path: Path):
    # [FUNCTION DOC] Runs main() on an unparseable log with exit code 1; asserts reward.txt is '0.000000'.
    log = _write(tmp_path / "out.log", "garbage\n")
    f2p = _write(tmp_path / "f2p.json", json.dumps(["t_fix"]))
    p2p = _write(tmp_path / "p2p.json", json.dumps([]))
    out_dir = tmp_path / "verifier"
    main(["--log", log, "--f2p", f2p, "--p2p", p2p, "--exit-code", "1", "--out-dir", str(out_dir)])
    assert (out_dir / "reward.txt").read_text().strip() == "0.000000"
