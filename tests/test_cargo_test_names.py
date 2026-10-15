# [MODULE DOC] Tests that cargo/libtest test identities parse identically in the canonical
# log_parsers.cargo_parser.parse_cargo_test and the standalone copy embedded in
# pipelines/_pr_runtime_verifier.py: test modes are stripped, doctest line numbers ignored,
# worst status wins across an item's doctests, noise lines ignored, and long lines do not
# cause regex backtracking. Also runs the standalone verifier script end-to-end in a subprocess.
"""Cargo test identities must survive generation-time and standalone parsing."""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest

from repo2rlenv.log_parsers.cargo_parser import parse_cargo_test
from repo2rlenv.pipelines import _pr_runtime_verifier as verifier


@pytest.fixture(
    params=[parse_cargo_test, verifier.parse_cargo_test], ids=["canonical", "standalone"]
)
def parser(request):
    # [FUNCTION DOC] Fixture parametrized over both parser implementations (ids 'canonical',
    # 'standalone').
    # Inputs: request (pytest param request). Output: the parser callable (str -> dict
    # name->status).
    return request.param


# `cargo test --no-fail-fast` on rustc 1.98.1: unit tests, an integration test
# binary and doctests, covering every libtest test mode.
_REAL_LOG = """\
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.13s
     Running unittests src/lib.rs (target/debug/deps/cargo_modes-9311986e2699f121)

running 8 tests
test tests::ignored_plain ... ignored
test tests::ignored_should_panic ... ignored
test tests::ignored_with_reason ... ignored, needs network
test tests::adds ... ok
test tests::does_not_panic - should panic ... FAILED
test tests::broken ... FAILED
test tests::panics_on_zero - should panic ... ok
test tests::panics_with_message - should panic ... ok

failures:

---- tests::does_not_panic stdout ----
note: test did not panic as expected at src/lib.rs:77:8
---- tests::broken stdout ----

thread 'tests::broken' (1925040) panicked at src/lib.rs:60:9:
assertion `left == right` failed
  left: 3
 right: 4
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    tests::broken
    tests::does_not_panic

test result: FAILED. 3 passed; 2 failed; 3 ignored; 0 measured; 0 filtered out; finished in 0.00s

error: test failed, to rerun pass `--lib`
     Running tests/integration.rs (target/debug/deps/integration-080ad6c4078d82ed)

running 2 tests
test integration_adds ... ok
test integration_panics - should panic ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests cargo_modes

running 8 tests
test src/lib.rs - add (line 38) ... ignored
test src/lib.rs - add (line 34) - compile ... ok
test src/lib.rs - add (line 30) - compile fail ... ok
test src/lib.rs - math::div (line 10) ... ok
test src/lib.rs - math::div (line 16) ... ok
test src/lib.rs - add (line 26) ... ok
test src/lib.rs - (line 3) ... ok
test src/lib.rs - add (line 42) ... FAILED

failures:

---- src/lib.rs - add (line 42) stdout ----
Test executable failed (exit status: 101).

failures:
    src/lib.rs - add (line 42)

test result: FAILED. 6 passed; 1 failed; 1 ignored; 0 measured; 0 filtered out; finished in 1.19s

error: doctest failed, to rerun pass `--doc`
"""


def test_real_cargo_log_keeps_every_libtest_result(parser):
    # [FUNCTION DOC] A real multi-binary cargo log maps every unit/integration/doctest result
    # to the expected PASSED/FAILED/SKIPPED dict. Inputs: parser fixture.
    assert parser(_REAL_LOG) == {
        "tests::ignored_plain": "SKIPPED",
        "tests::ignored_should_panic": "SKIPPED",
        "tests::ignored_with_reason": "SKIPPED",
        "tests::adds": "PASSED",
        "tests::does_not_panic": "FAILED",
        "tests::broken": "FAILED",
        "tests::panics_on_zero": "PASSED",
        "tests::panics_with_message": "PASSED",
        "integration_adds": "PASSED",
        "integration_panics": "PASSED",
        "src/lib.rs - add": "FAILED",
        "src/lib.rs - math::div": "PASSED",
        "src/lib.rs": "PASSED",
    }


@pytest.mark.parametrize("mode", ["should panic", "compile fail", "compile"])
@pytest.mark.parametrize(("word", "status"), [("ok", "PASSED"), ("FAILED", "FAILED")])
def test_test_mode_is_not_part_of_the_identity(parser, mode, word, status):
    # [FUNCTION DOC] Parametrized: ' - should panic' / ' - compile fail' / ' - compile'
    # suffixes are dropped from names for ok/FAILED. Inputs: parser, mode, word, status.
    assert parser(f"test tests::divide_by_zero - {mode} ... {word}\n") == {
        "tests::divide_by_zero": status
    }


def test_doctest_identity_ignores_line_numbers(parser):
    # [FUNCTION DOC] Two doctest lines differing only in '(line N)' produce the same identity
    # 'src/lib.rs - math::div'. Inputs: parser.
    before = "test src/lib.rs - math::div (line 10) - compile fail ... ok\n"
    after = "test src/lib.rs - math::div (line 17) - compile fail ... ok\n"
    assert parser(before) == parser(after) == {"src/lib.rs - math::div": "PASSED"}


@pytest.mark.parametrize(
    ("words", "status"),
    [
        (["ok", "FAILED", "ignored"], "FAILED"),
        (["ok", "ignored"], "PASSED"),
        (["ignored", "ignored"], "SKIPPED"),
    ],
)
def test_worst_status_among_an_items_doctests_wins_in_any_order(parser, words, status):
    # [FUNCTION DOC] For every permutation of doctest results for one item, FAILED beats
    # PASSED beats SKIPPED. Inputs: parser, words, status.
    for order in itertools.permutations(words):
        log = "".join(
            f"test src/lib.rs - add (line {10 + 4 * i}) ... {word}\n"
            for i, word in enumerate(order)
        )
        assert parser(log) == {"src/lib.rs - add": status}


def test_padded_names_and_ignore_reasons(parser):
    # [FUNCTION DOC] Whitespace-padded names and 'ignored, <reason>' lines parse correctly.
    # Inputs: parser.
    log = (
        "test tests::short          ... ok\n"
        "test tests::panics         - should panic ... ok\n"
        "test tests::slow ... ignored, needs network\n"
    )
    assert parser(log) == {
        "tests::short": "PASSED",
        "tests::panics": "PASSED",
        "tests::slow": "SKIPPED",
    }


def test_repeated_unit_test_keeps_last_write(parser):
    # [FUNCTION DOC] When a test name appears twice, the last status wins. Inputs: parser.
    log = "test tests::flaky ... FAILED\ntest tests::flaky ... ok\n"
    assert parser(log) == {"tests::flaky": "PASSED"}


@pytest.mark.parametrize(
    "log",
    [
        "",
        "test result: FAILED. 3 passed; 2 failed; 3 ignored; 0 measured; 0 filtered out\n",
        "failures:\n    tests::broken\n    src/lib.rs - add (line 42)\n",
        "---- src/lib.rs - add (line 42) stdout ----\n",
        "test tests::unfinished ... \n",
        "test tests::unknown ... running\n",
    ],
)
def test_summary_failure_blocks_and_unrelated_lines(parser, log):
    # [FUNCTION DOC] Parametrized non-result lines (summaries, failure lists, stdout headers,
    # unfinished/unknown statuses) yield an empty dict. Inputs: parser, log.
    assert parser(log) == {}


def test_long_lines_do_not_stall_parsing():
    # [FUNCTION DOC] Runs both parsers in a child Python process (10s timeout) on pathological
    # ~200k-char lines
    # to guard against catastrophic regex backtracking; the child must exit 0. Side effects:
    # spawns a subprocess. Raises AssertionError or subprocess.TimeoutExpired on failure.
    # Isolate the timeout so a backtracking regression cannot hang the suite.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from repo2rlenv.log_parsers.cargo_parser import parse_cargo_test\n"
            "from repo2rlenv.pipelines._pr_runtime_verifier import parse_cargo_test as standalone\n"
            "noise = ['test' + ' ' * 200000 + 'x', 'test x' + ' ' * 200000 + 'y',\n"
            "         'test ' + ' ... ' * 100000, 'test x' + ' ...' * 100000]\n"
            "log = '\\n'.join(noise) + '\\ntest tests::adds ... ok\\n'\n"
            "for parser in (parse_cargo_test, standalone):\n"
            "    assert parser(log) == {'tests::adds': 'PASSED'}\n",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# A real fix whose new test is `#[should_panic]`, captured before the fix, with
# the gold patch, and with an agent patch that is correct but adds a helper
# above `div`, which moves its doctest from line 3 to line 9.
_PRE_LOG = """\
     Running tests/div.rs (target/debug/deps/div-5716e1d9d44460c2)

running 2 tests
test divides ... ok
test rejects_zero - should panic ... FAILED

failures:

---- rejects_zero stdout ----
note: test did not panic as expected at tests/div.rs:8:4

failures:
    rejects_zero

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

error: test failed, to rerun pass `--test div`
   Doc-tests scenario

running 1 test
test src/lib.rs - div (line 3) ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s

error: 1 target failed:
    `--test div`
"""
_GOLD_LOG = """\
     Running tests/div.rs (target/debug/deps/div-5716e1d9d44460c2)

running 2 tests
test divides ... ok
test rejects_zero - should panic ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests scenario

running 1 test
test src/lib.rs - div (line 3) ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s
"""
_AGENT_LOG = _GOLD_LOG.replace("(line 3)", "(line 9)")


def test_shifted_doctest_keeps_a_correct_patch_at_full_reward(tmp_path: Path):
    # [FUNCTION DOC] Derives F2P/P2P from pre/gold logs, then copies the verifier module to
    # tmp_path and runs
    # it isolated (python -I) against an agent log where the doctest moved lines; the written
    # rewards/reward-details.json must report reward 1.0, resolved, no regressions.
    # Inputs: tmp_path. Side effects: writes files and spawns a subprocess (30s timeout).
    pre, gold = parse_cargo_test(_PRE_LOG), parse_cargo_test(_GOLD_LOG)
    f2p = [
        name for name, status in pre.items() if status == "FAILED" and gold.get(name) == "PASSED"
    ]
    p2p = [
        name for name, status in pre.items() if status == "PASSED" and gold.get(name) == "PASSED"
    ]
    assert f2p == ["rejects_zero"]
    assert p2p == ["divides", "src/lib.rs - div"]

    # Run the verifier in isolation, with no installed repo2rlenv imports.
    standalone = tmp_path / "verifier.py"
    standalone.write_text(Path(verifier.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "out.log").write_text(_AGENT_LOG, encoding="utf-8")
    (tmp_path / "f2p.json").write_text(json.dumps(f2p), encoding="utf-8")
    (tmp_path / "p2p.json").write_text(json.dumps(p2p), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(standalone),
            "--log",
            "out.log",
            "--f2p",
            "f2p.json",
            "--p2p",
            "p2p.json",
            "--runner",
            "cargo",
            "--exit-code",
            "0",
            "--out-dir",
            "rewards",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    details = json.loads((tmp_path / "rewards/reward-details.json").read_text())
    assert details["reward"] == 1.0
    assert details["resolved"] is True
    assert details["regressions"] == []
