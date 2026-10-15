# [MODULE DOC] Tests that both Jest-format parsers (canonical log_parsers.jest_parser.parse_jest and
# the standalone copy in pipelines._pr_runtime_verifier) also read Mocha's spec reporter: suite
# nesting forms 'Suite > sub > test' names, check-mark/'N)'/'-' lines map to PASSED/FAILED/SKIPPED,
# durations are stripped, epilogue failure blocks and multiple runs are handled, and Jest file
# headers still win. Also runs the standalone verifier end-to-end in a subprocess.
"""Mocha's spec reporter must be read by both parsers, not just jest's format."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from repo2rlenv.log_parsers.jest_parser import parse_jest
from repo2rlenv.pipelines import _pr_runtime_verifier as verifier


@pytest.fixture(params=[parse_jest, verifier.parse_jest], ids=["canonical", "standalone"])
def parser(request):
    # [FUNCTION DOC] Fixture parametrized over the canonical and standalone parse_jest functions;
    # returns the parser under test.
    return request.param


# Real `npx mocha` output (mocha 12.0.2): passes, a numbered failure, two
# pending tests, a slow test with its duration, and a second top-level suite.
_MIXED = (
    "\n"
    "\n"
    "  Math\n"
    "    ✔ is slow (40ms)\n"
    "    add()\n"
    "      ✔ adds two numbers\n"
    "      ✔ handles negatives (edge case)\n"
    "      1) breaks on purpose\n"
    "      - skipped test\n"
    "      - pending without body\n"
    "\n"
    "  Top level\n"
    "    ✔ returns 200 (OK)\n"
    "\n"
    "\n"
    "  4 passing (47ms)\n"
    "  2 pending\n"
    "  1 failing\n"
    "\n"
    "  1) Math\n"
    "       add()\n"
    "         breaks on purpose:\n"
    "\n"
    "      AssertionError [ERR_ASSERTION]: Expected values to be strictly equal:\n"
    "\n"
    "2 !== 3\n"
    "\n"
    "      + expected - actual\n"
    "\n"
    "      -2\n"
    "      +3\n"
    "      \n"
    "      at Context.<anonymous> (test/math.spec.js:6:50)\n"
    "      at process.processImmediate (node:internal/timers:574:21)\n"
    "\n"
    "\n"
    "\n"
)

# Suites nested three deep, plus a test that sits directly under the root suite.
_NESTED = (
    "\n"
    "\n"
    "  API\n"
    "    ✔ suite-level test\n"
    "    GET /users\n"
    "      ✔ returns 200 (OK)\n"
    "      1) fails hard\n"
    "    nested\n"
    "      deeper\n"
    "        ✔ works at depth\n"
    "\n"
    "\n"
    "  3 passing (5ms)\n"
    "  1 failing\n"
    "\n"
    "  1) API\n"
    "       GET /users\n"
    "         fails hard:\n"
    "\n"
    "      AssertionError [ERR_ASSERTION]: Expected values to be strictly deep-equal:\n"
    "+ actual - expected\n"
    "\n"
    "  {\n"
    "+   a: 1\n"
    "-   a: 2\n"
    "  }\n"
    "\n"
    "      + expected - actual\n"
    "\n"
    "       {\n"
    '      -  "a": 1\n'
    '      +  "a": 2\n'
    "       }\n"
    "      \n"
    "      at Context.<anonymous> (test2/api.spec.js:5:43)\n"
    "      at process.processImmediate (node:internal/timers:574:21)\n"
    "\n"
    "\n"
    "\n"
)

# Two `mocha` runs in one log, as `mocha a && mocha b` produces. The first
# run's epilogue repeats its failure titles before the second run starts.
_TWO_RUNS = (
    "\n"
    "\n"
    "  Math\n"
    "    ✔ is slow (41ms)\n"
    "    add()\n"
    "      ✔ adds two numbers\n"
    "      ✔ handles negatives (edge case)\n"
    "      1) breaks on purpose\n"
    "      - skipped test\n"
    "      - pending without body\n"
    "\n"
    "  Top level\n"
    "    ✔ returns 200 (OK)\n"
    "\n"
    "\n"
    "  4 passing (45ms)\n"
    "  2 pending\n"
    "  1 failing\n"
    "\n"
    "  1) Math\n"
    "       add()\n"
    "         breaks on purpose:\n"
    "\n"
    "      AssertionError [ERR_ASSERTION]: Expected values to be strictly equal:\n"
    "\n"
    "2 !== 3\n"
    "\n"
    "      + expected - actual\n"
    "\n"
    "      -2\n"
    "      +3\n"
    "      \n"
    "      at Context.<anonymous> (test/math.spec.js:6:50)\n"
    "      at process.processImmediate (node:internal/timers:574:21)\n"
    "\n"
    "\n"
    "\n"
    "\n"
    "\n"
    "  API\n"
    "    ✔ suite-level test\n"
    "    GET /users\n"
    "      ✔ returns 200 (OK)\n"
    "      1) fails hard\n"
    "    nested\n"
    "      deeper\n"
    "        ✔ works at depth\n"
    "\n"
    "\n"
    "  3 passing (5ms)\n"
    "  1 failing\n"
    "\n"
    "  1) API\n"
    "       GET /users\n"
    "         fails hard:\n"
    "\n"
    "      AssertionError [ERR_ASSERTION]: Expected values to be strictly deep-equal:\n"
    "+ actual - expected\n"
    "\n"
    "  {\n"
    "+   a: 1\n"
    "-   a: 2\n"
    "  }\n"
    "\n"
    "      + expected - actual\n"
    "\n"
    "       {\n"
    '      -  "a": 1\n'
    '      +  "a": 2\n'
    "       }\n"
    "      \n"
    "      at Context.<anonymous> (test2/api.spec.js:5:43)\n"
    "      at process.processImmediate (node:internal/timers:574:21)\n"
    "\n"
    "\n"
    "\n"
)


def test_spec_reporter_records_every_outcome(parser):
    # [FUNCTION DOC] Asserts a mixed Mocha run yields exactly the expected names/statuses, including
    # pending tests as SKIPPED and durations removed.
    assert parser(_MIXED) == {
        "Math > is slow": "PASSED",
        "Math > add() > adds two numbers": "PASSED",
        "Math > add() > handles negatives (edge case)": "PASSED",
        "Math > add() > breaks on purpose": "FAILED",
        "Math > add() > skipped test": "SKIPPED",
        "Math > add() > pending without body": "SKIPPED",
        "Top level > returns 200 (OK)": "PASSED",
    }


def test_nested_suites_qualify_the_name(parser):
    # [FUNCTION DOC] Asserts tests at several suite depths get fully qualified ' > '-joined names.
    assert parser(_NESTED) == {
        "API > suite-level test": "PASSED",
        "API > GET /users > returns 200 (OK)": "PASSED",
        "API > GET /users > fails hard": "FAILED",
        "API > nested > deeper > works at depth": "PASSED",
    }


def test_second_run_in_the_same_log_is_recorded(parser):
    # [FUNCTION DOC] Asserts two concatenated mocha runs yield 11 entries and no epilogue 'N)' lines
    # leak in as names.
    status = parser(_TWO_RUNS)
    assert status["Math > add() > breaks on purpose"] == "FAILED"
    assert status["API > GET /users > fails hard"] == "FAILED"
    assert status["API > nested > deeper > works at depth"] == "PASSED"
    # 7 from the first run, 4 from the second, and nothing from either epilogue.
    assert len(status) == 11
    assert not [name for name in status if name.startswith("1)")]


@pytest.mark.parametrize("glyph", ["\u2714", "\u2713", "\u221a"])
def test_check_mark_variants(parser, glyph):
    # log-symbols uses ✔ (√ on Windows); mocha 7 and older printed ✓.
    # [FUNCTION DOC] Parametrized over the check glyph variants (heavy check, check, square root):
    # asserts each is read as PASSED.
    log = f"  Suite\n    {glyph} passes\n\n  1 passing (4ms)\n"
    assert parser(log) == {"Suite > passes": "PASSED"}


def test_durations_and_pending_reasons_stay_out_of_the_name(parser):
    # [FUNCTION DOC] Asserts '(Nms)' suffixes are removed while text like 'waits 5ms' inside the
    # title is kept.
    log = (
        "  Suite\n"
        "    \u2714 slow one (1043ms)\n"
        "    \u2714 waits 5ms (12ms)\n"
        "    - pending one\n"
        "\n"
        "  2 passing (1s)\n"
        "  1 pending\n"
    )
    assert parser(log) == {
        "Suite > slow one": "PASSED",
        "Suite > waits 5ms": "PASSED",
        "Suite > pending one": "SKIPPED",
    }


def test_epilogue_failure_block_is_not_a_test(parser):
    # [FUNCTION DOC] Asserts the epilogue's suite header and assertion text do not become entries.
    status = parser(_MIXED)
    assert "Math > add() > breaks on purpose" in status
    # The epilogue prints `1) Math`, then the titles again, then a stack trace.
    assert "Math" not in status
    assert not [name for name in status if "AssertionError" in name]


def test_jest_output_is_still_read_as_jest(parser):
    # No mocha epilogue and a ✓ glyph, so this must keep its file prefix.
    # [FUNCTION DOC] Asserts a Jest log with a 'PASS file' header keeps the file prefix in names.
    log = "PASS  src/foo.test.js\n  Foo\n    \u2713 returns 200 (4 ms)\n"
    assert parser(log) == {"src/foo.test.js > Foo > returns 200": "PASSED"}


def test_long_lines_do_not_stall_parsing():
    # Isolate the timeout so a backtracking regression cannot hang the suite.
    # [FUNCTION DOC] Runs both parsers in a subprocess (10s timeout) on 200k-char pathological lines
    # to guard against regex catastrophic backtracking; asserts exit code 0.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from repo2rlenv.log_parsers.jest_parser import parse_jest\n"
            "from repo2rlenv.pipelines._pr_runtime_verifier import parse_jest as standalone\n"
            "noise = ['  ' + 'a' * 200000, '    \u2714 ' + 'b' * 200000,\n"
            "         '    1) ' + 'c' * 200000, '    - ' + ' ' * 200000 + 'd',\n"
            "         '    \u2714 x' + ' (1ms)' * 50000]\n"
            "log = '\\n'.join(noise) + '\\n  Suite\\n    \u2714 real test (2ms)\\n\\n  1 passing (3ms)\\n'\n"
            "for parser in (parse_jest, standalone):\n"
            "    assert parser(log)['Suite > real test'] == 'PASSED'\n",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# A real fix under mocha: `div` gains a zero check, so its test flips.
_PRE = (
    "\n"
    "\n"
    "  div()\n"
    "    ✔ divides\n"
    "    1) rejects a zero divisor\n"
    "\n"
    "\n"
    "  1 passing (3ms)\n"
    "  1 failing\n"
    "\n"
    "  1) div()\n"
    "       rejects a zero divisor:\n"
    "     AssertionError [ERR_ASSERTION]: Missing expected exception.\n"
    "      at Context.<anonymous> (test/div.spec.js:7:12)\n"
    "      at process.processImmediate (node:internal/timers:574:21)\n"
    "\n"
    "\n"
    "\n"
)

_POST = "\n\n  div()\n    ✔ divides\n    ✔ rejects a zero divisor\n\n\n  2 passing (2ms)\n\n"


def test_mocha_fix_yields_an_oracle_and_grades_it(tmp_path: Path):
    # [FUNCTION DOC] Derives F2P/P2P from pre/post mocha logs, copies the standalone verifier to
    # tmp_path, runs it with python -I on the post log, and asserts reward-details.json reports
    # reward 1.0 and resolved True. Side effects: writes files under tmp_path, spawns a subprocess.
    pre, post = parse_jest(_PRE), parse_jest(_POST)
    f2p = [name for name, st in pre.items() if st == "FAILED" and post.get(name) == "PASSED"]
    p2p = [name for name, st in pre.items() if st == "PASSED" and post.get(name) == "PASSED"]
    assert f2p == ["div() > rejects a zero divisor"]
    assert p2p == ["div() > divides"]

    # Run the verifier in isolation, with no installed repo2rlenv imports.
    standalone = tmp_path / "verifier.py"
    standalone.write_text(Path(verifier.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "out.log").write_text(_POST, encoding="utf-8")
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
            "--test-cmds",
            "npx mocha",
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


# --- run boundaries and format detection (review of #162) -------------------

# Two runs where the second one's first result is a failure, so the epilogue's
# own numbered block has to be told apart from the next run's listing.
_SECOND_RUN_STARTS_FAILING = (
    "\n"
    "  Math\n"
    "    \u2714 adds\n"
    "    1) breaks\n"
    "\n"
    "  1 passing (4ms)\n"
    "  1 failing\n"
    "\n"
    "  1) Math\n"
    "       breaks:\n"
    "     AssertionError: nope\n"
    "      at Context.<anonymous> (test/a.spec.js:4:1)\n"
    "\n"
    "\n"
    "  API\n"
    "    1) also breaks\n"
    "    \u2714 works\n"
    "\n"
    "  1 passing (3ms)\n"
    "  1 failing\n"
)

# A second run with nothing but failures, after a first run with none.
_SECOND_RUN_ONLY_FAILS = (
    "\n"
    "  Math\n"
    "    \u2714 adds\n"
    "\n"
    "  1 passing (4ms)\n"
    "\n"
    "\n"
    "  API\n"
    "    1) breaks\n"
    "    2) breaks harder\n"
    "\n"
    "  0 passing (2ms)\n"
    "  2 failing\n"
)


def test_second_run_starting_with_a_failure_is_recorded(parser):
    # [FUNCTION DOC] Asserts a second run whose first line is a numbered failure is not mistaken for
    # the previous run's epilogue.
    assert parser(_SECOND_RUN_STARTS_FAILING) == {
        "Math > adds": "PASSED",
        "Math > breaks": "FAILED",
        "API > also breaks": "FAILED",
        "API > works": "PASSED",
    }


def test_second_run_with_only_failures_is_recorded(parser):
    # [FUNCTION DOC] Asserts a second run consisting only of failures is recorded.
    assert parser(_SECOND_RUN_ONLY_FAILS) == {
        "Math > adds": "PASSED",
        "API > breaks": "FAILED",
        "API > breaks harder": "FAILED",
    }


def test_epilogue_failure_blocks_are_consumed_by_count(parser):
    # Three failures are repeated below `3 failing`, and none of them may be
    # read as a test of a later run.
    # [FUNCTION DOC] Asserts exactly 'N failing' epilogue blocks are skipped so none are read as new
    # tests.
    log = (
        "  Suite\n"
        "    1) a\n"
        "    2) b\n"
        "    3) c\n"
        "\n"
        "  0 passing (1ms)\n"
        "  3 failing\n"
        "\n"
        "  1) Suite\n       a:\n     AssertionError\n"
        "  2) Suite\n       b:\n     AssertionError\n"
        "  3) Suite\n       c:\n     AssertionError\n"
    )
    assert parser(log) == {"Suite > a": "FAILED", "Suite > b": "FAILED", "Suite > c": "FAILED"}


@pytest.mark.parametrize("glyph", ["\u221a", "\u2713"])
def test_jest_wins_when_the_log_has_a_file_header(parser, glyph):
    # jest prints √ on Windows, and mocha never prints a file header, so the
    # header decides. Reading this as mocha would drop the file prefix, keep
    # the ` (4 ms)` duration, and lose the × line entirely.
    # [FUNCTION DOC] Parametrized over glyphs: asserts a file header forces Jest parsing (file
    # prefix kept, duration stripped, cross mark read as FAILED).
    log = f"PASS  a.test.js\n  Suite\n    {glyph} passes (4 ms)\n    \u00d7 fails (2 ms)\n"
    assert parser(log) == {
        "a.test.js > Suite > passes": "PASSED",
        "a.test.js > Suite > fails": "FAILED",
    }


def test_mocha_windows_check_mark_is_read_as_mocha(parser):
    # No file header, and the epilogue identifies the runner.
    # [FUNCTION DOC] Asserts the Windows square-root check without a file header is read as mocha.
    log = "  Suite\n    \u221a passes (4ms)\n\n  1 passing (4ms)\n"
    assert parser(log) == {"Suite > passes": "PASSED"}


def test_jest_log_without_glyphs_is_untouched(parser):
    # [FUNCTION DOC] Asserts a 'FAIL file' Jest log is still parsed as Jest with its file prefix.
    log = "FAIL  b.test.js\n  Suite\n    \u2713 ok (1 ms)\n"
    assert parser(log) == {"b.test.js > Suite > ok": "PASSED"}
