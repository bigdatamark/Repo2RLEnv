# [MODULE DOC] Tests that pytest-xdist output (lines like '[gw1] [ 20%] PASSED <nodeid>', with the
# worker label and progress counter in front of the status) is parsed correctly by both the
# canonical log_parsers.pytest_parser.parse_pytest and the standalone copy in _pr_runtime_verifier.
# Uses captured real xdist logs (percent/classic/count styles, serial -n 0) as module constants, a
# backtracking guard run in a subprocess, and an end-to-end F2P/P2P derivation graded by the
# verifier script run isolated with `python -I`. Side effects: subprocesses and files under tmp_path
# only.
"""pytest-xdist worker labels must not hide results from either parser."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from repo2rlenv.log_parsers.pytest_parser import parse_pytest
from repo2rlenv.pipelines import _pr_runtime_verifier as verifier


@pytest.fixture(params=[parse_pytest, verifier.parse_pytest], ids=["canonical", "standalone"])
def parser(request):
    # [FUNCTION DOC] Pytest fixture parametrized over the canonical and standalone parse_pytest implementations.
    # Output: the parser callable.
    return request.param


_EXPECTED = {
    "test_calc.py::test_eval[1 + 1]": "PASSED",
    "test_calc.py::test_eval[2 + 2]": "PASSED",
    "test_calc.py::test_keep": "PASSED",
    "test_calc.py::test_broken": "FAILED",
    "test_calc.py::test_skipped": "SKIPPED",
}

# Real `pytest -v -n 2` output (pytest 9.1.1, pytest-xdist 3.8.0). Each result
# carries the worker that ran it, and the progress counter moves in front of
# the status. The bare `test_calc.py::test_keep` lines are xdist announcing
# what it dispatched, and carry no status.
_XDIST_PERCENT = (
    "============================= test session starts ==============================\n"
    "platform darwin -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0 -- /repo/.venv/bin/python\n"
    "rootdir: /repo\n"
    "plugins: xdist-3.8.0\n"
    "created: 2/2 workers\n"
    "2 workers [5 items]\n"
    "\n"
    "scheduling tests via LoadScheduling\n"
    "\n"
    "test_calc.py::test_keep \n"
    "test_calc.py::test_eval[1 + 1] \n"
    "[gw1] [ 20%] PASSED test_calc.py::test_keep \n"
    "[gw0] [ 40%] PASSED test_calc.py::test_eval[1 + 1] \n"
    "test_calc.py::test_broken \n"
    "test_calc.py::test_eval[2 + 2] \n"
    "[gw0] [ 60%] PASSED test_calc.py::test_eval[2 + 2] \n"
    "[gw1] [ 80%] FAILED test_calc.py::test_broken \n"
    "test_calc.py::test_skipped \n"
    "[gw1] [100%] SKIPPED test_calc.py::test_skipped \n"
    "\n"
    "=================================== FAILURES ===================================\n"
    "_________________________________ test_broken __________________________________\n"
    "[gw1] darwin -- Python 3.12.14 /repo/.venv/bin/python\n"
    "\n"
    "    def test_broken():\n"
    ">       assert 1 == 2\n"
    "E       assert 1 == 2\n"
    "\n"
    "test_calc.py:11: AssertionError\n"
    "=========================== short test summary info ============================\n"
    "FAILED test_calc.py::test_broken - assert 1 == 2\n"
    "==================== 1 failed, 3 passed, 1 skipped in 0.42s ====================\n"
)

# The same run with `-o console_output_style=classic` (no counter).
_XDIST_CLASSIC = (
    "============================= test session starts ==============================\n"
    "platform darwin -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0 -- /repo/.venv/bin/python\n"
    "rootdir: /repo\n"
    "plugins: xdist-3.8.0\n"
    "created: 2/2 workers\n"
    "2 workers [5 items]\n"
    "\n"
    "scheduling tests via LoadScheduling\n"
    "\n"
    "test_calc.py::test_keep \n"
    "test_calc.py::test_eval[1 + 1] \n"
    "[gw1] PASSED test_calc.py::test_keep \n"
    "[gw0] PASSED test_calc.py::test_eval[1 + 1] \n"
    "test_calc.py::test_broken \n"
    "test_calc.py::test_eval[2 + 2] \n"
    "[gw0] PASSED test_calc.py::test_eval[2 + 2] \n"
    "[gw1] FAILED test_calc.py::test_broken \n"
    "test_calc.py::test_skipped \n"
    "[gw1] SKIPPED test_calc.py::test_skipped \n"
    "\n"
    "=================================== FAILURES ===================================\n"
    "_________________________________ test_broken __________________________________\n"
    "[gw1] darwin -- Python 3.12.14 /repo/.venv/bin/python\n"
    "\n"
    "    def test_broken():\n"
    ">       assert 1 == 2\n"
    "E       assert 1 == 2\n"
    "\n"
    "test_calc.py:11: AssertionError\n"
    "=========================== short test summary info ============================\n"
    "FAILED test_calc.py::test_broken - assert 1 == 2\n"
    "==================== 1 failed, 3 passed, 1 skipped in 0.42s ====================\n"
)

# ...and with `console_output_style=count`.
_XDIST_COUNT = (
    "============================= test session starts ==============================\n"
    "platform darwin -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0 -- /repo/.venv/bin/python\n"
    "rootdir: /repo\n"
    "plugins: xdist-3.8.0\n"
    "created: 2/2 workers\n"
    "2 workers [5 items]\n"
    "\n"
    "scheduling tests via LoadScheduling\n"
    "\n"
    "test_calc.py::test_keep \n"
    "test_calc.py::test_eval[1 + 1] \n"
    "[gw1] [1/5] PASSED test_calc.py::test_keep \n"
    "[gw0] [2/5] PASSED test_calc.py::test_eval[1 + 1] \n"
    "test_calc.py::test_broken \n"
    "test_calc.py::test_eval[2 + 2] \n"
    "[gw0] [3/5] PASSED test_calc.py::test_eval[2 + 2] \n"
    "[gw1] [4/5] FAILED test_calc.py::test_broken \n"
    "test_calc.py::test_skipped \n"
    "[gw1] [5/5] SKIPPED test_calc.py::test_skipped \n"
    "\n"
    "=================================== FAILURES ===================================\n"
    "_________________________________ test_broken __________________________________\n"
    "[gw1] darwin -- Python 3.12.14 /repo/.venv/bin/python\n"
    "\n"
    "    def test_broken():\n"
    ">       assert 1 == 2\n"
    "E       assert 1 == 2\n"
    "\n"
    "test_calc.py:11: AssertionError\n"
    "=========================== short test summary info ============================\n"
    "FAILED test_calc.py::test_broken - assert 1 == 2\n"
    "==================== 1 failed, 3 passed, 1 skipped in 0.42s ====================\n"
)

# `pytest -v -n 0`: xdist installed but disabled, so no worker labels.
_SERIAL = (
    "============================= test session starts ==============================\n"
    "platform darwin -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0 -- /repo/.venv/bin/python\n"
    "rootdir: /repo\n"
    "plugins: xdist-3.8.0\n"
    "collecting ... collected 5 items\n"
    "\n"
    "test_calc.py::test_eval[1 + 1] PASSED                                    [ 20%]\n"
    "test_calc.py::test_eval[2 + 2] PASSED                                    [ 40%]\n"
    "test_calc.py::test_keep PASSED                                           [ 60%]\n"
    "test_calc.py::test_broken FAILED                                         [ 80%]\n"
    "test_calc.py::test_skipped SKIPPED (later)                               [100%]\n"
    "\n"
    "=================================== FAILURES ===================================\n"
    "_________________________________ test_broken __________________________________\n"
    "\n"
    "    def test_broken():\n"
    ">       assert 1 == 2\n"
    "E       assert 1 == 2\n"
    "\n"
    "test_calc.py:11: AssertionError\n"
    "=========================== short test summary info ============================\n"
    "FAILED test_calc.py::test_broken - assert 1 == 2\n"
    "==================== 1 failed, 3 passed, 1 skipped in 0.42s ====================\n"
)


@pytest.mark.parametrize(
    "log", [_XDIST_PERCENT, _XDIST_CLASSIC, _XDIST_COUNT], ids=["percent", "classic", "count"]
)
def test_worker_labelled_results_are_recorded(parser, log):
    # [FUNCTION DOC] Asserts each xdist log style parses to _EXPECTED, ignoring status-less dispatch announcement
    # lines and the summary footer.
    assert parser(log) == _EXPECTED


def test_serial_output_is_unchanged(parser):
    # [FUNCTION DOC] Asserts the non-xdist serial (-n 0) verbose log still parses to _EXPECTED.
    assert parser(_SERIAL) == _EXPECTED


@pytest.mark.parametrize("counter", ["", "[ 20%] ", "[1/5] ", "[ 17 / 17 ] "])
def test_worker_label_shapes(parser, counter):
    # [FUNCTION DOC] Asserts a '[gwN]' label followed by various (or no) progress counters then 'PASSED <nodeid>'
    # parses to that node PASSED.
    node = "tests/test_calc.py::test_eval[1 + 1]"
    assert parser(f"[gw11] {counter}PASSED {node} \n") == {node: "PASSED"}


def test_unlabelled_lines_keep_their_meaning(parser):
    # A test id that merely starts with a bracket is not a worker label.
    # [FUNCTION DOC] Asserts a bracketed prefix that is not a real worker label ('[gwen]') yields nothing, while an
    # ordinary verbose line still parses.
    assert parser("[gwen] PASSED tests/test_a.py::test_x\n") == {}
    assert parser("tests/test_a.py::test_x PASSED [ 50%]\n") == {
        "tests/test_a.py::test_x": "PASSED"
    }


def test_long_worker_labels_do_not_stall_parsing():
    # Isolate the timeout so a backtracking regression cannot hang the suite.
    # [FUNCTION DOC] Backtracking guard: in a subprocess with a 10 s timeout, runs both parsers over pathological
    # worker-label noise lines and asserts the trailing real node is parsed PASSED.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from repo2rlenv.log_parsers.pytest_parser import parse_pytest\n"
            "from repo2rlenv.pipelines._pr_runtime_verifier import parse_pytest as standalone\n"
            "noise = ['[gw1]' + ' ' * 200000 + 'x', '[gw' + '1' * 100000 + '] PASSED a::b',\n"
            "         '[gw1] ' + '[ 1%] ' * 50000 + 'PASSED a::b']\n"
            "node = 'tests/test_calc.py::test_eval[1 + 1]'\n"
            "log = '\\n'.join(noise) + '\\n[gw0] [ 50%] PASSED ' + node + '\\n'\n"
            "for parser in (parse_pytest, standalone):\n"
            "    assert parser(log)[node] == 'PASSED'\n",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# A real fix captured under `-v -n 2`: `div` gains a zero check. Every test
# passes afterwards, so pytest prints no summary section at all.
_SCENARIO_PRE = (
    "============================= test session starts ==============================\n"
    "platform darwin -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0 -- /repo/.venv/bin/python\n"
    "rootdir: /repo\n"
    "plugins: xdist-3.8.0\n"
    "created: 2/2 workers\n"
    "2 workers [3 items]\n"
    "\n"
    "scheduling tests via LoadScheduling\n"
    "\n"
    "test_calc.py::test_rejects_zero[pair0] \n"
    "test_calc.py::test_divides \n"
    "[gw0] [ 33%] PASSED test_calc.py::test_divides \n"
    "test_calc.py::test_rejects_zero[pair1] \n"
    "[gw1] [ 66%] FAILED test_calc.py::test_rejects_zero[pair0] \n"
    "[gw0] [100%] FAILED test_calc.py::test_rejects_zero[pair1] \n"
    "\n"
    "=================================== FAILURES ===================================\n"
    "___________________________ test_rejects_zero[pair0] ___________________________\n"
    "[gw1] darwin -- Python 3.12.14 /repo/.venv/bin/python\n"
    "\n"
    "pair = (1, 0)\n"
    "\n"
    '    @pytest.mark.parametrize("pair", [(1, 0), (5, 0)])\n'
    "    def test_rejects_zero(pair):\n"
    '        with pytest.raises(ValueError, match="divide by zero"):\n'
    ">           div(*pair)\n"
    "\n"
    "test_calc.py:12: \n"
    "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \n"
    "\n"
    "a = 1, b = 0\n"
    "\n"
    "    def div(a, b):\n"
    ">       return a / b\n"
    "               ^^^^^\n"
    "E       ZeroDivisionError: division by zero\n"
    "\n"
    "calc.py:2: ZeroDivisionError\n"
    "___________________________ test_rejects_zero[pair1] ___________________________\n"
    "[gw0] darwin -- Python 3.12.14 /repo/.venv/bin/python\n"
    "\n"
    "pair = (5, 0)\n"
    "\n"
    '    @pytest.mark.parametrize("pair", [(1, 0), (5, 0)])\n'
    "    def test_rejects_zero(pair):\n"
    '        with pytest.raises(ValueError, match="divide by zero"):\n'
    ">           div(*pair)\n"
    "\n"
    "test_calc.py:12: \n"
    "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \n"
    "\n"
    "a = 5, b = 0\n"
    "\n"
    "    def div(a, b):\n"
    ">       return a / b\n"
    "               ^^^^^\n"
    "E       ZeroDivisionError: division by zero\n"
    "\n"
    "calc.py:2: ZeroDivisionError\n"
    "=========================== short test summary info ============================\n"
    "FAILED test_calc.py::test_rejects_zero[pair0] - ZeroDivisionError: division b...\n"
    "FAILED test_calc.py::test_rejects_zero[pair1] - ZeroDivisionError: division b...\n"
    "========================= 2 failed, 1 passed in 0.42s ==========================\n"
)

_SCENARIO_POST = (
    "============================= test session starts ==============================\n"
    "platform darwin -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0 -- /repo/.venv/bin/python\n"
    "rootdir: /repo\n"
    "plugins: xdist-3.8.0\n"
    "created: 2/2 workers\n"
    "2 workers [3 items]\n"
    "\n"
    "scheduling tests via LoadScheduling\n"
    "\n"
    "test_calc.py::test_rejects_zero[pair0] \n"
    "test_calc.py::test_divides \n"
    "[gw1] [ 33%] PASSED test_calc.py::test_rejects_zero[pair0] \n"
    "[gw0] [ 66%] PASSED test_calc.py::test_divides \n"
    "test_calc.py::test_rejects_zero[pair1] \n"
    "[gw0] [100%] PASSED test_calc.py::test_rejects_zero[pair1] \n"
    "\n"
    "============================== 3 passed in 0.42s ===============================\n"
)


def test_distributed_fix_yields_an_oracle_and_grades_it(tmp_path: Path):
    # [FUNCTION DOC] Derives F2P/P2P from captured xdist pre/post logs, asserts the expected sets, then runs a copy
    # of the verifier isolated (with --test-cmds 'pytest -v -n 2') and asserts reward 1.0, resolved,
    # f2p_passed 2. Side effects: files in tmp_path, subprocess.
    pre, post = parse_pytest(_SCENARIO_PRE), parse_pytest(_SCENARIO_POST)
    f2p = sorted(name for name, st in pre.items() if st == "FAILED" and post.get(name) == "PASSED")
    p2p = sorted(name for name, st in pre.items() if st == "PASSED" and post.get(name) == "PASSED")
    assert f2p == [
        "test_calc.py::test_rejects_zero[pair0]",
        "test_calc.py::test_rejects_zero[pair1]",
    ]
    assert p2p == ["test_calc.py::test_divides"]

    # Run the verifier in isolation, with no installed repo2rlenv imports.
    standalone = tmp_path / "verifier.py"
    standalone.write_text(Path(verifier.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "out.log").write_text(_SCENARIO_POST, encoding="utf-8")
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
            "pytest -v -n 2",
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
    assert details["f2p_passed"] == 2
