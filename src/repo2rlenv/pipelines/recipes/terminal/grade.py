# [MODULE DOC] Standalone deterministic grader copied into terminal tasks as /tests/grade.py and run
# inside the verifier container (no dependency on repo2rlenv). It runs /tests/test_outputs.py under
# pytest with a timeout from /tests/contract.json, parses the JUnit XML with the co-shipped
# /tests/test_results.py, and writes a binary reward to /logs/verifier/reward.txt plus diagnostics
# to result.json.
"""Standalone deterministic terminal-task reward; no runtime dependency on us."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # [FUNCTION DOC] Grade the task. Steps: create /logs/verifier, pre-write reward.txt as "0"; load
    # contract.json; dynamically import /tests/test_results.py as module "test_results"; run
    # isolated pytest on /tests/test_outputs.py (-q, short tracebacks, cache provider disabled,
    # plugin autoload disabled, JUnit to /logs/verifier/results.xml) in a new process session with
    # stdout/stderr captured to stdout.txt/stderr.txt; wait up to contract timeout_sec. On timeout
    # writes result.json {passed: false, reason: timeout} and returns (reward stays 0). Always
    # SIGKILLs the process group afterwards. Otherwise parses JUnit; passes only if the exit code is
    # 0, the set of observed test base names (last "::" segment, parametrization stripped) equals
    # contract test_names, and every reported test passed. Writes result.json (passed, returncode,
    # statuses) or (passed false, reason) on OSError/ValueError, then reward "1" or "0". Output:
    # None. Side effects: subprocess, filesystem.
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    reward = logs / "reward.txt"
    reward.write_text("0\n")
    contract = json.loads(Path("/tests/contract.json").read_text())
    spec = importlib.util.spec_from_file_location("test_results", "/tests/test_results.py")
    parser = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parser
    spec.loader.exec_module(parser)
    report = logs / "results.xml"
    with (logs / "stdout.txt").open("w") as stdout, (logs / "stderr.txt").open("w") as stderr:
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-m",
                "pytest",
                "/tests/test_outputs.py",
                "-q",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                f"--junitxml={report}",
            ],
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        )
        try:
            code = process.wait(timeout=contract["timeout_sec"])
        except subprocess.TimeoutExpired:
            (logs / "result.json").write_text(json.dumps({"passed": False, "reason": "timeout"}))
            return
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    try:
        parsed = parser.parse_junit(report.read_text(), returncode=code)
        observed = {name.rsplit("::", 1)[-1].split("[", 1)[0] for name in parsed.statuses}
        passed = (
            code == 0
            and observed == set(contract["test_names"])
            and parsed.passed == set(parsed.statuses)
        )
        result = {"passed": passed, "returncode": code, "statuses": parsed.statuses}
    except (OSError, ValueError) as exc:
        passed = False
        result = {"passed": False, "reason": str(exc)}
    (logs / "result.json").write_text(json.dumps(result))
    reward.write_text("1\n" if passed else "0\n")


if __name__ == "__main__":
    main()
