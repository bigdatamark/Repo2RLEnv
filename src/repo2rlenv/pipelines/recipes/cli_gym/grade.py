# [MODULE DOC] Trusted verifier entrypoint copied into each CLI-Gym task as /tests/grade.py and run
# inside the task container. Writes reward 0.0 first, then awards 1.0 only if protected repository
# sources are unchanged, the original tests (restored from /tests/repository-tests) run under pytest
# and every required healthy test id passes with exit code 0. Standalone: uses only the stdlib plus
# the sibling /tests/test_results.py loaded by path.
"""Trusted entrypoint: restore original tests and require nonempty healthy test IDs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # [FUNCTION DOC] Grades the task. Creates /logs/verifier and writes reward.txt = 0.0. Loads
    # /tests/test_results.py as module test_results (registers it in sys.modules) and reads
    # /tests/contract.json. For each protected_source entry, if /workspace/<path> is missing or its
    # sha256 differs, writes reason.txt and returns. Replaces /workspace/tests (symlink, file or
    # dir) with a copy of /tests/repository-tests, runs "python -m pytest tests -q --tb=short
    # --junitxml= /logs/verifier/results.xml" in /workspace with output to test-output.txt and the
    # contract timeout; returns (reward 0) on OSError/timeout, missing junit file or parse
    # ValueError. Writes 1.0 when required tests are nonempty, the parsed returncode is 0 and
    # required is a subset of the passed ids. Output: None. Side effects: filesystem writes in
    # /logs/verifier and /workspace/tests, subprocess.
    output = Path("/logs/verifier")
    output.mkdir(parents=True, exist_ok=True)
    reward = output / "reward.txt"
    reward.write_text("0.0\n")
    spec = importlib.util.spec_from_file_location("test_results", "/tests/test_results.py")
    parser = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parser
    spec.loader.exec_module(parser)
    contract = json.loads(Path("/tests/contract.json").read_text())
    for relative, digest in contract["protected_source"].items():
        path = Path("/workspace") / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            (output / "reason.txt").write_text(
                "Repository source changed; this is an environment repair task"
            )
            return
    tests = Path("/workspace/tests")
    if tests.is_symlink() or tests.is_file():
        tests.unlink()
    elif tests.exists():
        shutil.rmtree(tests)
    shutil.copytree("/tests/repository-tests", tests)
    results = output / "results.xml"
    with (output / "test-output.txt").open("w") as log:
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests", "-q", "--tb=short", f"--junitxml={results}"],
                cwd="/workspace",
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=contract["timeout_sec"],
            )
        except (OSError, subprocess.TimeoutExpired):
            return
    if not results.is_file():
        return
    try:
        parsed = parser.parse_junit(results.read_text(), returncode=result.returncode)
    except ValueError:
        return
    required = set(contract["required_tests"])
    if required and parsed.returncode == 0 and required <= parsed.passed:
        reward.write_text("1.0\n")


if __name__ == "__main__":
    main()
