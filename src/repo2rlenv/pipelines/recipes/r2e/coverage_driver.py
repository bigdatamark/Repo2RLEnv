# [MODULE DOC] In-container pytest driver used as test instrumentation: runs the full pytest
# command while measuring branch coverage of one target source file only during tests from the
# generated test module, writing /tmp/coverage.json and initializing /tmp/observations.json.
"""Measure target branch coverage from generated tests during a full pytest run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import coverage
import pytest


class GeneratedTestCoverage:
    # [CLASS DOC] pytest plugin that starts/stops a branch-coverage collector around each test whose
    # node id belongs to the generated test file.
    def __init__(self, source: str, generated_test_path: str = "tests/test_r2e_generated.py"):
        # [FUNCTION DOC] Inputs: source (repo-relative target path; coverage includes only
        # /workspace/<source>); generated_test_path (default tests/test_r2e_generated.py). Creates an
        # in-memory branch Coverage object with no config file.
        self.generated_test_path = generated_test_path
        self.coverage = coverage.Coverage(
            branch=True, data_file=None, config_file=False, include=["/workspace/" + source]
        )

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        # [FUNCTION DOC] Hookwrapper around each test's run protocol: if the item's file part of nodeid
        # equals generated_test_path, coverage runs for the duration of the test. Yields once.
        measured = item.nodeid.split("::", 1)[0] == self.generated_test_path
        if measured:
            self.coverage.start()
        try:
            yield
        finally:
            if measured:
                self.coverage.stop()


def main() -> None:
    # [FUNCTION DOC] CLI entry: parses SOURCE and --generated-test-path, passes remaining args to
    # pytest.main with the plugin, writes "[]" to /tmp/observations.json first, then writes the JSON
    # coverage report to /tmp/coverage.json (an empty files map if no data). Exits with pytest's code
    # via SystemExit.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("source")
    parser.add_argument("--generated-test-path", default="tests/test_r2e_generated.py")
    args, pytest_args = parser.parse_known_args(sys.argv[1:])
    plugin = GeneratedTestCoverage(args.source, args.generated_test_path)
    Path("/tmp/observations.json").write_text("[]")
    code = pytest.main(pytest_args, plugins=[plugin])
    try:
        plugin.coverage.json_report(outfile="/tmp/coverage.json")
    except coverage.exceptions.NoDataError:
        Path("/tmp/coverage.json").write_text(json.dumps({"files": {}}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
