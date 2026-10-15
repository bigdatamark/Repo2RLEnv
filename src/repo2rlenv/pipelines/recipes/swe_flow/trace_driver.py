# [MODULE DOC] Standalone pytest plugin + runner copied into the repository container. Uses
# sys.setprofile to record, per test, which in-repo functions were called (core nodes), the call
# edges between them, and which were called directly from test files (targets). Writes
# /tmp/traces.json and exits with pytest's code. argv[1] is a JSON settings object (source_paths,
# max_tests, seed); remaining args go to pytest.
"""Standalone pytest call tracer, copied into a remote repository container."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import pytest


class CallTrace:
    # [CLASS DOC] Pytest plugin object holding trace state: roots (/workspace/<source_path>),
    # settings, current test id, records per test, and a cache of code-object identities.
    def __init__(self, settings: dict):
        # [FUNCTION DOC] Store settings and initialise roots, current (None), records and
        # identities. Input: settings dict with source_paths (also seed, max_tests used later).
        self.roots = [Path("/workspace") / root for root in settings["source_paths"]]
        self.settings = settings
        self.current = None
        self.records = {}
        self.identities = {}

    def identity(self, code):
        # [FUNCTION DOC] Map a code object to "relative_path:first_line:name" if its file lies under
        # a source root, else None. Results are cached by id(code), except that names starting with
        # "<" (lambdas, comprehensions) return None without caching.
        code_id = id(code)
        if code_id in self.identities:
            return self.identities[code_id]
        path = Path(code.co_filename)
        if not path.is_absolute() or not any(
            path == root or root in path.parents for root in self.roots
        ):
            self.identities[code_id] = None
            return None
        if code.co_name.startswith("<"):
            return None
        result = f"{path.relative_to('/workspace')}:{code.co_firstlineno}:{code.co_name}"
        self.identities[code_id] = result
        return result

    def profile(self, frame, event, arg):
        # [FUNCTION DOC] sys.setprofile callback. Ignores non-"call" events and calls outside a
        # test. Every 10000 calls checks the 5-second per-test deadline; when exceeded marks the
        # record truncated and disables profiling. Adds in-repo callees to core_nodes; adds a
        # (caller, callee) edge when the caller is in-repo, otherwise marks the callee a target when
        # the caller's file path contains "/tests/".
        if event != "call" or self.current is None:
            return
        self.calls += 1
        if self.calls % 10000 == 0 and time.monotonic() >= self.deadline:
            self.records[self.current]["truncated"] = True
            sys.setprofile(None)
            return
        callee = self.identity(frame.f_code)
        if callee is None:
            return
        record = self.records[self.current]
        record["core_nodes"].add(callee)
        caller = self.identity(frame.f_back.f_code) if frame.f_back else None
        if caller:
            record["edges"].add((caller, callee))
        elif frame.f_back and "/tests/" in frame.f_back.f_code.co_filename:
            record["target_nodes"].add(callee)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        # [FUNCTION DOC] Hookwrapper around each test: creates a fresh record, resets the call
        # counter and deadline (now + 5s), installs profile for the test's duration and restores the
        # previous profiler afterwards. Inputs: item, nextitem. Yields to pytest.
        self.current = item.nodeid
        self.records[self.current] = {
            "core_nodes": set(),
            "target_nodes": set(),
            "edges": set(),
            "truncated": False,
        }
        self.calls = 0
        self.deadline = time.monotonic() + 5
        previous = sys.getprofile()
        sys.setprofile(self.profile)
        try:
            yield
        finally:
            sys.setprofile(previous)
            self.current = None

    def pytest_collection_modifyitems(self, session, config, items):
        # [FUNCTION DOC] Deterministically sample at most max_tests items (shuffle by seed), report
        # the rest via pytest_deselected, and filter items in place preserving original order.
        selected = list(items)
        random.Random(self.settings["seed"]).shuffle(selected)
        selected = set(selected[: self.settings["max_tests"]])
        config.hook.pytest_deselected(items=[item for item in items if item not in selected])
        items[:] = [item for item in items if item in selected]


def main() -> None:
    # [FUNCTION DOC] Run pytest with a CallTrace plugin, dump records (sets converted to sorted
    # lists) with test_id to /tmp/traces.json, then raise SystemExit with pytest's exit code.
    tracer = CallTrace(json.loads(sys.argv[1]))
    code = pytest.main(sys.argv[2:], plugins=[tracer])
    Path("/tmp/traces.json").write_text(
        json.dumps(
            [
                {
                    "test_id": key,
                    **{
                        name: sorted(value) if isinstance(value, set) else value
                        for name, value in record.items()
                    },
                }
                for key, record in tracer.records.items()
            ],
            sort_keys=True,
        )
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
