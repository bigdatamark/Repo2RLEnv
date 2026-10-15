# [MODULE DOC] Private pytest entry point copied into SWE-smith verifier containers. It snapshots
# every callable unittest.TestCase attribute whose name starts with "assert" (both the function
# object and its __code__) before running pytest, runs pytest with repository addopts cleared, then
# detects any replacement or in-place code mutation of those assertion helpers (or of the TestCase
# class binding). Tampering yields exit code 254 plus a stderr message; otherwise pytest's exit code
# is returned. The host still parses rewards; this is a bounded anti-tampering guard, not full
# isolation. Running as a script passes argv[1:] to run() and exits with its result.
"""Private pytest entry point with a bounded framework-tampering check.

The host still owns reward parsing. This guard detects replacement or in-place
editing of unittest assertions, a demonstrated attack against repository tests.
It does not claim to isolate arbitrary test and solution code in one interpreter.
"""

from __future__ import annotations

import sys
import unittest


def run(arguments: list[str]) -> int:
    # [FUNCTION DOC] Run pytest with the given arguments and verify the unittest assertion framework
    # was not tampered with. Inputs: arguments (list of str) - pytest selectors/options appended
    # after "-o addopts=" (which neutralizes repository addopts such as coverage plugins). Output:
    # int exit code - pytest's own code, or 254 if any captured assert* method object or its
    # __code__ changed, or the module's unittest.TestCase no longer refers to the captured class.
    # Side effects: imports pytest lazily (after the snapshot), runs tests in-process, writes the
    # list of changed names to stderr on tampering. Raises: whatever pytest.main propagates
    # (normally nothing).
    import pytest

    # Capture before pytest imports any repository modules. Retain code objects
    # too: an attacker can mutate function.__code__ without replacing a method.
    original = {
        name: (value, getattr(value, "__code__", None))
        for name, value in vars(unittest.TestCase).items()
        if name.startswith("assert") and callable(value)
    }
    case = unittest.TestCase
    # The harness supplies selectors and reporting explicitly. Repository addopts
    # often enable optional coverage/report plugins that are intentionally not
    # auto-loaded in this isolated verifier; they are not test dependencies.
    code = int(pytest.main(["-o", "addopts=", *arguments]))
    changed = []
    for name, (method, implementation) in original.items():
        current = getattr(case, name, None)
        if current is not method or getattr(current, "__code__", None) is not implementation:
            changed.append(name)
    if unittest.TestCase is not case or changed:
        sys.stderr.write("Verifier framework was modified: " + ", ".join(changed) + "\n")
        return 254
    return code


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
