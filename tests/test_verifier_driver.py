# [MODULE DOC] Tests for the swe_smith test_driver.run guard: if the test run tampers with
# unittest assertion methods (replacement or code swap), the driver returns 254 regardless of the
# apparent pytest result; otherwise pytest's return code passes through.
from __future__ import annotations

import unittest

import pytest

from repo2rlenv.pipelines.recipes.swe_smith.test_driver import run


@pytest.mark.parametrize("mode", ["replacement", "code"])
def test_framework_tampering_overrides_an_apparently_passing_suite(monkeypatch, mode):
    # [FUNCTION DOC] Parametrized
    # replacement/code tampering inside a fake pytest.main returning 0 makes run return 254.
    def passing_but_tampered(arguments):
        # [FUNCTION DOC] HELPER:
        # Fake pytest.main that tampers with TestCase.assertLess (by replacement or by swapping its
        # code
        # object) and returns 0.
        if mode == "replacement":
            monkeypatch.setattr(unittest.TestCase, "assertLess", lambda *args: None)
        else:
            monkeypatch.setattr(
                unittest.TestCase.assertLess, "__code__", (lambda *args: None).__code__
            )
        return 0

    monkeypatch.setattr(pytest, "main", passing_but_tampered)
    assert run(["trusted-fixture"]) == 254


@pytest.mark.parametrize("returncode", [0, 1, 2, 5])
def test_guard_preserves_unmodified_test_execution_outcome(monkeypatch, returncode):
    # [FUNCTION DOC] Parametrized return codes pass
    # through unchanged when no tampering occurs.
    monkeypatch.setattr(pytest, "main", lambda arguments: returncode)
    assert run(["trusted-fixture"]) == returncode
