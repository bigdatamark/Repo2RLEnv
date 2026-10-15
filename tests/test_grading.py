# [MODULE DOC] Tests for reward.py:grade_test_execution (SWE-bench-style FAIL_TO_PASS /
# PASS_TO_PASS grading) and the ExecutionReport it returns: rates, FULL/PARTIAL/NO resolution and
# per-test success/failure lists.
"""SWE-bench-style FAIL_TO_PASS / PASS_TO_PASS grading."""

from __future__ import annotations

from repo2rlenv.reward import ExecutionReport, grade_test_execution


def test_all_pass_yields_full_resolution():
    # [FUNCTION DOC] Asserts all F2P and P2P passing gives rates 1.0 and FULL.
    report = grade_test_execution(
        fail_to_pass=["t1", "t2"],
        pass_to_pass=["t3"],
        test_status={"t1": "PASSED", "t2": "PASSED", "t3": "PASSED"},
    )
    assert report.f2p_rate == 1.0
    assert report.p2p_rate == 1.0
    assert report.resolution_status == "FULL"


def test_partial_f2p_with_full_p2p_is_partial():
    # [FUNCTION DOC] Asserts half of F2P passing with all P2P passing gives f2p 0.5 and PARTIAL.
    report = grade_test_execution(
        fail_to_pass=["t1", "t2"],
        pass_to_pass=["t3"],
        test_status={"t1": "PASSED", "t2": "FAILED", "t3": "PASSED"},
    )
    assert report.f2p_rate == 0.5
    assert report.p2p_rate == 1.0
    assert report.resolution_status == "PARTIAL"


def test_failing_p2p_breaks_resolution():
    # [FUNCTION DOC] Asserts any P2P failure gives NO even with full F2P.
    """Any P2P failure (regression) means NO, even if F2P succeeds fully."""
    report = grade_test_execution(
        fail_to_pass=["t1"],
        pass_to_pass=["t2"],
        test_status={"t1": "PASSED", "t2": "FAILED"},
    )
    assert report.f2p_rate == 1.0
    assert report.p2p_rate == 0.0
    assert report.resolution_status == "NO"


def test_missing_test_counts_as_failure():
    # [FUNCTION DOC] Asserts an F2P test absent from test_status counts as failed (rate 0, NO).
    """Tests not present in the parser output count as failures (silent skip)."""
    report = grade_test_execution(
        fail_to_pass=["t1"],
        pass_to_pass=[],
        test_status={},  # the test runner crashed before any results
    )
    assert report.f2p_rate == 0.0
    assert report.resolution_status == "NO"


def test_empty_sets_grade_as_full_by_default():
    # [FUNCTION DOC] Asserts empty F2P and P2P sets grade vacuously as rates 1.0 and FULL.
    """No F2P + no P2P ⇒ vacuously full. SWE-bench's grading does the same."""
    report = grade_test_execution(
        fail_to_pass=[],
        pass_to_pass=[],
        test_status={"t1": "PASSED"},
    )
    assert report.f2p_rate == 1.0
    assert report.p2p_rate == 1.0
    assert report.resolution_status == "FULL"


def test_report_dataclass_shape():
    # [FUNCTION DOC] Asserts the ExecutionReport lists split tests into F2P/P2P success and
    # failure.
    """Caller (Harbor verifier) reads the four lists for per-test reporting."""
    report = grade_test_execution(
        fail_to_pass=["a", "b"],
        pass_to_pass=["c"],
        test_status={"a": "PASSED", "b": "FAILED", "c": "PASSED"},
    )
    assert isinstance(report, ExecutionReport)
    assert report.fail_to_pass_success == ["a"]
    assert report.fail_to_pass_failure == ["b"]
    assert report.pass_to_pass_success == ["c"]
    assert report.pass_to_pass_failure == []
