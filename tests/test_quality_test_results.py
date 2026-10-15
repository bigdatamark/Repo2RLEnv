# [MODULE DOC] Tests for quality.test_results.parse_junit and execution_contrast: empty or contradictory JUnit
# reports are rejected and FAIL_TO_PASS/PASS_TO_PASS contrasts require identical collection.
from __future__ import annotations

import pytest

from repo2rlenv.quality.test_results import execution_contrast, parse_junit


def xml(failure="", extra=""):
    # [FUNCTION DOC] HELPER: Returns a JUnit testsuite XML string with testcase test_library::behavior containing
    # the optional failure markup, followed by extra markup.
    return f'<testsuite><testcase classname="test_library" name="behavior">{failure}</testcase>{extra}</testsuite>'


def test_empty_and_forged_success_are_rejected():
    # [FUNCTION DOC] Verifies parse_junit rejects an empty suite, a failure with returncode 0 ('contradict'), and a
    # clean report with returncode 2 ('normally').
    with pytest.raises(ValueError, match="Empty"):
        parse_junit("<testsuite/>", returncode=0)
    with pytest.raises(ValueError, match="contradict"):
        parse_junit(xml("<failure/>"), returncode=0)
    with pytest.raises(ValueError, match="normally"):
        parse_junit(xml(), returncode=2)


def test_missing_collection_cannot_count_as_a_mutation():
    # [FUNCTION DOC] Verifies execution_contrast raises ('collection') when the defective run collected fewer
    # tests.
    healthy = parse_junit(
        xml(extra='<testcase classname="test_library" name="regression"/>'), returncode=0
    )
    defective = parse_junit(xml("<failure/>"), returncode=1)
    with pytest.raises(ValueError, match="collection"):
        execution_contrast(healthy, defective)


def test_valid_contrast_keeps_regressions():
    # [FUNCTION DOC] Verifies a valid contrast returns FAIL_TO_PASS behavior and PASS_TO_PASS regression.
    extra = '<testcase classname="test_library" name="regression"/>'
    healthy = parse_junit(xml(extra=extra), returncode=0)
    defective = parse_junit(xml("<failure/>", extra), returncode=1)
    assert execution_contrast(healthy, defective) == {
        "FAIL_TO_PASS": ["test_library::behavior"],
        "PASS_TO_PASS": ["test_library::regression"],
    }
