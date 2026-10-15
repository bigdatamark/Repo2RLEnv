# [MODULE DOC] Strict parsing of JUnit XML test reports into per-test statuses, and computation of
# fail-to-pass / pass-to-pass contrasts between a healthy and a defective run. Used by repository
# recipes and their exported verifiers; every inconsistency raises ValueError.
"""Strict test evidence used by repository recipes and their exported verifiers."""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree


@dataclass(frozen=True)
class TestResults:
    # [CLASS DOC] Frozen record of a test run: statuses maps '<classname>::<name>' to one of
    # passed/failed/error/skipped; returncode is the runner's exit code. __test__ = False stops
    # pytest from collecting it.
    __test__ = False
    statuses: dict[str, str]
    returncode: int

    @property
    def passed(self) -> set[str]:
        # [FUNCTION DOC] Property: the set of test identities whose status is 'passed'.
        return {name for name, status in self.statuses.items() if status == "passed"}


def parse_junit(text: str, *, returncode: int) -> TestResults:
    # [FUNCTION DOC] Parses a JUnit XML report into TestResults. Inputs: text (XML), returncode
    # (keyword; runner exit). Raises ValueError if the text contains a DOCTYPE/ENTITY declaration,
    # the returncode is not 0 or 1, the XML is invalid, the root is not testsuite/testsuites, a
    # testcase lacks name or classname, an identity is duplicated, there are no testcases, or
    # returncode is 0 while some test failed/errored. Status precedence: error, failure, skipped,
    # else passed. Output: TestResults.
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("JUnit must not contain a DTD or entity declaration")
    if returncode not in {0, 1}:
        raise ValueError(f"Test runner did not complete normally: exit {returncode}")
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise ValueError("Invalid JUnit report") from exc
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError("Expected a JUnit testsuite or testsuites root")
    statuses = {}
    for case in root.iter("testcase"):
        name, classname = case.get("name"), case.get("classname")
        if not name or not classname:
            raise ValueError("Test cases require a name and class identity")
        identity = f"{classname}::{name}"
        if identity in statuses:
            raise ValueError(f"Duplicate test identity: {identity}")
        if case.find("error") is not None:
            status = "error"
        elif case.find("failure") is not None:
            status = "failed"
        elif case.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"
        statuses[identity] = status
    if not statuses:
        raise ValueError("Empty test result is not execution evidence")
    if returncode == 0 and any(status in {"failed", "error"} for status in statuses.values()):
        raise ValueError("JUnit failures contradict the runner exit code")
    return TestResults(statuses, returncode)


def execution_contrast(healthy: TestResults, defective: TestResults) -> dict[str, list[str]]:
    # [FUNCTION DOC] Computes the F2P/P2P split. Inputs: healthy and defective TestResults. Raises
    # ValueError if the healthy run is not exit 0 with passes, if the two runs collected different
    # test sets, or if no healthy-passing test fails/errors in the defective run or its exit code is
    # not 1. Output: {'FAIL_TO_PASS': sorted newly failing tests, 'PASS_TO_PASS': sorted other
    # healthy passes}.
    if healthy.returncode != 0 or not healthy.passed:
        raise ValueError("Reference must execute successfully with passing tests")
    if healthy.statuses.keys() != defective.statuses.keys():
        raise ValueError("Mutation changed test collection; contrast is not comparable")
    failing = sorted(
        name for name in healthy.passed if defective.statuses[name] in {"failed", "error"}
    )
    if not failing or defective.returncode != 1:
        raise ValueError("No intended fail-to-pass contrast")
    return {"FAIL_TO_PASS": failing, "PASS_TO_PASS": sorted(healthy.passed - set(failing))}
