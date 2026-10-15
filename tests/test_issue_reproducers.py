# [MODULE DOC] Tests for swe_smith issue_violations: reproducer code blocks in generated issue
# reports are statically checked (pyflakes, never executed) and leaked private-test evidence is
# rejected. Skipped entirely if pyflakes is not installed.
from __future__ import annotations

import pytest

pytest.importorskip("pyflakes")

from repo2rlenv.pipelines.recipes.swe_smith.issue import IssueReport, issue_violations


def violations(issue):
    # [FUNCTION DOC] HELPER: Wraps the issue text in an IssueReport (fixed reason 'Observed
    # behavior') and runs issue_violations against a fixed contrast with one FAIL_TO_PASS test id.
    # Inputs: issue (str markdown). Output: list of violation strings (empty = acceptable). Side
    # effects: none.
    return issue_violations(
        IssueReport(issue=issue, reason="Observed behavior"),
        {"contrast": {"FAIL_TO_PASS": ["tests.test_api::test_result"]}},
    )


def test_reproducer_typo_is_rejected_before_export():
    # [FUNCTION DOC] Asserts a python block referencing an undefined name yields exactly one
    # violation mentioning "undefined name 'apprx'".
    result = violations('```python\nimport math\napprox = math.pi\nprint(f"{apprx}")\n```')
    assert len(result) == 1
    assert "undefined name 'apprx'" in result[0]


def test_valid_reproducer_is_parsed_without_importing_or_executing_it():
    # [FUNCTION DOC] Asserts a reproducer importing a nonexistent module (and an unused import)
    # produces no violations, proving it is only parsed, not imported or executed.
    assert not violations(
        "```python\nimport nonexistent_repository_module as api\n"
        "import unused_module\nresult = api.function()\nprint(result)\n```"
    )


def test_invalid_python_syntax_is_actionable():
    # [FUNCTION DOC] Asserts a 'py' fenced block with an unclosed paren produces a violation
    # containing 'invalid syntax on line 1'.
    assert "invalid syntax on line 1" in violations("```py\nvalue = (\n```")[0]


def test_private_test_provenance_is_not_part_of_a_bug_report():
    # [FUNCTION DOC] Asserts prose that reveals private test evidence ('The test evidence requires
    # ...') is flagged with at least one violation.
    assert violations("The test evidence requires less than one percent error.")
