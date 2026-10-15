# [MODULE DOC] SWE-smith issue (bug report) generation from failing-test evidence through the shared
# metered LLM client, with deterministic validation (no private test names, no mention of the test
# suite, Python examples must parse and not use undefined names) and bounded retries.
"""SWE-smith's test-evidence issue generation, through the shared metered client."""

from __future__ import annotations

import ast
import json
import re
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.campaigns.llm import metered_complete
from repo2rlenv.quality.authoring_context import bounded_context
from repo2rlenv.quality.python_evidence import test_excerpts
from repo2rlenv.spec.input import LLMSpec


class IssueReport(BaseModel):
    # [CLASS DOC] Pydantic model (extra forbidden): issue (markdown text, may be empty when
    # unsupported) and reason.
    model_config = ConfigDict(extra="forbid")
    issue: str
    reason: str


def _reproducer_errors(markdown: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Statically check python/py fenced code blocks in markdown. Reports
    # syntax errors and pyflakes UndefinedName/UndefinedLocal messages, without executing code.
    # Input: markdown. Output: list of error strings.
    """Check self-contained Python examples without executing generated code."""
    from pyflakes.checker import Checker
    from pyflakes.messages import UndefinedLocal, UndefinedName

    errors = []
    blocks = re.findall(r"^```(?:python|py)\s*\n(.*?)^```\s*$", markdown, re.M | re.S)
    for index, source in enumerate(blocks, start=1):
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            errors.append(f"Python example {index} has invalid syntax on line {exc.lineno}")
            continue
        for message in Checker(tree).messages:
            if isinstance(message, (UndefinedName, UndefinedLocal)):
                errors.append(
                    f"Python example {index}, line {message.lineno}: "
                    + message.message % message.message_args
                )
    return errors


def issue_violations(report: IssueReport, candidate: dict) -> list[str]:
    # [FUNCTION DOC] List validation problems for an IssueReport. Checks: empty issue (includes
    # reason), over 4000 chars, mentions of private test function/class names derived from
    # FAIL_TO_PASS ids (parts starting with test_ or ending with Tests), test-referring phrases, and
    # reproducer errors. Inputs: report, candidate (contrast). Output: list of strings (empty =
    # valid).
    issues = []
    text = report.issue
    if not text.strip():
        issues.append("No supported issue: " + report.reason)
    if len(text) > 4000:
        issues.append("Use a concise bug report under 4000 characters")
    names = {
        part
        for identity in candidate["contrast"]["FAIL_TO_PASS"]
        for part in identity.replace("::", ".").split(".")
        if part.startswith("test_") or part.endswith("Tests")
    }
    if any(name in text for name in names):
        issues.append("Remove private test function and class names")
    if any(
        phrase in text.lower()
        for phrase in (
            "test suite",
            "test-suite",
            "failing test",
            "failed test",
            "tests assert",
            "test shows",
            "test evidence",
        )
    ):
        issues.append("Describe observable behavior without referring to the supplied tests")
    issues.extend(_reproducer_errors(text))
    return issues


def issue_context(generation: Path, candidate: dict) -> str:
    # [FUNCTION DOC] Build bounded LLM context for a candidate: up to 24 unique failing test methods
    # (parametrization stripped) with source excerpts from generation/base, the defective run's
    # stdout log, and counts. Inputs: generation, candidate. Output: str from bounded_context.
    """Select failing test methods and imports without executing repository code."""
    base = generation / "base"
    identities = candidate["contrast"]["FAIL_TO_PASS"]
    methods = list(dict.fromkeys(identity.split("[", 1)[0] for identity in identities))
    excerpts = test_excerpts(base, methods[:24])
    log = (generation / "candidates" / candidate["id"] / "defective" / "stdout.txt").read_text()
    return bounded_context(
        {
            "test_source": excerpts,
            "test_execution": log,
            "failing_instances": len(identities),
            "failing_methods": len(methods),
            "sampled_methods": min(24, len(methods)),
        }
    )


def write_issue(
    generation: Path,
    candidate: dict,
    model: LLMSpec,
    ledger: BudgetLedger,
    receipt: Path,
    *,
    operation_id: str,
    reservation_usd: str = "0.50",
    max_attempts: int = 2,
    review_feedback: list[str] | None = None,
    resume: bool = False,
) -> IssueReport:
    # [FUNCTION DOC] Generate a validated IssueReport with up to max_attempts LLM calls.
    # Inputs: generation, candidate, model (LLMSpec), ledger, receipt (Path), keyword operation_id,
    # reservation_usd ("0.50"), max_attempts (1-3, default 2), review_feedback (optional list added
    # to the prompt), resume. Later attempts use suffixed operation ids and receipt stems and
    # include the previous validation errors (or a JSON-format reminder on parse failure) as
    # feedback.
    # Output: the first violation-free IssueReport. Raises ValueError if max_attempts is out of
    # range or all attempts fail. Side effects: metered LLM calls and receipts.
    context = issue_context(generation, candidate)
    if max_attempts not in range(1, 4):
        raise ValueError("Issue generation allows one to three explicit attempts")
    feedback = (
        "\nAddress these review findings: " + json.dumps(review_feedback) if review_feedback else ""
    )
    for attempt in range(1, max_attempts + 1):
        response = metered_complete(
            model,
            ledger=ledger,
            operation_id=operation_id if attempt == 1 else f"{operation_id}:attempt-{attempt}",
            reservation_usd=reservation_usd,
            receipt=receipt
            if attempt == 1
            else receipt.with_stem(f"{receipt.stem}-attempt-{attempt}"),
            system=files(__package__).joinpath("issue_prompt.md").read_text()
            + (
                "\nEXPANSION QUALITY CONTRACT: Prefer a short developer bug report about "
                "observable behavior. Do not invent a standalone reproducer by substituting "
                "values into a stateful test: earlier operations may have changed that state. "
                "Only give concrete input/output pairs when the supplied evidence establishes "
                "that exact pair. Otherwise describe the invariant, such as matching Python "
                "list indexing, without guessed numbers or comments claiming an observed result. "
                "Do not prescribe the implementation, helper calls or exact patch."
            ),
            user=context + feedback,
            max_tokens=4096,
            response_schema=IssueReport.model_json_schema(),
            resume=resume,
        )
        try:
            result = IssueReport.model_validate_json(response.content)
            violations = issue_violations(result, candidate)
            if not violations:
                return result
            feedback = (
                "\nRevise your previous response to address these validation errors: "
                + json.dumps(violations)
            )
        except ValueError:
            feedback = (
                "\nReturn a valid JSON object with exactly the string fields issue and reason."
            )
    raise ValueError("Issue generation exhausted its bounded attempts: " + feedback)
