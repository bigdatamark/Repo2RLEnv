# [MODULE DOC] Standalone verifier script copied into SCALER task test images. Reads
# /workspace/answer.txt, compares it with /tests/reference.json either by a typed contract
# (string/number/array with Decimal tolerances) or by math-verify, and writes
# /logs/verifier/reward.txt (+1 or -1) and result.json. Requires math_verify only in main().
"""Standalone answer-file adapter for SCALER's active math-verify reward route."""

from __future__ import annotations

import json
import re
import stat
from decimal import Decimal, localcontext
from pathlib import Path


def unbox(text: str) -> str:
    # [FUNCTION DOC] Return the content of the last boxed answer, or the stripped text if none.
    # Tracks brace depth from the final occurrence of the boxed marker. Raises ValueError if
    # anything other than whitespace or dollar signs follows the closing brace, or if the box is
    # unclosed.
    """Extract the final boxed answer without treating strings as numbers."""
    text = text.strip()
    start = text.rfind(r"\boxed{")
    if start < 0:
        return text
    first = start + len(r"\boxed{")
    depth = 1
    for index in range(first, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            if text[index + 1 :].strip().strip("$"):
                raise ValueError("Unexpected content after the final answer")
            return text[first:index].strip()
    raise ValueError("Unclosed boxed answer")


def numeric_equal(expected: str, actual: str, contract: dict) -> bool:
    # [FUNCTION DOC] Compare two decimal strings under the contract's tolerances.
    # Inputs: expected, actual (str), contract (absolute_tolerance/relative_tolerance, default 0).
    # Returns False unless both match a plain decimal/scientific regex and all values are finite;
    # uses high-precision Decimal arithmetic. True when |gold-pred| <= max(absolute,
    # relative*|gold|).
    # Raises ValueError on negative tolerances.
    """Compare finite decimal outputs under explicitly declared tolerances."""
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    if not re.fullmatch(number, expected) or not re.fullmatch(number, actual):
        return False
    with localcontext() as context:
        context.prec = max(80, len(expected) + len(actual) + 10)
        gold, pred = Decimal(expected), Decimal(actual)
        absolute = Decimal(str(contract.get("absolute_tolerance", "0")))
        relative = Decimal(str(contract.get("relative_tolerance", "0")))
        if not all(value.is_finite() for value in (gold, pred, absolute, relative)):
            return False
        if absolute < 0 or relative < 0:
            raise ValueError("Negative answer tolerance")
        return abs(gold - pred) <= max(absolute, relative * abs(gold))


def typed_equal(reference: str, answer: str, contract: dict) -> bool:
    # [FUNCTION DOC] Compare a reference and answer under an explicit output_type contract. Both are
    # unboxed first. string: exact equality; number: numeric_equal; array: both parsed as JSON lists
    # with numbers kept as strings, equal length, every pair numeric_equal (non-string elements
    # fail). Output: bool. Raises ValueError for unsupported types; JSON errors propagate.
    expected, actual = unbox(reference), unbox(answer)
    kind = contract["output_type"]
    if kind == "string":
        return expected == actual
    if kind == "number":
        return numeric_equal(expected, actual, contract)
    if kind == "array":
        gold = json.loads(expected, parse_int=str, parse_float=str)
        pred = json.loads(actual, parse_int=str, parse_float=str)
        if not isinstance(gold, list) or not isinstance(pred, list) or len(gold) != len(pred):
            return False
        return all(
            isinstance(left, str)
            and isinstance(right, str)
            and numeric_equal(left, right, contract)
            for left, right in zip(gold, pred, strict=True)
        )
    raise ValueError("Unsupported answer type")


def main() -> None:
    # [FUNCTION DOC] Verifier entry point. Creates /logs/verifier, writes reward "-1" first. Returns
    # early (reward -1) if answer.txt is missing, a symlink, not regular, or over 64 KiB. Otherwise
    # uses typed_equal when answer-contract.json declares output_type, else math-verify (LaTeX gold
    # vs expression/LaTeX prediction). Any exception records score -1 and the error type in
    # result.json. On success writes result.json {score, accuracy} and reward 1 or -1. Output: None.
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    reward = logs / "reward.txt"
    reward.write_text("-1\n")
    path = Path("/workspace/answer.txt")
    if (
        not path.exists()
        or path.is_symlink()
        or not stat.S_ISREG(path.stat().st_mode)
        or path.stat().st_size > 65536
    ):
        return
    verify = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    try:
        answer = path.read_text()
        reference = json.loads(Path("/tests/reference.json").read_text())
        contract_path = Path("/tests/answer-contract.json")
        contract = json.loads(contract_path.read_text()) if contract_path.exists() else {}
        if contract.get("output_type"):
            accuracy = float(typed_equal(reference, answer, contract))
        else:
            accuracy, _ = verify([reference], [answer])
    except (Exception, TimeoutException) as exc:
        (logs / "result.json").write_text(json.dumps({"score": -1, "error": type(exc).__name__}))
        return
    score = 1 if accuracy else -1
    (logs / "result.json").write_text(json.dumps({"score": score, "accuracy": accuracy}))
    reward.write_text(str(score) + "\n")


if __name__ == "__main__":
    main()
