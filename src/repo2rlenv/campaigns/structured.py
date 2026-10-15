# [MODULE DOC] Structured (Pydantic-schema) LLM completion on top of metered_complete, with bounded
# correction: only a completed response that fails schema or content validation may trigger a new,
# separately metered repair request. Provider failures are never replayed here.
"""Correct completed, invalid structured responses without replaying provider effects."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from repo2rlenv.campaigns.llm import metered_complete
from repo2rlenv.execution.lifecycle import save_record


def validated_complete[Result: BaseModel](
    schema: type[Result],
    model,
    *,
    ledger,
    receipt: Path,
    operation_id: str,
    system: str,
    payload: dict,
    reservation_usd: str,
    max_tokens: int,
    resume: bool,
    max_attempts: int = 3,
    validate: Callable[[Result], None] | None = None,
) -> Result:
    # [FUNCTION DOC] Generic over Result (a BaseModel subclass). Requests JSON matching schema and
    # returns the parsed Result. Inputs: schema (model class), model (LLMSpec), ledger, receipt
    # (Path), operation_id, system prompt, payload (dict serialized as the user message),
    # reservation_usd, max_tokens, resume (passed to metered_complete), max_attempts (1-3, default
    # 3), validate (optional extra callable raising ValueError/SyntaxError on bad content).
    # Attempt 0 uses the given receipt and operation_id; attempt n uses receipt
    # '<stem>-repair-<n>.json' and operation '<id>:repair:<n>', each reserving reservation_usd
    # again. Every call passes schema.model_json_schema() as response_schema. On
    # ValueError/SyntaxError (pydantic ValidationError is a ValueError) the next user message is the
    # original payload under 'input' plus validation_error (truncated to 4000 chars),
    # previous_response and a fixed correction instruction, and a '<stem>-validation.json' record
    # with the error and attempt number is saved.
    # Returns the first valid Result. Raises ValueError if max_attempts is outside 1..3 or all
    # attempts fail (message includes the last error); propagates metered_complete errors. Side
    # effects: LLM calls, ledger writes, receipt files.
    """Only schema/content errors from a completed response authorize correction."""
    if not 1 <= max_attempts <= 3:
        raise ValueError("Structured correction supports one to three attempts")
    feedback = None
    for attempt in range(max_attempts):
        path = (
            receipt if attempt == 0 else receipt.with_name(f"{receipt.stem}-repair-{attempt}.json")
        )
        response = metered_complete(
            model,
            ledger=ledger,
            receipt=path,
            operation_id=operation_id if attempt == 0 else f"{operation_id}:repair:{attempt}",
            reservation_usd=reservation_usd,
            max_tokens=max_tokens,
            resume=resume,
            system=system,
            user=json.dumps(payload if feedback is None else {"input": payload, **feedback}),
            response_schema=schema.model_json_schema(),
        )
        try:
            result = schema.model_validate_json(response.content)
            if validate is not None:
                validate(result)
            return result
        except (ValueError, SyntaxError) as exc:
            feedback = {
                "validation_error": str(exc)[:4000],
                "previous_response": response.content,
                "instruction": "Correct this completed response, preserving its task and behavior. Return complete valid JSON/code. There are at most two correction attempts; fix the specific error rather than redesigning the task.",
            }
            save_record(
                path.with_name(path.stem + "-validation.json"),
                {"error": str(exc), "attempt": attempt + 1},
            )
    raise ValueError(
        f"Structured response exhausted {max_attempts} attempts: {feedback['validation_error']}"
    )
