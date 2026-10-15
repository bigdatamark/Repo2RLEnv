# [MODULE DOC] Tests for campaigns.structured.validated_complete: schema-invalid model output is
# repaired by follow-up calls with distinct operation ids, provider errors are not retried, and
# repairs stop at the attempt limit. metered_complete is monkeypatched; no real LLM calls.
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from repo2rlenv.campaigns.structured import validated_complete
from repo2rlenv.pipelines.recipes.terminal.templates import TestProgram as GeneratedTestProgram


def test_invalid_completed_test_program_is_corrected_without_repeating_other_stages(
    monkeypatch, tmp_path
):
    # [FUNCTION DOC] First fake response is an invalid test program, second valid; asserts the valid
    # code is returned, operation ids are initial then ':repair:1', the repair prompt includes
    # validation_error, and a -validation.json receipt is written.
    calls = []
    valid = "\n".join(f"def test_{n}():\n    assert {n} == {n}\n" for n in range(5))

    def complete(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Fake metered_complete recording kwargs; returns invalid JSON code
        # on the first call, valid afterwards.
        calls.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {"code": "def test_one():\ntry:\n" * 8 if len(calls) == 1 else valid}
            )
        )

    monkeypatch.setattr("repo2rlenv.campaigns.structured.metered_complete", complete)
    result = validated_complete(
        GeneratedTestProgram,
        None,
        ledger=None,
        receipt=tmp_path / "initial-model.json",
        operation_id="design:one:initial",
        system="Generate tests",
        payload={"task": "owned fixture"},
        reservation_usd="0.75",
        max_tokens=1000,
        resume=False,
    )
    assert result.code == valid
    assert [call["operation_id"] for call in calls] == [
        "design:one:initial",
        "design:one:initial:repair:1",
    ]
    assert "validation_error" in json.loads(calls[1]["user"])
    assert (tmp_path / "initial-model-validation.json").exists()


def test_provider_timeout_does_not_trigger_a_new_paid_request(monkeypatch, tmp_path):
    # [FUNCTION DOC] A TimeoutError from the provider propagates after exactly one call.
    calls = []

    def complete(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Fake that records the call and raises TimeoutError.
        calls.append(kwargs)
        raise TimeoutError("Uncertain provider outcome")

    monkeypatch.setattr("repo2rlenv.campaigns.structured.metered_complete", complete)
    with pytest.raises(TimeoutError):
        validated_complete(
            GeneratedTestProgram,
            None,
            ledger=None,
            receipt=tmp_path / "model.json",
            operation_id="owned",
            system="test",
            payload={},
            reservation_usd="1",
            max_tokens=1000,
            resume=False,
        )
    assert len(calls) == 1


def test_schema_repairs_stop_at_the_configured_limit(monkeypatch, tmp_path):
    # [FUNCTION DOC] Always-invalid output raises ValueError 'exhausted 3 attempts' after 3 calls.
    calls = []

    def complete(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Fake always returning code with no executable tests.
        calls.append(kwargs)
        return SimpleNamespace(content=json.dumps({"code": "# no executable tests\n" * 8}))

    monkeypatch.setattr("repo2rlenv.campaigns.structured.metered_complete", complete)
    with pytest.raises(ValueError, match="exhausted 3 attempts"):
        validated_complete(
            GeneratedTestProgram,
            None,
            ledger=None,
            receipt=tmp_path / "model.json",
            operation_id="owned",
            system="test",
            payload={},
            reservation_usd="1",
            max_tokens=1000,
            resume=False,
        )
    assert len(calls) == 3
