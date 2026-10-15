# [MODULE DOC] Tests for campaigns.llm.metered_complete: LLM calls reserve budget in a
# BudgetLedger, write a response receipt file, keep the hold on uncertain outcomes, settle only on
# evidenced non- zero cost, persist nested usage, and support exact-request resume.
# campaigns.llm.complete is monkeypatched.
from __future__ import annotations

import pytest
from pydantic import BaseModel

from repo2rlenv.campaigns.budget import BudgetLedger, OperationAlreadyRecorded
from repo2rlenv.campaigns.llm import metered_complete
from repo2rlenv.llm import LLMResponse
from repo2rlenv.spec.input import LLMSpec


def call(tmp_path, ledger):
    # [FUNCTION DOC] HELPER: Invokes metered_complete with a test LLMSpec, operation_id 'issue:1',
    # reservation 0.50, receipt tmp_path/response.json and fixed prompts. Output: the LLMResponse;
    # propagates errors.
    return metered_complete(
        LLMSpec(provider="test", model="test"),
        ledger=ledger,
        operation_id="issue:1",
        reservation_usd="0.50",
        receipt=tmp_path / "response.json",
        system="system",
        user="user",
        max_tokens=100,
    )


def test_request_timeout_holds_budget_and_prevents_redispatch(tmp_path, monkeypatch):
    # [FUNCTION DOC] Asserts a TimeoutError leaves 0.50 reserved with status 'uncertain'; a retry
    # raises FileExistsError (receipt exists) and, after deleting the receipt,
    # OperationAlreadyRecorded.
    ledger = BudgetLedger(tmp_path / "budget.db", limit_usd="1")

    def timeout(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Fake complete that always raises TimeoutError.
        raise TimeoutError("Unknown provider outcome")

    monkeypatch.setattr("repo2rlenv.campaigns.llm.complete", timeout)
    with pytest.raises(TimeoutError):
        call(tmp_path, ledger)
    assert ledger.status()["reserved_usd"] == "0.500000"
    assert ledger.status()["operations"][0]["status"] == "uncertain"
    with pytest.raises(FileExistsError):
        call(tmp_path, ledger)
    (tmp_path / "response.json").unlink()
    with pytest.raises(OperationAlreadyRecorded):
        call(tmp_path, ledger)


@pytest.mark.parametrize(
    "cost,usage,settled",
    [(0.02, {"prompt_tokens": 2}, True), (0, {"prompt_tokens": 2}, False), (0.02, None, False)],
)
def test_only_evidenced_nonzero_estimates_release_reservation(
    tmp_path, monkeypatch, cost, usage, settled
):
    # [FUNCTION DOC] Parametrized over cost/usage: asserts the operation is settled (accounted
    # 0.02) only when cost is non-zero and usage is present; otherwise stays 'uncertain' with 0
    # accounted.
    ledger = BudgetLedger(tmp_path / "budget.db", limit_usd="1")
    monkeypatch.setattr(
        "repo2rlenv.campaigns.llm.complete",
        lambda *a, **kw: LLMResponse("result", usage=usage, cost_usd=cost),
    )
    assert call(tmp_path, ledger).content == "result"
    status = ledger.status()
    assert status["operations"][0]["status"] == ("settled" if settled else "uncertain")
    assert status["accounted_usd"] == ("0.020000" if settled else "0.000000")


def test_nested_provider_usage_is_persisted_without_losing_the_response(tmp_path, monkeypatch):
    # [FUNCTION DOC] Asserts usage containing a Pydantic model is serialized into the receipt JSON
    # alongside the response content.
    import json

    class TokenDetails(BaseModel):
        # [CLASS DOC] HELPER: Pydantic model with reasoning_tokens, standing in for nested provider
        # usage details.
        reasoning_tokens: int

    ledger = BudgetLedger(tmp_path / "budget.db", limit_usd="1")
    response = LLMResponse(
        "report",
        usage={"completion_tokens_details": TokenDetails(reasoning_tokens=30)},
        cost_usd=0.02,
    )
    monkeypatch.setattr("repo2rlenv.campaigns.llm.complete", lambda *a, **kw: response)
    call(tmp_path, ledger)
    data = json.loads((tmp_path / "response.json").read_text())
    assert data["response"]["usage"]["completion_tokens_details"]["reasoning_tokens"] == 30
    assert data["response"]["content"] == "report"


def test_resume_reuses_only_the_exact_request_without_charging_again(tmp_path, monkeypatch):
    # [FUNCTION DOC] Asserts resume=True returns the stored response without a second provider call
    # or charge, and raises ValueError ('different request') when the request differs.
    ledger = BudgetLedger(tmp_path / "budget.db", limit_usd="1")
    calls = []

    def complete(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Fake complete recording kwargs and returning a report with usage
        # and cost 0.02.
        calls.append(kwargs)
        return LLMResponse("report", usage={"prompt_tokens": 20}, cost_usd=0.02)

    monkeypatch.setattr("repo2rlenv.campaigns.llm.complete", complete)
    kwargs = dict(
        ledger=ledger,
        operation_id="resume:1",
        reservation_usd="0.50",
        receipt=tmp_path / "response.json",
        system="system",
        user="user",
        max_tokens=100,
    )
    model = LLMSpec(provider="test", model="test")
    metered_complete(model, **kwargs)
    assert metered_complete(model, **kwargs, resume=True).content == "report"
    assert len(calls) == 1
    assert ledger.status()["accounted_usd"] == "0.020000"
    kwargs["user"] = "changed"
    with pytest.raises(ValueError, match="different request"):
        metered_complete(model, **kwargs, resume=True)
