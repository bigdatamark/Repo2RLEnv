# [MODULE DOC] Metered structured LLM calls and per-run spending inside a shared campaign budget.
# RunBudget scopes every reservation to a run prefix on the campaign's transactional BudgetLedger
# (and can nest: a RunBudget's ledger may itself be a RunBudget). JsonModel issues
# schema-constrained model requests through campaigns.llm.metered_complete with resumable on-disk
# receipts. prompt() loads packaged prompt templates from the prompts/ resource directory.
"""Metered structured calls and per-run spending within the shared campaign."""

from __future__ import annotations

import time
from decimal import Decimal
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel

from repo2rlenv.auth import resolve_llm_api_key
from repo2rlenv.campaigns.budget import BudgetExceeded, BudgetLedger
from repo2rlenv.campaigns.llm import metered_complete
from repo2rlenv.spec.input import LLMSpec


class RunBudget:
    # [CLASS DOC] Per-run view over a BudgetLedger (or parent RunBudget). Every operation id must
    # contain the run prefix, and each reservation adds a (prefix, limit) scope so the run cannot
    # exceed its own USD limit while still respecting the campaign ledger. shared=True marks this
    # budget as a shared parent whose children may wait for sibling holds to settle.
    """All effects still reserve on the campaign's transactional ledger."""

    def __init__(self, ledger: BudgetLedger, prefix: str, limit: str, *, shared: bool = False):
        # [FUNCTION DOC] Stores ledger, prefix, limit (parsed to Decimal from a USD string) and
        # shared flag; path mirrors ledger.path. Raises decimal.InvalidOperation for a non-numeric
        # limit.
        self.ledger, self.prefix, self.limit = ledger, prefix, Decimal(limit)
        self.path = ledger.path
        self.shared = shared

    def totals(self) -> dict[str, str]:
        # [FUNCTION DOC] Returns {'accounted_usd', 'reserved_usd'} as decimal strings for operations
        # whose id contains the prefix: accounted sums actual_micros, reserved sums reserved_micros
        # of operations not yet settled. Reads ledger status only.
        operations = [op for op in self.ledger.status()["operations"] if self.prefix in op["id"]]
        spent = sum(op["actual_micros"] or 0 for op in operations)
        held = sum(op["reserved_micros"] for op in operations if op["status"] != "settled")
        return {
            "accounted_usd": str(Decimal(spent) / 1000000),
            "reserved_usd": str(Decimal(held) / 1000000),
        }

    def reserve(self, operation_id, amount_usd, description, *, scopes=()):
        # [FUNCTION DOC] Reserves amount_usd for operation_id on the underlying ledger, adding this
        # run's (prefix, limit) to any extra scopes. Raises ValueError if the id lacks the prefix;
        # propagates BudgetExceeded from the ledger.
        if self.prefix not in operation_id:
            raise ValueError("Quality operation must belong to this run")
        self.ledger.reserve(
            operation_id,
            amount_usd,
            description,
            scopes=(*scopes, (self.prefix, self.limit)),
        )

    def reserve_with_wait(
        self, operation_id, amount_usd, description, *, timeout_sec: int = 0, on_wait=None
    ):
        # [FUNCTION DOC] Like reserve (without extra scopes) but retries for up to timeout_sec (int
        # 0-300, else ValueError) while the denial is purely 'reservation_pressure' on shared parent
        # scopes that sibling holds could free (see _shared_siblings_can_settle). Calls on_wait(exc)
        # before each sleep of at most 1 second. Re-raises the last BudgetExceeded when the deadline
        # passes or the denial cannot be resolved by waiting. Returns None on success. Side effects:
        # ledger reservation, sleeping.
        """Wait only for definite pre-dispatch pressure from shared sibling holds.

        Ordinary reserve callers never wait. Each failed transaction is closed
        before inspecting siblings or sleeping; only reserve can admit work.
        """
        if type(timeout_sec) is not int or not 0 <= timeout_sec <= 300:
            raise ValueError("Reservation wait must be an integer from 0 to 300 seconds")
        deadline = time.monotonic() + timeout_sec
        last_denial = None
        while True:
            if last_denial is not None and time.monotonic() >= deadline:
                raise last_denial
            try:
                self.reserve(operation_id, amount_usd, description)
                return
            except BudgetExceeded as exc:
                last_denial = exc
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._shared_siblings_can_settle(exc):
                    raise
                if on_wait is not None:
                    on_wait(exc)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(1.0, remaining))

    def _shared_siblings_can_settle(self, exc: BudgetExceeded) -> bool:
        # [FUNCTION DOC] HELPER: Decides whether a BudgetExceeded is worth waiting on. Walks up the
        # RunBudget chain recording, for each shared ancestor, which child prefix this run lives
        # under. Returns False unless every denial is reservation_pressure on such a shared scope
        # and, for each, the scope's limit minus used (actual + unsettled reservations) plus the
        # reserved holds of sibling (non-ancestor-child) operations covers the requested amount.
        # Reads ledger status only.
        shared_children = {}
        child, scope = None, self
        while isinstance(scope, RunBudget):
            if scope.shared and child is not None:
                shared_children[scope.prefix] = child.prefix
            child, scope = scope, scope.ledger
        if not exc.denials or any(
            denial.scope not in shared_children or denial.reason != "reservation_pressure"
            for denial in exc.denials
        ):
            return False
        operations = self.status()["operations"]
        for denial in exc.denials:
            scoped = [op for op in operations if denial.scope in op["id"]]
            used = sum(
                (op["actual_micros"] or 0)
                + (op["reserved_micros"] if op["status"] != "settled" else 0)
                for op in scoped
            )
            siblings = sum(
                op["reserved_micros"]
                for op in scoped
                if op["status"] == "reserved" and shared_children[denial.scope] not in op["id"]
            )
            # A sibling may already have settled since the refusal. Retry the
            # atomic reservation then too, without releasing any holds ourselves.
            if denial.limit_micros - used + siblings < denial.requested_micros:
                return False
        return True

    def status(self):
        # [FUNCTION DOC] Returns the underlying ledger's status() dict.
        return self.ledger.status()

    def settle(self, *args, **kwargs):
        # [FUNCTION DOC] Delegates settle(...) to the underlying ledger and returns its result.
        return self.ledger.settle(*args, **kwargs)

    def mark_uncertain(self, *args, **kwargs):
        # [FUNCTION DOC] Delegates mark_uncertain(...) to the underlying ledger and returns its
        # result.
        return self.ledger.mark_uncertain(*args, **kwargs)


def prompt(name: str) -> str:
    # [FUNCTION DOC] Returns the text of the packaged prompt template prompts/<name>.md in this
    # package. Raises FileNotFoundError if absent.
    return files(__package__).joinpath("prompts", name + ".md").read_text()


class ModelRequestError(RuntimeError):
    # [CLASS DOC] RuntimeError raised when a provider request failed in a way that needs human
    # diagnosis; callers must not automatically redispatch it (the receipt file is the evidence).
    """A provider effect needs diagnosis; never automatically redispatch it."""


def response_schema(model: type[BaseModel]) -> dict:
    # [FUNCTION DOC] Returns the model's Pydantic JSON schema rewritten for strict structured
    # output: recursively removes every 'default', and for each object with properties marks all
    # properties required and sets additionalProperties false. Pure apart from the fresh schema
    # dict.
    """Require provider output fields even when readers accept legacy defaults."""
    schema = model.model_json_schema()

    def visit(value):
        # [FUNCTION DOC] HELPER: Recursive in-place walker over dicts/lists applying the
        # strict-schema rewrite described above.
        if isinstance(value, dict):
            value.pop("default", None)
            if value.get("type") == "object" and "properties" in value:
                value["required"] = list(value["properties"])
                value["additionalProperties"] = False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    return schema


class JsonModel:
    # [CLASS DOC] Schema-constrained LLM client bound to a receipt directory, a RunBudget, a
    # per-call USD reservation and a max token count.
    def __init__(self, directory: Path, budget: RunBudget, *, reservation: str, max_tokens: int):
        # [FUNCTION DOC] Stores directory, budget, reservation (USD string) and max_tokens. No side
        # effects.
        self.directory, self.budget = directory, budget
        self.reservation, self.max_tokens = reservation, max_tokens

    def ask(self, schema: type[BaseModel], model: LLMSpec, system: str, user: str, key: str):
        # [FUNCTION DOC] Sends one structured request and returns the parsed schema instance.
        # Inputs: schema (Pydantic model class), model (LLMSpec), system and user prompts, key
        # (receipt name and operation id suffix). Uses receipt <directory>/<key>.json with
        # resume=True, so an existing receipt is replayed without a new call; operation id is
        # '<prefix>:model:<key>'. Raises ValueError if no receipt exists and no API key resolves
        # (nothing dispatched); re-raises BudgetExceeded; wraps any other exception as
        # ModelRequestError naming the receipt; pydantic ValidationError if the content does not
        # match the schema. Side effects: network call, ledger reservation/settlement, receipt file.
        receipt = self.directory / f"{key}.json"
        if not receipt.exists() and not resolve_llm_api_key(model.provider, model.api_key_env):
            raise ValueError(f"No API key for {model.provider}; no model request was dispatched")
        try:
            response = metered_complete(
                model,
                ledger=self.budget,
                operation_id=f"{self.budget.prefix}:model:{key}",
                reservation_usd=self.reservation,
                receipt=receipt,
                system=system,
                user=user,
                max_tokens=self.max_tokens,
                response_schema=response_schema(schema),
                resume=True,
            )
        except BudgetExceeded:
            raise
        except Exception as exc:
            raise ModelRequestError(
                f"{model.qualified_name} request failed ({type(exc).__name__}); inspect {receipt}"
            ) from None
        return schema.model_validate_json(response.content)
