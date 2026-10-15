# [MODULE DOC] Adapter that lets the author bridge (which expects reserve/settle/spent) charge the
# shared transactional campaign ledger (quality.loop.client.RunBudget), writing a JSON receipt per
# settled author model call.
"""Adapt the isolated author bridge to the shared transactional campaign ledger."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from repo2rlenv.execution.lifecycle import save_record
from repo2rlenv.quality.loop.client import RunBudget


class AuthorBudget:
    # [CLASS DOC] Budget adapter for one author stage. Holds the RunBudget, a receipts directory and
    # a stage name; numbers operations sequentially per instance.
    def __init__(self, budget: RunBudget, directory: Path, stage: str):
        # [FUNCTION DOC] Inputs: budget (RunBudget ledger), directory (Path where receipts are
        # written), stage (str stage label). Sets turn counter to 0. No I/O.
        self.budget, self.directory, self.stage = budget, directory, stage
        self.turn = 0

    @property
    def spent(self) -> float:
        # [FUNCTION DOC] Property: total spend across the whole ledger as float, summing
        # RunBudget.totals() values as Decimals (not only this stage's spend).
        return float(sum(Decimal(value) for value in self.budget.totals().values()))

    def reserve(self, amount: float, description: str) -> str:
        # [FUNCTION DOC] Increments the turn counter and reserves `amount` USD in the ledger under
        # operation id "<budget.prefix>:author:<stage>:<turn>" with the given description. Returns
        # the operation id. Ledger may raise if the reservation exceeds limits.
        self.turn += 1
        operation = f"{self.budget.prefix}:author:{self.stage}:{self.turn}"
        self.budget.reserve(operation, str(amount), description)
        return operation

    def settle(self, operation: str, amount: float) -> None:
        # [FUNCTION DOC] Writes a receipt JSON (operation, cost_usd, source, trace path =
        # directory.parent/trace.jsonl) to directory/<turn>.json via save_record, then settles the
        # operation in the ledger with the actual amount and the receipt's absolute path as
        # evidence. Returns None.
        receipt = self.directory / (operation.rsplit(":", 1)[-1] + ".json")
        save_record(
            receipt,
            {
                "operation": operation,
                "cost_usd": amount,
                "source": "author provider response",
                "trace": str(self.directory.parent / "trace.jsonl"),
            },
        )
        self.budget.settle(operation, str(amount), evidence=str(receipt.resolve()))
