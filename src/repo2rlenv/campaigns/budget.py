# [MODULE DOC] Transactional spend ledger shared by all paid operations of one campaign, stored in a
# SQLite file.
# Amounts are integer micro-dollars (USD * 1e6, rounded up). Tables: budget (single row id=1 holding
# limit_micros), operations (id, description, reserved_micros, actual_micros, status in
# reserved/uncertain/settled, evidence) and, lazily, budget_increases (audit log of authorized limit
# raises keyed by evidence).
# Lifecycle of an operation: reserve -> (mark_uncertain) -> settle. Unsettled reservations (reserved
# or uncertain) hold capacity; settled ones count their actual cost, which may exceed the
# reservation. Every mutation runs inside a BEGIN IMMEDIATE transaction so concurrent controllers
# serialize.
"""Transactional spend reservations shared by all operations in a campaign.

Amounts are stored as integer micro-dollars. Uncertain provider outcomes retain
their reservation until explicitly reconciled; recorded actual cost may exceed a
reservation, but that overrun must prevent future dispatch rather than disappear.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

_SCALE = Decimal(1_000_000)


class BudgetExceeded(RuntimeError):
    # [CLASS DOC] Exception (subclass of RuntimeError) raised by BudgetLedger.reserve when a
    # reservation would exceed the global cap or any scoped cap. Carries the per-cap BudgetDenial
    # evidence.
    """The proposed operation would exceed the remaining allowance."""

    def __init__(self, message: str, *, denials: tuple[BudgetDenial, ...] = ()):
        # [FUNCTION DOC] Stores the message via RuntimeError and the denials tuple (default empty)
        # on self.denials. Inputs: message (str); denials (keyword, tuple of BudgetDenial). No side
        # effects.
        super().__init__(message)
        self.denials = denials


@dataclass(frozen=True)
class BudgetDenial:
    # [CLASS DOC] Frozen dataclass describing why one cap refused a reservation, captured inside the
    # same transaction. Fields (all micro-dollars except scope): scope (operation-ID substring of a
    # scoped cap, or None for the global campaign cap), requested_micros, limit_micros,
    # accounted_micros (sum of actual costs), reserved_micros (reservations of unsettled operations,
    # including uncertain), uncertain_micros (subset in uncertain state).
    """Capacity evidence from the same transaction that refused reservation."""

    scope: str | None  # None is the global campaign cap.
    requested_micros: int
    limit_micros: int
    accounted_micros: int
    reserved_micros: int
    uncertain_micros: int

    @property
    def remaining_micros(self) -> int:
        # [FUNCTION DOC] Property: limit minus accounted minus reserved, in micro-dollars. May be
        # negative if actual costs overran. No side effects.
        return self.limit_micros - self.accounted_micros - self.reserved_micros

    @property
    def reason(self) -> str:
        # [FUNCTION DOC] Property classifying the denial as a short string, checked in order:
        # 'request_exceeds_limit' if the request alone exceeds the limit; 'accounted_exhaustion' if
        # request plus accounted spend exceeds it; 'uncertain_reservations' if request plus
        # accounted plus uncertain holds exceed it; otherwise 'reservation_pressure' (only ordinary
        # pending reservations block it, so waiting may help).
        if self.requested_micros > self.limit_micros:
            return "request_exceeds_limit"
        if self.requested_micros + self.accounted_micros > self.limit_micros:
            return "accounted_exhaustion"
        if (
            self.requested_micros + self.accounted_micros + self.uncertain_micros
            > self.limit_micros
        ):
            return "uncertain_reservations"
        return "reservation_pressure"

    def as_dict(self) -> dict:
        # [FUNCTION DOC] Returns a JSON-friendly dict with scope, reason and each amount (requested,
        # limit, accounted, reserved, uncertain, remaining) formatted as six-decimal USD strings via
        # _usd. No side effects.
        return {
            "scope": self.scope,
            "reason": self.reason,
            "requested_usd": _usd(self.requested_micros),
            "limit_usd": _usd(self.limit_micros),
            "accounted_usd": _usd(self.accounted_micros),
            "reserved_usd": _usd(self.reserved_micros),
            "uncertain_usd": _usd(self.uncertain_micros),
            "remaining_usd": _usd(self.remaining_micros),
        }


class OperationAlreadyRecorded(RuntimeError):
    # [CLASS DOC] Exception (RuntimeError) signalling that an operation ID already exists in the
    # ledger (on reserve) or was settled with different amount/evidence (on settle); the caller must
    # inspect it rather than redispatch.
    """The operation must be inspected instead of dispatched again."""


def _micros(value: Decimal | str | float) -> int:
    # [FUNCTION DOC] HELPER: Converts a Decimal/str/float USD amount to integer micro-dollars,
    # rounding up (ceiling). Goes through str() so floats are interpreted by their repr. Raises
    # ValueError for non-finite or negative amounts.
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("Budget amounts must be finite and nonnegative")
    return int((amount * _SCALE).to_integral_value(rounding=ROUND_CEILING))


def _usd(value: int) -> str:
    # [FUNCTION DOC] HELPER: Formats integer micro-dollars as a USD decimal string with exactly six
    # fractional digits (e.g. 1500000 -> '1.500000').
    return format(Decimal(value) / _SCALE, ".6f")


class BudgetLedger:
    # [CLASS DOC] SQLite-backed spend ledger for one campaign file. Attribute: path (Path of the
    # database). All public methods open a fresh connection per call through _transaction, so
    # instances are cheap and safe to share across processes.
    def __init__(self, path: Path, *, limit_usd: Decimal | str | float | None = None):
        # [FUNCTION DOC] Opens/creates the ledger at path. Inputs: path (Path to the sqlite file;
        # parent directories are created); limit_usd (keyword, optional limit). Creates the budget
        # and operations tables if absent. For a new ledger (no budget row) limit_usd is required
        # and is inserted; for an existing ledger a given limit_usd must equal the stored one.
        # Raises ValueError if a new ledger has no limit, or if a supplied limit differs from the
        # stored limit (no implicit reset). Side effects: filesystem (mkdir, sqlite file).
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, limit_micros INTEGER NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS operations ("
                "id TEXT PRIMARY KEY, description TEXT NOT NULL, "
                "reserved_micros INTEGER NOT NULL, actual_micros INTEGER, "
                "status TEXT NOT NULL CHECK(status IN ('reserved','uncertain','settled')), "
                "evidence TEXT NOT NULL DEFAULT '')"
            )
            row = db.execute("SELECT limit_micros FROM budget WHERE id=1").fetchone()
            if row is None:
                if limit_usd is None:
                    raise ValueError("A new ledger requires an explicit spending limit")
                db.execute("INSERT INTO budget VALUES (1, ?)", (_micros(limit_usd),))
            elif limit_usd is not None and row[0] != _micros(limit_usd):
                raise ValueError(
                    "Existing campaign limit differs; refusing an implicit budget reset"
                )

    @contextmanager
    def _transaction(self):
        # [FUNCTION DOC] HELPER: Context manager yielding a sqlite3 connection (30s busy timeout,
        # Row factory) inside BEGIN IMMEDIATE. Commits on normal exit; rolls back and re-raises on
        # any exception (including BaseException); always closes the connection.
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _totals(db: sqlite3.Connection) -> tuple[int, int, int]:
        # [FUNCTION DOC] HELPER: Static. Given an open connection, returns (limit_micros, sum of
        # actual_micros over all operations, sum of reserved_micros over operations whose status is
        # not settled). Raises TypeError-like errors if the budget row is missing.
        limit = db.execute("SELECT limit_micros FROM budget WHERE id=1").fetchone()[0]
        spent, reserved = db.execute(
            "SELECT COALESCE(SUM(actual_micros),0), "
            "COALESCE(SUM(CASE WHEN status != 'settled' THEN reserved_micros ELSE 0 END),0) "
            "FROM operations"
        ).fetchone()
        return limit, spent, reserved

    def reserve(
        self,
        operation_id: str,
        amount_usd: Decimal | str | float,
        description: str,
        *,
        scopes: tuple[tuple[str, Decimal | str | float], ...] = (),
    ) -> None:
        # [FUNCTION DOC] Atomically reserves amount_usd for a new operation. Inputs: operation_id
        # (non-blank unique ID), amount_usd (positive amount), description (non-blank), scopes
        # (keyword; tuple of (substring, limit) pairs - each substring must occur in operation_id
        # and defines a scoped cap over all operations whose ID contains it).
        # Within one transaction: rejects an existing ID with OperationAlreadyRecorded; then for the
        # global cap and each scoped cap computes accounted spend, unsettled reservations and
        # uncertain reservations of the matching operations, and records a BudgetDenial whenever
        # accounted + reserved + amount exceeds that cap. If any cap denies, raises BudgetExceeded
        # with a combined message and all denials (nothing inserted). Otherwise inserts the
        # operation with status 'reserved'.
        # Raises ValueError for blank ID/description, zero amount, or a scope substring not in the
        # ID (also via _micros for negative/non-finite). Side effects: sqlite write. Note: scope
        # matching is substring (instr), not strict prefix, despite the variable name.
        if not operation_id.strip() or not description.strip():
            raise ValueError("Reservation requires an operation ID and description")
        amount = _micros(amount_usd)
        if amount == 0:
            raise ValueError("Paid operations require a positive reservation")
        limits = []
        for prefix, limit in scopes:
            if not prefix or prefix not in operation_id:
                raise ValueError("Operation must belong to each budget scope")
            limits.append((prefix, _micros(limit)))
        with self._transaction() as db:
            if db.execute("SELECT 1 FROM operations WHERE id=?", (operation_id,)).fetchone():
                raise OperationAlreadyRecorded(operation_id)
            limit = db.execute("SELECT limit_micros FROM budget WHERE id=1").fetchone()[0]
            denials = []
            # Inspect every cap atomically: a shared hold must not mask an exhausted
            # candidate, quality or global allowance that cannot be waited away.
            for prefix, scoped_limit in [(None, limit), *limits]:
                spent, reserved, uncertain = db.execute(
                    "SELECT COALESCE(SUM(actual_micros),0), "
                    "COALESCE(SUM(CASE WHEN status != 'settled' THEN reserved_micros ELSE 0 END),0), "
                    "COALESCE(SUM(CASE WHEN status = 'uncertain' THEN reserved_micros ELSE 0 END),0) "
                    "FROM operations WHERE ? IS NULL OR instr(id, ?) > 0",
                    (prefix, prefix),
                ).fetchone()
                if spent + reserved + amount > scoped_limit:
                    denials.append(
                        BudgetDenial(prefix, amount, scoped_limit, spent, reserved, uncertain)
                    )
            if denials:
                messages = [
                    f"{'Budget scope ' + denial.scope if denial.scope else operation_id}: "
                    f"requested ${_usd(amount)}, remaining ${_usd(denial.remaining_micros)} "
                    f"({denial.reason})"
                    for denial in denials
                ]
                raise BudgetExceeded(
                    "; ".join(messages) + "; completed evidence retained", denials=tuple(denials)
                )
            db.execute(
                "INSERT INTO operations(id,description,reserved_micros,status) VALUES (?,?,?,'reserved')",
                (operation_id, description, amount),
            )

    def increase_limit(self, new_limit_usd, *, expected_limit_usd, evidence: str) -> None:
        # [FUNCTION DOC] Records an explicitly authorized raise of the global limit without touching
        # operations. Inputs: new_limit_usd; expected_limit_usd (keyword, the limit the caller
        # believes is current - optimistic concurrency guard); evidence (keyword, non-blank
        # authorization text, also the idempotency key).
        # Creates the budget_increases table if needed. If the same evidence was already recorded
        # with the same (expected, new) pair it returns silently (idempotent); with a different pair
        # raises ValueError. Raises ValueError if new <= expected, evidence blank, or the current
        # limit differs from expected. Otherwise updates the limit and logs the increase. Side
        # effects: sqlite writes.
        """Record an explicitly authorized increase without changing operations."""
        new_limit, expected = _micros(new_limit_usd), _micros(expected_limit_usd)
        if new_limit <= expected or not evidence.strip():
            raise ValueError("A budget increase requires a higher limit and authorization evidence")
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS budget_increases ("
                "evidence TEXT PRIMARY KEY, previous_micros INTEGER NOT NULL, "
                "new_micros INTEGER NOT NULL, recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            prior = db.execute(
                "SELECT previous_micros,new_micros FROM budget_increases WHERE evidence=?",
                (evidence,),
            ).fetchone()
            if prior is not None:
                if tuple(prior) == (expected, new_limit):
                    return
                raise ValueError("Authorization evidence already records a different increase")
            current = db.execute("SELECT limit_micros FROM budget WHERE id=1").fetchone()[0]
            if current != expected:
                raise ValueError(
                    "Budget limit changed; inspect it before authorizing another increase"
                )
            db.execute("UPDATE budget SET limit_micros=? WHERE id=1", (new_limit,))
            db.execute(
                "INSERT INTO budget_increases(evidence,previous_micros,new_micros) VALUES (?,?,?)",
                (evidence, expected, new_limit),
            )

    def mark_uncertain(self, operation_id: str, evidence: str) -> None:
        # [FUNCTION DOC] Moves an outstanding (reserved or uncertain) operation to status
        # 'uncertain' and overwrites its evidence, meaning the provider outcome is unknown and the
        # reservation must stay held until reconciled. Raises ValueError for blank evidence, an
        # unknown ID, or an already settled operation. Side effects: sqlite write.
        if not evidence.strip():
            raise ValueError("An uncertain operation needs recovery evidence")
        with self._transaction() as db:
            row = db.execute("SELECT status FROM operations WHERE id=?", (operation_id,)).fetchone()
            if row is None or row[0] == "settled":
                raise ValueError("Only an outstanding reservation can become uncertain")
            db.execute(
                "UPDATE operations SET status='uncertain', evidence=? WHERE id=?",
                (evidence, operation_id),
            )

    def settle(
        self, operation_id: str, actual_usd: Decimal | str | float, *, evidence: str
    ) -> None:
        # [FUNCTION DOC] Records the actual cost of an operation and marks it 'settled'. Inputs:
        # operation_id; actual_usd (may exceed the reservation); evidence (keyword, non-blank
        # usage/billing/no-dispatch proof). Idempotent: if already settled with identical amount and
        # evidence, returns; if settled differently raises OperationAlreadyRecorded. Raises
        # ValueError for blank evidence or unknown ID. Side effects: sqlite write.
        amount = _micros(actual_usd)
        if not evidence.strip():
            raise ValueError("Settlement needs usage, billing or no-dispatch evidence")
        with self._transaction() as db:
            row = db.execute(
                "SELECT status,actual_micros,evidence FROM operations WHERE id=?", (operation_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"No reservation for {operation_id}")
            if row[0] == "settled":
                if row[1] == amount and row[2] == evidence:
                    return
                raise OperationAlreadyRecorded(f"Conflicting settlement for {operation_id}")
            db.execute(
                "UPDATE operations SET actual_micros=?, status='settled', evidence=? WHERE id=?",
                (amount, evidence, operation_id),
            )

    def status(self) -> dict:
        # [FUNCTION DOC] Returns a snapshot dict: limit_usd, accounted_usd (sum of actual costs),
        # reserved_usd (unsettled reservations), remaining_usd (limit - accounted - reserved; may be
        # negative) as six-decimal strings, plus operations (list of row dicts with all columns,
        # ordered by id). Runs in a write-locking transaction; no data changes.
        with self._transaction() as db:
            limit, spent, reserved = self._totals(db)
            operations = [dict(row) for row in db.execute("SELECT * FROM operations ORDER BY id")]
        return {
            "limit_usd": _usd(limit),
            "accounted_usd": _usd(spent),
            "reserved_usd": _usd(reserved),
            "remaining_usd": _usd(limit - spent - reserved),
            "operations": operations,
        }
