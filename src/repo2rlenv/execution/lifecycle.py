# [MODULE DOC] Worker lifecycle with durable receipts: atomic JSON record writing, provisioning that
# records the worker identity before dispatch and reserves budget, termination that keeps cost
# uncertain until reconciled, and a Docker daemon readiness probe.
"""Persist worker identity before dispatch and retain uncertain spend for recovery."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.execution.base import RemoteWorker, WorkerSpec, connect_worker, create_worker


def now() -> str:
    # [FUNCTION DOC] Returns the current UTC time as an ISO-8601 string.
    return datetime.now(UTC).isoformat()


def save_record(path: Path, record: dict) -> None:
    # [FUNCTION DOC] Atomically writes record as sorted, indented JSON (plus newline) to path:
    # creates the parent, writes and fsyncs a temporary file in the same directory, then os.replace.
    # Removes the temporary file on failure.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def provision_worker(
    spec: WorkerSpec, directory: Path, ledger: BudgetLedger, *, reserve_usd: str
) -> tuple[RemoteWorker, Path]:
    # [FUNCTION DOC] Creates a remote worker with receipts. Exclusively creates
    # directory/<spec.name>.json (FileExistsError if it exists, so a name is never reused), reserves
    # reserve_usd under operation "worker:<provider>:<name>" (records reservation_failed and re-
    # raises on error), records creating, calls create_worker, then records running with worker_id.
    # On any creation exception records creation_uncertain, marks the ledger operation uncertain and
    # re-raises. Returns (worker, receipt path).
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec.name}.json"
    # The stable name is also the provider recovery handle if a create response
    # is lost. Never blindly create another worker for this operation.
    with path.open("x") as handle:
        json.dump({"state": "claimed", "spec": spec.model_dump(), "claimed_at": now()}, handle)
    operation_id = f"worker:{spec.provider}:{spec.name}"
    record = {
        "state": "claimed",
        "spec": spec.model_dump(),
        "operation_id": operation_id,
        "ledger": str(ledger.path.resolve()),
        "claimed_at": now(),
    }
    try:
        ledger.reserve(operation_id, reserve_usd, f"{spec.provider} worker {spec.name}")
    except Exception:
        record["state"] = "reservation_failed"
        save_record(path, record)
        raise
    record.update(state="creating", started_at=now())
    save_record(path, record)
    try:
        worker = create_worker(spec)
        record.update(state="running", worker_id=worker.id, ready_at=now())
        save_record(path, record)
        return worker, path
    except BaseException:
        record["state"] = "creation_uncertain"
        save_record(path, record)
        ledger.mark_uncertain(operation_id, f"Inspect provider worker by name; receipt={path}")
        raise


def stop_worker(path: Path, ledger: BudgetLedger) -> dict:
    # [FUNCTION DOC] Terminates the worker named by a receipt. Raises ValueError if the receipt
    # belongs to another ledger or has no worker_id. Returns immediately if already terminated.
    # Records terminating, reconnects and terminates; on failure records termination_uncertain,
    # marks uncertain and re-raises. On success records terminated and marks the ledger operation
    # uncertain pending cost reconciliation. Returns the record.
    record = json.loads(path.read_text())
    if Path(record["ledger"]).resolve() != ledger.path.resolve():
        raise ValueError("Worker receipt belongs to another budget ledger")
    if record["state"] == "terminated":
        return record
    worker_id = record.get("worker_id")
    if worker_id is None:
        raise ValueError(
            "Worker creation is uncertain; resolve its provider identity before cleanup"
        )
    record["state"] = "terminating"
    save_record(path, record)
    try:
        connect_worker(record["spec"]["provider"], worker_id).terminate()
    except BaseException:
        record["state"] = "termination_uncertain"
        save_record(path, record)
        ledger.mark_uncertain(record["operation_id"], f"Termination not confirmed; receipt={path}")
        raise
    record.update(state="terminated", stopped_at=now())
    save_record(path, record)
    # Cleanup is confirmed, but its price must come from explicit evidence.
    ledger.mark_uncertain(
        record["operation_id"], f"Worker terminated; cost reconciliation pending; receipt={path}"
    )
    return record


def prepare_docker(worker: RemoteWorker) -> None:
    # [FUNCTION DOC] Ensures the worker's Docker daemon responds: creates /work and /evidence,
    # starts dockerd (overlay2, log to /evidence/dockerd.log) if docker info fails, and waits up to
    # 60s. Raises RemoteCommandError if not ready (90s exec timeout).
    """Probe the real daemon; package presence alone is not build readiness."""
    script = """set -eu
mkdir -p /work /evidence
if ! docker info >/dev/null 2>&1; then
  nohup dockerd --storage-driver=overlay2 >/evidence/dockerd.log 2>&1 </dev/null &
fi
for attempt in $(seq 1 60); do
  if docker info >/dev/null 2>&1; then exit 0; fi
  sleep 1
done
exit 1
"""
    worker.exec(["sh", "-c", script], timeout=90).checked("Docker daemon readiness")
