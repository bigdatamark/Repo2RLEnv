# [MODULE DOC] CPU trial backend: lazily provisions (or reuses) one Modal/Daytona worker per run,
# installs the repo2rlenv runtime wheel and Docker on it, and runs Harbor trials remotely through
# execution.harbor.run_trial with budget reservations, importing each result as a TrialRecord.
"""Lazy, reusable Modal/Daytona execution for the quality loop."""

from __future__ import annotations

import json
from pathlib import Path

from repo2rlenv.execution.artifacts import check_runtime_wheel, install_runtime, runtime_python
from repo2rlenv.execution.base import WorkerSpec, connect_worker
from repo2rlenv.execution.harbor import run_trial
from repo2rlenv.execution.lifecycle import prepare_docker, provision_worker, stop_worker
from repo2rlenv.quality.loop.artifacts import import_trial
from repo2rlenv.quality.loop.client import RunBudget
from repo2rlenv.quality.loop.models import LoopOptions, TrialRecord


class RemoteTrials:
    # [CLASS DOC] Remote trial runner bound to an output directory, RunBudget and LoopOptions. Owns
    # the worker it provisions (stopped in close) unless an existing worker receipt was supplied.
    def __init__(
        self,
        directory: Path,
        budget: RunBudget,
        options: LoopOptions,
        *,
        wheel: Path,
        provider: str = "modal",
        worker_receipt: Path | None = None,
        worker_reservation_usd: str = "3.00",
    ):
        # [FUNCTION DOC] Stores directory, budget, options, wheel path, provider (default modal),
        # optional worker receipt and worker reservation USD (default '3.00'). owns_worker is True
        # when no receipt was given. No side effects; the worker is created lazily.
        self.directory, self.budget, self.options = directory, budget, options
        self.wheel, self.provider, self.receipt = wheel, provider, worker_receipt
        self.worker_reservation = worker_reservation_usd
        self.owns_worker = worker_receipt is None
        self.worker = None
        self.python = None

    def identity(self) -> dict:
        # [FUNCTION DOC] Returns the execution identity: provider, runtime wheel sha256 (via
        # check_runtime_wheel, which may raise), resolved worker receipt path and worker
        # reservation.
        return {
            "provider": self.provider,
            "runtime_sha256": check_runtime_wheel(self.wheel),
            "worker_receipt": str(self.receipt.resolve()) if self.receipt else None,
            "worker_reservation_usd": self.worker_reservation,
        }

    def _ready(self):
        # [FUNCTION DOC] HELPER: Ensures a usable worker once. Checks the wheel; without a receipt,
        # scans workers/<prefix>-w<N>.json, skipping terminated ones, and either provisions a new
        # worker (2 CPU, 4096 MB, 7200 s, reserving worker_reservation_usd) or adopts the first
        # non-terminated receipt. A receipt-based worker must be running and on this campaign's
        # ledger (else ValueError) and is connected. Then prepares Docker and installs the runtime
        # wheel, setting self.python. Side effects: remote provisioning and installs, budget
        # reservation, receipts under the run directory.
        if self.worker is not None:
            return
        check_runtime_wheel(self.wheel)
        if self.receipt is None:
            workers = self.directory / "workers"
            # Old terminated workers are not revived. Uncertain creates are never repeated.
            index = 0
            while True:
                name = f"{self.budget.prefix}-w{index}"
                path = workers / f"{name}.json"
                if not path.exists():
                    self.worker, self.receipt = provision_worker(
                        WorkerSpec(
                            provider=self.provider,
                            name=name,
                            cpus=2,
                            memory_mb=4096,
                            timeout_sec=7200,
                        ),
                        workers,
                        self.budget,
                        reserve_usd=self.worker_reservation,
                    )
                    break
                record = json.loads(path.read_text())
                if record["state"] == "terminated":
                    index += 1
                    continue
                self.receipt = path
                break
        if self.worker is None:
            record = json.loads(self.receipt.read_text())
            if (
                record["state"] != "running"
                or Path(record["ledger"]).resolve() != self.budget.path.resolve()
            ):
                raise ValueError("Remote validation needs a running worker in this campaign")
            self.worker = connect_worker(record["spec"]["provider"], record["worker_id"])
        prepare_docker(self.worker)
        self.python = runtime_python(
            install_runtime(self.worker, self.wheel, self.directory / "runtime")
        )

    def run(self, task: Path, role: str, key: str) -> TrialRecord:
        # [FUNCTION DOC] Runs one trial and returns its TrialRecord; reuses trials/<key>/trial.json
        # if present. Agent is nop/oracle/oracle/terminus-2 for baseline/oracle/probe/rollout;
        # rollouts use the solver model and may wait shared_reservation_wait_sec for budget. Trial
        # id is '<prefix>-<key>'. Propagates errors from provisioning, run_trial and import.
        output = self.directory / "trials" / key
        if (output / "trial.json").exists():
            return self._read(output, task, role)
        self._ready()
        run_trial(
            self.worker,
            task,
            output,
            trial_id=f"{self.budget.prefix}-{key}",
            agent={
                "baseline": "nop",
                "oracle": "oracle",
                "probe": "oracle",
                "rollout": "terminus-2",
            }[role],
            model=self.options.solver_model if role == "rollout" else None,
            ledger=self.budget,
            reservation_usd=self.options.solver_reservation_usd,
            reservation_wait_sec=self.options.shared_reservation_wait_sec
            if role == "rollout"
            else 0,
            max_turns=self.options.max_turns,
            max_tokens=self.options.solver_tokens,
            timeout_sec=self.options.trial_timeout_sec,
            python=self.python,
        )
        return self._read(output, task, role)

    @staticmethod
    def _read(output: Path, task: Path, role: str) -> TrialRecord:
        # [FUNCTION DOC] HELPER (static): Imports a trial output (probes as oracle evidence), sets
        # the role and, for probes, probe_installed = whether agent/oracle.txt contains
        # __QUALITY_PROBE_COMPLETED__. Raises ValueError from import_trial.
        record = import_trial(output, task, "oracle" if role == "probe" else role)
        updates = {"role": role}
        if role == "probe":
            log = Path(record.result).parent / "agent/oracle.txt"
            updates["probe_installed"] = (
                log.is_file() and "__QUALITY_PROBE_COMPLETED__" in log.read_text()
            )
        return record.model_copy(update=updates)

    def close(self):
        # [FUNCTION DOC] Stops the worker via stop_worker if this instance owns it and one was
        # started, then clears it. Caller-supplied workers are left running.
        if self.owns_worker and self.worker is not None and self.receipt is not None:
            stop_worker(self.receipt, self.budget)
            self.worker = None
