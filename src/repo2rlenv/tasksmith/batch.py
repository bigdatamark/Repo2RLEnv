# [MODULE DOC] Batch controller for Tasksmith PR campaigns. A BatchPlan lists GitHub PR candidates;
# run_batch freezes the plan, runtime wheel and the owned controller code into the batch directory,
# then dispatches each PR to a fresh isolated Python child process (via a thread pool of
# supervisors) that runs one Tasksmith instance under nested shared budgets in a SQLite ledger.
# Results, receipts and generated tasks are retained; uncertain work is never redispatched, and a
# report.json tracks verified / generated_unverified / blocked counts. Also runnable as a script.
"""Bounded process-isolated PR campaigns with retained artifacts and shared budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from decimal import Decimal
from pathlib import Path, PurePosixPath

from pydantic import Field, model_validator

from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.execution.artifacts import check_runtime_wheel
from repo2rlenv.execution.lifecycle import save_record
from repo2rlenv.locking import lock_file
from repo2rlenv.quality.loop.artifacts import digest
from repo2rlenv.quality.loop.client import RunBudget
from repo2rlenv.quality.loop.models import LoopResult, ProbeManifest
from repo2rlenv.tasksmith.models import Options, Panel, Record
from repo2rlenv.tasksmith.reuse import prepared_probe_definitions


def _url(value: str) -> str:
    # [FUNCTION DOC] HELPER: Validates that value is a GitHub PR URL
    # (https://github.com/<owner>/<repo>/pull/<n>, optional trailing slash) and returns it without
    # the trailing slash. Raises ValueError otherwise.
    if not re.fullmatch(r"https://github.com/[\w.-]+/[\w.-]+/pull/[1-9][0-9]*/?", value):
        raise ValueError("Batch inputs must be GitHub PR URLs")
    return value.rstrip("/")


class Candidate(Record):
    # [CLASS DOC] Pydantic Record describing one PR in a batch: url, Tasksmith Options, optional
    # generation_run (prior run dir) with reuse_evidence flag, optional frozen source_record,
    # prepared_task directory and prepared_probes manifest.
    url: str
    options: Options
    generation_run: Path | None = None
    reuse_evidence: bool = False
    source_record: dict | None = None
    prepared_task: Path | None = None
    prepared_probes: ProbeManifest | None = Field(
        default=None,
        description="Exact task-bound probe definitions for fresh replay; no prior trial evidence.",
    )

    @model_validator(mode="after")
    def inputs(self):
        # [FUNCTION DOC] After-validator: normalises the URL; requires generation_run when
        # reuse_evidence is set; resolves generation_run; a prepared_task requires source_record and
        # forbids generation_run (resolved to absolute); validates prepared probes against the task
        # via prepared_probe_definitions with the option's max_probes and required focus. Raises
        # ValueError on violations. Returns self.
        self.url = _url(self.url)
        if self.reuse_evidence and self.generation_run is None:
            raise ValueError("Evidence reuse requires a generation run")
        if self.generation_run is not None:
            self.generation_run = self.generation_run.resolve()
        if self.prepared_task is not None:
            if not self.source_record or self.generation_run is not None:
                raise ValueError(
                    "Prepared tasks require frozen source evidence and cannot use generation_run"
                )
            self.prepared_task = self.prepared_task.resolve()
        prepared_probe_definitions(
            self.prepared_task,
            self.prepared_probes,
            max_probes=self.options.quality.max_probes,
            required_focus=self.options.required_probe_focus,
        )
        return self


class BatchPlan(Record):
    # [CLASS DOC] Pydantic Record for a whole batch: name (lowercase slug <=31 chars), 1-1000
    # candidates, prior_verified result paths, target_verified (1-1000, default 50), max_parallel
    # and max_gpu_parallel (1-8, default 2), max_spend_usd (decimal string, default "200.00").
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,30}$")
    candidates: list[Candidate] = Field(min_length=1, max_length=1000)
    # Each result supplies its own source URL through the checksum-bound task.
    prior_verified: list[Path] = Field(default_factory=list)
    target_verified: int = Field(default=50, ge=1, le=1000)
    max_parallel: int = Field(default=2, ge=1, le=8)
    max_gpu_parallel: int = Field(default=2, ge=1, le=8)
    max_spend_usd: str = "200.00"

    @model_validator(mode="after")
    def limits(self):
        # [FUNCTION DOC] After-validator: rejects duplicate candidate URLs and non-finite or non-
        # positive spend limits (ValueError); resolves prior_verified paths. Returns self.
        if len({item.url for item in self.candidates}) != len(self.candidates):
            raise ValueError("Batch PRs must be unique, including trailing-slash variants")
        limit = Decimal(self.max_spend_usd)
        if not limit.is_finite() or limit <= 0:
            raise ValueError("Batch spending limit must be finite and positive")
        self.prior_verified = [path.resolve() for path in self.prior_verified]
        return self


def verified_result(path: Path) -> dict:
    # [FUNCTION DOC] Loads a quality LoopResult JSON at path and checks that its task is labelled
    # "verified" by label_from_quality (ValueError otherwise). Reads the task's task.toml
    # metadata.repo2env.source_url. Returns dict with url, absolute task path, bundle_hash, absolute
    # quality_result path and its sha256 digest. May raise KeyError/OSError/validation errors.
    """Count only a usable result with bound controls, probes and a judged rollout."""
    import tomllib

    from repo2rlenv.quality.labels import label_from_quality

    result = LoopResult.model_validate_json(path.read_text())
    task = Path(result.task_path)
    if label_from_quality(task, path, provenance="unknown").status != "verified":
        raise ValueError("A verified task with complete bound evidence is required")
    metadata = tomllib.loads((task / "task.toml").read_text())["metadata"]["repo2env"]
    return {
        "url": _url(metadata["source_url"]),
        "task": str(task.resolve()),
        "bundle_hash": result.bundle_hash,
        "quality_result": str(path.resolve()),
        "quality_sha256": digest(path),
    }


def _controller_identity() -> str:
    # [FUNCTION DOC] HELPER: Returns a sha256 over a sorted JSON map of relative path -> digest for
    # every .py/.md/.mjs/.js file under the repo2rlenv package directory (fingerprint of the running
    # controller code).
    """Freeze owned controller code and prompts, including agent bridges."""
    root = Path(__file__).parents[1]
    hashes = {
        path.relative_to(root).as_posix(): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".md", ".mjs", ".js"}
    }
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def _freeze_controller(wheel: Path, destination: Path) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Extracts the repo2rlenv/ package files from a wheel zip into
    # destination as a read-only tree without importing them, and returns {member name: sha256}.
    # Raises ValueError if >20000 files or >100 MiB, if any path is absolute, contains "..", is non-
    # normalised, duplicated or a symlink, or if batch.py / __init__.py are missing. If destination
    # already exists it is verified instead (no symlinks, identical file hashes and inventory, else
    # ValueError). Otherwise files are written into a temp dir next to destination with mode 0555
    # (executable) or 0444, directories set to 0555, then atomically renamed into place.
    """Copy only owned package files from the exact wheel, without importing them."""
    with zipfile.ZipFile(wheel) as archive:
        members = [
            item
            for item in archive.infolist()
            if item.filename.startswith("repo2rlenv/") and not item.is_dir()
        ]
        if len(members) > 20000 or sum(item.file_size for item in members) > 100 * 1024 * 1024:
            raise ValueError("Owned controller exceeds the extraction limit")
        content, modes = {}, {}
        for item in members:
            path = PurePosixPath(item.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != item.filename
                or item.filename in content
                or (item.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("Runtime wheel contains unsafe or duplicate controller paths")
            content[item.filename] = archive.read(item)
            modes[item.filename] = 0o555 if (item.external_attr >> 16) & 0o111 else 0o444
    if "repo2rlenv/tasksmith/batch.py" not in content or "repo2rlenv/__init__.py" not in content:
        raise ValueError("Runtime wheel does not contain the owned batch controller")
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in content.items()}
    if destination.exists():
        if destination.is_symlink() or any(path.is_symlink() for path in destination.rglob("*")):
            raise ValueError("Frozen controller cannot contain symlinks")
        for name, expected in hashes.items():
            if not (destination / name).is_file() or digest(destination / name) != expected:
                raise ValueError("Frozen controller files changed")
        actual = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
        }
        if actual != set(hashes):
            raise ValueError("Frozen controller inventory changed")
        return hashes
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".controller-", dir=destination.parent) as temporary:
        copied = Path(temporary) / "controller"
        for name, value in content.items():
            path = copied / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            path.chmod(modes[name])
        for path in sorted(copied.rglob("*"), reverse=True):
            if path.is_dir():
                path.chmod(0o555)
        copied.rename(destination)
        destination.chmod(0o555)
    return hashes


_CHILD_BOOTSTRAP = """
import json, sys
from pathlib import Path
controller, request = map(Path, sys.argv[1:])
sys.path.insert(0, str(controller))
import repo2rlenv
if not Path(repo2rlenv.__file__).resolve().is_relative_to(controller.resolve()):
    raise RuntimeError('Controller was imported outside the frozen package')
from repo2rlenv.tasksmith.batch import _run_candidate
data = json.loads(request.read_text())
_run_candidate(data['configuration'], data['candidate'])
"""


def _supervise_candidate(configuration: dict, item: dict) -> dict:
    # [FUNCTION DOC] HELPER: Runs in a supervisor thread. Creates candidates/<key> (mode 0700),
    # refuses (ValueError) if a dispatch.json or batch-result.json already exists, (re)verifies the
    # frozen controller, writes dispatch.json with configuration and candidate, and runs `python -I
    # -B -c <bootstrap>` which imports only the frozen controller and calls _run_candidate.
    # stdout/stderr go to newly created (exclusive, 0600) controller.stdout/.stderr files. Raises
    # RuntimeError on nonzero exit; returns the child's batch-result.json contents.
    """A supervisor thread only waits; all Tasksmith work occurs in a fresh process."""
    directory = Path(configuration["directory"]) / "candidates" / _key(item["url"])
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    request = directory / "dispatch.json"
    if request.exists() or (directory / "batch-result.json").exists():
        raise ValueError("Child dispatch already exists; inspect its receipts before retrying")
    _freeze_controller(Path(configuration["wheel"]), Path(configuration["controller_root"]))
    save_record(request, {"configuration": configuration, "candidate": item})
    handles = []
    try:
        for name in ("controller.stdout", "controller.stderr"):
            descriptor = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            handles.append(os.fdopen(descriptor, "wb"))
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _CHILD_BOOTSTRAP,
                configuration["controller_root"],
                str(request),
            ],
            stdout=handles[0],
            stderr=handles[1],
            check=False,
        )
    finally:
        for handle in handles:
            handle.close()
    if completed.returncode:
        raise RuntimeError("Isolated controller failed; inspect its retained private logs")
    return json.loads((directory / "batch-result.json").read_text())


def _prefix(directory: Path) -> str:
    # [FUNCTION DOC] HELPER: Returns the ledger prefix "tsbatch-" + first 12 hex chars of sha256 of
    # the resolved directory path.
    return "tsbatch-" + hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:12]


def _key(url: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the first 12 hex chars of sha256(url), used as a per-candidate
    # id.
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def _cleanup_pending(directory: Path) -> list[str]:
    # [FUNCTION DOC] HELPER: Scans workers/*.json and **/allocations/*.json receipts under directory
    # (excluding *.cost.json) and returns absolute paths of those whose "state" is not terminated /
    # build_failed / reservation_failed (i.e. provider resources possibly still live).
    receipts = [*directory.glob("workers/*.json"), *directory.glob("**/allocations/*.json")]
    return [
        str(path.resolve())
        for path in receipts
        if not path.name.endswith(".cost.json")
        and json.loads(path.read_text()).get("state")
        not in {"terminated", "build_failed", "reservation_failed"}
    ]


def _outcome(directory: Path, url: str) -> dict:
    # [FUNCTION DOC] HELPER: Builds the result row for one candidate directory: lists all task dirs
    # (task.toml parents), status "generated_unverified" if any else "blocked"; if exactly one
    # quality result exists and verified_result accepts it for the same URL, status becomes
    # "verified" with the proof, otherwise a reason is recorded. Adds pipeline_status from
    # report.json (when present) and cleanup_pending receipts. Returns the dict.
    # Keep every revision, failed build and generated bundle. No target imports.
    tasks = sorted({str(path.parent.resolve()) for path in directory.rglob("task.toml")})
    row = {"url": url, "status": "generated_unverified" if tasks else "blocked", "tasks": tasks}
    reviews = sorted(directory.glob("candidates/*/quality/result.json"))
    if len(reviews) == 1:
        try:
            proof = verified_result(reviews[0])
            if proof["url"] != url:
                raise ValueError("Reviewed task belongs to a different PR")
            row.update(status="verified", verified=proof)
        except (ValueError, KeyError, OSError) as exc:
            row["reason"] = f"{type(exc).__name__}: quality evidence is incomplete or invalid"
    report = directory / "report.json"
    if report.is_file():
        rows = json.loads(report.read_text()).get("candidates", [])
        row["pipeline_status"] = rows[0].get("status") if len(rows) == 1 else "incomplete"
    row["cleanup_pending"] = _cleanup_pending(directory)
    return row


def _run_candidate(configuration: dict, item: dict) -> dict:
    # [FUNCTION DOC] HELPER: Entry point executed inside the isolated child process. Validates the
    # Candidate, returns an existing batch-result.json if present; verifies the controller hash and
    # runtime wheel identity match the frozen configuration (ValueError otherwise). Creates a
    # Tasksmith runner with a per-candidate prefix and a RunBudget nested inside the shared batch
    # RunBudget, then runs it on a single-PR Panel with the candidate's reuse/prepared inputs.
    # Exceptions are caught and summarised as a reason. For the modal provider, reconciles
    # terminated worker receipts whose ledger operation is not yet settled. Writes and returns the
    # outcome row with budget totals.
    """One child process owns one Tasksmith instance and its remote worker."""
    from repo2rlenv.tasksmith.matrix_runner import reconcile_worker
    from repo2rlenv.tasksmith.runner import Tasksmith

    candidate = Candidate.model_validate(item)
    root = Path(configuration["directory"])
    directory = root / "candidates" / _key(candidate.url)
    result_path = directory / "batch-result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if _controller_identity() != configuration["controller_sha256"]:
        raise ValueError("Controller changed since this batch was frozen")
    wheel = Path(configuration["wheel"])
    if check_runtime_wheel(wheel) != configuration["runtime"]:
        raise ValueError("Frozen runtime changed")
    runner = Tasksmith(directory, Path(configuration["campaign"]), candidate.options, wheel)
    runner.prefix = _prefix(root) + "-" + _key(candidate.url)
    shared = RunBudget(
        runner.ledger, _prefix(root), configuration["plan"]["max_spend_usd"], shared=True
    )
    runner.budget = RunBudget(shared, runner.prefix, candidate.options.max_spend_usd)
    failure = None
    try:
        runner.run(
            Panel(name="pr-" + _key(candidate.url), prs=[candidate.url]),
            generation_run=candidate.generation_run,
            reuse_evidence=candidate.reuse_evidence,
            source_records=[candidate.source_record] if candidate.source_record else None,
            prepared_task=candidate.prepared_task,
            prepared_probes=candidate.prepared_probes,
        )
    except Exception as exc:
        failure = f"{type(exc).__name__}: inspect the preserved candidate receipts and logs"
    finally:
        # Only confirmed Modal termination permits allocation-time reconciliation.
        # Other providers and uncertain creates retain their full reservation.
        if candidate.options.provider == "modal":
            operations = {item["id"]: item for item in runner.ledger.status()["operations"]}
            for receipt in (directory / "workers").glob("*.json"):
                if receipt.name.endswith(".cost.json"):
                    continue
                record = json.loads(receipt.read_text())
                operation = operations.get(record.get("operation_id"), {})
                if record.get("state") == "terminated" and operation.get("status") != "settled":
                    reconcile_worker(receipt, runner.budget)
    row = _outcome(directory, candidate.url)
    if failure:
        row["reason"] = failure
    row["budget"] = runner.budget.totals()
    save_record(result_path, row)
    return row


def _available(budget: RunBudget) -> Decimal:
    # [FUNCTION DOC] HELPER: Returns the Decimal remaining spend: min(batch limit - batch totals,
    # ledger-reported remaining_usd).
    batch_used = sum(Decimal(value) for value in budget.totals().values())
    return min(budget.limit - batch_used, Decimal(budget.status()["remaining_usd"]))


def _report(configuration: dict, entries: dict, budget: RunBudget) -> dict:
    # [FUNCTION DOC] HELPER: Builds and saves report.json: combines prior verified proofs with newly
    # verified entries, raises ValueError on duplicate URLs or bundle hashes, and counts verified /
    # prior / new / generated_unverified / blocked plus budget totals and available USD. Returns it.
    verified = [
        *configuration["prior_verified"],
        *[row["verified"] for row in entries.values() if row.get("status") == "verified"],
    ]
    # Never turn repeated PRs or repeated bundles into additional successes.
    urls = {item["url"] for item in verified}
    bundles = {item["bundle_hash"] for item in verified}
    if len(urls) != len(verified) or len(bundles) != len(verified):
        raise ValueError("Duplicate accepted PR or bundle detected")
    result = {
        "schema_version": "1",
        "target_verified": configuration["plan"]["target_verified"],
        "verified": len(verified),
        "prior_verified": len(configuration["prior_verified"]),
        "new_verified": sum(row.get("status") == "verified" for row in entries.values()),
        "generated_unverified": sum(
            row.get("status") == "generated_unverified" for row in entries.values()
        ),
        "blocked": sum(row.get("status") == "blocked" for row in entries.values()),
        "candidates": entries,
        "budget": budget.totals(),
        "available_usd": str(_available(budget)),
    }
    save_record(Path(configuration["directory"]) / "report.json", result)
    return result


def run_batch(
    plan: BatchPlan,
    directory: Path,
    campaign: Path,
    wheel: Path,
    *,
    on_event=print,
    expected_prior_verified: list[dict] | None = None,
    preallocation_check: Callable[[], None] | None = None,
):
    # [FUNCTION DOC] Runs or resumes a batch and returns the report dict (with stop_reason and
    # pending). Inputs: plan (BatchPlan), directory (batch output dir), campaign (dir holding
    # budget.sqlite3), wheel (runtime wheel), on_event (callback for progress strings, default
    # print), expected_prior_verified (optional list that freshly computed prior proofs must equal),
    # preallocation_check (optional callable run before any budget allocation). Takes an exclusive
    # file lock on directory/.lock. Validates prior results (no duplicate URLs), writes or verifies
    # configuration.json (frozen plan, paths, runtime and controller hash; mismatch raises
    # ValueError), copies the wheel and freezes the controller. Reloads saved entries: rows left
    # "running" become their child result or "blocked/interrupted_controller"; verified proofs are
    # re-checked; cleanup_pending lists refreshed. If any entry needs reconciliation it stops
    # immediately with stop_reason "reconciliation_required". Otherwise loops with a thread pool of
    # max_parallel supervisors, admitting pending PRs while slots remain (bounded by
    # target_verified), the worker reservation fits the available budget and the GPU parallel cap
    # allows; admission halts once a drain-request.json exists or reconciliation is needed.
    # Supervisor exceptions mark the PR blocked. Final stop_reason: reconciliation_required,
    # target_reached, drained, budget_headroom or panel_exhausted. Side effects: filesystem,
    # subprocesses, ledger.
    # KNOWN ISSUE: on_event defaults to print(), violating the project rule that library code must
    # use repo2rlenv.ui.console instead of print.
    """Resume undispatched PRs; uncertain child/provider work is never redispatched."""
    directory, campaign, wheel = directory.resolve(), campaign.resolve(), wheel.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        lock_file(lock)
        prior = [verified_result(path) for path in plan.prior_verified]
        if len({item["url"] for item in prior}) != len(prior):
            raise ValueError("Prior verified results contain duplicate PRs")
        # Campaign preparation can bind its approved inventory without repeating
        # expensive evidence scans. This fresh check remains before allocation.
        if expected_prior_verified is not None and prior != expected_prior_verified:
            raise ValueError("Fresh prior evidence differs from the approved inventory")
        frozen_wheel = directory / "runtime" / wheel.name
        controller_root = directory / "runtime" / "controller"
        configuration = {
            "schema_version": "1",
            "plan": plan.model_dump(mode="json"),
            "directory": str(directory),
            "campaign": str(campaign),
            "wheel": str(frozen_wheel),
            "runtime": check_runtime_wheel(wheel),
            "controller_sha256": _controller_identity(),
            "controller_root": str(controller_root),
            "prior_verified": prior,
        }
        manifest = directory / "configuration.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != configuration:
                raise ValueError("Frozen batch inputs changed; use a new batch directory")
            if check_runtime_wheel(frozen_wheel) != configuration["runtime"]:
                raise ValueError("Frozen runtime changed")
        else:
            frozen_wheel.parent.mkdir(exist_ok=True)
            shutil.copyfile(wheel, frozen_wheel)
            save_record(manifest, configuration)
        _freeze_controller(frozen_wheel, controller_root)
        if preallocation_check is not None:
            preallocation_check()
        ledger = BudgetLedger(campaign / "budget.sqlite3")
        budget = RunBudget(ledger, _prefix(directory), plan.max_spend_usd)
        saved = directory / "report.json"
        entries = json.loads(saved.read_text())["candidates"] if saved.exists() else {}
        if not set(entries) <= {item.url for item in plan.candidates}:
            raise ValueError("Saved batch results contain a PR outside the frozen panel")
        for url, row in list(entries.items()):
            if row["status"] == "running" or row.get("reason_code") == "interrupted_controller":
                path = directory / "candidates" / _key(url) / "batch-result.json"
                entries[url] = (
                    json.loads(path.read_text())
                    if path.exists()
                    else {
                        **row,
                        "status": "blocked",
                        "reason_code": "interrupted_controller",
                        "reason": "Interrupted controller; reconcile child and provider receipts before any retry",
                    }
                )
            if entries[url]["status"] == "verified":
                proof = entries[url]["verified"]
                if proof["url"] != url or verified_result(Path(proof["quality_result"])) != proof:
                    raise ValueError("A previously accepted child result changed")
            if entries[url].get("cleanup_pending"):
                missing = [
                    path for path in entries[url]["cleanup_pending"] if not Path(path).is_file()
                ]
                entries[url]["cleanup_pending"] = sorted(
                    set(missing + _cleanup_pending(directory / "candidates" / _key(url)))
                )
        existing = {item["url"] for item in prior}
        pending = [
            item for item in plan.candidates if item.url not in entries and item.url not in existing
        ]
        report = _report(configuration, entries, budget)
        if any(
            row.get("reason_code") == "interrupted_controller" or row.get("cleanup_pending")
            for row in entries.values()
        ):
            report.update(
                stop_reason="reconciliation_required", pending=[item.url for item in pending]
            )
            save_record(saved, report)
            return report
        active = {}
        gpu_urls = {item.url for item in plan.candidates if item.options.gpus}
        reconcile = False
        drain = False
        with ThreadPoolExecutor(max_workers=plan.max_parallel) as pool:
            while pending or active:
                # Admission can be stopped without signalling child processes.
                # The marker is a one-way request for this invocation; retain it
                # alongside completed receipts when preparing a continuation.
                drain = drain or (directory / "drain-request.json").exists()
                slots = min(
                    plan.max_parallel - len(active),
                    plan.target_verified - report["verified"] - len(active),
                )
                if reconcile or drain:
                    slots = 0
                while slots > 0 and pending:
                    # Admission is advisory; nested ledger reservations enforce both
                    # caps atomically when a child actually dispatches an effect.
                    affordable = next(
                        (
                            index
                            for index, item in enumerate(pending)
                            if Decimal(item.options.worker_reservation_usd) <= _available(budget)
                            and (
                                not item.options.gpus
                                or sum(url in gpu_urls for url in active.values())
                                < plan.max_gpu_parallel
                            )
                        ),
                        None,
                    )
                    if affordable is None:
                        break
                    item = pending.pop(affordable)
                    entries[item.url] = {"url": item.url, "status": "running"}
                    report = _report(configuration, entries, budget)
                    future = pool.submit(
                        _supervise_candidate, configuration, item.model_dump(mode="json")
                    )
                    active[future] = item.url
                    on_event(f"Started {item.url}; {len(active)} isolated PR workers")
                    slots -= 1
                if not active:
                    break
                completed, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    url = active.pop(future)
                    try:
                        entries[url] = future.result()
                    except Exception as exc:
                        entries[url] = {
                            "url": url,
                            "status": "blocked",
                            "reason_code": "interrupted_controller",
                            "reason": f"{type(exc).__name__}: inspect retained child receipts before retrying",
                        }
                    reconcile = (
                        reconcile
                        or bool(entries[url].get("cleanup_pending"))
                        or entries[url].get("reason_code") == "interrupted_controller"
                    )
                    on_event(f"{url}: {entries[url]['status']}")
                report = _report(configuration, entries, budget)
        report["stop_reason"] = (
            "reconciliation_required"
            if reconcile
            else "target_reached"
            if report["verified"] >= plan.target_verified
            else "drained"
            if drain
            else "budget_headroom"
            if pending
            else "panel_exhausted"
        )
        report["pending"] = [item.url for item in pending]
        save_record(saved, report)
        return report


def main():
    # [FUNCTION DOC] Script entry point: parses positional plan (JSON) and output paths plus
    # required --campaign and --runtime-wheel, runs run_batch, and prints the report without the
    # candidates map as indented JSON.
    # KNOWN ISSUE: uses print() directly instead of repo2rlenv.ui.console (project invariant).
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--runtime-wheel", type=Path, required=True)
    args = parser.parse_args()
    plan = BatchPlan.model_validate_json(args.plan.read_text())
    result = run_batch(plan, args.output, args.campaign, args.runtime_wheel)
    print(
        json.dumps({key: value for key, value in result.items() if key != "candidates"}, indent=2)
    )


if __name__ == "__main__":
    main()
