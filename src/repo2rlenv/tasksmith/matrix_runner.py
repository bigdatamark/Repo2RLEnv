# [MODULE DOC] Controller-side orchestration of remote bootstrap readiness runs. The CPU matrix runs
# bootstrap_matrix on one provisioned remote worker and snapshots its Docker cache; the GPU path
# builds a CUDA image on Modal and validates it in a network-blocked L4 sandbox. Target code never
# executes on the controller. Compute cost is reconciled into the shared budget ledger with a
# conservative estimate after confirmed termination.
"""Controller for remote bootstrap readiness; target code never executes here."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.execution.artifacts import (
    check_runtime_wheel,
    install_runtime,
    runtime_python,
    unpack_evidence,
)
from repo2rlenv.execution.base import WorkerSpec
from repo2rlenv.execution.jobs import launch_job, observe_job
from repo2rlenv.execution.lifecycle import (
    now,
    prepare_docker,
    provision_worker,
    save_record,
    stop_worker,
)
from repo2rlenv.quality.loop.client import RunBudget
from repo2rlenv.tasksmith.bootstrap_matrix import RepositoryBootstrap, dockerfile


def reconcile_worker(receipt: Path, budget: RunBudget, *, gpu_rate: str = "0") -> dict:
    # [FUNCTION DOC] Settles a worker's ledger reservation with a conservative compute estimate.
    # Reads the JSON receipt (requires state "terminated" or "build_failed", else ValueError),
    # computes seconds between started_at and stopped_at, a per-second rate from spec cpus and
    # memory_mb plus gpu_rate (str USD/s, default "0"), and estimate = seconds * rate * 2 + 1.00
    # rounded up to 1e-6. Writes <receipt>.cost.json evidence and settles record["operation_id"] in
    # budget with it. Returns the evidence dict.
    record = json.loads(receipt.read_text())
    if record["state"] not in {"terminated", "build_failed"}:
        raise ValueError("Compute reconciliation requires confirmed termination")
    seconds = Decimal(
        str(
            (
                datetime.fromisoformat(record["stopped_at"])
                - datetime.fromisoformat(record["started_at"])
            ).total_seconds()
        )
    )
    spec = record["spec"]
    # Conservative full allocation, doubled, plus an explicit image-build allowance.
    rate = (
        Decimal("0.00003942") * spec["cpus"]
        + Decimal("0.00000667") * spec["memory_mb"] / 1024
        + Decimal(gpu_rate)
    )
    estimate = (seconds * rate * 2 + Decimal("1.00")).quantize(
        Decimal("0.000001"), rounding=ROUND_CEILING
    )
    evidence = {
        "kind": "conservative_estimate_not_invoice",
        "seconds": str(seconds),
        "rate_per_second": str(rate),
        "safety_factor": 2,
        "build_allowance_usd": "1.00",
        "accounted_usd": str(estimate),
        "pricing_source": "https://modal.com/pricing",
    }
    path = receipt.with_suffix(".cost.json")
    save_record(path, evidence)
    budget.settle(record["operation_id"], str(estimate), evidence=str(path.resolve()))
    return evidence


def run_cpu_matrix(
    config: Path, output: Path, campaign: Path, wheel: Path, *, on_event=print
) -> dict:
    # [FUNCTION DOC] Runs the CPU readiness matrix remotely and returns the report dict. Inputs:
    # config (JSON list of RepositoryBootstrap), output dir, campaign dir (budget.sqlite3), wheel
    # (runtime wheel), on_event (progress callback, default print). Validates the wheel and unique
    # repositories, computes an identity (repos, recipes, runtime, smoke script hash). If
    # configuration.json exists: identity must match (ValueError) and a completed previous report
    # whose worker receipts are all terminated is returned; any other existing state raises
    # ValueError. Otherwise saves the identity, provisions a 4-CPU/16 GB worker reserving $12 under
    # a shared $30 RunBudget, prepares Docker, installs the runtime, uploads the config and runs
    # bootstrap_matrix as a remote job (12600s), raising RuntimeError if it does not complete.
    # Downloads build-logs.tar.gz and readiness.tar.gz, unpacks evidence, loads the remote report,
    # snapshots the worker filesystem (7-day TTL) and marks the report completed with snapshot
    # id/TTL/timestamp. Always stops the worker, reconciles compute and saves report.json with
    # budget totals.
    # KNOWN ISSUE: on_event defaults to print(), contrary to the project's console-only output rule.
    runtime_identity = check_runtime_wheel(wheel)
    specs = [RepositoryBootstrap.model_validate(item) for item in json.loads(config.read_text())]
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Bootstrap repositories must be unique")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "configuration.json"
    identity = {
        "repositories": [spec.model_dump() for spec in specs],
        "resource": "cpu",
        "runtime": runtime_identity,
        "recipes": {spec.name: dockerfile(spec, "cpu") for spec in specs},
        "smoke_sha256": hashlib.sha256(
            Path(__file__).with_name("bootstrap_smoke.py").read_bytes()
        ).hexdigest(),
    }
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("Bootstrap inputs changed; preserve this run and use a new output")
        if (output / "report.json").exists():
            previous = json.loads((output / "report.json").read_text())
            receipts = [
                path
                for path in (output / "workers").glob("*.json")
                if not path.name.endswith(".cost.json")
            ]
            if (
                previous.get("status") == "completed"
                and receipts
                and all(
                    json.loads(path.read_text()).get("state") == "terminated"
                    for path in receipts
                    if not path.name.endswith(".cost.json")
                )
            ):
                return previous
        raise ValueError(
            "Incomplete bootstrap exists; inspect its worker and supervisor before retrying"
        )
    save_record(manifest, identity)
    shared_prefix = (
        "tsboot-" + hashlib.sha256(str(output.parent.resolve()).encode()).hexdigest()[:12]
    )
    prefix = shared_prefix + "-" + hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:6]
    ledger = BudgetLedger(campaign / "budget.sqlite3")
    budget = RunBudget(ledger, shared_prefix, "30.00")
    worker, receipt = provision_worker(
        WorkerSpec(name=prefix, cpus=4, memory_mb=16384, timeout_sec=14400),
        output / "workers",
        budget,
        reserve_usd="12.00",
    )
    report = {"resource": "cpu", "status": "incomplete", "repositories": []}
    try:
        on_event(f"CPU bootstrap worker {worker.id} ready")
        prepare_docker(worker)
        python = runtime_python(install_runtime(worker, wheel, output / "runtime"))
        worker.upload(config, "/work/hf-bootstrap.json")
        launch_job(
            worker,
            "/work/hf-bootstrap-job",
            [
                python,
                "-m",
                "repo2rlenv.tasksmith.bootstrap_matrix",
                "/work/hf-bootstrap.json",
                "/work/hf-bootstrap",
            ],
            timeout_sec=12600,
            python=python,
        )
        observed = observe_job(worker, "/work/hf-bootstrap-job", timeout_sec=12700)
        if observed["state"] != "completed":
            raise RuntimeError("CPU bootstrap supervisor failed; inspect the retained logs")
        worker.exec(
            [
                "tar",
                "-czf",
                "/work/hf-bootstrap.tar.gz",
                "-C",
                "/work",
                "hf-bootstrap",
                "hf-bootstrap-cache",
            ],
            timeout=120,
        ).checked("Package bootstrap evidence")
        worker.download("/work/hf-bootstrap.tar.gz", output / "build-logs.tar.gz")
        # Download only evidence; images and target execution stay on the provider.
        worker.exec(
            ["tar", "-czf", "/work/readiness.tar.gz", "-C", "/work", "hf-bootstrap"], timeout=120
        ).checked("Package readiness")
        archive = output / "readiness.tar.gz"
        worker.download("/work/readiness.tar.gz", archive)
        unpack_evidence(archive, output, root_name="hf-bootstrap")
        report = json.loads((output / "hf-bootstrap/report.json").read_text())
        # Snapshot an idle Docker daemon, with no active builds or trial containers.
        worker.exec(["sync"], timeout=30).checked("Flush builder filesystem")
        snapshot = worker.sandbox.snapshot_filesystem(timeout=55, ttl=7 * 24 * 3600)
        report.update(
            resource="cpu",
            snapshot_id=snapshot.object_id,
            snapshot_ttl_sec=7 * 24 * 3600,
            snapshot_created_at=now(),
            status="completed",
        )
        save_record(output / "report.json", report)
        on_event("CPU Docker cache snapshot recorded")
    finally:
        stop_worker(receipt, budget)
        report["compute"] = reconcile_worker(receipt, budget)
        report.update(budget.totals())
        save_record(output / "report.json", report)
    return report


def run_gpu_repository(
    spec: RepositoryBootstrap, output: Path, campaign: Path, *, on_event=print
) -> dict:
    # [FUNCTION DOC] Builds one repository's CUDA image on Modal and runs the GPU smoke in an L4
    # sandbox; returns the report dict (status ready/failed/incomplete). Inputs: spec
    # (RepositoryBootstrap), output dir, campaign dir, on_event (default print). If operation.json
    # exists, returns the saved report when source, smoke hash and recipe match and the state is
    # terminated/build_failed, else raises ValueError (never redispatches). Otherwise writes the
    # clone-mode GPU Dockerfile, reserves $6 in the shared ledger, records an operation receipt,
    # builds the image and creates a network-blocked sandbox (2 CPUs, 8 GB, L4, 1800s), uploads and
    # runs bootstrap_smoke.py, saves smoke stdout/stderr and pip-freeze.txt, and marks ready with
    # image id, parsed smoke JSON and Dockerfile hash. ImageBuildError marks build_failed and
    # reconciles cost; other exceptions mark the report failed (not re-raised). Finally terminates
    # the sandbox (on failure marks termination_uncertain in the ledger and re-raises) and
    # reconciles with a GPU rate, or marks creation_uncertain if no sandbox was created and the
    # build did not fail; saves report.json with budget totals.
    # KNOWN ISSUE: on_event defaults to print(), contrary to the project's console-only output rule.
    """Build on the provider, then reserve a native L4 sandbox for real CUDA checks."""
    import modal

    from repo2rlenv.execution.modal import APP_NAME, ModalWorker

    output.mkdir(parents=True, exist_ok=True)
    shared_prefix = (
        "tsboot-" + hashlib.sha256(str(output.parent.parent.resolve()).encode()).hexdigest()[:12]
    )
    prefix = (
        shared_prefix + "-gpu-" + hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:6]
    )
    budget = RunBudget(BudgetLedger(campaign / "budget.sqlite3"), shared_prefix, "30.00")
    recipe = output / "Dockerfile"
    recipe_text = dockerfile(spec, "gpu", clone=True)
    smoke_hash = hashlib.sha256(
        Path(__file__).with_name("bootstrap_smoke.py").read_bytes()
    ).hexdigest()
    if (output / "operation.json").exists():
        previous = json.loads((output / "operation.json").read_text())
        saved = output / "report.json"
        if (
            previous.get("source") == spec.model_dump()
            and previous.get("smoke_sha256") == smoke_hash
            and recipe.read_text() == recipe_text
            and previous["state"] in {"terminated", "build_failed"}
            and saved.exists()
        ):
            return json.loads(saved.read_text())
        raise ValueError("GPU attempt already exists; inspect its receipt instead of redispatching")
    recipe.write_text(recipe_text)
    operation = "worker:modal:" + prefix
    budget.reserve(operation, "6.00", f"Build and GPU smoke for {spec.name}")
    record = {
        "state": "creating",
        "operation_id": operation,
        "started_at": now(),
        "source": spec.model_dump(),
        "smoke_sha256": smoke_hash,
        "spec": {"cpus": 2, "memory_mb": 8192, "gpu": "L4", "name": prefix},
    }
    receipt = output / "operation.json"
    save_record(receipt, record)
    sandbox = None
    report = {"source": spec.model_dump(), "resource": "gpu", "status": "incomplete"}
    try:
        image = modal.Image.from_dockerfile(recipe, context_dir=output)
        on_event(f"Build {spec.name} CUDA image and validate on L4")
        sandbox = modal.Sandbox.create(
            "sleep",
            "1800",
            app=modal.App.lookup(APP_NAME, create_if_missing=True),
            name=prefix,
            image=image,
            gpu="L4",
            cpu=(2, 2),
            memory=8192,
            timeout=1800,
            block_network=True,
        )
        record.update(
            state="running", worker_id=sandbox.object_id, ready_at=now(), image_id=image.object_id
        )
        save_record(receipt, record)
        worker = ModalWorker(sandbox)
        worker.upload(Path(__file__).with_name("bootstrap_smoke.py"), "/tmp/bootstrap-smoke.py")
        result = worker.exec(["python", "/tmp/bootstrap-smoke.py", spec.name, "gpu"], timeout=180)
        (output / "smoke.stdout").write_text(result.stdout)
        (output / "smoke.stderr").write_text(result.stderr)
        worker.download("/opt/bootstrap-freeze.txt", output / "pip-freeze.txt")
        result.checked("GPU behavioral smoke")
        report.update(
            status="ready",
            image_id=image.object_id,
            smoke=json.loads(result.stdout),
            dockerfile_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),
        )
    except modal.exception.ImageBuildError as exc:
        # Modal hydrates the image before creating a sandbox. A failed build
        # has a known outcome, but still incurs image-build compute.
        record.update(state="build_failed", stopped_at=now(), no_sandbox_dispatched=True)
        save_record(receipt, record)
        report.update(status="failed", error=f"ImageBuildError: {exc}")
        report["compute"] = reconcile_worker(receipt, budget)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if sandbox is not None:
            record["state"] = "terminating"
            save_record(receipt, record)
            try:
                sandbox.terminate(wait=True)
            except BaseException:
                record["state"] = "termination_uncertain"
                save_record(receipt, record)
                budget.mark_uncertain(operation, f"Termination uncertain; receipt={receipt}")
                raise
            record.update(state="terminated", stopped_at=now())
            save_record(receipt, record)
            report["compute"] = reconcile_worker(receipt, budget, gpu_rate="0.000222")
        elif record["state"] != "build_failed":
            record["state"] = "creation_uncertain"
            save_record(receipt, record)
            budget.mark_uncertain(operation, f"Inspect provider by name; receipt={receipt}")
        report.update(budget.totals())
        save_record(output / "report.json", report)
    return report
