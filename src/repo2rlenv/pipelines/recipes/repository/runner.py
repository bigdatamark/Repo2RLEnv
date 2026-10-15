# [MODULE DOC] Shared local controller for owned recipes that generate candidates on a remote
# worker. It claims a run receipt under the campaign directory (hash-pinned to the configuration and
# runtime wheel), installs the runtime on an already-running worker, launches the recipe's worker
# module as a remote job, downloads and unpacks the evidence, then calls the subclass's
# author_export for each candidate until the target count, recording exported bundle hashes for
# resumability.
"""Shared controller for remote repository generators and owned task authors."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from repo2rlenv.bootstrap.spec import LanguageHint
from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.campaigns.events import ProgressEvent
from repo2rlenv.emitter.bundle import inspect_bundle
from repo2rlenv.execution.artifacts import (
    check_runtime_wheel,
    install_runtime,
    runtime_python,
    unpack_evidence,
)
from repo2rlenv.execution.base import connect_worker
from repo2rlenv.execution.jobs import launch_job, observe_job
from repo2rlenv.execution.lifecycle import prepare_docker, save_record
from repo2rlenv.pipelines.base import PipelineResult
from repo2rlenv.spec.input import GenerationInput, RepositorySource


class RemoteGenerationPipeline:
    # [CLASS DOC] Base class for remote-generation recipes (implements the Pipeline protocol). Class
    # attributes: recipe_id and worker_module (set by subclasses), requires_bootstrap False
    # (bootstrap happens in the worker), supported_languages {PYTHON}, experimental True,
    # native_supported False. Subclasses implement author_export and may override
    # worker_configuration and source_identity.
    recipe_id: str
    worker_module: str
    requires_bootstrap = False  # The existing bootstrap runs inside the worker.
    supported_languages = frozenset({LanguageHint.PYTHON})
    experimental = True
    native_supported = False

    def __init__(self, input: GenerationInput, options, bootstrap=None):
        # [FUNCTION DOC] Validate the input and store it.
        # Inputs: input (GenerationInput), options (recipe options model), bootstrap (ignored).
        # Raises ValueError if input.pipeline.recipe differs from recipe_id or input.execution is
        # None. Sets self.input, self.options and a no-op on_event callback.
        if input.pipeline.recipe != self.recipe_id:
            raise ValueError(f"This pipeline requires --recipe {self.recipe_id}")
        if input.execution is None:
            raise ValueError("Owned generation requires remote execution settings")
        self.input, self.options = input, options
        self.on_event: Callable[[ProgressEvent], None] = lambda event: None

    def set_event_callback(self, callback: Callable[[ProgressEvent], None]) -> None:
        # [FUNCTION DOC] Replace the progress callback (receives ProgressEvent objects). Input:
        # callback. Output: None.
        self.on_event = callback

    def event(self, stage: str, state: str, message: str, **metrics) -> None:
        # [FUNCTION DOC] Emit a ProgressEvent tagged with recipe_id through on_event.
        # Inputs: stage, state, message (str), **metrics (attached as metrics). Output: None.
        self.on_event(
            ProgressEvent(
                recipe=self.recipe_id, stage=stage, state=state, message=message, metrics=metrics
            )
        )

    def worker_configuration(self, run: Path) -> dict:
        # [FUNCTION DOC] Build the JSON config uploaded to the worker.
        # Input: run (Path, unused here). Output: dict with repo and options dumped in JSON mode.
        return {
            "repo": self.input.repo.model_dump(mode="json"),
            "options": self.options.model_dump(mode="json"),
        }

    def source_identity(self) -> dict:
        # [FUNCTION DOC] Extra identity folded into the run configuration hash (e.g. input file
        # digests). Default: empty dict.
        return {}

    def run(self, out_dir: Path) -> PipelineResult:
        # [FUNCTION DOC] Execute or resume the full remote run; return a PipelineResult.
        # Input: out_dir (Path for exported tasks).
        # Behavior: run dir is campaign_dir/runs/run_id with receipt run.json. Configuration hash =
        # sha256 of the input dump (minus execution.resume) plus runtime wheel sha256 and
        # source_identity. Existing receipt requires execution.resume and a matching hash; otherwise
        # the receipt is created exclusively in state "claimed". Unless already downloaded, requires
        # the worker receipt to be "running" on the same campaign budget ledger and (if dispatched)
        # the same worker id, and at least 60s left in the worker window (timeout = min(execution
        # timeout, remaining-15)). On "claimed": installs the runtime, creates remote dirs, uploads
        # worker-config.json, records state "remote_dispatched", and launches python -m
        # worker_module config output. Then observes the job (forwarding events), saves remote-
        # job.json, downloads stderr and raises RuntimeError on failure, tars and downloads
        # evidence, unpacks it, sets state "downloaded". Then (state "exporting") iterates
        # candidates until options.target exports: previously exported tasks are re-verified by
        # bundle hash (ValueError if changed); author_export ValueErrors count as skip
        # "task_materialization". Records each export's path and bundle hash, finally state
        # "completed" with skip counts.
        # Output: PipelineResult(candidates=attempted, emitted, skipped, out_dir, skip_reasons).
        # Raises: FileExistsError, ValueError, RuntimeError as above. Side effects: filesystem
        # receipts, remote worker commands, uploads/downloads, progress events.
        execution = self.input.execution
        run = execution.campaign_dir / "runs" / execution.run_id
        receipt = run / "run.json"
        wheel_hash = check_runtime_wheel(execution.runtime_wheel)
        configuration = self.input.model_dump(mode="json")
        configuration["execution"].pop("resume", None)
        configuration["runtime_sha256"] = wheel_hash
        identity = self.source_identity()
        if identity:
            configuration["source_identity"] = identity
        config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
        if receipt.exists():
            if not execution.resume:
                raise FileExistsError("Run already exists; resume it explicitly")
            record = json.loads(receipt.read_text())
            if record["config_sha256"] != config_hash:
                raise ValueError("Run configuration or runtime code changed; use a new run ID")
        else:
            run.mkdir(parents=True, exist_ok=True)
            with receipt.open("x"):
                pass
            record = {
                "state": "claimed",
                "config_sha256": config_hash,
                "configuration": configuration,
                "exported": {},
            }
            save_record(receipt, record)
        ledger = BudgetLedger(execution.campaign_dir / "budget.sqlite3")
        generation = run / execution.run_id
        if record["state"] not in {"downloaded", "exporting", "completed"}:
            worker_record = json.loads(execution.worker_receipt.read_text())
            if (
                worker_record["state"] != "running"
                or Path(worker_record["ledger"]).resolve() != ledger.path.resolve()
            ):
                raise ValueError("Run requires a running worker from the same campaign")
            if record.get("worker_id", worker_record["worker_id"]) != worker_record["worker_id"]:
                raise ValueError("A dispatched run must resume on its original worker")
            deadline = datetime.fromisoformat(worker_record["started_at"]) + timedelta(
                seconds=worker_record["spec"]["timeout_sec"]
            )
            remaining = int((deadline - datetime.now(UTC)).total_seconds())
            if remaining < 60:
                raise ValueError(
                    "Worker execution window expired; reconcile this run before dispatch"
                )
            timeout = min(execution.timeout_sec, remaining - 15)
            worker = connect_worker(worker_record["spec"]["provider"], worker_record["worker_id"])
            prepare_docker(worker)
            remote = "/work/generation/" + execution.run_id
            remote_output = "/evidence/generation/" + execution.run_id
            if record["state"] == "claimed":
                self.event("runtime", "started", "Install the verified owned runtime")
                install_runtime(worker, execution.runtime_wheel, run)
                worker.exec(
                    ["mkdir", "-p", "/work/generation", "/evidence/generation"], timeout=30
                ).checked("Generation parents")
                worker.exec(["mkdir", remote], timeout=30).checked("Claim remote generation")
                save_record(
                    run / "worker-config.json",
                    self.worker_configuration(run),
                )
                worker.upload(run / "worker-config.json", remote + "/config.json")
                record.update(
                    state="remote_dispatched",
                    worker_id=worker.id,
                    remote=remote,
                    remote_output=remote_output,
                )
                save_record(receipt, record)
                launch_job(
                    worker,
                    remote,
                    [
                        runtime_python(wheel_hash),
                        "-m",
                        self.worker_module,
                        remote + "/config.json",
                        remote_output,
                    ],
                    timeout_sec=timeout,
                    python=runtime_python(wheel_hash),
                )
            self.event("generation", "started", "Generate candidates in the remote worker")
            status = observe_job(
                worker,
                remote,
                event_file=remote_output + "/events.jsonl",
                timeout_sec=timeout + 10,
                on_event=self.on_event,
            )
            save_record(run / "remote-job.json", status)
            if status["state"] != "completed" or status.get("returncode") != 0:
                worker.download(remote + "/stderr.txt", run / "worker.stderr")
                raise RuntimeError("Remote generation failed; inspect the recorded worker stderr")
            worker.exec(
                [
                    "tar",
                    "-czf",
                    remote + "/generation.tar.gz",
                    "-C",
                    "/evidence/generation",
                    execution.run_id,
                ],
                timeout=120,
            ).checked("Archive generation evidence")
            worker.download(remote + "/generation.tar.gz", run / "generation.tar.gz")
            if not generation.exists():
                unpack_evidence(run / "generation.tar.gz", run, root_name=execution.run_id)
            record["state"] = "downloaded"
            save_record(receipt, record)
        result = json.loads((generation / "generation.json").read_text())
        skipped = Counter(item["reason"] for item in result["rejected"])
        record["state"] = "exporting"
        save_record(receipt, record)
        for candidate in result["candidates"]:
            if len(record["exported"]) >= self.options.target:
                break
            candidate_id = candidate["id"]
            if candidate_id in record["exported"]:
                task = Path(record["exported"][candidate_id]["path"])
                if (
                    inspect_bundle(task)["bundle_hash"]
                    != record["exported"][candidate_id]["bundle_hash"]
                ):
                    raise ValueError("An exported task changed after its receipt was written")
                continue
            self.event(
                "author",
                "started",
                "Author the task from recorded execution evidence",
                exported=len(record["exported"]),
            )
            try:
                task = self.author_export(generation, candidate, ledger, run, out_dir)
            except ValueError as exc:
                skipped["task_materialization"] += 1
                self.event("author", "failed", str(exc))
                continue
            record["exported"][candidate_id] = {
                "path": str(task.resolve()),
                "bundle_hash": inspect_bundle(task)["bundle_hash"],
            }
            save_record(receipt, record)
            self.event(
                "export", "completed", task.name, exported=len(record["exported"]), accepted=0
            )
        record.update(state="completed", skipped=dict(skipped), attempted=result["attempted"])
        save_record(receipt, record)
        return PipelineResult(
            candidates=result["attempted"],
            emitted=len(record["exported"]),
            skipped=sum(skipped.values()),
            out_dir=out_dir,
            skip_reasons=dict(skipped),
        )

    def author_export(self, generation, candidate, ledger, run, out_dir) -> Path:
        # [FUNCTION DOC] Abstract hook: author and export one candidate to out_dir and return the
        # task directory. Inputs: generation (unpacked evidence dir), candidate (dict), ledger
        # (BudgetLedger), run (run dir), out_dir. Raises NotImplementedError here.
        raise NotImplementedError


class RepositoryGenerationPipeline(RemoteGenerationPipeline):
    # [CLASS DOC] RemoteGenerationPipeline restricted to public GitHub repository sources and
    # requiring an LLM for issue/instruction writing.
    """Repository-specific input and model contract over the shared remote lifecycle."""

    def __init__(self, input: GenerationInput, options, bootstrap=None):
        # [FUNCTION DOC] Run base validation, then require a RepositorySource whose repo URL starts
        # with https://github.com/ and whose access is not private, and a non-None input.llm. Raises
        # ValueError otherwise.
        super().__init__(input, options, bootstrap)
        if (
            not isinstance(input.source, RepositorySource)
            or not input.repo.url.startswith("https://github.com/")
            or input.repo.access == "private"
        ):
            raise ValueError(
                "This recipe currently supports public GitHub Python repository profiles"
            )
        if input.llm is None:
            raise ValueError("Repository generation requires an issue-writing LLM")
