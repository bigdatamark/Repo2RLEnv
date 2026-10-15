# [MODULE DOC] Implements the pr_to_env pipeline (owned SWE-gen recipe, RFC 0015): turns a caller-
# supplied list of GitHub PR URLs into Harbor tasks. For each PR it fetches source metadata, runs
# the swe_gen worker remotely to bootstrap and reverse the PR's source changes (producing a
# defective snapshot plus contrast evidence), has an LLM write the task instruction, exports a draft
# repository task, gates it with fresh Harbor nop/oracle trials (rewards must be 0 and 1), and
# exports the accepted task to the output directory. Progress and per-PR outcomes are persisted in a
# resumable run receipt under the campaign directory.
"""Caller-supplied PRs to Harbor tasks through the owned SWE-gen recipe."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from repo2rlenv.bootstrap.spec import LanguageHint
from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.campaigns.events import ProgressEvent
from repo2rlenv.emitter.bundle import inspect_bundle
from repo2rlenv.execution.artifacts import check_runtime_wheel, install_runtime, runtime_python
from repo2rlenv.execution.base import connect_worker
from repo2rlenv.execution.generation import run_generator
from repo2rlenv.execution.harbor import run_trial
from repo2rlenv.execution.lifecycle import prepare_docker, save_record
from repo2rlenv.github import GitHubError
from repo2rlenv.pipelines.base import PipelineResult
from repo2rlenv.pipelines.recipes.catalog import get_recipe
from repo2rlenv.pipelines.recipes.repository.export import export_repository_task
from repo2rlenv.pipelines.recipes.swe_gen.instruction import write_instruction
from repo2rlenv.pipelines.recipes.swe_gen.source import fetch_source
from repo2rlenv.spec.input import GenerationInput, PipelineName, PRSource
from repo2rlenv.spec.recipe_options import PRRecipeOptions


class PRToEnvPipeline:
    # [CLASS DOC] Pipeline class registered as PipelineName.PR_TO_ENV. Class attributes:
    # requires_bootstrap False (bootstrap happens inside the remote worker), supported_languages
    # only Python, experimental True, native_supported False (only usable via the swe_gen recipe).
    # Instance attributes: input (GenerationInput), options (PRRecipeOptions), on_event (progress
    # callback, default no-op).
    name = PipelineName.PR_TO_ENV
    requires_bootstrap = False
    supported_languages = frozenset({LanguageHint.PYTHON})
    experimental = True
    native_supported = False

    def __init__(self, input: GenerationInput, options: PRRecipeOptions, bootstrap=None):
        # [FUNCTION DOC] Validates and stores configuration. Inputs: input (GenerationInput; must
        # have pipeline.recipe == "swe_gen", a PRSource source, an execution spec and an llm spec),
        # options (PRRecipeOptions), bootstrap (ignored; accepted for constructor compatibility).
        # Raises ValueError when any requirement is not met. Sets on_event to a no-op.
        if input.pipeline.recipe != "swe_gen" or not isinstance(input.source, PRSource):
            raise ValueError("pr_to_env requires explicit PR URLs and --recipe swe_gen")
        if input.execution is None or input.llm is None:
            raise ValueError("SWE-gen requires a remote worker and instruction model")
        self.input, self.options = input, options
        self.on_event = lambda event: None

    def set_event_callback(self, callback) -> None:
        # [FUNCTION DOC] Replaces the progress callback (called with ProgressEvent objects). Input:
        # callback (callable). Output: None.
        self.on_event = callback

    def run(self, out_dir: Path) -> PipelineResult:
        # [FUNCTION DOC] Runs (or resumes) the recipe and returns a PipelineResult. Input: out_dir
        # (Path where accepted tasks are exported). Steps: computes the run directory
        # campaign_dir/runs/run_id and receipt run.json; fingerprints the input settings (minus the
        # execution.resume flag) together with the runtime wheel digest (check_runtime_wheel). If
        # the receipt exists, requires resume=True and an identical fingerprint (else ValueError);
        # otherwise creates it with state "running". Re-checks every already-exported task's bundle
        # hash (ValueError if changed) and returns immediately if state is "completed". Loads the
        # budget ledger and worker receipt, requiring a running worker bound to this campaign's
        # ledger (ValueError), connects to the worker, prepares Docker, installs the runtime wheel,
        # and computes a deadline as the earlier of worker start + worker timeout and now +
        # execution timeout. Per URL (up to options.max_candidates, stopping once options.target
        # tasks exist, skipping URLs already recorded): raises TimeoutError if fewer than 900s
        # remain; fetches (or reloads cached) source.json via fetch_source, recording skip
        # "unsupported_source" on ValueError/GitHubError; runs the swe_gen worker module via
        # run_generator (skip "bootstrap_or_contrast" when it returns None); writes the instruction
        # via write_instruction (skip "substantiality" when not substantial); exports a draft task,
        # runs nop and oracle Harbor trials and skips "harbor_packaging" unless both completed with
        # rewards [0, 1]; finally exports to out_dir and records path, bundle info and trial result
        # paths. The receipt is saved after each change and set to "completed" at the end. Output:
        # PipelineResult(candidates = emitted + skipped, emitted, skipped, skip_reasons, out_dir).
        # Side effects: filesystem writes under the campaign and out_dir, remote worker execution,
        # LLM spend via the ledger, progress events.
        execution = self.input.execution
        run = execution.campaign_dir / "runs" / execution.run_id
        receipt = run / "run.json"
        digest = check_runtime_wheel(execution.runtime_wheel)
        settings = self.input.model_dump(mode="json")
        settings["execution"].pop("resume")
        fingerprint = hashlib.sha256(
            json.dumps({"input": settings, "runtime": digest}, sort_keys=True).encode()
        ).hexdigest()
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if not execution.resume or record["fingerprint"] != fingerprint:
                raise ValueError("Run exists; resume with the identical configuration and runtime")
        else:
            run.mkdir(parents=True, exist_ok=True)
            record = {
                "state": "running",
                "fingerprint": fingerprint,
                "configuration": settings,
                "tasks": {},
                "skipped": {},
            }
            save_record(receipt, record)

        def result():
            # [FUNCTION DOC] HELPER: Builds the PipelineResult from the current receipt record,
            # counting skip reasons from record["skipped"]. No inputs; output PipelineResult.
            skipped = Counter(record["skipped"].values())
            return PipelineResult(
                candidates=len(record["tasks"]) + sum(skipped.values()),
                emitted=len(record["tasks"]),
                skipped=sum(skipped.values()),
                skip_reasons=dict(skipped),
                out_dir=out_dir,
            )

        for task in record["tasks"].values():
            if inspect_bundle(Path(task["path"]))["bundle_hash"] != task["bundle_hash"]:
                raise ValueError("An exported task has changed")
        if record["state"] == "completed":
            return result()
        ledger = BudgetLedger(execution.campaign_dir / "budget.sqlite3")
        worker_record = json.loads(execution.worker_receipt.read_text())
        if (
            worker_record["state"] != "running"
            or Path(worker_record["ledger"]).resolve() != ledger.path.resolve()
        ):
            raise ValueError("Generation requires a running worker from this campaign")
        worker = connect_worker(worker_record["spec"]["provider"], worker_record["worker_id"])
        prepare_docker(worker)
        python = runtime_python(install_runtime(worker, execution.runtime_wheel, run))
        deadline = min(
            datetime.fromisoformat(worker_record["started_at"])
            + timedelta(seconds=worker_record["spec"]["timeout_sec"]),
            datetime.now(UTC) + timedelta(seconds=execution.timeout_sec),
        )
        recipe = get_recipe("swe_gen")
        prefix = hashlib.sha256(execution.run_id.encode()).hexdigest()[:10]

        def event(stage, state, message):
            # [FUNCTION DOC] HELPER: Emits a ProgressEvent via self.on_event for this recipe with
            # the given stage, state and message, and metrics exported = number of recorded tasks,
            # accepted = 0. Output: None.
            self.on_event(
                ProgressEvent(
                    recipe=recipe.id,
                    stage=stage,
                    state=state,
                    message=message,
                    metrics={"exported": len(record["tasks"]), "accepted": 0},
                )
            )

        for index, url in enumerate(self.input.source.urls[: self.options.max_candidates]):
            if len(record["tasks"]) >= self.options.target:
                break
            if url in record["tasks"] or url in record["skipped"]:
                continue
            if (deadline - datetime.now(UTC)).total_seconds() < 900:
                raise TimeoutError("Insufficient worker window for another PR; preserve the run")
            candidate_dir = run / "candidates" / hashlib.sha256(url.encode()).hexdigest()[:20]
            candidate_dir.mkdir(parents=True, exist_ok=True)
            source_path = candidate_dir / "source.json"
            event("source", "started", url)
            try:
                source = (
                    json.loads(source_path.read_text())
                    if source_path.exists()
                    else fetch_source(url, self.options)
                )
                save_record(source_path, source)
            except (ValueError, GitHubError) as exc:
                record["skipped"][url] = "unsupported_source"
                save_record(candidate_dir / "rejection.json", {"reason": str(exc)})
                save_record(receipt, record)
                event("source", "failed", str(exc))
                continue
            job_id = f"pr-{prefix}-{index:03d}"
            event("bootstrap", "started", source["title"])
            generation = run_generator(
                worker,
                python=python,
                module="repo2rlenv.pipelines.recipes.swe_gen.worker",
                config={"source": source, "options": self.options.model_dump(mode="json")},
                directory=candidate_dir / "generation",
                job_id=job_id,
                timeout_sec=min(1200, int((deadline - datetime.now(UTC)).total_seconds()) - 15),
                resume=execution.resume,
            )
            if generation is None:
                record["skipped"][url] = "bootstrap_or_contrast"
                save_record(receipt, record)
                event("contrast", "failed", "Retained bootstrap and test evidence")
                continue
            candidate = json.loads((generation / "candidate.json").read_text())
            event("instruction", "started", source["title"])
            instruction = write_instruction(
                candidate,
                self.options,
                self.input.llm,
                ledger,
                candidate_dir / "instruction.json",
                operation_id=f"instruction:{job_id}",
                resume=execution.resume,
            )
            if not instruction.is_substantial:
                record["skipped"][url] = "substantiality"
                save_record(receipt, record)
                event("instruction", "failed", instruction.reason)
                continue
            kwargs = dict(
                base=generation / "base",
                defective={
                    path: (generation / "defective-source" / path).read_bytes()
                    for path in source["source_files"]
                },
                reference={
                    path: (generation / "base" / path).read_bytes()
                    for path in source["source_files"]
                },
                options=self.options,
                instruction=instruction.instruction,
                name="swe-gen-" + source["id"],
                org=self.input.output.org,
                contrast=candidate["contrast"],
                metadata={
                    "recipe": recipe.id,
                    "recipe_version": "1",
                    "pipeline": recipe.pipeline,
                    "upstream_revision": recipe.upstream["commit"],
                    "repository": source["repo"],
                    "source_revision": source["head"],
                    "source_pr": url,
                    "force_generate_instruction": self.options.force_generate_instruction,
                },
                resume=execution.resume,
            )
            task = export_repository_task(destination=candidate_dir / "draft", **kwargs)
            event("harbor", "started", "Fresh baseline and reference execution")
            trials = [
                run_trial(
                    worker,
                    task,
                    candidate_dir / agent,
                    trial_id=f"{job_id}-{agent}",
                    agent=agent,
                    python=python,
                    resume=execution.resume,
                )
                for agent in ("nop", "oracle")
            ]
            if not all(trial.completed for trial in trials) or [
                trial.reward for trial in trials
            ] != [0, 1]:
                record["skipped"][url] = "harbor_packaging"
                save_record(receipt, record)
                event("harbor", "failed", "Retained task and trial evidence")
                continue
            task = export_repository_task(destination=out_dir, **kwargs)
            record["tasks"][url] = {
                "path": str(task.resolve()),
                **inspect_bundle(task),
                "baseline": str(trials[0].result),
                "oracle": str(trials[1].result),
            }
            save_record(receipt, record)
            event("export", "completed", task.name)
        record["state"] = "completed"
        save_record(receipt, record)
        return result()
