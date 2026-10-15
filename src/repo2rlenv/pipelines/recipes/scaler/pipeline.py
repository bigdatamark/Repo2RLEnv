# [MODULE DOC] SCALER pipeline (reasoning_synth / scaler): a RemoteGenerationPipeline that ships a
# family JSON to the remote worker, which executes generators and reference programs, then exports
# each instance and accepts it only if Harbor trials give the native -1 (nop agent) and +1 (oracle
# agent) rewards. No LLM calls.
"""Released SCALER family expansion, without model calls or training dependencies."""

from __future__ import annotations

import hashlib
import json

from repo2rlenv.execution.artifacts import runtime_python
from repo2rlenv.execution.base import connect_worker
from repo2rlenv.execution.harbor import run_trial
from repo2rlenv.pipelines.recipes.repository.runner import RemoteGenerationPipeline
from repo2rlenv.pipelines.recipes.scaler.export import export_task
from repo2rlenv.pipelines.recipes.scaler.families import load_families
from repo2rlenv.spec.input import FamilySource, PipelineName


class ScalerPipeline(RemoteGenerationPipeline):
    # [CLASS DOC] Remote pipeline bound to PipelineName.REASONING_SYNTH, recipe "scaler", worker
    # module scaler.worker; supported_languages None (language-agnostic).
    name = PipelineName.REASONING_SYNTH
    recipe_id = "scaler"
    worker_module = "repo2rlenv.pipelines.recipes.scaler.worker"
    supported_languages = None

    def __init__(self, input, options, bootstrap=None):
        # [FUNCTION DOC] Run base validation, require a FamilySource input (ValueError otherwise),
        # and load families plus their sha256 via load_families (may raise ValueError).
        super().__init__(input, options, bootstrap)
        if not isinstance(input.source, FamilySource):
            raise ValueError("SCALER requires a native family JSON mapping")
        self.families, self.source_sha256 = load_families(input.source.path)

    def source_identity(self):
        # [FUNCTION DOC] Return {"families_sha256": digest} so the run config hash changes when the
        # family file changes.
        return {"families_sha256": self.source_sha256}

    def worker_configuration(self, run):
        # [FUNCTION DOC] Return the worker config: families mapping, source sha256 and options
        # dumped to JSON mode. Input run is unused.
        return {
            "families": self.families,
            "source_sha256": self.source_sha256,
            "options": self.options.model_dump(mode="json"),
        }

    def author_export(self, generation, candidate, ledger, run, out_dir):
        # [FUNCTION DOC] Export a candidate as a draft under run/tasks/<id>/draft, run Harbor trials
        # on the campaign worker with agents "nop" and "oracle" (trial ids derived from a sha256 of
        # run_id:candidate id, runtime python from the wheel hash), and require both trials
        # completed with rewards exactly [-1, 1]; then export the final task into out_dir.
        # Output: Path of the final task. Raises ValueError otherwise. Side effects: remote trials,
        # writes.
        execution = self.input.execution
        directory = run / "tasks" / candidate["id"]
        task = export_task(
            candidate, directory / "draft", self.input.output.org, resume=execution.resume
        )
        receipt = json.loads(execution.worker_receipt.read_text())
        worker = connect_worker(receipt["spec"]["provider"], receipt["worker_id"])
        python = runtime_python(hashlib.sha256(execution.runtime_wheel.read_bytes()).hexdigest())
        prefix = hashlib.sha256(f"{execution.run_id}:{candidate['id']}".encode()).hexdigest()[:20]
        self.event(
            "harbor", "started", f"{candidate['family']} · difficulty {candidate['difficulty']}"
        )
        trials = [
            run_trial(
                worker,
                task,
                directory / agent,
                trial_id=f"scaler-{prefix}-{agent}",
                agent=agent,
                python=python,
                resume=execution.resume,
            )
            for agent in ("nop", "oracle")
        ]
        if not all(trial.completed for trial in trials) or [trial.reward for trial in trials] != [
            -1,
            1,
        ]:
            raise ValueError(
                "SCALER export did not retain its native -1/+1 baseline/reference reward"
            )
        return export_task(candidate, out_dir, self.input.output.org, resume=execution.resume)
