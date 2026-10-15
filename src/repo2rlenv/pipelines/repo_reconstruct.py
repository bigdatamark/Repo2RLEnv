# [MODULE DOC] repo_reconstruct pipeline (recipe swe_flow): owned SWE-Flow reconstruction on the
# shared RepositoryGenerationPipeline. The remote worker (swe_flow.worker) produces scheduled
# development steps; this module authors docstrings and an instruction, skeletonizes the scheduled
# functions to build the defective state, verifies it with Harbor nop/oracle trials, then exports.
"""Repository reconstruction through the owned SWE-Flow development schedule."""

from __future__ import annotations

import hashlib
import json

from repo2rlenv.execution.artifacts import runtime_python
from repo2rlenv.execution.base import connect_worker
from repo2rlenv.execution.harbor import run_trial
from repo2rlenv.pipelines.recipes.catalog import get_recipe
from repo2rlenv.pipelines.recipes.repository.export import export_repository_task
from repo2rlenv.pipelines.recipes.repository.runner import RepositoryGenerationPipeline
from repo2rlenv.pipelines.recipes.swe_flow.author import author
from repo2rlenv.pipelines.recipes.swe_flow.schedule import skeletonize
from repo2rlenv.spec.input import PipelineName


class RepoReconstructPipeline(RepositoryGenerationPipeline):
    # [CLASS DOC] RepositoryGenerationPipeline subclass for PipelineName.REPO_RECONSTRUCT: recipe_id
    # "swe_flow", worker_module "repo2rlenv.pipelines.recipes.swe_flow.worker". Overrides
    # author_export.
    name = PipelineName.REPO_RECONSTRUCT
    recipe_id = "swe_flow"
    worker_module = "repo2rlenv.pipelines.recipes.swe_flow.worker"

    def author_export(self, generation, candidate, ledger, run, out_dir):
        # [FUNCTION DOC] Author, verify and export one reconstruction candidate. Inputs: generation
        # (Path with base/ checkout), candidate (dict: id, source_files, schedule incl. step,
        # functions, contrast, repo, ref), ledger, run (Path), out_dir (Path). Calls swe_flow
        # author() for docstrings and instruction (models under run/tasks/<id>/models); builds
        # defective file bytes by skeletonize() of each changed source file per the schedule and
        # docstrings, and reference bytes from the base; assembles export_repository_task arguments
        # (name "swe-flow-<id>", org, contrast, metadata incl. recipe id, pipeline, recipe_version
        # "1", upstream commit, repository, source revision, development step and scheduled function
        # count). Exports a draft, runs Harbor "nop" and "oracle" trials (ids "flow-<20
        # hex>-<agent>") on the campaign worker and requires rewards [0, 1] with both completed.
        # Output: Path of the task exported to out_dir. Raises: ValueError on failed checks; helper
        # errors. Side effects: LLM spend, remote execution, filesystem writes.
        execution = self.input.execution
        key = candidate["id"]
        directory = run / "tasks" / key
        documents, instruction = author(
            candidate,
            self.input.llm,
            ledger,
            directory / "models",
            operation_prefix=f"{execution.run_id}:{key}",
            resume=execution.resume,
        )
        recipe = get_recipe(self.recipe_id)
        base = generation / "base"
        changed = candidate["source_files"]
        arguments = dict(
            base=base,
            defective={
                relative: skeletonize(
                    (base / relative).read_text(),
                    path=relative,
                    schedule=candidate["schedule"],
                    docstrings=documents,
                ).encode()
                for relative in changed
            },
            reference={relative: (base / relative).read_bytes() for relative in changed},
            options=self.options,
            instruction=instruction,
            name="swe-flow-" + key,
            org=self.input.output.org,
            contrast=candidate["contrast"],
            metadata={
                "recipe": recipe.id,
                "pipeline": recipe.pipeline,
                "recipe_version": "1",
                "upstream_revision": recipe.upstream["commit"],
                "repository": candidate["repo"],
                "source_revision": candidate["ref"],
                "development_step": candidate["schedule"]["step"],
                "scheduled_function_count": len(candidate["functions"]),
            },
            resume=execution.resume,
        )
        task = export_repository_task(destination=directory / "draft", **arguments)
        receipt = json.loads(execution.worker_receipt.read_text())
        worker = connect_worker(receipt["spec"]["provider"], receipt["worker_id"])
        python = runtime_python(hashlib.sha256(execution.runtime_wheel.read_bytes()).hexdigest())
        prefix = hashlib.sha256(f"{execution.run_id}:{key}".encode()).hexdigest()[:20]
        trials = [
            run_trial(
                worker,
                task,
                directory / agent,
                trial_id=f"flow-{prefix}-{agent}",
                agent=agent,
                resume=execution.resume,
                python=python,
            )
            for agent in ("nop", "oracle")
        ]
        if not all(trial.completed for trial in trials) or [trial.reward for trial in trials] != [
            0,
            1,
        ]:
            raise ValueError("Reconstruction task did not pass Harbor baseline/reference checks")
        return export_repository_task(destination=out_dir, **arguments)
