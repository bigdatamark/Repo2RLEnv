# [MODULE DOC] CodeMidasPipeline: controller for the owned CodeMidas recipe on the repo_reconstruct
# pipeline. For each candidate anchor produced by the remote worker it runs agentic LLM stages
# (feature design with sandboxed shell observation, test authoring executed remotely for contrast,
# independent review with optional contract repair), removes the feature's bodies, exports a
# sanitized repository task, gates it with 2 nop and 4 oracle Harbor trials, and retains a labeled
# copy in the output directory.
"""Agentic source discovery, observed tests and independent review to Harbor."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from importlib.resources import files

from repo2rlenv.execution.artifacts import runtime_python
from repo2rlenv.execution.base import connect_worker
from repo2rlenv.execution.generation import run_generator
from repo2rlenv.execution.harbor import run_trial
from repo2rlenv.execution.lifecycle import save_record
from repo2rlenv.pipelines.recipes.codemidas.models import Feature, Review, Verifier
from repo2rlenv.pipelines.recipes.codemidas.source import (
    existing_reconstructions,
    implementation_pair,
    reconstruction_identity,
    validate_assertions,
)
from repo2rlenv.pipelines.recipes.repository.export import export_repository_task
from repo2rlenv.pipelines.recipes.repository.runner import RepositoryGenerationPipeline
from repo2rlenv.quality.loop.client import RunBudget
from repo2rlenv.spec.input import PipelineName
from repo2rlenv.tasksmith.author.artifact import artifact_stage
from repo2rlenv.tasksmith.author.bridge import ProviderOutputError
from repo2rlenv.tasksmith.author.budget import AuthorBudget


class ConstructionExhausted(RuntimeError):
    # [CLASS DOC] RuntimeError subclass signalling that a candidate was deliberately rejected or ran
    # out of repair allowance (not a provider fault). Converted to ValueError by author_export.
    """Stop this candidate without treating a known rejection as a provider fault."""


class CodeMidasPipeline(RepositoryGenerationPipeline):
    # [CLASS DOC] Pipeline class for PipelineName.REPO_RECONSTRUCT with recipe_id "codemidas" and
    # worker_module repo2rlenv.pipelines.recipes.codemidas.worker; inherits run/resume machinery
    # from RepositoryGenerationPipeline.
    name = PipelineName.REPO_RECONSTRUCT
    recipe_id = "codemidas"
    worker_module = "repo2rlenv.pipelines.recipes.codemidas.worker"

    def __init__(self, input, options, bootstrap=None):
        # [FUNCTION DOC] Calls the base constructor, then raises ValueError unless the LLM's
        # qualified name equals options.author_model with no custom endpoint and no fallback. If
        # options.stack_manifest is set, reads it and requires the input repo URL to be
        # https://github.com/<repo_path> and the ref to equal the row's commit_id (ValueError
        # otherwise).
        super().__init__(input, options, bootstrap)
        if (
            input.llm.qualified_name != options.author_model
            or input.llm.endpoint
            or input.llm.fallback
        ):
            raise ValueError(
                "CodeMidas requires its explicit direct OpenAI author model, without fallback"
            )
        if options.stack_manifest:
            from repo2rlenv.pipelines.recipes.codemidas.stack import read_manifest

            row = read_manifest(options.stack_manifest)["row"]
            if (
                input.repo.url != "https://github.com/" + row["repo_path"]
                or input.repo.ref != row["commit_id"]
            ):
                raise ValueError(
                    "Repository identity and revision must match the Stack row exactly"
                )

    def source_identity(self):
        # [FUNCTION DOC] Returns extra identity for the run fingerprint: {stack_manifest_sha256:
        # sha256 of the manifest file} when a Stack manifest is configured, else an empty dict.
        if self.options.stack_manifest:
            return {
                "stack_manifest_sha256": hashlib.sha256(
                    self.options.stack_manifest.read_bytes()
                ).hexdigest()
            }
        return {}

    def worker_configuration(self, run):
        # [FUNCTION DOC] Extends the base worker configuration for run with stack_source = the
        # parsed Stack manifest when configured. Output: config dict.
        config = super().worker_configuration(run)
        if self.options.stack_manifest:
            from repo2rlenv.pipelines.recipes.codemidas.stack import read_manifest

            config["stack_source"] = read_manifest(self.options.stack_manifest)
        return config

    def author_export(self, generation, candidate, ledger, run, out_dir):
        # [FUNCTION DOC] Synchronous wrapper: runs _author_export with asyncio.run and converts
        # ConstructionExhausted / ProviderOutputError into ValueError (so the base runner records a
        # skip). Inputs as _author_export. Output: Path of the exported task.
        try:
            return asyncio.run(self._author_export(generation, candidate, ledger, run, out_dir))
        except (ConstructionExhausted, ProviderOutputError) as exc:
            raise ValueError(str(exc)) from exc

    async def _author_export(self, generation, candidate, ledger, run, out_dir):
        # [FUNCTION DOC] HELPER (async): Builds one task for a candidate. Inputs: generation
        # (prepare output dir with base/), candidate (dict with id, anchor, path, repo, ref,
        # image_digest, private_paths, optional source_kind/source_provenance), ledger, run (run
        # dir), out_dir. Sets up worker connection, runtime python, a per-candidate RunBudget
        # (candidate_budget_usd), a deadline of min(2400s, execution timeout) and persisted shell
        # observations. Stage "design" yields a Feature (validated by check_feature); raises
        # ValueError if the resulting reconstruction identity already exists in out_dir. Then up to
        # max_rounds rounds: "tests" stage yields a Verifier validated by check_verifier (remote
        # execution); "review" stage (reviewer_model, with shell) yields a Review; stop when
        # approved; when repair_target is "contract" and rounds remain, re-run design as a contract
        # repair preserving the same symbols; feed review issues into the next tests round. If never
        # approved raises ConstructionExhausted. Loads execution evidence from disk if none in
        # memory (ValueError if absent). Builds defective/reference maps and the public profile,
        # exports a draft with the generated test file and metadata, runs nop x2 (reward 0) and
        # oracle x4 (reward 1) Harbor trials (ValueError on any mismatch), retains the task into
        # out_dir/<name> and writes quality.json. Output: Path.
        execution = self.input.execution
        key = candidate["id"]
        directory = run / "tasks" / key
        receipt = json.loads(execution.worker_receipt.read_text())
        worker = connect_worker(receipt["spec"]["provider"], receipt["worker_id"])
        python = runtime_python(hashlib.sha256(execution.runtime_wheel.read_bytes()).hexdigest())
        prefix = hashlib.sha256(f"{execution.run_id}:{key}".encode()).hexdigest()[:20]
        budget = RunBudget(ledger, f"codemidas:{prefix}", str(self.options.candidate_budget_usd))
        deadline = time.time() + min(2400, execution.timeout_sec)
        observations_path = directory / "observations.json"
        shell_calls = (
            json.loads(observations_path.read_text())["observations"]
            if observations_path.exists()
            else []
        )

        async def shell(command, timeout_sec=120):
            # [FUNCTION DOC] HELPER (async LLM tool): Runs a bounded shell command (str up to 20000
            # chars, else ValueError) in a throwaway container of the candidate image on the worker:
            # no network, read-only root with 128m /tmp tmpfs, 1 CPU, test_memory_mb memory, 128
            # pids, wrapped in "timeout" clamped to 1-120s. Appends an observation {id, command,
            # returncode, last 20000 chars of stdout} and persists observations.json; always
            # force-removes the container. Output: observation as JSON string.
            if not isinstance(command, str) or len(command) > 20000:
                raise ValueError("Use a bounded shell command")
            timeout_sec = max(1, min(int(timeout_sec), 120))
            name = "codemidas-observe-" + uuid.uuid4().hex
            argv = [
                "docker",
                "run",
                "--rm",
                "--name",
                name,
                "--network",
                "none",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,size=128m",
                "--cpus",
                "1",
                "--memory",
                f"{self.options.test_memory_mb}m",
                "--pids-limit",
                "128",
                "-w",
                "/workspace",
                candidate["image_digest"],
                "timeout",
                str(timeout_sec),
                "sh",
                "-c",
                command,
            ]
            try:
                result = await asyncio.to_thread(worker.exec, argv, timeout=timeout_sec + 20)
                observation = {
                    "id": len(shell_calls) + 1,
                    "command": command,
                    "returncode": result.returncode,
                    "output": result.stdout[-20000:],
                }
                shell_calls.append(observation)
                save_record(directory / "observations.json", {"observations": shell_calls})
                return json.dumps(observation)
            finally:
                await asyncio.to_thread(worker.exec, ["docker", "rm", "-f", name], timeout=30)

        async def stage(
            schema,
            label,
            inputs,
            prompt_name,
            model,
            validate=None,
            use_shell=False,
            initial_draft_key=None,
        ):
            # [FUNCTION DOC] HELPER (async): Runs one artifact_stage. Inputs: schema, label (stage
            # dir name), inputs (context dict), prompt_name (system prompt file <name>.md), model,
            # validate (optional async validator), use_shell (expose shell tool), initial_draft_key.
            # Emits a started event; if context.json exists, requires it to equal inputs ignoring
            # "observations" and "execution" keys (ValueError) and reuses the stored context, else
            # saves it. For design/tests prompts exposes a reject_candidate tool. Uses the candidate
            # budget, max_turns and deadline. Output: validated artifact.
            self.event(label, "started", candidate["anchor"])
            root = directory / label
            context_path = root / "context.json"
            if context_path.exists():
                previous = json.loads(context_path.read_text())

                def immutable(value):
                    # [FUNCTION DOC] HELPER: Returns the dict without the "observations" and
                    # "execution" keys (used to compare stage contexts on resume).
                    return {
                        k: v for k, v in value.items() if k not in {"observations", "execution"}
                    }

                if immutable(previous) != immutable(inputs):
                    raise ValueError("Stage context changed; use a new run identity")
                inputs = previous
            else:
                save_record(context_path, inputs)

            async def reject_candidate(reason):
                # [FUNCTION DOC] HELPER (async LLM tool): Returns an instruction string if reason is
                # not a 20-2000 char string; otherwise writes rejection.json and raises
                # ConstructionExhausted(reason).
                if not isinstance(reason, str) or not 20 <= len(reason) <= 2000:
                    return "Give a concrete reason between 20 and 2000 characters."
                save_record(root / "rejection.json", {"reason": reason})
                raise ConstructionExhausted(reason)

            rejection_tools = []
            rejection_handlers = {}
            if prompt_name in {"design", "tests"}:
                rejection_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "reject_candidate",
                            "description": (
                                "Stop an unsuitable construction with a concrete observed "
                                "reason. Do not invent behavior absent from the reference."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {"reason": {"type": "string"}},
                                "required": ["reason"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ]
                rejection_handlers = {"reject_candidate": reject_candidate}
            return await artifact_stage(
                schema=schema,
                stage=label,
                inputs=inputs,
                system=files(__package__).joinpath(prompt_name + ".md").read_text(),
                prompt=json.dumps(inputs),
                root=root,
                budget=AuthorBudget(budget, root / "costs", label),
                model=model,
                runtime="openai",
                max_cost=self.options.candidate_budget_usd,
                max_turns=self.options.max_turns,
                deadline=deadline,
                shell=shell if use_shell else None,
                validate=validate,
                initial_draft_key=initial_draft_key,
                extra_tools=rejection_tools,
                extra_handlers=rejection_handlers,
            )

        async def check_feature(value):
            # [FUNCTION DOC] HELPER (async validator): Ensures the feature's symbols form a valid
            # implementation_pair (ValueError otherwise) and include the candidate's anchor symbol
            # at its path (ValueError otherwise).
            implementation_pair(generation / "base", value, self.options.source_paths)
            if not any(
                symbol.path == candidate["path"] and symbol.qualified_name == candidate["anchor"]
                for symbol in value.symbols
            ):
                raise ValueError(
                    "Include the anchor implementation in the selected coherent feature"
                )

        feature = await stage(
            Feature,
            "design",
            {"candidate": candidate, "source_paths": self.options.source_paths},
            "design",
            self.options.author_model,
            check_feature,
            True,
        )
        defective, _ = implementation_pair(generation / "base", feature, self.options.source_paths)
        feature_hash = reconstruction_identity(candidate["repo"], candidate["ref"], defective)
        if feature_hash in existing_reconstructions(out_dir):
            raise ValueError("This missing implementation already has a task in the collection")
        evaluations = [
            json.loads(path.read_text())
            for path in sorted(directory.glob("execution-*/*/evaluation.json"))
        ]

        async def check_verifier(value):
            # [FUNCTION DOC] HELPER (async validator): Validates assertion mapping, raises
            # ConstructionExhausted once max_rounds executions exist, then runs the worker in
            # "evaluate" mode remotely (timeout 2 * test_timeout_sec + 90) in execution-<n>; raises
            # ConstructionExhausted if the job produced no output; appends the evaluation and raises
            # ValueError with the evaluation JSON when it shows no contrast (so the model revises).
            validate_assertions(feature, value)
            if len(evaluations) >= self.options.max_rounds:
                raise ConstructionExhausted("CodeMidas construction repair allowance exhausted")
            attempt = len(evaluations)
            output = await asyncio.to_thread(
                run_generator,
                worker,
                python=python,
                module=self.worker_module,
                config={
                    "mode": "evaluate",
                    "options": self.options.model_dump(mode="json"),
                    "feature": feature.model_dump(),
                    "verifier": value.model_dump(),
                    "generation": "/evidence/generation/" + execution.run_id,
                    "image_digest": candidate["image_digest"],
                },
                directory=directory / f"execution-{attempt}",
                job_id=f"cm-{prefix}-{attempt}",
                timeout_sec=self.options.test_timeout_sec * 2 + 90,
                resume=execution.resume,
            )
            if output is None:
                raise ConstructionExhausted(
                    "Remote construction failed with a recorded outcome; inspect "
                    f"{directory / f'execution-{attempt}'}. This candidate is not exportable."
                )
            evaluation = json.loads((output / "evaluation.json").read_text())
            evaluations.append(evaluation)
            if not evaluation["contrast"]:
                raise ValueError(json.dumps(evaluation))

        test_inputs = {
            "feature": feature.model_dump(),
            "candidate": candidate,
            "learner_instruction": feature.task_instruction(),
        }
        for revision in range(self.options.max_rounds):
            suffix = "" if not revision else f"-revision-{revision}"
            verifier = await stage(
                Verifier,
                "tests" + suffix,
                test_inputs,
                "tests",
                self.options.author_model,
                check_verifier,
                True,
                "previous_verifier" if revision else None,
            )
            review = await stage(
                Review,
                "review" + suffix,
                {
                    "feature": feature.model_dump(),
                    "learner_instruction": feature.task_instruction(),
                    "verifier": verifier.model_dump(),
                    "observations": shell_calls,
                    "execution": evaluations[-1],
                    "source": {
                        symbol.path: (generation / "base" / symbol.path).read_text()
                        for symbol in feature.symbols
                    },
                },
                "review",
                self.options.reviewer_model,
                use_shell=True,
            )
            if review.approved:
                break
            if review.repair_target == "contract" and revision + 1 < self.options.max_rounds:
                selected = feature.symbols

                async def check_revision(value, selected=selected):
                    # [FUNCTION DOC] HELPER (async validator): Runs check_feature and additionally
                    # requires the revised feature's symbols to equal the previously selected ones
                    # (ValueError).
                    await check_feature(value)
                    if value.symbols != selected:
                        raise ValueError("A contract repair must preserve the selected feature")

                feature = await stage(
                    Feature,
                    f"design-revision-{revision + 1}",
                    {
                        "candidate": candidate,
                        "source_paths": self.options.source_paths,
                        "previous_feature": feature.model_dump(),
                        "review_feedback": review.issues,
                        "repair_policy": (
                            "Correct only the identified contract discrepancy using observed "
                            "public API behavior. Preserve selected symbols and feature scope. "
                            "No solver has run; do not tailor requirements to model success."
                        ),
                    },
                    "design",
                    self.options.author_model,
                    check_revision,
                    True,
                    "previous_feature",
                )
            test_inputs = {
                "feature": feature.model_dump(),
                "learner_instruction": feature.task_instruction(),
                "candidate": candidate,
                "previous_verifier": verifier.model_dump(),
                "review_feedback": review.issues,
            }
        else:
            raise ConstructionExhausted(
                "Independent assertion review rejected: " + "; ".join(review.issues)
            )
        if not evaluations:  # Artifact resume still requires its bound execution receipt.
            outputs = sorted(directory.glob("execution-*/*/evaluation.json"))
            if not outputs:
                raise ValueError("Committed tests lack execution evidence")
            evaluations = [json.loads(path.read_text()) for path in outputs]
        defective, reference = implementation_pair(
            generation / "base", feature, self.options.source_paths
        )
        from repo2rlenv.pipelines.recipes.codemidas.sanitize import public_profile

        profile = public_profile(generation / "base", self.options, candidate["private_paths"])
        kwargs = dict(
            base=generation / "base",
            defective=defective,
            reference=reference,
            options=profile,
            instruction=feature.task_instruction(),
            name="codemidas-" + key,
            org=self.input.output.org,
            contrast=evaluations[-1]["contrast"],
            verifier_source={"test_codemidas_generated.py": verifier.test_code.encode()},
            resume=execution.resume,
            metadata={
                "recipe": "codemidas",
                "recipe_version": "1",
                "pipeline": "repo_reconstruct",
                "paper": "https://arxiv.org/abs/2609.22068",
                "repository": candidate["repo"],
                "source_revision": candidate["ref"],
                "source_kind": candidate.get("source_kind", "github"),
                "source_provenance": candidate.get("source_provenance", {}),
                "author_model": self.options.author_model,
                "reviewer_model": self.options.reviewer_model,
                "assertion_review": "passed",
                "instruction_contract_version": "1",
                "reconstruction_identity": feature_hash,
            },
        )
        task = export_repository_task(destination=directory / "draft", **kwargs)
        for agent, count, reward in (("nop", 2, 0), ("oracle", 4, 1)):
            for index in range(count):
                self.event("harbor", "started", f"{agent} {index + 1}/{count}")
                evidence = await asyncio.to_thread(
                    run_trial,
                    worker,
                    task,
                    directory / f"{agent}-{index}",
                    trial_id=f"cm-{prefix}-{agent}-{index}",
                    agent=agent,
                    python=python,
                    resume=execution.resume,
                    timeout_sec=900,
                )
                if not evidence.completed or evidence.reward != reward:
                    raise ValueError(f"Harbor {agent} trial {index} failed its consistency gate")
        # Preserve the exact verified bytes; stage evidence is stored next to the task.
        from repo2rlenv.pipelines.recipes.codemidas.retention import retain

        output = retain(task, out_dir / task.name, directory)
        save_record(
            directory / "quality.json",
            {
                "execution_verified": True,
                "baseline_runs": 2,
                "reference_runs": 4,
                "assertion_review": review.model_dump(),
                "rollout_review": "not_run",
                "curriculum_selection": "not_run",
                "task": str(output.resolve()),
                "cost": budget.totals(),
            },
        )
        return output
