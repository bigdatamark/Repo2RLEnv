# [PACKAGE DOC] Pipelines package: imports every native pipeline class and exposes the PIPELINES
# registry mapping pipeline name (pr_diff, pr_runtime, commit_runtime, code_instruct,
# equivalence_tests, cve_patches, repo_mutate, repo_reconstruct, pr_to_env, terminal_synth,
# task_evolve) to its class, plus re-exports of the Pipeline protocol and PipelineResult. Importing
# it imports all pipeline modules.
"""Pipeline implementations + the standardized contract."""

from __future__ import annotations

from repo2rlenv.pipelines.base import Pipeline, PipelineResult
from repo2rlenv.pipelines.code_instruct import CodeInstructPipeline
from repo2rlenv.pipelines.commit_runtime import CommitRuntimePipeline
from repo2rlenv.pipelines.cve_patches import CVEPatchesPipeline
from repo2rlenv.pipelines.equivalence_tests import EquivalenceTestsPipeline
from repo2rlenv.pipelines.pr_diff import PRDiffPipeline
from repo2rlenv.pipelines.pr_runtime import PRRuntimePipeline
from repo2rlenv.pipelines.pr_to_env import PRToEnvPipeline
from repo2rlenv.pipelines.repo_mutate import RepoMutatePipeline
from repo2rlenv.pipelines.repo_reconstruct import RepoReconstructPipeline
from repo2rlenv.pipelines.task_evolve import TaskEvolutionPipeline
from repo2rlenv.pipelines.terminal_synth import TerminalSynthesisPipeline

PIPELINES: dict[str, type[Pipeline]] = {
    "pr_diff": PRDiffPipeline,
    "pr_runtime": PRRuntimePipeline,
    "commit_runtime": CommitRuntimePipeline,
    "code_instruct": CodeInstructPipeline,
    "equivalence_tests": EquivalenceTestsPipeline,
    "cve_patches": CVEPatchesPipeline,
    "repo_mutate": RepoMutatePipeline,
    "repo_reconstruct": RepoReconstructPipeline,
    "pr_to_env": PRToEnvPipeline,
    "terminal_synth": TerminalSynthesisPipeline,
    "task_evolve": TaskEvolutionPipeline,
}

__all__ = [
    "PIPELINES",
    "CVEPatchesPipeline",
    "CodeInstructPipeline",
    "CommitRuntimePipeline",
    "EquivalenceTestsPipeline",
    "PRDiffPipeline",
    "PRRuntimePipeline",
    "PRToEnvPipeline",
    "Pipeline",
    "PipelineResult",
    "RepoMutatePipeline",
    "RepoReconstructPipeline",
    "TaskEvolutionPipeline",
    "TerminalSynthesisPipeline",
]
