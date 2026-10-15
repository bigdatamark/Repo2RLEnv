# [MODULE DOC] task_evolve pipeline (recipe seta_evol): evolves supplied Harbor tasks via a retained
# curriculum strategy by loading parents from the task source and running the shared terminal.runner
# loop with SETA-evol design and builder prompts.
"""Evolve supplied Harbor tasks through an explicit curriculum strategy."""

from __future__ import annotations

from pathlib import Path

from repo2rlenv.pipelines.base import PipelineResult
from repo2rlenv.spec.input import GenerationInput, PipelineName, TaskSource
from repo2rlenv.spec.recipe_options import TaskEvolutionOptions


class TaskEvolutionPipeline:
    # [CLASS DOC] Pipeline implementation for PipelineName.TASK_EVOLVE. Class attributes:
    # requires_bootstrap False, supported_languages None, experimental True, native_supported False.
    # Holds input, options and an on_event callback (no-op by default).
    name = PipelineName.TASK_EVOLVE
    requires_bootstrap = False
    supported_languages = None
    experimental = True
    native_supported = False

    def __init__(self, input: GenerationInput, options: TaskEvolutionOptions, bootstrap=None):
        # [FUNCTION DOC] Validate and store inputs. Inputs: input (GenerationInput), options
        # (TaskEvolutionOptions), bootstrap (ignored). Raises: ValueError unless recipe is
        # "seta_evol" with a TaskSource, and unless input.execution and input.llm are both set.
        if input.pipeline.recipe != "seta_evol" or not isinstance(input.source, TaskSource):
            raise ValueError("task_evolve requires the SETA evolution recipe and task input")
        if input.execution is None or input.llm is None:
            raise ValueError("Evolution requires a remote worker and authoring model")
        self.input, self.options = input, options
        self.on_event = lambda event: None

    def set_event_callback(self, callback) -> None:
        # [FUNCTION DOC] Replace the progress-event callback used by run(). Input: callback
        # (callable taking a ProgressEvent). Output: None.
        self.on_event = callback

    def run(self, out_dir: Path) -> PipelineResult:
        # [FUNCTION DOC] Lazily import the seta_evol recipe and run_synthesis; load parent tasks
        # with recipe.load_parents(source path, options) and run synthesis with
        # designer=recipe.design and builder_prompt=recipe.builder_prompt(). Output: PipelineResult.
        from repo2rlenv.pipelines.recipes.seta_evol import recipe
        from repo2rlenv.pipelines.recipes.terminal.runner import run_synthesis

        parents = recipe.load_parents(self.input.source.path, self.options)
        return run_synthesis(
            self.input,
            self.options,
            out_dir,
            self.on_event,
            inputs=parents,
            designer=recipe.design,
            builder_prompt=recipe.builder_prompt(),
        )
