# [MODULE DOC] DataArcPipeline: the terminal_synth / dataarc recipe entry point. Loads Harbor seed
# tasks, expands them into strategy records and delegates synthesis, execution and Harbor emission
# to the shared terminal runner (run_synthesis).
"""DataArc's released terminal synthesis branch under the shared Harbor boundary."""

from __future__ import annotations

from pathlib import Path

from repo2rlenv.pipelines.base import PipelineResult
from repo2rlenv.pipelines.recipes.dataarc import recipe
from repo2rlenv.pipelines.recipes.terminal.runner import run_synthesis
from repo2rlenv.spec.input import PipelineName, SeedSource


class DataArcPipeline:
    # [CLASS DOC] Pipeline implementation for recipe "dataarc" under PipelineName.TERMINAL_SYNTH.
    # Class flags: no bootstrap, any language, experimental, not native-supported. Holds input,
    # options and an on_event callback (no-op by default).
    name = PipelineName.TERMINAL_SYNTH
    requires_bootstrap = False
    supported_languages = None
    experimental = True
    native_supported = False

    def __init__(self, input, options, bootstrap=None):
        # [FUNCTION DOC] Validates and stores the generation input.
        # Inputs: input (GenerationInput); options (DataArc options); bootstrap (unused, default None).
        # Raises: ValueError if recipe is not "dataarc", source is not a SeedSource, or execution/llm is
        # missing. Side effects: sets self.input, self.options, self.on_event.
        if input.pipeline.recipe != "dataarc" or not isinstance(input.source, SeedSource):
            raise ValueError("DataArc requires a directory of Harbor seed tasks")
        if input.execution is None or input.llm is None:
            raise ValueError("DataArc requires remote execution and an authoring model")
        self.input, self.options = input, options
        self.on_event = lambda event: None

    def set_event_callback(self, callback) -> None:
        # [FUNCTION DOC] Replaces the progress event callback. Input: callback. Output: None.
        self.on_event = callback

    def run(self, out_dir: Path) -> PipelineResult:
        # [FUNCTION DOC] Runs synthesis into out_dir: loads seeds via recipe.load_seeds from the seed
        # source path and calls run_synthesis with recipe.design and recipe.materialize.
        # Output: PipelineResult. Raises/side effects: those of load_seeds and run_synthesis.
        return run_synthesis(
            self.input,
            self.options,
            out_dir,
            self.on_event,
            inputs=recipe.load_seeds(self.input.source.path, self.options),
            designer=recipe.design,
            materializer=recipe.materialize,
        )
