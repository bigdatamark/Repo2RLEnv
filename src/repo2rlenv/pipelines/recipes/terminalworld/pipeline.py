# [MODULE DOC] Pipeline entry point for terminal_reconstruct / terminalworld: wires load_recordings
# (inputs), recipe.design (with min_score) and materialize into the shared
# terminal.runner.run_synthesis loop.
"""Reconstruct terminal tasks from public recordings and observed execution effects."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from repo2rlenv.pipelines.base import PipelineResult
from repo2rlenv.pipelines.recipes.terminal.runner import run_synthesis
from repo2rlenv.pipelines.recipes.terminalworld.materialize import materialize
from repo2rlenv.pipelines.recipes.terminalworld.recipe import design
from repo2rlenv.pipelines.recipes.terminalworld.source import load_recordings
from repo2rlenv.spec.input import PipelineName, RecordingSource


class TerminalWorldPipeline:
    # [CLASS DOC] Pipeline implementation for PipelineName.TERMINAL_RECONSTRUCT with recipe
    # "terminalworld". Class attributes: requires_bootstrap False, supported_languages None,
    # experimental True, native_supported False. Holds input, options and an on_event callback
    # (defaults to a no-op).
    name = PipelineName.TERMINAL_RECONSTRUCT
    requires_bootstrap = False
    supported_languages = None
    experimental = True
    native_supported = False

    def __init__(self, input, options, bootstrap=None):
        # [FUNCTION DOC] Validate and store the inputs. Inputs: input (GenerationInput), options
        # (recipe options with min_score etc.), bootstrap (ignored, default None). Raises:
        # ValueError unless the recipe is "terminalworld" with a RecordingSource, and unless both
        # input.execution and input.llm are set.
        if input.pipeline.recipe != "terminalworld" or not isinstance(
            input.source, RecordingSource
        ):
            raise ValueError("TerminalWorld requires native recording input directories")
        if input.execution is None or input.llm is None:
            raise ValueError("TerminalWorld requires a remote worker and authoring model")
        self.input, self.options = input, options
        self.on_event = lambda event: None

    def set_event_callback(self, callback) -> None:
        # [FUNCTION DOC] Replace the progress-event callback used by run(). Input: callback
        # (callable taking a ProgressEvent). Output: None.
        self.on_event = callback

    def run(self, out_dir: Path) -> PipelineResult:
        # [FUNCTION DOC] Run synthesis into out_dir by calling run_synthesis with
        # inputs=load_recordings(source path), designer=design partially applied with
        # min_score=options.min_score, and materializer=materialize. Output: PipelineResult. Raises:
        # whatever loading or run_synthesis raises.
        return run_synthesis(
            self.input,
            self.options,
            out_dir,
            self.on_event,
            inputs=load_recordings(self.input.source.path),
            designer=partial(design, min_score=self.options.min_score),
            materializer=materialize,
        )
