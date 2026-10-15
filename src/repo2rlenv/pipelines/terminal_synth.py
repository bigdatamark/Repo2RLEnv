# [MODULE DOC] terminal_synth pipeline for the SETA seed2synth recipe: synthesizes terminal tasks
# from explicit seed data via the shared terminal.runner.run_synthesis loop with its defaults
# (load_seeds, SETA design and builder prompt).
"""Terminal task synthesis from explicit seed data."""

from __future__ import annotations

from pathlib import Path

from repo2rlenv.bootstrap.spec import LanguageHint
from repo2rlenv.pipelines.base import PipelineResult
from repo2rlenv.spec.input import GenerationInput, PipelineName, SeedSource
from repo2rlenv.spec.recipe_options import TerminalSynthesisOptions


class TerminalSynthesisPipeline:
    # [CLASS DOC] Pipeline implementation for PipelineName.TERMINAL_SYNTH with recipe
    # "seta_seed2synth". Class attributes: requires_bootstrap False, supported_languages {PYTHON},
    # experimental True, native_supported False. Holds input, options and an on_event callback (no-
    # op by default).
    name = PipelineName.TERMINAL_SYNTH
    requires_bootstrap = False
    supported_languages = frozenset({LanguageHint.PYTHON})
    experimental = True
    native_supported = False

    def __init__(self, input: GenerationInput, options: TerminalSynthesisOptions, bootstrap=None):
        # [FUNCTION DOC] Validate and store inputs. Inputs: input (GenerationInput), options
        # (TerminalSynthesisOptions), bootstrap (ignored). Raises: ValueError unless recipe is
        # "seta_seed2synth" with a SeedSource, and unless input.execution and input.llm are both
        # set.
        if input.pipeline.recipe != "seta_seed2synth" or not isinstance(input.source, SeedSource):
            raise ValueError("terminal_synth currently requires the SETA recipe and seed input")
        if input.execution is None or input.llm is None:
            raise ValueError("Terminal synthesis requires a remote worker and authoring model")
        self.input, self.options = input, options
        self.on_event = lambda event: None

    def set_event_callback(self, callback) -> None:
        # [FUNCTION DOC] Replace the progress-event callback used by run(). Input: callback
        # (callable taking a ProgressEvent). Output: None.
        self.on_event = callback

    def run(self, out_dir: Path) -> PipelineResult:
        # [FUNCTION DOC] Lazily import run_synthesis and call it with default
        # seeds/designer/builder. Output: PipelineResult.
        from repo2rlenv.pipelines.recipes.terminal.runner import run_synthesis

        return run_synthesis(self.input, self.options, out_dir, self.on_event)
