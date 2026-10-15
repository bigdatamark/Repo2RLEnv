# [MODULE DOC] Pipeline entry point for terminal_synth with recipe "tmax": samples seeds from a
# sampler JSON, designs with templates-based TMax prompts, runs the initial-state preflight, and
# executes the shared terminal.runner.run_synthesis loop with an unprivileged agent user named
# "user".
"""TMax legacy task generation through its retained sampler and distinct stages."""

from __future__ import annotations

from pathlib import Path

from repo2rlenv.pipelines.base import PipelineResult
from repo2rlenv.pipelines.recipes.terminal.preflight import initial_state
from repo2rlenv.pipelines.recipes.terminal.runner import run_synthesis
from repo2rlenv.pipelines.recipes.tmax import recipe
from repo2rlenv.pipelines.recipes.tmax.sampler import sample_inputs
from repo2rlenv.spec.input import PipelineName, SeedSource


class TMaxPipeline:
    # [CLASS DOC] Pipeline implementation for PipelineName.TERMINAL_SYNTH with recipe "tmax". Class
    # attributes: requires_bootstrap False, supported_languages None, experimental True,
    # native_supported False. Holds input, options and an on_event callback (no-op by default).
    name = PipelineName.TERMINAL_SYNTH
    requires_bootstrap = False
    supported_languages = None
    experimental = True
    native_supported = False

    def __init__(self, input, options, bootstrap=None):
        # [FUNCTION DOC] Validate and store inputs. Inputs: input (GenerationInput), options (recipe
        # options), bootstrap (ignored). Raises: ValueError unless recipe is "tmax" with a
        # SeedSource (the sampler JSON), and unless both input.execution and input.llm are set.
        if input.pipeline.recipe != "tmax" or not isinstance(input.source, SeedSource):
            raise ValueError("TMax requires its sampler JSON as a seeds input")
        if input.execution is None or input.llm is None:
            raise ValueError("TMax requires remote execution and an authoring model")
        self.input, self.options = input, options
        self.on_event = lambda event: None

    def set_event_callback(self, callback) -> None:
        # [FUNCTION DOC] Replace the progress-event callback used by run(). Input: callback
        # (callable taking a ProgressEvent). Output: None.
        self.on_event = callback

    def run(self, out_dir: Path) -> PipelineResult:
        # [FUNCTION DOC] Call run_synthesis with inputs=sample_inputs(source path),
        # designer=recipe.design, builder_prompt=recipe.builder_prompt(), preflight=initial_state
        # and agent_user="user". Output: PipelineResult. Raises: whatever sampling or run_synthesis
        # raises.
        return run_synthesis(
            self.input,
            self.options,
            out_dir,
            self.on_event,
            inputs=sample_inputs(self.input.source.path),
            designer=recipe.design,
            builder_prompt=recipe.builder_prompt(),
            preflight=initial_state,
            agent_user="user",
        )
