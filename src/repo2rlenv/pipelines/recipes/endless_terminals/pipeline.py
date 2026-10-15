# [MODULE DOC] EndlessTerminalsPipeline: entry point for terminal_synth / endless_terminals.
# Samples taxonomy inputs from the sampler JSON and delegates to the shared terminal runner,
# with initial-state preflight tests executed separately and the agent running as "user".
"""Owned Endless Terminals pipeline with separately executed initial-state tests."""

from __future__ import annotations

from pathlib import Path

from repo2rlenv.pipelines.base import PipelineResult
from repo2rlenv.pipelines.recipes.endless_terminals import recipe
from repo2rlenv.pipelines.recipes.endless_terminals.sampler import sample_inputs
from repo2rlenv.pipelines.recipes.terminal.preflight import initial_state
from repo2rlenv.pipelines.recipes.terminal.runner import run_synthesis
from repo2rlenv.spec.input import PipelineName, SeedSource


class EndlessTerminalsPipeline:
    # [CLASS DOC] Pipeline for recipe "endless_terminals" under PipelineName.TERMINAL_SYNTH. Flags: no
    # bootstrap, any language, experimental, not native-supported. Holds input, options, on_event.
    name = PipelineName.TERMINAL_SYNTH
    requires_bootstrap = False
    supported_languages = None
    experimental = True
    native_supported = False

    def __init__(self, input, options, bootstrap=None):
        # [FUNCTION DOC] Validates and stores the input. Inputs: input (GenerationInput); options; bootstrap
        # (unused). Raises: ValueError if recipe is not "endless_terminals", source is not a SeedSource
        # (the sampler JSON path), or execution/llm is missing. Side effects: sets attributes.
        if input.pipeline.recipe != "endless_terminals" or not isinstance(input.source, SeedSource):
            raise ValueError("Endless Terminals requires its sampler JSON as a seeds input")
        if input.execution is None or input.llm is None:
            raise ValueError("Endless Terminals requires remote execution and an authoring model")
        self.input, self.options = input, options
        self.on_event = lambda event: None

    def set_event_callback(self, callback) -> None:
        # [FUNCTION DOC] Replaces the progress event callback. Output: None.
        self.on_event = callback

    def run(self, out_dir: Path) -> PipelineResult:
        # [FUNCTION DOC] Runs run_synthesis into out_dir with inputs from sample_inputs(source path),
        # recipe.design, recipe.builder_prompt(), initial_state preflight and agent_user "user".
        # Output: PipelineResult. Raises/side effects: those of the sampler and runner.
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
