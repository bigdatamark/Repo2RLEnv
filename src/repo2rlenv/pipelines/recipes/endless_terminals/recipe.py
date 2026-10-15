# [MODULE DOC] Endless Terminals prompt wiring: loads the retained template/initial/final/
# environment prompt files from package data and hands them to the shared terminal templates.
"""Retained Endless Terminals prompts and native template/test/environment order."""

from __future__ import annotations

from importlib.resources import files

from repo2rlenv.pipelines.recipes.terminal import templates


def design(seed, **kwargs):
    # [FUNCTION DOC] Design step: calls templates.design for a sampled seed with the template,
    # initial-state and final-state prompts and capabilities [seed["category"]]; extra kwargs are
    # forwarded. Output: whatever templates.design returns. Raises: KeyError if category is missing.
    # Side effects: reads package resources (and any LLM calls made by templates.design).
    resources = files(__package__)
    return templates.design(
        seed,
        template_prompt=resources.joinpath("template_prompt.md").read_text(),
        initial_prompt=resources.joinpath("initial_prompt.md").read_text(),
        final_prompt=resources.joinpath("final_prompt.md").read_text(),
        capabilities=[seed["category"]],
        **kwargs,
    )


def builder_prompt() -> str:
    # [FUNCTION DOC] Returns the builder system prompt: templates.builder_prompt applied to the
    # packaged environment_prompt.md text. Side effects: reads package resources.
    return templates.builder_prompt(
        files(__package__).joinpath("environment_prompt.md").read_text()
    )
