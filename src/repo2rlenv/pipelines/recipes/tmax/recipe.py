# [MODULE DOC] TMax design and builder prompts: wraps terminal.templates.design / builder_prompt
# with TMax's retained prompt resources (template_prompt.md via sampler.template_prompt,
# initial_prompt.md, final_prompt.md, environment_prompt.md).
"""TMax's retained prompts over shared initial/final-state generation stages."""

from __future__ import annotations

from importlib.resources import files

from repo2rlenv.pipelines.recipes.terminal import templates
from repo2rlenv.pipelines.recipes.tmax.sampler import template_prompt


def design(seed, **kwargs):
    # [FUNCTION DOC] Designer for run_synthesis. Inputs: seed (sampled dict with domain and
    # primitive_skills) and keyword arguments forwarded to templates.design (model, ledger, receipt,
    # operation_id, resume). Uses the domain-filled template prompt, the retained initial/final
    # prompts and primitive_skills as capabilities. Output: TemplateDesign. Raises: KeyError on
    # missing seed fields; templates.design errors.
    resources = files(__package__)
    return templates.design(
        seed,
        template_prompt=template_prompt(seed),
        initial_prompt=resources.joinpath("initial_prompt.md").read_text(),
        final_prompt=resources.joinpath("final_prompt.md").read_text(),
        capabilities=seed["primitive_skills"],
        **kwargs,
    )


def builder_prompt() -> str:
    # [FUNCTION DOC] Return templates.builder_prompt applied to the retained environment_prompt.md
    # text. Output: str.
    return templates.builder_prompt(
        files(__package__).joinpath("environment_prompt.md").read_text()
    )
