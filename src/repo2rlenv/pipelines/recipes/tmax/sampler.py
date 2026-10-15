# [MODULE DOC] Seeded, bounded sampling of TMax task seeds from the retained legacy taxonomy
# (taxonomy.json: SKILL_TAXONOMY, TASK_LANGUAGES with weights, TASK_COMPLEXITY, COMMAND_COMPLEXITY,
# DOMAIN_SCENARIOS, REAL_SOFTWARE_ANCHORS, DOMAIN_MODULES) plus the domain-specific template prompt.
"""Seeded sampling from TMax's retained legacy taxonomy and complexity axes."""

from __future__ import annotations

import json
import random
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SamplerInput(BaseModel):
    # [CLASS DOC] Pydantic model (extra forbidden) for the sampler JSON file: corpus_kind (only
    # "legacy"), count (1-1000, default 40), seed (int, default 24), domains and languages (optional
    # allow-lists).
    model_config = ConfigDict(extra="forbid")
    corpus_kind: Literal["legacy"] = "legacy"
    count: int = Field(default=40, ge=1, le=1000)
    seed: int = 24
    domains: list[str] | None = None
    languages: list[str] | None = None


def sample_inputs(path: Path) -> list[dict]:
    # [FUNCTION DOC] Sample seed records. Input: path to a SamplerInput JSON. Validates that
    # domain/language allow-lists, when given, are nonempty subsets of the taxonomy (ValueError
    # otherwise). Using Random(seed), each draw picks a domain, a skill type, 3-5 skills from all of
    # that domain's skills, a complexity from the first three TASK_COMPLEXITY entries, a command
    # complexity, a scenario, a weighted language, and with probability 0.35 a real-software anchor
    # if one exists. All random choices are made before filtering, so a draw's values do not depend
    # on the filters. Draws outside the allow-lists are skipped. Output: list of exactly count dicts
    # (source "tmax_legacy_taxonomy", title "<domain>: <skill_type> (<draw>)", sampled fields,
    # corpus/fixture/verifier kinds, sampler_seed, draw). Raises: ValueError if count records are
    # not found within count*100 draws; ValidationError on bad input JSON.
    settings = SamplerInput.model_validate_json(path.read_text())
    data = json.loads(files(__package__).joinpath("taxonomy.json").read_text())
    taxonomy = data["SKILL_TAXONOMY"]
    languages, weights = zip(*data["TASK_LANGUAGES"], strict=True)
    if settings.domains is not None and (
        not settings.domains or not set(settings.domains) <= taxonomy.keys()
    ):
        raise ValueError("Unknown or empty TMax domain selection")
    if settings.languages is not None and (
        not settings.languages or not set(settings.languages) <= set(languages)
    ):
        raise ValueError("Unknown or empty TMax language selection")
    rng = random.Random(settings.seed)
    records = []
    for draw in range(settings.count * 100):
        domain = rng.choice(list(taxonomy))
        types = taxonomy[domain]
        skill_type = rng.choice(list(types))
        all_skills = [skill for group in types.values() for skill in group]
        skills = rng.sample(all_skills, min(rng.randint(3, 5), len(all_skills)))
        complexity = rng.choice(data["TASK_COMPLEXITY"][:3])
        commands = rng.choice(data["COMMAND_COMPLEXITY"])
        scenario = rng.choice(data["DOMAIN_SCENARIOS"][domain])
        language = rng.choices(languages, weights=weights, k=1)[0]
        anchors = data["REAL_SOFTWARE_ANCHORS"].get(domain)
        anchor = rng.choice(anchors) if anchors and rng.random() < 0.35 else None
        if settings.domains is not None and domain not in settings.domains:
            continue
        if settings.languages is not None and language not in settings.languages:
            continue
        records.append(
            {
                "source": "tmax_legacy_taxonomy",
                "title": f"{domain}: {skill_type} ({draw})",
                "domain": domain,
                "skill_type": skill_type,
                "primitive_skills": skills,
                "task_complexity": complexity,
                "command_complexity": commands,
                "scenario": scenario,
                "language": language,
                "anchor": anchor,
                "corpus_kind": "legacy",
                "fixture_kind": "text_only",
                "verifier_kind": "exact_text",
                "sampler_seed": settings.seed,
                "draw": draw,
            }
        )
        if len(records) == settings.count:
            return records
    raise ValueError("Sampler selection is too restrictive for its bounded draw budget")


def template_prompt(seed: dict) -> str:
    # [FUNCTION DOC] Return template_prompt.md with {{domain_label}} replaced by the title-cased
    # domain (underscores to spaces), {{module}} by DOMAIN_MODULES[domain], and {{v2_block}}
    # removed. Input: seed dict with "domain". Output: str. Raises: KeyError for an unknown domain.
    data = json.loads(files(__package__).joinpath("taxonomy.json").read_text(encoding="utf-8"))
    return (
        files(__package__)
        .joinpath("template_prompt.md")
        .read_text(encoding="utf-8")
        .replace("{{domain_label}}", seed["domain"].replace("_", " ").title())
        .replace("{{module}}", data["DOMAIN_MODULES"][seed["domain"]])
        .replace("{{v2_block}}", "")
    )
