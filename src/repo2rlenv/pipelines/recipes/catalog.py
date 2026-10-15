# [MODULE DOC] Dependency-free recipe catalog. Loads recipe metadata from the packaged catalog.json
# and maps implemented recipe ids to lazily imported pipeline classes via the IMPLEMENTATIONS dict
# ("module.path:ClassName" strings). Being catalogued does not mean a recipe is runnable; only ids
# present in IMPLEMENTATIONS are implemented.
"""Dependency-free discovery of recipe scope; catalogued does not mean runnable."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

from pydantic import BaseModel, ConfigDict


class RecipeInfo(BaseModel):
    # [CLASS DOC] Frozen Pydantic model (extra fields forbidden) for one catalog entry: id, title,
    # pipeline (PipelineName value it belongs to), source_kinds (accepted input source kinds),
    # summary, upstream (provenance dict, e.g. repository/paper/commit/version), rfc, target
    # (default task count), recipe_version (default "1").
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    title: str
    pipeline: str
    source_kinds: list[str]
    summary: str
    upstream: dict[str, Any]
    rfc: str
    target: int
    recipe_version: str = "1"


# Entries are added only with an owned implementation and tests. Optional cloud
# or model libraries are imported when running, never while listing recipes.
IMPLEMENTATIONS: dict[str, str] = {
    "frontiersmith": "repo2rlenv.pipelines.recipes.frontiersmith.pipeline:FrontierSmithPipeline",
    "codemidas": "repo2rlenv.pipelines.recipes.codemidas.pipeline:CodeMidasPipeline",
    "scaler": "repo2rlenv.pipelines.recipes.scaler.pipeline:ScalerPipeline",
    "swe_next": "repo2rlenv.pipelines.recipes.swe_next.pipeline:SWENextPipeline",
    "r2e_gym": "repo2rlenv.pipelines.recipes.r2e_gym.pipeline:R2EGymPipeline",
    "dataarc": "repo2rlenv.pipelines.recipes.dataarc.pipeline:DataArcPipeline",
    "cli_gym": "repo2rlenv.pipelines.recipes.cli_gym.pipeline:CLIGymPipeline",
    "terminalworld": "repo2rlenv.pipelines.recipes.terminalworld.pipeline:TerminalWorldPipeline",
    "endless_terminals": "repo2rlenv.pipelines.recipes.endless_terminals.pipeline:EndlessTerminalsPipeline",
    "tmax": "repo2rlenv.pipelines.recipes.tmax.pipeline:TMaxPipeline",
    "r2e": "repo2rlenv.pipelines.recipes.r2e.pipeline:R2EPipeline",
    "swe_flow": "repo2rlenv.pipelines.repo_reconstruct:RepoReconstructPipeline",
    "swe_gen": "repo2rlenv.pipelines.pr_to_env:PRToEnvPipeline",
    "swe_smith": "repo2rlenv.pipelines.repo_mutate:RepoMutatePipeline",
    "seta_seed2synth": "repo2rlenv.pipelines.terminal_synth:TerminalSynthesisPipeline",
    "seta_evol": "repo2rlenv.pipelines.task_evolve:TaskEvolutionPipeline",
}


@lru_cache(maxsize=1)
def recipes() -> tuple[RecipeInfo, ...]:
    # [FUNCTION DOC] Loads and validates all catalog entries once (lru_cache, maxsize 1). Reads
    # catalog.json from this package's resources and validates each record as RecipeInfo. Output:
    # tuple of RecipeInfo. Raises ValueError on duplicate ids, Pydantic ValidationError on bad
    # records, json/IO errors if the resource is missing or malformed.
    data = json.loads(files(__package__).joinpath("catalog.json").read_text())
    result = tuple(RecipeInfo.model_validate(record) for record in data)
    if len({record.id for record in result}) != len(result):
        raise ValueError("Recipe catalog contains duplicate identifiers")
    return result


def get_recipe(recipe_id: str) -> RecipeInfo:
    # [FUNCTION DOC] Looks up a recipe by id. Input: recipe_id (str). Output: RecipeInfo. Raises
    # ValueError for unknown ids (message points to 'repo2rlenv pipelines list').
    for recipe in recipes():
        if recipe.id == recipe_id:
            return recipe
    raise ValueError(f"Unknown recipe {recipe_id!r}; use 'repo2rlenv pipelines list'")


def describe(recipe: RecipeInfo) -> dict[str, Any]:
    # [FUNCTION DOC] Returns a recipe as a plain dict: all model fields plus "status"
    # ("experimental" if the id is in IMPLEMENTATIONS else "planned") and "implemented" (bool).
    # Input: RecipeInfo.
    return {
        **recipe.model_dump(),
        "status": "experimental" if recipe.id in IMPLEMENTATIONS else "planned",
        "implemented": recipe.id in IMPLEMENTATIONS,
    }
