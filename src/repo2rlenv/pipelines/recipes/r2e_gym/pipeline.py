# [MODULE DOC] Registers the R2E-Gym recipe as a HistoryPipeline variant. All commit mining, bug-
# edit filtering and post-change test extraction live in the shared HistoryPipeline; this module
# only binds the pipeline name and recipe id.
"""Commit mining, bug-edit filters and extracted post-change test files."""

from __future__ import annotations

from repo2rlenv.pipelines.recipes.history.pipeline import HistoryPipeline
from repo2rlenv.spec.input import PipelineName


class R2EGymPipeline(HistoryPipeline):
    # [CLASS DOC] HistoryPipeline bound to PipelineName.COMMIT_RUNTIME with recipe_id "r2e_gym". No
    # overrides; behavior is entirely inherited from recipes.history.pipeline.
    name = PipelineName.COMMIT_RUNTIME
    recipe_id = "r2e_gym"
