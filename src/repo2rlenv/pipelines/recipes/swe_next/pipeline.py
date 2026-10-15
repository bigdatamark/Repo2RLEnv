# [MODULE DOC] Registers the SWE-Next recipe as a HistoryPipeline variant; merged-PR mining, first-
# parent comparisons and test extraction are inherited.
"""Merged PRs, first-parent comparisons and original-path test extraction."""

from __future__ import annotations

from repo2rlenv.pipelines.recipes.history.pipeline import HistoryPipeline
from repo2rlenv.spec.input import PipelineName


class SWENextPipeline(HistoryPipeline):
    # [CLASS DOC] HistoryPipeline bound to PipelineName.PR_RUNTIME with recipe_id "swe_next". No
    # overrides.
    name = PipelineName.PR_RUNTIME
    recipe_id = "swe_next"
