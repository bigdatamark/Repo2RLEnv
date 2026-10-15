# [PACKAGE DOC] Package repo2rlenv.quality.loop: the composable Harbor task review-and-repair loop,
# independent of any generation recipe. It ingests a Harbor task plus trial evidence
# (baseline/oracle/rollout/ probe), builds a bounded evidence pack for an LLM reviewer, optionally
# authors and validates repairs and semantic probes on remote workers, and reports a LoopResult.
# Re-exports LoopOptions, LoopResult (from models) and QualityLoop (from runner). Importing the
# package imports runner.
"""Composable Harbor review and repair; independent of generation recipes."""

from repo2rlenv.quality.loop.models import LoopOptions, LoopResult
from repo2rlenv.quality.loop.runner import QualityLoop

__all__ = ["LoopOptions", "LoopResult", "QualityLoop"]
