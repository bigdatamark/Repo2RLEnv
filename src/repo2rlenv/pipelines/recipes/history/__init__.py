# [PACKAGE DOC] Shared history-mining machinery for commit/PR-history recipes (swe_next under
# pr_runtime, r2e_gym under commit_runtime). Modules: source (controller-side merged-PR listing),
# selection (Python entity diffing and test correspondence), test_suite (staging selected tests),
# worker (remote candidate selection and old/new test contrast) and pipeline (HistoryPipeline
# authoring and Harbor validation). Recipe-specific policy stays in each recipe's options/prompts.
"""Shared historical snapshot mechanics; recipe policies remain explicit."""

from __future__ import annotations
