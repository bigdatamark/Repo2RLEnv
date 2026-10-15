# [MODULE DOC] End-to-end pr_diff test against public huggingface/trl via live GitHub. Skipped when
# gh is missing or unauthenticated. Flaky by construction: fails with diff_too_small skips if the
# latest PRs are small.
"""End-to-end test against a public repo (huggingface/trl).

Requires:
  - gh CLI installed and authenticated (or GITHUB_TOKEN env)
  - network access

If gh isn't available, the test is skipped.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from repo2rlenv.pipelines.pr_diff import PRDiffPipeline
from repo2rlenv.reward import calculate_diff_similarity_reward
from repo2rlenv.spec.input import (
    GenerationInput,
    LLMSpec,
    OutputSpec,
    PipelineName,
    PipelineSpec,
    RepoSpec,
)
from repo2rlenv.spec.options import PRDiffOptions

pytestmark = pytest.mark.skipif(
    not shutil.which("gh"),
    reason="gh CLI not available — skipping end-to-end tests",
)


def _gh_authenticated() -> bool:
    # [FUNCTION DOC] HELPER: Returns True if 'gh auth status' exits 0 within 5s, else False (also
    # on subprocess/OS error). Runs at import time for skipif. Side effects: subprocess.
    try:
        r = subprocess.run(["gh", "auth", "status"], capture_output=True, timeout=5, check=False)
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


@pytest.mark.skipif(not _gh_authenticated(), reason="gh not authenticated")
def test_e2e_public_trl(tmp_path: Path):
    # [FUNCTION DOC] Runs PRDiffPipeline (limit=2, max_files_per_pr=10) into tmp_path and asserts
    # >=1 candidate and >=1 emitted task; each task dir must have task.toml, instruction.md and a
    # non-empty solution/patch.diff that self-scores diff-similarity reward 1.0. Inputs: tmp_path.
    # Side effects: network, writes files.
    """Mine 2 PRs from huggingface/trl and verify each is a valid Harbor task."""
    gen_input = GenerationInput(
        repo=RepoSpec(url="huggingface/trl", access="public"),
        pipeline=PipelineSpec(name=PipelineName.PR_DIFF, options={}),
        llm=LLMSpec(provider="anthropic", model="claude-sonnet-4-6"),
        output=OutputSpec(
            destination=str(tmp_path),
            org="hfeval",
            dataset_name="trl-r2e",
        ),
    )
    options = PRDiffOptions(limit=2, max_files_per_pr=10)
    pipeline = PRDiffPipeline(gen_input, options)

    result = pipeline.run(tmp_path)

    assert result.candidates >= 1, "should have at least one PR candidate"
    assert result.emitted >= 1, f"should emit ≥1 task, got 0 (skips: {result.skip_reasons})"

    # Verify each emitted task
    for task_dir in tmp_path.iterdir():
        if not task_dir.is_dir():
            continue
        assert (task_dir / "task.toml").is_file()
        assert (task_dir / "instruction.md").is_file()
        oracle = task_dir / "solution" / "patch.diff"
        assert oracle.is_file()
        assert oracle.stat().st_size > 0, f"empty oracle in {task_dir}"

        # Self-similarity sanity check
        oracle_text = oracle.read_text(encoding="utf-8")
        reward, _ = calculate_diff_similarity_reward(oracle_text, oracle_text)
        assert reward == 1.0, f"oracle should self-score 1.0 in {task_dir}"
