# [MODULE DOC] Unit tests for the pr_diff pipeline: _strip_info_leak (removes Closes/Fixes/See refs, markdown
# and bare GitHub/redirect links, trailer lines, collapses blank runs), _build_instruction (squash
# suffix and '(fixes #N)' stripped from title, placeholder for empty bodies, overall shape),
# build_pr_diff_environment_dockerfile (python:3.12-slim clone at base commit, no oracle baked,
# optional GITHUB_TOKEN build arg scrubbed from the remote), _pr_diff_aux_files (verifier, oracle,
# instruction under tests/), build_pr_diff_eval_script (git add -A, diff vs base, SCRIPT_DIR paths,
# stale reward removal, exit 0) and that the verifier source is stdlib-only. One test executes the
# emitted test.sh with real git/bash in tmp_path (POSIX only).
"""Tests for `pipelines.pr_diff` — instruction-text construction.

Focus is on the v0.8.3 Arc 1 optimization: broaden the info-leak strip
so PR descriptions don't hint at the patch the agent is supposed to
produce. We strip:

  - Closes / Fixes / Resolves #N  (including multi like "Closes #1, #2")
  - See / Refs / Follow-up to #N
  - Markdown issue links `[#N](url)`
  - Bare github.com /pull/, /issues/, /commit/ URLs
  - Trailer lines (Co-authored-by, Signed-off-by, Reviewed-by, Acked-by)
  - Squash-merge "(#N)" suffix in the title
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from repo2rlenv.github import PullRequestSummary
from repo2rlenv.pipelines.pr_diff import (
    _build_instruction,
    _pr_diff_aux_files,
    _strip_info_leak,
    build_pr_diff_environment_dockerfile,
    build_pr_diff_eval_script,
)


def _pr(*, title: str = "Fix the bug", body: str = "") -> PullRequestSummary:
    # [FUNCTION DOC] HELPER: Builds a minimal merged PullRequestSummary (#1, base 'main', zero/one SHAs, one
    # changed file a.py). Inputs (keyword-only): title (default 'Fix the bug'), body (default '').
    # Output: PullRequestSummary.
    """Build a minimal PullRequestSummary stub."""
    return PullRequestSummary(
        number=1,
        title=title,
        body=body,
        state="closed",
        merged_at="2026-01-01T00:00:00Z",
        base_ref="main",
        base_sha="0" * 40,
        head_sha="1" * 40,
        is_draft=False,
        url="https://github.com/example/repo/pull/1",
        changed_files=["a.py"],
    )


# ---------------------------------------------------------------------------
# _strip_info_leak
# ---------------------------------------------------------------------------


def test_strips_closes_fixes_resolves() -> None:
    # [FUNCTION DOC] Checks 'Closes #42' is removed while surrounding prose stays.
    out = _strip_info_leak("Fix it.  Closes #42")
    assert "Closes" not in out
    assert "#42" not in out
    assert "Fix it" in out


def test_strips_multi_closes() -> None:
    # [FUNCTION DOC] Checks a multi-ref 'Fixes #1, #2, #3' list is removed entirely.
    out = _strip_info_leak("rolls up everything. Fixes #1, #2, #3 — and that's all.")
    assert "#1" not in out and "#2" not in out and "#3" not in out
    assert "Fixes" not in out
    assert "rolls up everything" in out


def test_strips_see_refs_follow_up() -> None:
    # [FUNCTION DOC] Loops over See/Refs/ref/see also/Follow-up to/follow up to variants and checks '#99' is
    # stripped from each.
    cases = [
        "See #99",
        "Refs #99",
        "ref #99",
        "see also #99",
        "Follow-up to #99",
        "follow up to #99",
    ]
    for c in cases:
        out = _strip_info_leak(f"Context. {c}. Done.")
        assert "#99" not in out, f"failed to strip {c!r} → {out!r}"


def test_strips_markdown_issue_link() -> None:
    # [FUNCTION DOC] Checks a '[#1234](github url)' markdown link is removed.
    out = _strip_info_leak("Background in [#1234](https://github.com/x/y/issues/1234).")
    assert "#1234" not in out
    assert "https://github.com" not in out
    assert "Background in" in out


def test_strips_bare_github_pull_url() -> None:
    # [FUNCTION DOC] Checks bare github.com pull, issues and commit URLs are removed.
    cases = [
        "https://github.com/foo/bar/pull/42",
        "https://github.com/foo/bar/issues/42",
        "https://github.com/foo/bar/commit/abc123def456",
    ]
    for url in cases:
        out = _strip_info_leak(f"See {url} for details.")
        assert "github.com" not in out, f"failed for {url}"


def test_strips_trailer_lines() -> None:
    # [FUNCTION DOC] Checks Co-authored-by/Signed-off-by/Reviewed-by trailer lines are removed and the real text
    # kept.
    body = (
        "The real bug is in the parser.\n"
        "\n"
        "Co-authored-by: Someone Else <e@example.com>\n"
        "Signed-off-by: Approver <a@example.com>\n"
        "Reviewed-by: Reviewer <r@example.com>\n"
    )
    out = _strip_info_leak(body)
    assert "Co-authored-by" not in out
    assert "Signed-off-by" not in out
    assert "Reviewed-by" not in out
    assert "The real bug is in the parser" in out


def test_keeps_legitimate_text() -> None:
    # [FUNCTION DOC] Checks text with no leak patterns passes through unchanged.
    body = (
        "Users report that `Choice('a','b').convert('c')` raises.\n"
        "It should raise BadParameter not a plain ValueError."
    )
    # No leak patterns → body should pass through verbatim (modulo trailing trim)
    out = _strip_info_leak(body)
    assert out == body


def test_squeezes_whitespace_after_strip() -> None:
    # [FUNCTION DOC] Checks runs of 3+ newlines collapse so no triple newline remains.
    body = "Useful text.\n\n\n\nMore useful text."
    out = _strip_info_leak(body)
    # Triple+ blank lines collapse to a single blank line
    assert "\n\n\n" not in out


# ---------------------------------------------------------------------------
# _build_instruction
# ---------------------------------------------------------------------------


def test_squash_suffix_stripped_from_title() -> None:
    # [FUNCTION DOC] Checks a squash-merge '(#1234)' suffix is removed from the Title line.
    pr = _pr(title="Fix the bug (#1234)", body="")
    instr = _build_instruction(pr)
    assert "**Title:** Fix the bug\n" in instr
    assert "(#1234)" not in instr


def test_title_drops_parenthesized_closes_marker() -> None:
    # [FUNCTION DOC] Checks a '(fixes #1800)' marker in the title is removed while the rest of the title is kept.
    """Real-world title from stretchr/testify#1888:
    "assert: fix NotSubset error messages using %#v instead of %q (fixes #1800)"
    """
    pr = _pr(
        title="assert: fix NotSubset error messages using %#v instead of %q (fixes #1800)",
        body="",
    )
    instr = _build_instruction(pr)
    assert "fixes #1800" not in instr
    assert "#1800" not in instr
    assert "assert: fix NotSubset" in instr


def test_strips_redirect_github_url() -> None:
    # [FUNCTION DOC] Checks redirect.github.com pull URLs (from Dependabot notes) are removed.
    """Dependabot release notes commonly embed `https://redirect.github.com/...` —
    these still leak the answer.
    """
    body = "Pulls in github-script fixes via https://redirect.github.com/x/y/pull/1929 — done."
    out = _strip_info_leak(body)
    assert "redirect.github.com" not in out
    assert "pull/1929" not in out


def test_strips_closes_with_markdown_issue_link() -> None:
    # [FUNCTION DOC] Regression: checks 'Closes [#1234](url)' removes both keyword and link, keeping surrounding
    # prose.
    """`Closes [#1234](url)` — the markdown-link form of a Closes/Fixes ref.

    Before the fix, _CLOSES_RE only handled bare `Closes #N`, so the
    markdown-link form left the `Closes ` keyword orphaned in the output.
    """
    body = "Background prose. Closes [#1234](https://github.com/x/y/issues/1234). More."
    out = _strip_info_leak(body)
    assert "Closes" not in out
    assert "#1234" not in out
    assert "github.com" not in out
    assert "Background prose" in out
    assert "More." in out


def test_strips_multi_closes_with_markdown_links() -> None:
    # [FUNCTION DOC] Checks 'Fixes [#1](url), [#2](url)' is removed entirely.
    """`Fixes [#1](url), [#2](url)` — closes-list of markdown-link refs."""
    body = (
        "Refactors the parser. "
        "Fixes [#1](https://github.com/x/y/issues/1), [#2](https://github.com/x/y/issues/2). End."
    )
    out = _strip_info_leak(body)
    assert "Fixes" not in out
    assert "#1" not in out and "#2" not in out
    assert "github.com" not in out
    assert "Refactors the parser" in out


def test_strips_descriptive_markdown_gh_link() -> None:
    # [FUNCTION DOC] Checks a descriptive-text markdown link to a GitHub PR is removed without leaving orphaned
    # brackets.
    """`[some descriptive text](https://github.com/x/y/pull/N)` — markdown link
    whose link-text isn't `[#N]` but whose URL still leaks the answer.

    Before the fix, the bare-URL regex stripped the URL but left orphaned
    `[some descriptive text]()` brackets behind.
    """
    body = (
        "There's a deep discussion in "
        "[my detailed analysis](https://github.com/x/y/pull/1234) you should read."
    )
    out = _strip_info_leak(body)
    assert "github.com" not in out
    assert "pull/1234" not in out
    # No orphaned empty markdown brackets
    assert "[" not in out and "]()" not in out
    assert "There's a deep discussion in" in out


def test_strips_see_with_markdown_issue_link() -> None:
    # [FUNCTION DOC] Checks 'See [#42](url)' is removed including the 'See' keyword.
    body = "Context here. See [#42](https://github.com/x/y/issues/42)."
    out = _strip_info_leak(body)
    assert "See" not in out
    assert "#42" not in out
    assert "Context here." in out


def test_empty_body_emits_placeholder() -> None:
    # [FUNCTION DOC] Checks an empty PR body yields the '(no description provided in source PR)' placeholder.
    pr = _pr(title="No description here", body="")
    instr = _build_instruction(pr)
    assert "(no description provided in source PR)" in instr


def test_body_only_links_collapses_to_placeholder() -> None:
    # [FUNCTION DOC] Checks a body consisting only of 'Closes #99' also yields the placeholder.
    """If the body is just a closes-link and nothing else, we still emit
    the placeholder rather than an empty Description section."""
    pr = _pr(title="Trivial", body="Closes #99")
    instr = _build_instruction(pr)
    assert "(no description provided in source PR)" in instr


def test_full_instruction_shape() -> None:
    # [FUNCTION DOC] Checks a full instruction has '# Issue', the title, '## Task', edit-in-place and diff-
    # similarity wording, and no closes refs or GitHub URLs.
    pr = _pr(
        title="Fix Choice.convert to raise BadParameter",
        body=(
            "The current implementation raises ValueError. "
            "Closes #1234. See https://github.com/pallets/click/issues/1234 for context."
        ),
    )
    instr = _build_instruction(pr)
    assert "# Issue" in instr
    assert "Fix Choice.convert" in instr
    assert "Closes #1234" not in instr
    assert "github.com" not in instr
    assert "## Task" in instr
    # New harbor-runnable task — agent edits files directly; verifier captures
    # changes via git diff and scores them with SWE-RL-style similarity.
    assert "Edit files in place" in instr
    assert "diff-similarity" in instr


# ---------------------------------------------------------------------------
# build_pr_diff_environment_dockerfile + build_pr_diff_eval_script
# ---------------------------------------------------------------------------


def test_dockerfile_starts_from_python_slim() -> None:
    # [FUNCTION DOC] Checks the env Dockerfile is FROM python:3.12-slim, installs git, blob-less clones the repo
    # into /workspace and hard-resets to the base commit.
    df = build_pr_diff_environment_dockerfile(
        repo_url="https://github.com/pallets/click.git",
        base_commit="abc1234567890",
    )
    assert "FROM python:3.12-slim" in df
    assert "apt-get install" in df and "git" in df
    assert "git clone --filter=blob:none https://github.com/pallets/click.git /workspace" in df
    assert "git reset --hard abc1234567890" in df


def test_dockerfile_does_not_contain_the_oracle() -> None:
    # [FUNCTION DOC] Checks the Dockerfile contains no oracle text, its base64, any /verifier/ path or base64
    # decode (oracle only ships in tests/ at verify time).
    """The oracle must NOT be baked into the agent's image — it ships as a
    tests/ aux file Harbor delivers only at verify time. Baking it let the
    agent read /verifier/oracle.patch and `git apply` it for a free 1.0."""
    import base64

    oracle = "diff --git a/foo.py b/foo.py\n@@ -1 +1 @@\n-old\n+secret_fix\n"
    df = build_pr_diff_environment_dockerfile(
        repo_url="https://github.com/x/y.git",
        base_commit="deadbeef",
    )
    # Neither the oracle text nor its base64 encoding appears in the image.
    assert "secret_fix" not in df
    assert base64.b64encode(oracle.encode()).decode() not in df
    # No /verifier bakes at all.
    assert "/verifier/" not in df
    assert "base64 -d" not in df


def test_aux_files_ship_verifier_oracle_instruction() -> None:
    # [FUNCTION DOC] Checks _pr_diff_aux_files returns exactly tests/verifier.py, tests/oracle.patch (verbatim) and
    # tests/instruction.md, with a main function in the verifier.
    """The verifier, oracle, and instruction ride in tests/ (verify-time only)."""
    aux = _pr_diff_aux_files(oracle_diff="diff --git a/x b/x\n+fix\n", instruction="# Issue\ndo it")
    assert set(aux) == {"tests/verifier.py", "tests/oracle.patch", "tests/instruction.md"}
    assert aux["tests/oracle.patch"] == "diff --git a/x b/x\n+fix\n"
    assert "def main" in aux["tests/verifier.py"]


def test_eval_script_shebang_and_paths() -> None:
    # [FUNCTION DOC] Checks the eval script starts with a bash shebang, stages untracked files with git add -A
    # before diffing --cached against the base into /tmp/predicted.patch, and reads
    # verifier/oracle/instruction from $SCRIPT_DIR rather than baked /verifier/ paths.
    es = build_pr_diff_eval_script(base_commit="abc1234567890")
    assert es.startswith("#!/bin/bash")
    # `git add -A` BEFORE `git diff --cached <base>` ensures new
    # (untracked) files added by the agent are captured in the diff —
    # without this, PRs that add files would silently downscore.
    assert "git add -A" in es
    assert "git diff --cached abc1234567890 > /tmp/predicted.patch" in es
    # The verifier + oracle come from tests/ ($SCRIPT_DIR), not the image.
    assert 'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"' in es
    assert '"$SCRIPT_DIR/verifier.py"' in es
    assert '"$SCRIPT_DIR/oracle.patch"' in es
    assert '"$SCRIPT_DIR/instruction.md"' in es
    # The old baked location must be gone (but /logs/verifier/ stays).
    assert "/verifier/verifier.py" not in es
    assert "/verifier/oracle.patch" not in es


def test_eval_script_clears_stale_reward_files() -> None:
    # [FUNCTION DOC] Checks the script removes reward.txt, reward.json and reward-details.json before grading
    # (anti-forgery).
    """A tampering agent must not be able to pre-write reward.txt and have it
    stand — the script drops any reward file before the verifier runs."""
    es = build_pr_diff_eval_script(base_commit="abc")
    assert (
        "rm -f /logs/verifier/reward.txt /logs/verifier/reward.json "
        "/logs/verifier/reward-details.json"
    ) in es


@pytest.mark.skipif(os.name == "nt", reason="The emitted verifier runs in a Linux sandbox")
@pytest.mark.parametrize("apply_oracle", [False, True], ids=["no-op", "oracle"])
def test_verify_time_assets_grade_edits_and_remove_forged_rewards(tmp_path: Path, apply_oracle):
    # [FUNCTION DOC] Parametrized (no-op vs oracle applied): builds a real git repo in tmp_path with a one-line
    # change as oracle, pre-writes forged reward files, writes the aux files and the eval script
    # with /workspace, /logs/verifier and /tmp/predicted.patch relocated into tmp_path, then runs
    # bash with R2E_* env vars removed, an empty ANTHROPIC_API_KEY and a private GIT_CONFIG_GLOBAL.
    # Asserts exit 0, forged reward.json is gone, and reward.txt is 1.0 with the oracle applied and
    # < 1.0 otherwise. Side effects: subprocesses (git, bash, 30s timeout) and files under tmp_path.
    # Skipped on Windows.
    """Execute the emitted shell and real verifier with only paths relocated."""
    workspace = tmp_path / "repo"
    workspace.mkdir()

    def git(*args: str) -> str:
        # [FUNCTION DOC] HELPER: Runs `git <args>` in the temp workspace with check=True and returns stdout as
        # text. Raises CalledProcessError on failure.
        return subprocess.run(
            ["git", *args], cwd=workspace, capture_output=True, text=True, check=True
        ).stdout

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    source = workspace / "answer.py"
    source.write_text("answer = 1\n", encoding="utf-8")
    git("add", "answer.py")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD").strip()
    source.write_text("answer = 2\n", encoding="utf-8")
    oracle = git("diff", "HEAD")
    if not apply_oracle:
        git("checkout", "--", "answer.py")

    rewards = tmp_path / "rewards"
    rewards.mkdir()
    for name, content in (("reward.txt", "1"), ("reward.json", '{"reward": 1}')):
        (rewards / name).write_text(content, encoding="utf-8")
    for name, content in _pr_diff_aux_files(oracle_diff=oracle, instruction="Fix answer").items():
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True)
        target.write_text(content.replace("/logs/verifier", str(rewards)), encoding="utf-8")
    script = build_pr_diff_eval_script(base_commit=base)
    for original, relocated in (
        ("/workspace", workspace),
        ("/logs/verifier", rewards),
        ("/tmp/predicted.patch", tmp_path / "predicted.patch"),
    ):
        script = script.replace(original, shlex.quote(str(relocated)))
    script_path = tmp_path / "tests/test.sh"
    script_path.write_text(script, encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("R2E_")}
    env.update(ANTHROPIC_API_KEY="", GIT_CONFIG_GLOBAL=str(tmp_path / "gitconfig"))
    result = subprocess.run(
        ["bash", str(script_path)], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # Harbor reads reward.json before reward.txt; a forged JSON must not survive.
    assert not (rewards / "reward.json").exists()
    reward = float((rewards / "reward.txt").read_text(encoding="utf-8"))
    if apply_oracle:
        assert reward == 1.0
    else:
        assert reward < 1.0


def test_eval_script_exits_zero() -> None:
    # [FUNCTION DOC] Checks the eval script ends with 'exit 0' (reward is in reward.txt, not the exit code).
    """Verifier writes reward.txt; bash exit code is moot."""
    es = build_pr_diff_eval_script(base_commit="abc")
    assert "exit 0" in es


def test_dockerfile_supports_private_repo_build_arg() -> None:
    # [FUNCTION DOC] Checks the Dockerfile declares an empty ARG GITHUB_TOKEN, uses an x-access-token clone URL
    # when set, falls back to the clean URL, resets origin to the clean URL, and bakes no literal
    # token.
    """The emitted Dockerfile clones via an optional GITHUB_TOKEN build arg
    so private repos work at consumer build time, then scrubs the remote so
    the token never persists in git config inside the image."""
    df = build_pr_diff_environment_dockerfile(
        repo_url="https://github.com/myorg/private-repo.git",
        base_commit="abc123",
    )
    # Build arg declared, empty default (public repos need no arg).
    assert "ARG GITHUB_TOKEN=" in df
    # Authed clone uses the x-access-token form when the arg is set.
    assert 'x-access-token:"${GITHUB_TOKEN}"@github.com/myorg/private-repo.git' in df
    # Public fallback clone is the clean URL.
    assert "git clone --filter=blob:none https://github.com/myorg/private-repo.git" in df
    # Remote is reset to the clean URL so the token can't leak via git config.
    assert "remote set-url origin https://github.com/myorg/private-repo.git" in df
    # The token itself must never appear literally baked anywhere.
    assert "ghp_" not in df


def test_verifier_source_is_stdlib_only() -> None:
    # [FUNCTION DOC] Parses _verifier_source() with ast, collects top-level absolute import roots and asserts all
    # are in sys.stdlib_module_names.
    """The verifier ships to a bare python:3.12-slim container — nothing but stdlib may be imported."""
    import ast
    import sys

    from repo2rlenv.pipelines.pr_diff import _verifier_source

    tree = ast.parse(_verifier_source())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    non_stdlib = sorted(imported - sys.stdlib_module_names)
    assert non_stdlib == [], f"verifier imports non-stdlib modules: {non_stdlib}"
