# [MODULE DOC] pr_runtime pipeline plus the shared runtime-task toolkit reused by commit_runtime and
# cve_patches. The pipeline lists merged PRs/MRs via the source provider, fetches each diff, splits
# it into a source patch (oracle) and a test patch, applies metadata/structural/Lite filters,
# validates FAIL_TO_PASS / PASS_TO_PASS sets inside the bootstrap Docker image
# (pr_runtime_validate.validate_pr), and emits Harbor tasks with a Dockerfile reset to the PR base
# commit (git history scrubbed), a graded tests/test.sh, plain verifier/F2P/P2P artifacts and an
# egress-guard compose overlay. Shared helpers: leak scrubbing and body reflow for instructions,
# test-path classification, diff splitting/counting, eval script and Dockerfile builders, and test-
# command normalization/targeting per runner (pytest, go, cargo, jest/mocha/vitest). Side effects:
# network via provider, Docker, files under out_dir, logging.
"""Sandbox-verified PR mining (SWE-bench-style).

For each merged PR within scope:
  1. Pull metadata + unified diff via `gh pr list` / `gh pr diff`
  2. Split the diff into `patch` (source files) and `test_patch` (test files)
     using the same keyword-on-path heuristic SWE-bench uses
  3. If validation enabled: run the bootstrap container twice (once with
     test_patch only, once with both patches) to compute FAIL_TO_PASS and
     PASS_TO_PASS sets — the verified oracle
  4. Emit a Harbor task with environment/Dockerfile (FROM <bootstrap_image>),
     tests/test.sh (the eval script), and solution/patch.diff (gold patch)

Unlike `pr_diff`, this pipeline requires a working Docker image from the
bootstrap phase. `cmd_generate` triggers `ensure_bootstrap()` automatically
when `requires_bootstrap=True`.

----------------------------------------------------------------------------
Acknowledgment
----------------------------------------------------------------------------
This pipeline mirrors the data-collection + validation approach of:

  SWE-bench: Can Language Models Resolve Real-world Github Issues?
  (Jimenez et al., ICLR '24, arXiv:2310.06770)
  https://github.com/SWE-bench/SWE-bench        (MIT)

  SWE-bench-Live: A Live Benchmark for Issue Resolving
  (Zhang et al., NIPS '25, arXiv:2505.23419)
  https://github.com/microsoft/SWE-bench-Live   (MIT)

We adapt the patch-split heuristic (collect/utils.py:extract_patches),
the eval-script structure (harness/test_spec/utils.py:make_eval_script_list_common),
and the F2P/P2P grading semantics (harness/grading.py). No code is copied;
we don't depend on the `swebench` PyPI package.

Released under Apache-2.0 along with the rest of Repo2RLEnv.
----------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from repo2rlenv import github
from repo2rlenv.auth import resolve_repo_token
from repo2rlenv.bootstrap.spec import BootstrapResult
from repo2rlenv.emitter.harbor import HarborTask, write_harbor_task
from repo2rlenv.github import GitHubError, PullRequestSummary
from repo2rlenv.gitlab import GitLabError
from repo2rlenv.pipelines._env_guard import (
    egress_guard_compose,
    fix_source_hosts,
    git_history_scrub,
)
from repo2rlenv.pipelines.base import PipelineResult
from repo2rlenv.provider import provider_for
from repo2rlenv.sources import Capability
from repo2rlenv.spec.input import GenerationInput, PipelineName
from repo2rlenv.spec.options import PRRuntimeOptions

logger = logging.getLogger(__name__)

# PR/MR data calls dispatch by source (GitHub vs GitLab); either can raise.
_PROVIDER_ERRORS = (GitHubError, GitLabError)


_CLOSES_RE = re.compile(r"\b(?:closes|fixes|resolves)\s+#\d+\b", re.IGNORECASE)

# `git clean -fdx` excludes. Keep language dependency/build dirs so resetting
# to a PR's base_commit doesn't wipe installed deps the test suite needs:
#   node_modules (npm/yarn) · target (cargo) · vendor (go) · .venv/venv/.tox
#   (python) · .gradle/build (jvm). Without these, Node/Rust repos would yield
#   ZERO tasks because the suite can't even import its deps after the clean.
_GIT_CLEAN_EXCLUDES = (
    "-e .venv -e venv -e __pycache__ -e .tox "
    "-e node_modules -e target -e vendor -e .gradle -e .next -e .pytest_cache"
)

# Which issue does this PR close? Used to source the problem statement from
# the issue (bug report) instead of the PR body (fix description).
# Matches `Fixes #123`, `Closes #123`, AND the markdown-link form
# `fixes [#123](url)` that PR authors commonly use.
_LINKED_ISSUE_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+\[?#(\d+)", re.IGNORECASE
)

# Boilerplate sections that bloat a PR body without describing the problem,
# OR leak the solution. We drop everything from the first such header on.
# Crucially this includes test sections: a "Tests added / updated" block
# names the exact FAIL_TO_PASS tests that grade the task (an eval leak).
_BODY_NOISE_HEADER_RE = re.compile(
    r"^\s*#{0,6}\s*"
    r"(?:checklist|change\s*log|changelog|release\s*notes?|how\s+to\s+test|"
    r"test\s*plan|tests?\s+added(?:\s*/?\s*updated)?|tests?\s+added\s+or\s+updated|"
    r"testing|types?\s+of\s+changes?|pr\s+checklist|reviewer\s+notes?)\s*:?\s*$",
    re.IGNORECASE,
)

# Solution-leak patterns stripped from the problem statement before use.
_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "cherry-pick c3535905", "backport commit abc1234", "commit deadbeef1234"
    re.compile(
        r"\b(?:cherry[-\s]?pick(?:ed|ing|s)?|back[-\s]?port(?:ed|ing|s)?|commit)\b"
        r"[^\n]*?\b[0-9a-f]{7,40}\b",
        re.IGNORECASE,
    ),
    # Bare full 40-char SHAs (almost always a git ref pointing at the fix)
    re.compile(r"\b[0-9a-f]{40}\b"),
    # Markdown links to a github PR/issue/commit (point straight at the fix)
    re.compile(r"\[[^\]]*\]\(https?://github\.com/[^\s)]*?/(?:pull|issues|commit)/[^\s)]*\)"),
    # Bare github PR/issue/commit URLs (incl. redirect.github.com)
    re.compile(r"https?://(?:\w+\.)?github\.com/[^\s)]*?/(?:pull|issues|commit)/\S+"),
    # Closes/Fixes/Resolves/See/Refs #N (and cross-repo owner/repo#N)
    re.compile(
        r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?|see|refs?|ref|follow[-\s]?up(?:\s+to)?)\b"
        r"[:\s]*(?:[\w.-]+/[\w.-]+)?#\d+",
        re.IGNORECASE,
    ),
    # Markdown issue/PR refs `[#1234](url)` and bare `[#1234]`
    re.compile(r"\[#\d+\]\([^)]*\)"),
    # Parenthesized PR/issue ref at the end of a commit subject — common
    # "merged via squash" trailer: `Fix X (#1234)`. Strip incl. preceding
    # whitespace so the title cleans up to "Fix X".
    re.compile(r"\s*\(#\d+\)"),
    # Cross-repo issue refs without a closes keyword (`gorilla#739`).
    # Distinct from owner/repo#N (which has a slash) — that's already
    # covered by the closes pattern above when paired with a keyword.
    re.compile(r"\b[a-zA-Z][\w.-]*#\d+\b"),
)


def _linked_issue_number(pr_body: str) -> int | None:
    # [FUNCTION DOC] HELPER: Returns the issue number referenced by a close/fix/resolve keyword
    # followed by "#N" or "[#N" in the text, else None. Input: pr_body (str or None). Output: int or
    # None.
    """Return the issue number this PR closes, if any (`Closes #123`)."""
    m = _LINKED_ISSUE_RE.search(pr_body or "")
    return int(m.group(1)) if m else None


def _strip_info_leak(text: str) -> str:
    # [FUNCTION DOC] HELPER: Applies every _LEAK_PATTERNS regex in order, deleting cherry-
    # pick/backport/commit SHA mentions, bare 40-char SHAs, markdown and bare GitHub
    # pull/issues/commit links, close/fix/see/ref #N refs (incl. owner/repo#N), [#N](url) links, "
    # (#N)" squash suffixes and name#N cross-repo refs. Does not trim whitespace. Input: text (str).
    # Output: str. Side effects: none.
    """Remove solution-pointing references (SHAs, fix-PR links, #refs)."""
    out = text
    for pat in _LEAK_PATTERNS:
        out = pat.sub("", out)
    return out


# A markdown comment block (PR templates wrap guidance in <!-- ... -->).
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Collapse 3+ blank lines down to a single blank line.
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
# Cap the instruction body so a 5000-word PR essay doesn't dominate context.
_MAX_BODY_CHARS = 4000


def _reflow_pr_body(body: str) -> str:
    # [FUNCTION DOC] HELPER: Tidies a PR/issue body: removes HTML comments, truncates at the first
    # boilerplate header line (checklist, changelog, release notes, how to test, test plan, tests
    # added/updated, testing, types of changes, reviewer notes), collapses 3+ newlines, strips, and
    # caps at 4000 chars appending a "(truncated)" marker. Input: body (str). Output: str.
    """Tidy a verbose PR body into a focused problem statement.

    - drop HTML comment blocks (PR-template guidance)
    - drop everything from the first checklist/changelog/template header on
    - collapse runs of blank lines
    - cap length

    Conservative: only trims obvious boilerplate. Never rewrites prose (no
    LLM call here — deterministic + cheap).
    """
    body = _HTML_COMMENT_RE.sub("", body)
    kept: list[str] = []
    for line in body.splitlines():
        if _BODY_NOISE_HEADER_RE.match(line):
            break  # the rest is checklist/changelog noise
        kept.append(line)
    out = _MULTI_BLANK_RE.sub("\n\n", "\n".join(kept)).strip()
    if len(out) > _MAX_BODY_CHARS:
        out = out[:_MAX_BODY_CHARS].rstrip() + "\n\n…(truncated)"
    return out


# Path-component classifier for "is this a test file?".
#
# SWE-bench's heuristic is a substring match on the full path — which over-fires:
# `docs/testing.md`, `src/click/testing.py`, etc. all become "test files".
# We instead match on PATH COMPONENTS (split by /) and explicitly exclude
# documentation paths.
_TEST_DIR_NAMES = {"test", "tests", "testing", "e2e", "__tests__"}
_DOC_PREFIX_DIRS = {"docs", "doc", "documentation", "examples", "example"}


def _path_is_test(path: str) -> bool:
    # [FUNCTION DOC] HELPER: Classifies a repo path as a test file: False for empty paths or paths
    # whose first component is a docs/examples root; True if any directory component is
    # test/tests/testing/e2e/ __tests__; else True for basenames test_*.py/.js/.ts, *_test.py,
    # *_test.go, *.test.ts/js, *.spec.ts/js. Case-insensitive. Input: path (str). Output: bool.
    """True if the file is a real test file, false for docs / src files
    that merely contain a test keyword in their path.

    Rules:
      1. Files under a documentation root (`docs/`, `examples/`, ...) are
         NEVER test files, even if their name contains "test".
      2. Files inside a test directory component (any path part in
         `_TEST_DIR_NAMES`) are test files.
      3. Files with a pytest-style basename (`test_*.py`, `*_test.py`,
         `*_test.go`) are test files.
    """
    if not path:
        return False
    parts = [p.lower() for p in path.split("/") if p]
    if not parts:
        return False
    # Rule 1: skip anything under a docs root
    if parts[0] in _DOC_PREFIX_DIRS:
        return False
    # Rule 2: any directory component is a known test dir
    # (excluding the last component, which is the file name)
    for component in parts[:-1]:
        if component in _TEST_DIR_NAMES:
            return True
    # Rule 3: filename-level test markers
    basename = parts[-1]
    return (
        (basename.startswith("test_") and basename.endswith((".py", ".js", ".ts")))
        or basename.endswith(("_test.py", "_test.go", ".test.ts", ".test.js"))
        or basename.endswith((".spec.ts", ".spec.js"))
    )


# Match `diff --git a/<path> b/<path>` block boundaries to split a unified diff
_DIFF_HEADER_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)


def split_patch_and_test_patch(unified_diff: str) -> tuple[str, str]:
    # [FUNCTION DOC] Splits a unified diff into (source patch, test patch) by per-file "diff --git"
    # blocks: a block goes to the test patch if either its a/ or b/ path satisfies _path_is_test,
    # else to the source patch. Blank input returns ("", ""); input with no diff headers is returned
    # entirely as the source patch. Input: unified_diff (str). Output: tuple[str, str].
    # KNOWN ISSUE: the docstring describes SWE-bench's substring rule ("path contains
    # test/tests/e2e/ testing") but the code uses the stricter path-component/basename classifier
    # _path_is_test.
    """Split a PR's unified diff into (source patch, test patch).

    SWE-bench rule: a file hunk goes into `test_patch` iff its path contains
    one of `test/tests/e2e/testing`; everything else goes into `patch`.

    We walk the diff by `diff --git` markers (one per file in the PR) so we
    keep each file's hunks intact.
    """
    if not unified_diff.strip():
        return "", ""

    # Find each "diff --git a/X b/Y" header and the byte offset where its block starts
    matches = list(_DIFF_HEADER_RE.finditer(unified_diff))
    if not matches:
        # Empty / malformed — return whole thing as patch
        return unified_diff, ""

    patch_parts: list[str] = []
    test_parts: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(unified_diff)
        block = unified_diff[start:end]
        path_a = m.group(1)
        path_b = m.group(2)
        # If EITHER path looks like a test file (covers renames, new files), it's a test
        if _path_is_test(path_a) or _path_is_test(path_b):
            test_parts.append(block)
        else:
            patch_parts.append(block)

    return "".join(patch_parts), "".join(test_parts)


def _build_instruction(
    pr: PullRequestSummary,
    owner: str = "",
    name: str = "",
    *,
    token: str | None = None,
    provider=github,
) -> str:
    # [FUNCTION DOC] HELPER: Builds the markdown task instruction for a PR. Inputs: pr; owner, name
    # (default ""); token (keyword-only, default None); provider (keyword-only, module/object with
    # fetch_issue, default the github module). If the PR body links an issue and owner/name are
    # given, fetches the issue and uses its title/body (fallbacks to PR values when empty). Body
    # then has Closes #N removed, leak patterns stripped, and is reflowed (placeholder when empty);
    # title is leak-stripped (falls back to the raw PR title). Output: "# Issue" / Title / "##
    # Description" / "## Task" (base = first 12 chars of base_sha). Side effects: possible network
    # issue fetch.
    """Build the task instruction (problem statement).

    Sources the problem from the **linked issue** (the bug *report*) when the
    PR closes one, falling back to the PR body otherwise. The PR body is
    written by the fixer and routinely leaks the solution (commit SHAs to
    cherry-pick, the fix approach, the names of the grading tests); the issue
    describes the *symptom*, which is what we want the agent to work from.
    This mirrors SWE-bench, which builds problem statements from issue text.

    Whatever text we use is then run through the info-leak strip + reflow so
    stray fix-PR links / #refs / SHAs / test-section noise don't leak through.
    """
    title = pr.title
    body = pr.body or ""

    issue_num = _linked_issue_number(pr.body or "")
    if issue_num is not None and owner and name:
        fetched = provider.fetch_issue(owner, name, issue_num, token=token)
        if fetched:
            i_title, i_body = fetched
            title = i_title or pr.title
            body = i_body or body

    body = _CLOSES_RE.sub("", body)
    body = _strip_info_leak(body)
    body = _reflow_pr_body(body).strip()
    if not body:
        body = "(no description provided in source issue/PR)"
    title = _strip_info_leak(title).strip() or pr.title
    return (
        f"# Issue\n\n"
        f"**Title:** {title}\n\n"
        f"## Description\n\n"
        f"{body}\n\n"
        f"## Task\n\n"
        f"Modify the repository so that the issue described above is resolved. "
        f"The task's test suite verifies your patch by applying it on top of "
        f"the base commit `{pr.base_sha[:12]}` and running the modified tests."
    )


def _word_count(text: str) -> int:
    # [FUNCTION DOC] HELPER: Counts regex word tokens in text (None treated as empty). Output: int.
    return len(re.findall(r"\b\w+\b", text or ""))


# PR titles that signal a non-bug chore — not a meaningful SWE task, and their
# bodies routinely leak the fix (commit SHA to cherry-pick, the source PR).
_NON_BUG_TITLE_RE = re.compile(
    r"\b(?:back[-\s]?port|cherry[-\s]?pick|revert|"
    r"bump|release|changelog|forward[-\s]?merge|"
    r"prepare\s+(?:for\s+)?release|version\s+bump|re-?sync)\b"
    # merge-forward / branch-sync PRs: "merge branch", "merge stable into main",
    # "merge X into Y", "sync stable", "sync main", "merge master" — these are
    # broad branch syncs, not focused bug fixes, and produce huge noisy diffs.
    r"|\bmerge\b[^\n]*\binto\b"
    r"|\bmerge\s+(?:branch|stable|main|master|develop|upstream|release)\b"
    r"|\bsync\s+(?:stable|main|master|branch|develop|upstream)\b",
    re.IGNORECASE,
)


def _is_non_bug_pr(title: str) -> bool:
    # [FUNCTION DOC] HELPER: True when the title matches _NON_BUG_TITLE_RE (backport, cherry-pick,
    # revert, bump, release, changelog, forward-merge, version bump, resync, merge ... into, merge
    # branch/main/..., sync main/...). Input: title (str or None). Output: bool.
    """True if the PR title marks a backport / cherry-pick / release / revert."""
    return bool(_NON_BUG_TITLE_RE.search(title or ""))


def _verifier_source() -> str:
    # [FUNCTION DOC] HELPER: Reads and returns the text of the sibling _pr_runtime_verifier.py.
    # Output: str. Raises: OSError if missing. Side effects: file read.
    # KNOWN ISSUE: the docstring says the source is baked into test.sh as base64, but it is now
    # shipped as the plain tests/verifier.py artifact via _runtime_aux_files.
    """Read the standalone graded verifier's source for base64 embedding.

    Mirrors pr_diff: the verifier (`_pr_runtime_verifier.py`) runs inside
    the task container, so we read its source at gen time and bake it into
    `tests/test.sh` as a base64 blob.
    """
    return (Path(__file__).parent / "_pr_runtime_verifier.py").read_text(encoding="utf-8")


def build_environment_dockerfile(bootstrap_image: str, base_commit: str) -> str:
    # [FUNCTION DOC] Builds the per-task Dockerfile: FROM bootstrap_image (tag for local images,
    # registry digest for pushed ones; chosen by the caller), WORKDIR /workspace, best-effort
    # install of git + ca-certificates and python3 via apt-get or apk, mark safe.directory, fetch
    # base_commit (or unshallow) best effort, hard reset to base_commit and git clean -fdx keeping
    # dependency dirs (_GIT_CLEAN_EXCLUDES), then append _env_guard.git_history_scrub(base_commit).
    # Inputs: bootstrap_image (str); base_commit (SHA str). Output: str. Side effects: none.
    """Build the per-task environment/Dockerfile.

    The bootstrap image has the repo at the bootstrap-time HEAD (whatever
    ref was used during `repo2rlenv bootstrap`). Each PR has its own
    `base_commit` — usually NOT bootstrap HEAD. If Harbor builds the agent
    image from the bootstrap image as-is, the agent + verifier see
    HEAD-state source files, but the gold patch (and any model patch) was
    written against `base_commit`-state files. Patches would fail to apply
    against the wrong line context.

    Fix: at build time, fetch `base_commit` (in case shallow clone doesn't
    have it) and reset the working tree to it. This makes Harbor's
    "apply model patch then run test.sh" flow correct.

    `bootstrap_image` should be the tag (e.g. `local/r2e-bootstrap/foo:abc`)
    for local-only bootstraps and the registry-qualified digest
    (`ghcr.io/owner/foo@sha256:...`) for pushed images. Docker BuildKit's
    `FROM <name>@sha256:...` syntax tries to fetch from a registry — local
    digest references don't work. The caller (`_build_task`) picks the
    right form based on `BootstrapResult.pushed_to_registry`.
    """
    return (
        f"# Auto-generated by Repo2RLEnv pr_runtime\n"
        f"FROM {bootstrap_image}\n"
        f"WORKDIR /workspace\n"
        f"# Defensive: ensure git is on PATH at build time. Bootstrap base\n"
        f"# images vary — some (Python slim) don't ship git; some agents\n"
        f"# install it; others don't. Re-installing is a no-op when already\n"
        f"# present. Tries apt-get first (Debian/Ubuntu), then apk (Alpine).\n"
        f"# Install git + ca-certificates. Minimal images (node:alpine) ship\n"
        f"# without CA certs, so HTTPS git fetch fails verification.\n"
        f"RUN (command -v git >/dev/null 2>&1 && [ -e /etc/ssl/certs/ca-certificates.crt ]) || \\\n"
        f"    (apt-get update && apt-get install -y --no-install-recommends git ca-certificates \\\n"
        f"     && rm -rf /var/lib/apt/lists/*) || \\\n"
        f"    (apk add --no-cache git ca-certificates && update-ca-certificates) || true\n"
        f"# Defensive: the graded F2P/P2P verifier is Python.\n"
        f"# Language-specific bootstrap images (Go/Rust/Node) may not\n"
        f"# ship python3 — install it so test.sh can score F2P/P2P. No-op when\n"
        f"# python3 is already present (every Python-repo image has it).\n"
        f"RUN command -v python3 >/dev/null 2>&1 || \\\n"
        f"    (apt-get update && apt-get install -y --no-install-recommends python3 \\\n"
        f"     && rm -rf /var/lib/apt/lists/*) || \\\n"
        f"    apk add --no-cache python3 || true\n"
        f"# Position the working tree at the PR's base commit so subsequent\n"
        f"# model-patch applications align with the line context the patch\n"
        f"# was authored against. The fetch is a no-op if the commit is\n"
        f"# already in the shallow clone.\n"
        f"RUN git config --global --add safe.directory /workspace \\\n"
        f"    && git fetch --depth 1 origin {base_commit} 2>/dev/null \\\n"
        f"       || git fetch --unshallow origin 2>/dev/null || true\n"
        f"RUN git reset --hard {base_commit} && git clean -fdx {_GIT_CLEAN_EXCLUDES}\n"
        # ANTI-CHEAT: the working tree is at base_commit, but .git still holds
        # the future (origin/main, tags, the fix commit + its hidden test),
        # which an agent can read offline. Strip it down to base_commit.
        + git_history_scrub(base_commit)
    )


def _path_prelude_for_language(language: str | None) -> str:
    # [FUNCTION DOC] HELPER: Returns an "export PATH=..." line prepending known toolchain dirs for
    # go, rust, node or java (case-insensitive), or "" for other/None languages. Input: language
    # (str or None).
    # Output: str.
    """Shell snippet that prepends common toolchain dirs to $PATH.

    The bootstrap agent often installs language toolchains (Go, Rust,
    Node) into well-known paths (`/usr/local/go/bin`, `~/.cargo/bin`,
    nvm dirs) but doesn't always persist a corresponding `export PATH`
    to a shell init file. When Harbor's verifier runs `bash test.sh` in
    a non-interactive shell, those binaries vanish from PATH → exit 127
    on `go test` / `cargo test` / `node` → false-negative reward 0.

    The fix at emission time: prepend the known install locations for
    the bootstrap-detected language so the verifier shell always finds
    the runner binary. Missing dirs are no-ops; the cost is one extra
    line in test.sh.
    """
    extras = {
        "go": ["/usr/local/go/bin", "$HOME/go/bin"],
        "rust": ["$HOME/.cargo/bin"],
        "node": ["/usr/local/lib/node_modules/.bin", "$HOME/.nvm/versions/node/*/bin"],
        "java": ["/usr/lib/jvm/default-java/bin"],
    }
    dirs = extras.get((language or "").lower(), [])
    if not dirs:
        return ""
    joined = ":".join(dirs)
    return f'export PATH="{joined}:$PATH"\n'


def build_eval_script(
    base_commit: str,
    test_patch: str,
    test_cmds: list[str],
    *,
    language: str | None = None,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
) -> str:
    # [FUNCTION DOC] Builds tests/test.sh run by Harbor after the agent's edits. Inputs:
    # base_commit; test_patch (hidden test diff); test_cmds (list joined with &&); keyword-only
    # language (for PATH prelude), fail_to_pass and pass_to_pass (lists or None). Script: bash with
    # set -uxo pipefail, optional PATH prelude, cd /workspace, mark safe.directory, mkdir
    # /logs/verifier, checkout test files touched by the test patch from base_commit (tolerating
    # failure), apply the test patch from a heredoc with git apply --reject and fail closed (reward
    # 0 + reward-details.json with "test_patch_apply_failed", exit 0) if it does not apply, run the
    # test block in a subshell capturing output to /logs/verifier/test_output.log, record exit code,
    # and print the log. Without fail_to_pass: binary reward 1.0/0.0 from the exit code. With
    # fail_to_pass: runs $SCRIPT_DIR/verifier.py with the log, f2p.json, p2p.json, quoted test
    # commands, exit code and out dir, falling back to the binary reward if the verifier fails. Both
    # paths reset the test files again and exit 0. Output: str. Side effects: none.
    """Build the `tests/test.sh` content that Harbor runs after the model patch.

    Adapted from SWE-bench's `harness/test_spec/utils.py:make_eval_script_list_common`.
    The flow:
      1. cd /workspace + mark safe.directory (for non-root git operations)
      2. Prepend known toolchain paths for the detected language (compensates
         for bootstrap agents that install Go/Rust/Node outside /usr/bin
         without exporting PATH in any persisted shell init file)
      3. Reset test files to base_commit (so re-running stays clean) — this is
         the anti-tamper guard: the agent CANNOT pass by editing/deleting the
         tests, because we restore them to base and re-apply the test_patch
      4. Apply the test_patch (via heredoc + git apply --reject)
      5. Run test_cmds, capturing stdout+stderr to a log file
      6. Score the reward:
         - GRADED (default, when fail_to_pass is provided): bake the standalone
           graded verifier + the F2P/P2P test-name lists, parse the captured
           log, and write reward = f2p_rate * p2p_rate to reward.txt (+ full
           breakdown incl. the strict SWE-bench ``resolved`` bool to reward-details.json)
         - BINARY (fallback, e.g. ``skip_validation`` with no F2P known): write
           1.0 if the suite exited 0 else 0.0
      7. Reset test files again on the way out

    The model's predicted patch is applied by Harbor *before* this script runs.
    """
    test_files = _files_in_patch(test_patch)
    heredoc = "EOF_R2E_TEST_PATCH"
    reset = (
        f"git checkout {base_commit} -- {' '.join(test_files)}"
        if test_files
        else "echo 'no test files to reset'"
    )
    apply = f"git apply --verbose --reject - <<'{heredoc}'\n{test_patch}\n{heredoc}"
    test_block = " && ".join(test_cmds) if test_cmds else "echo 'no test_cmds configured'"
    path_prelude = _path_prelude_for_language(language)

    # Fail CLOSED if the hidden test_patch doesn't apply: an agent that edited
    # tests or the file layout could make `git apply` fail, and we must NOT
    # then score against stale/native tests. Write reward 0 + a clear status
    # and stop. (Reset tolerates test files absent at base — that's `|| true`.)
    apply_guard = (
        ""
        if not test_patch.strip()
        else (
            f"{apply}\n"
            "R2E_APPLY_RC=$?\n"
            'if [ "$R2E_APPLY_RC" -ne 0 ]; then\n'
            '  echo "0.000000" > /logs/verifier/reward.txt\n'
            "  printf '%s' "
            '\'{"reward": 0.0, "resolved": false, "parse_status": '
            '"test_patch_apply_failed"}\' > /logs/verifier/reward-details.json\n'
            '  echo "R2E: test_patch failed to apply (rc=$R2E_APPLY_RC) — failing closed" >&2\n'
            "  exit 0\n"
            "fi\n"
        )
    )
    head = (
        "#!/bin/bash\n"
        "set -uxo pipefail\n"
        f"{path_prelude}"  # may be empty
        # Harbor mounts the task's tests/ dir at /tests; resolve it so we can
        # read the sibling verifier.py + f2p.json + p2p.json artifacts.
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        "cd /workspace\n"
        "git config --global --add safe.directory /workspace\n"
        "mkdir -p /logs/verifier\n"
        f"{reset} || true\n"  # tolerate test files that didn't exist at base
        f"{apply_guard}"
        # Capture the suite output so the verifier can parse per-test status.
        # `( ... )` subshell + redirect keeps the whole `a && b` block's output;
        # pipefail preserves the real test exit code.
        f"( {test_block} ) > /logs/verifier/test_output.log 2>&1\n"
        "TEST_EXIT_CODE=$?\n"
        "cat /logs/verifier/test_output.log\n"
    )

    if not fail_to_pass:
        # No F2P oracle available (e.g. --skip-validation). Fall back to the
        # binary exit-code reward.
        return (
            head + '[ "$TEST_EXIT_CODE" -eq 0 ] && echo "1.0" > /logs/verifier/reward.txt '
            '|| echo "0.0" > /logs/verifier/reward.txt\n'
            f"{reset} || true\n"
            "exit 0\n"
        )

    # Graded path: a thin orchestrator that runs the verifier + F2P/P2P lists
    # shipped as PLAIN task artifacts (tests/verifier.py, tests/f2p.json,
    # tests/p2p.json — written by `_runtime_aux_files`). No base64 blobs:
    # the task semantics are inspectable, and test.sh stays small regardless
    # of how many F2P/P2P tests a task has.
    cmds_str = " ".join(test_cmds).replace("'", "'\\''")
    return (
        head + 'python3 "$SCRIPT_DIR/verifier.py" '
        "--log /logs/verifier/test_output.log "
        '--f2p "$SCRIPT_DIR/f2p.json" --p2p "$SCRIPT_DIR/p2p.json" '
        f"--test-cmds '{cmds_str}' --exit-code \"$TEST_EXIT_CODE\" "
        "--out-dir /logs/verifier || "
        # If python3 is somehow unavailable, never leave reward.txt empty.
        '{ [ "$TEST_EXIT_CODE" -eq 0 ] && echo "1.0" > /logs/verifier/reward.txt '
        '|| echo "0.0" > /logs/verifier/reward.txt; }\n'
        f"{reset} || true\n"  # cleanup; failure here doesn't change verdict
        "exit 0\n"  # reward.txt is the verdict, not the bash exit code
    )


def _runtime_aux_files(fail_to_pass: list[str], pass_to_pass: list[str]) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Returns the plain tests/ artifacts for the graded verifier:
    # tests/verifier.py (source), tests/f2p.json and tests/p2p.json (pretty JSON lists; P2P None ->
    # []). Inputs: fail_to_pass, pass_to_pass (list[str]). Output: dict[str, str]. Side effects:
    # reads verifier source file.
    """The plain task artifacts the graded test.sh reads from /tests.

    Shipping these as files (vs base64 inside test.sh) keeps task semantics
    inspectable and test.sh small. Harbor exposes tests/ at /tests in the
    container.
    """
    return {
        "tests/verifier.py": _verifier_source(),
        "tests/f2p.json": json.dumps(fail_to_pass, indent=2),
        "tests/p2p.json": json.dumps(pass_to_pass or [], indent=2),
    }


def _files_in_patch(unified_diff: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns unique b/ paths from "diff --git" headers in order; [] for
    # blank input.
    # Input: unified_diff (str). Output: list[str].
    """Extract the unique 'b/' file paths touched by a unified diff."""
    if not unified_diff.strip():
        return []
    seen: list[str] = []
    for m in _DIFF_HEADER_RE.finditer(unified_diff):
        b = m.group(2)
        if b not in seen:
            seen.append(b)
    return seen


# Lines in a test_patch that introduce a new test function/class.
# Python: `+def test_foo(...)`, `+    def test_bar(...)`, `+class TestX`
# JS/TS:  `+it(`, `+test(`, `+describe(` (not currently filtered)
# Go:     `+func TestFoo(`
_NEW_TEST_FUNC_RE = re.compile(
    r"^\+\s*(?:def\s+test_\w+|class\s+\w*[Tt]est\w*|func\s+Test\w+|it\s*\(|test\s*\(|describe\s*\()",
)


def _diff_loc_changed(unified_diff: str) -> int:
    # [FUNCTION DOC] HELPER: Counts +/- lines excluding "+++ "/"--- " headers; None treated as
    # empty.
    # Input: unified_diff. Output: int.
    """Count real +/- lines in a unified diff (excludes +++/--- file markers)."""
    n = 0
    for line in (unified_diff or "").splitlines():
        if (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++ ", "--- ")):
            n += 1
    return n


def _difficulty_bucket(f2p_count: int, loc_changed: int) -> str:
    # [FUNCTION DOC] HELPER: Coarse difficulty: "trivial" when loc <= 5 and F2P <= 1, else "small"
    # (loc <= 20), "medium" (<= 80), "large". Inputs: f2p_count, loc_changed (int). Output: str.
    """Coarse difficulty from oracle size — lets consumers slice train/eval.

    Combines the fix size (LOC) and how many distinct behaviours must be
    restored (F2P count). Mirrors Arc 1's pr_diff buckets on LOC.
    """
    if loc_changed <= 5 and f2p_count <= 1:
        return "trivial"
    if loc_changed <= 20:
        return "small"
    if loc_changed <= 80:
        return "medium"
    return "large"


def _count_new_test_funcs(test_patch: str) -> int:
    # [FUNCTION DOC] HELPER: Counts added lines (starting with +) that define a new test: Python def
    # test_* or class *Test*, Go func Test*, JS it(/test(/describe(. Input: test_patch (str).
    # Output: int (0 if blank).
    """Count new test-function definitions added in a unified diff.

    Used to filter out PRs whose test_patch is comment-only or docstring-only
    (cosmetic changes that can't produce a FAIL_TO_PASS oracle).
    """
    if not test_patch.strip():
        return 0
    return sum(1 for line in test_patch.splitlines() if _NEW_TEST_FUNC_RE.match(line))


def normalize_test_cmds_for_runtime(test_cmds: list[str]) -> list[str]:
    # [FUNCTION DOC] Rewrites bootstrap test commands so they actually run tests with per-test
    # output. For every command: strips trailing "| head/tail ...", "2>&1", ">/dev/null" redirects
    # and trailing pipe/ ampersand chars. pytest: removes --collect-only/--co and -q/--quiet, adds
    # -v if no verbosity flag. go test: inserts -v after "go test" if missing. cargo test: removes
    # -q/--quiet. jest/mocha/ vitest/npm|yarn|pnpm test: removes --silent; vitest: removes --verbose
    # and adds --reporter=verbose if no reporter; jest/mocha: adds --verbose if no verbose/reporter
    # flag.
    # Input: test_cmds (list[str]). Output: new list[str] (stripped). Side effects: none.
    """Adapt bootstrap-recorded test commands for actual per-PR execution.

    Bootstrap prefers fast/tolerant commands (e.g. `pytest --collect-only`)
    so it can declare success without running every test. For pr_runtime,
    we need commands that *run* tests and emit per-test pass/fail lines
    that our parsers can read.

    Transforms (per runner):
      pytest:
        - Drop `--collect-only` / `--co` so pytest actually runs tests
        - Drop `-q` / `--quiet`: suppresses per-test names; cancels `-v` in pytest 9
        - Add `-v` if no verbosity flag is present
      go test:
        - Add `-v` if missing (default `go test` doesn't print --- PASS lines)
      cargo test:
        - Default output is already parseable; no transform needed
      jest / mocha / npm test:
        - Add `--verbose` if not present, so per-test ✓/✕ lines are emitted
        - Some configs swallow stdout via `--silent`; we strip that
      vitest:
        - Use `--reporter=verbose` instead: vitest rejects `--verbose`, and its
          default reporter collapses fully passing files to one summary line
    """
    out: list[str] = []
    for cmd in test_cmds:
        cleaned = cmd

        # Strip shell pipes / redirects / tail-truncators that bootstrap agents
        # sometimes append (e.g. `pytest -q 2>&1 | head -50`) so we capture only
        # the test runner invocation. If we keep them, `targeted_test_cmds_for_pr`
        # appends test files AFTER the pipe → broken command.
        # `[^|]*` swallows whatever flags follow `head`/`tail` (`-50`, `-n 100`, etc.)
        # without crossing into another piped command.
        cleaned = re.sub(r"\s*\|\s*(?:head|tail)\s*[^|]*$", "", cleaned)
        cleaned = re.sub(r"\s*2>&1\b", "", cleaned)
        cleaned = re.sub(r"\s*&?>\s*/dev/null\b", "", cleaned)
        cleaned = cleaned.rstrip(" |&")

        # --- pytest ---
        if re.search(r"\bpytest\b", cleaned):
            cleaned = re.sub(r"\s+--collect-only\b", "", cleaned)
            cleaned = re.sub(r"\s+--co\b", "", cleaned)  # pytest's short form
            # Strip -q/--quiet: it suppresses per-test names that the log parser needs.
            # -q and -v cancel each other in pytest 9 (verbosity counter), so -q must go.
            cleaned = re.sub(r"\s+(?:-q|--quiet)\b", "", cleaned)
            if not re.search(r"\s-v\b|\s--verbose\b|-vv\b", cleaned):
                cleaned = cleaned.rstrip() + " -v"

        # --- go test ---
        elif re.search(r"\bgo\s+test\b", cleaned):
            if not re.search(r"\s-v\b", cleaned):
                # Insert -v right after `go test`; positional args go after
                cleaned = re.sub(r"\bgo\s+test\b", "go test -v", cleaned, count=1)

        # --- cargo test ---
        elif re.search(r"\bcargo\s+test\b", cleaned):
            # `cargo test` already prints `test NAME ... ok/FAILED/ignored`
            # by default — no transformation needed. If a user passed
            # `-q`, the per-test lines disappear; strip it.
            cleaned = re.sub(r"\s+(?:-q|--quiet)\b", "", cleaned)

        # --- jest / npm test / yarn test / pnpm test ---
        elif re.search(r"\b(?:jest|mocha|vitest|npm\s+test|yarn\s+test|pnpm\s+test)\b", cleaned):
            cleaned = re.sub(r"\s+--silent\b", "", cleaned)
            # Add a per-test reporter if the cmd is the runner itself (skip
            # wrappers where flags need to go after `--`)
            if re.search(r"\bvitest\b", cleaned):
                # `vitest --verbose` exits with `CACError: Unknown option`.
                cleaned = re.sub(r"\s+--verbose\b", "", cleaned)
                if not re.search(r"\s--reporter\b", cleaned):
                    cleaned = cleaned.rstrip() + " --reporter=verbose"
            elif re.search(r"\b(?:jest|mocha)\b", cleaned) and not re.search(
                r"\s--verbose\b|\s--reporter\b", cleaned
            ):
                cleaned = cleaned.rstrip() + " --verbose"

        out.append(cleaned.strip())
    return out


# Pytest-style file extensions we know how to target. Anything else triggers
# the fallback to "run the whole suite".
_PYTEST_TARGETABLE_EXT = (".py",)
_JEST_TARGETABLE_EXT = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")


def _go_packages_from_test_files(test_files: list[str]) -> list[str]:
    # [FUNCTION DOC] HELPER: Maps *_test.go file paths to unique "./<dir>" package paths ("./" for
    # root-level files), preserving order; non-Go-test files are ignored. Input: test_files. Output:
    # list[str].
    """Map `pkg/foo/bar_test.go` → `./pkg/foo` for Go's package-path CLI."""
    pkgs: list[str] = []
    for f in test_files:
        if not f.endswith("_test.go"):
            continue
        # Directory of the test file, prefixed with ./ for `go test` package syntax
        parts = f.rsplit("/", 1)
        pkg = "./" + parts[0] if len(parts) == 2 else "./"
        if pkg not in pkgs:
            pkgs.append(pkg)
    return pkgs


def targeted_test_cmds_for_pr(test_cmds: list[str], test_files: list[str]) -> list[str]:
    # [FUNCTION DOC] Narrows test commands to the files touched by the test patch. If test_files is
    # empty, returns test_cmds unchanged (same list). pytest commands: append .py test files unless
    # a positional path argument already follows the pytest token. go test: replace "./..." with the
    # package dirs, or append them if no ./path is present. jest/mocha/vitest: append JS/TS test
    # files unless a path argument exists. Other runners (e.g. cargo) are unchanged.
    # Inputs: test_cmds (list[str]); test_files (list[str]). Output: list[str].
    """Limit the test invocation to the file paths the PR's test_patch touches.

    Running the whole suite on every PR is 10-50× slower than running only
    the files the PR cares about. SWE-bench-Live's harness does the same.

    Per-runner rules:
      pytest:  append the changed .py test files as positional args
      jest:    append the changed .js/.ts test files as positional args
      go test: replace `./...` (or trailing nothing) with the package
               directories containing changed `*_test.go` files
      cargo:   no targeting — Rust's filter is name-substring, not file
               (we'd need to introspect test names; whole-suite is fine)

    Skips if the cmd already has a positional path arg.
    """
    if not test_files:
        return test_cmds

    py_files = [f for f in test_files if f.endswith(_PYTEST_TARGETABLE_EXT)]
    js_files = [f for f in test_files if f.endswith(_JEST_TARGETABLE_EXT)]
    go_pkgs = _go_packages_from_test_files(test_files)

    out: list[str] = []
    for cmd in test_cmds:
        # --- pytest ---
        if re.search(r"\bpytest\b", cmd) and py_files:
            tokens = cmd.split()
            pytest_idx = next(
                (i for i, t in enumerate(tokens) if t == "pytest" or t.endswith("/pytest")),
                -1,
            )
            if pytest_idx >= 0:
                tail = tokens[pytest_idx + 1 :]
                has_path_arg = any(
                    not t.startswith("-") and (t.endswith(".py") or "/" in t) for t in tail
                )
                if not has_path_arg:
                    cmd = cmd.rstrip() + " " + " ".join(py_files)

        # --- go test ---
        elif re.search(r"\bgo\s+test\b", cmd) and go_pkgs:
            # Replace `./...` with the targeted packages; if neither is present,
            # append the packages
            if "./..." in cmd:
                cmd = cmd.replace("./...", " ".join(go_pkgs))
            elif not re.search(r"\b\./\S+\b", cmd):
                cmd = cmd.rstrip() + " " + " ".join(go_pkgs)

        # --- jest / npx jest / mocha / vitest ---
        elif re.search(r"\b(?:jest|mocha|vitest)\b", cmd) and js_files:
            tokens = cmd.split()
            # If a positional file path is already present, don't double-up
            has_path = any(
                not t.startswith("-") and (t.endswith(_JEST_TARGETABLE_EXT) or "/" in t)
                for t in tokens[1:]
            )
            if not has_path:
                cmd = cmd.rstrip() + " " + " ".join(js_files)

        out.append(cmd)
    return out


class PRRuntimePipeline:
    # [CLASS DOC] Pipeline implementation for PipelineName.PR_RUNTIME. Class attributes: name,
    # required_capabilities={PULL_REQUESTS}, requires_bootstrap=True, experimental=False. Holds
    # input, PRRuntimeOptions, BootstrapResult, optional progress callback; run() also sets _token
    # and _provider used later when building instructions.
    """Sandbox-verified PR mining. Implements the `Pipeline` Protocol."""

    name: ClassVar[PipelineName] = PipelineName.PR_RUNTIME
    required_capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.PULL_REQUESTS})
    requires_bootstrap: ClassVar[bool] = True
    experimental: ClassVar[bool] = False  # stable

    def __init__(
        self,
        input: GenerationInput,
        options: PRRuntimeOptions,
        bootstrap: BootstrapResult | None = None,
    ):
        # [FUNCTION DOC] Stores input/options/bootstrap and clears the progress callback. Inputs:
        # input (GenerationInput); options (PRRuntimeOptions); bootstrap (BootstrapResult or None).
        # Raises: RuntimeError when bootstrap is None.
        if bootstrap is None:
            raise RuntimeError(
                "pr_runtime requires a BootstrapResult (set requires_bootstrap=True "
                "and let cmd_generate trigger it, or pass one explicitly)"
            )
        self.input = input
        self.options = options
        self.bootstrap = bootstrap
        self._progress_cb = None

    def set_progress_callback(self, cb) -> None:
        # [FUNCTION DOC] Registers the progress callback (keywords name, outcome, reason). Input:
        # cb. Output: None.
        self._progress_cb = cb

    def _emit_progress(self, name: str, outcome: str, reason: str = "") -> None:
        # [FUNCTION DOC] HELPER: Invokes the progress callback if set; its exceptions are logged at
        # debug and swallowed.
        # Inputs: name, outcome, reason (default ""). Output: None.
        if self._progress_cb is not None:
            try:
                self._progress_cb(name=name, outcome=outcome, reason=reason)
            except Exception as exc:
                logger.debug("progress callback failed: %s", exc)

    # ----- run loop -----------------------------------------------------------

    def run(self, out_dir: Path) -> PipelineResult:
        # [FUNCTION DOC] Executes the pipeline. Resolves the token and provider (stored on self),
        # raising RuntimeError for a private repo without a token; lists merged PRs with
        # limit/since/until/skip_drafts (RuntimeError on provider errors). For each PR: _pre_filter;
        # fetch diff (provider error -> "diff_fetch_failed" with a warning); split patches; skip
        # "empty_source_patch"/"no_test_patch"; _structural_quality_filter; _lite_filter. Unless
        # skip_validation, lazily starts one shared sandbox and runs validate_pr at base_sha with
        # normalized+targeted test commands; skips "no_fail_to_pass" (progress reason =
        # outcome.reason when present) if require_fail_to_pass and F2P count < min_fail_to_pass.
        # Builds and writes the task. Sandbox cleaned up in finally.
        # Input: out_dir (Path). Output: PipelineResult(candidates=len(prs), emitted, skipped,
        # out_dir, skip_reasons). Side effects: network, Docker, filesystem, logging, callbacks.
        out_dir.mkdir(parents=True, exist_ok=True)

        token = resolve_repo_token(self.input.repo, self.input.auth)
        self._token = token  # reused by _build_task to fetch linked-issue text
        self._provider = provider_for(self.input.repo)  # GitHub vs GitLab dispatch
        if self.input.repo.access == "private" and not token:
            raise RuntimeError(
                "private repo specified but no token resolved. "
                "Run `gh auth login` / set GITHUB_TOKEN (GitHub) or GITLAB_TOKEN (GitLab)."
            )

        owner, name = self.input.repo.owner_name
        logger.info("listing merged PRs for %s/%s (limit=%d)", owner, name, self.options.limit)
        try:
            prs = self._provider.list_merged_prs(
                owner,
                name,
                limit=self.options.limit,
                since=self.options.since,
                until=self.options.until,
                skip_drafts=self.options.skip_drafts,
                token=token,
            )
        except _PROVIDER_ERRORS as exc:
            raise RuntimeError(f"failed to list PRs: {exc}") from exc

        skip_reasons: dict[str, int] = {}
        emitted = 0
        sandbox = None  # lazy-init for the validation loop

        try:
            for pr in prs:
                pr_label = f"{owner}/{name}#{pr.number}"

                # Pre-validation skip filters
                reason = self._pre_filter(pr)
                if reason:
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    self._emit_progress(pr_label, "skip", reason)
                    continue

                # Fetch diff
                try:
                    diff = self._provider.fetch_pr_diff(owner, name, pr.number, token=token)
                except _PROVIDER_ERRORS as exc:
                    logger.warning("PR #%d: diff fetch failed: %s", pr.number, exc)
                    skip_reasons["diff_fetch_failed"] = skip_reasons.get("diff_fetch_failed", 0) + 1
                    self._emit_progress(pr_label, "error", "diff_fetch_failed")
                    continue

                patch, test_patch = split_patch_and_test_patch(diff)
                if not patch.strip():
                    skip_reasons["empty_source_patch"] = (
                        skip_reasons.get("empty_source_patch", 0) + 1
                    )
                    self._emit_progress(pr_label, "skip", "empty_source_patch")
                    continue
                if not test_patch.strip():
                    skip_reasons["no_test_patch"] = skip_reasons.get("no_test_patch", 0) + 1
                    self._emit_progress(pr_label, "skip", "no_test_patch")
                    continue

                # Structural quality filters (cheap, run before validation):
                # - CI-only PRs: source patch is 100% under .github/
                # - No new test functions: test_patch only edits comments/docstrings
                structural_reason = self._structural_quality_filter(patch, test_patch)
                if structural_reason:
                    skip_reasons[structural_reason] = skip_reasons.get(structural_reason, 0) + 1
                    self._emit_progress(pr_label, "skip", structural_reason)
                    continue

                # Lite-style structural filters
                lite_reason = self._lite_filter(pr, patch)
                if lite_reason:
                    skip_reasons[lite_reason] = skip_reasons.get(lite_reason, 0) + 1
                    self._emit_progress(pr_label, "skip", lite_reason)
                    continue

                # Validation (optional via skip_validation)
                fail_to_pass: list[str] = []
                pass_to_pass: list[str] = []
                validation_status = "skipped"
                if not self.options.skip_validation:
                    if sandbox is None:
                        sandbox = self._start_validation_sandbox()
                    from repo2rlenv.pipelines.pr_runtime_validate import validate_pr

                    targeted_cmds = targeted_test_cmds_for_pr(
                        normalize_test_cmds_for_runtime(self.bootstrap.test_cmds),
                        _files_in_patch(test_patch),
                    )
                    outcome = validate_pr(
                        sandbox=sandbox,
                        base_commit=pr.base_sha,
                        patch=patch,
                        test_patch=test_patch,
                        test_cmds=targeted_cmds,
                        language=self.bootstrap.language.value,
                        timeout=self.options.validation_timeout_sec,
                    )
                    fail_to_pass = outcome.fail_to_pass
                    pass_to_pass = outcome.pass_to_pass
                    validation_status = outcome.status
                    if (
                        self.options.require_fail_to_pass
                        and len(fail_to_pass) < self.options.min_fail_to_pass
                    ):
                        skip_reasons["no_fail_to_pass"] = skip_reasons.get("no_fail_to_pass", 0) + 1
                        self._emit_progress(pr_label, "skip", outcome.reason or "no_fail_to_pass")
                        continue

                # Emit the Harbor task
                task = self._build_task(
                    pr,
                    patch,
                    test_patch,
                    fail_to_pass=fail_to_pass,
                    pass_to_pass=pass_to_pass,
                    validation_status=validation_status,
                )
                write_harbor_task(task, out_dir)
                emitted += 1
                logger.info(
                    "emitted task %s (F2P=%d, P2P=%d)",
                    task.name,
                    len(fail_to_pass),
                    len(pass_to_pass),
                )
                self._emit_progress(task.name, "emit")
        finally:
            if sandbox is not None:
                sandbox.cleanup()

        return PipelineResult(
            candidates=len(prs),
            emitted=emitted,
            skipped=sum(skip_reasons.values()),
            out_dir=out_dir,
            skip_reasons=skip_reasons,
        )

    # ----- filters ------------------------------------------------------------

    def _pre_filter(self, pr: PullRequestSummary) -> str | None:
        # [FUNCTION DOC] HELPER: Diff-free gate returning "draft" (draft and skip_drafts),
        # "not_merged", "no_files", "non_bug_pr" (title matches non-bug patterns),
        # "problem_statement_too_short" (when min_problem_statement_words > 0 and PR body is
        # shorter), else None. Input: pr. Output: str/None.
        """Cheap filters that don't need the diff."""
        if pr.is_draft and self.options.skip_drafts:
            return "draft"
        if not pr.merged_at:
            return "not_merged"
        if not pr.changed_files:
            return "no_files"
        if _is_non_bug_pr(pr.title):
            # Backports / cherry-picks / release chores / reverts / version
            # bumps aren't real fix tasks, and their bodies leak commit SHAs
            # and fix-PR links. Drop them up front.
            return "non_bug_pr"
        if (
            self.options.min_problem_statement_words > 0
            and _word_count(pr.body or "") < self.options.min_problem_statement_words
        ):
            return "problem_statement_too_short"
        return None

    def _structural_quality_filter(self, source_patch: str, test_patch: str) -> str | None:
        # [FUNCTION DOC] HELPER: Returns "ci_only_patch" when skip_ci_only and all source files are
        # under .github/, "no_new_test_funcs" when require_new_test_funcs and the test patch adds no
        # test definition, else None. Inputs: source_patch, test_patch (str). Output: str or None.
        """Cheap diff-level filters that catch over-emitted task types.

        Returns a skip reason string, or None to keep.

        Two filters here, both shipping a lot of false positives in v0.3:
          1. CI-only PRs: source patch is 100% under .github/. These are
             tooling changes (zizmor scan, publish workflows) — no real
             code-fix signal.
          2. No new test functions: test_patch only edits comments /
             docstrings (e.g. typo cleanup PRs). Without a new test that
             FAILS at base_commit, there can be no FAIL_TO_PASS oracle.
        """
        source_files = _files_in_patch(source_patch)
        # Filter 1: CI-only — all source files are workflow YAMLs
        if (
            self.options.skip_ci_only
            and source_files
            and all(p.startswith(".github/") for p in source_files)
        ):
            return "ci_only_patch"
        # Filter 2: test_patch must add ≥1 new test function
        if self.options.require_new_test_funcs:
            n_new = _count_new_test_funcs(test_patch)
            if n_new < 1:
                return "no_new_test_funcs"
        return None

    def _lite_filter(self, pr: PullRequestSummary, source_patch: str) -> str | None:
        # [FUNCTION DOC] HELPER: SWE-bench Lite-style gate. Always: "too_many_source_files" if more
        # than max_source_files_per_pr. When options.lite_filter: "lite_not_single_source_file" (not
        # exactly 1 file), "lite_problem_too_short" (body < 40 words), "lite_has_image" (markdown
        # image or <img>), "lite_has_external_link" (http(s) URL not starting with github.com/),
        # "lite_has_commit_sha" (7-40 hex chars). Inputs: pr, source_patch. Output: str or None.
        # Note: the hex check also matches ordinary hex-looking words in the body (e.g. "deadbeef",
        # "1234567"), rejecting some legitimate PRs.
        """SWE-bench Lite-style structural filters."""
        source_files = _files_in_patch(source_patch)
        if len(source_files) > self.options.max_source_files_per_pr:
            return "too_many_source_files"
        if self.options.lite_filter:
            if len(source_files) != 1:
                return "lite_not_single_source_file"
            if _word_count(pr.body or "") < 40:
                return "lite_problem_too_short"
            # Reject if PR body contains images / external links / cross-PR/issue refs
            body = (pr.body or "").lower()
            if re.search(r"!\[[^\]]*\]\(|<img\s", body):
                return "lite_has_image"
            if re.search(r"\bhttps?://(?!github\.com/)", body):
                return "lite_has_external_link"
            if re.search(r"\b[a-f0-9]{7,40}\b", body):
                return "lite_has_commit_sha"
        return None

    # ----- sandbox -----------------------------------------------------------

    def _start_validation_sandbox(self):
        # [FUNCTION DOC] HELPER: Starts a DockerSandbox on the bootstrap image tag with an empty
        # temp marker dir (with .keep) as repo_dir (the repo is already inside the image). Output:
        # sandbox (caller cleans up). Side effects: temp dir (not removed), Docker container.
        """Spin up a DockerSandbox from the bootstrap image (shared across PRs)."""
        # The bootstrap image already contains the repo at the bootstrap-time HEAD.
        # We pass repo_dir=None? No — DockerSandbox.start requires a repo_dir to copy in.
        # The repo is already in the image; we'll just `git checkout` to each PR's base_commit
        # from inside the container, so the path here is a no-op marker.
        import tempfile

        from repo2rlenv.bootstrap.docker import DockerSandbox

        marker = Path(tempfile.mkdtemp(prefix="r2e-pr-runtime-"))
        (marker / ".keep").write_text("")  # docker cp <src>/. <dst> works on any non-empty dir
        # Pull just the tag, don't re-copy the repo (image already has it)
        sandbox = DockerSandbox.start(
            base_image=self.bootstrap.image_tag,
            repo_dir=marker,
            platform=self.input.bootstrap.platform,
        )
        return sandbox

    # ----- task builder -------------------------------------------------------

    def _build_task(
        self,
        pr: PullRequestSummary,
        patch: str,
        test_patch: str,
        *,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        validation_status: str,
    ) -> HarborTask:
        # [FUNCTION DOC] HELPER: Builds the HarborTask for a PR. Inputs: pr, patch (oracle),
        # test_patch; keyword-only fail_to_pass, pass_to_pass, validation_status. Id
        # "<owner>__<repo>-<number>". Eval script from build_eval_script with normalized+targeted
        # commands and F2P/P2P (graded when F2P non-empty); Dockerfile on image digest if pushed
        # else tag at base_sha. repo2env metadata: pipeline_version "0.3.0", PR URL reference,
        # optional synthesis_llm, pr_runtime block (reward_mode "graded"), reward_calibration with
        # counts, LOC and difficulty bucket. Instruction via _build_instruction using the stored
        # token/provider (defaults when run() was not called). Task difficulty is fixed to "medium"
        # (the computed bucket is only in metadata); category "bugfix". Aux files: verifier +
        # F2P/P2P JSON (only when F2P non-empty) plus egress-guard compose. Output: HarborTask. Side
        # effects: possible network issue fetch.
        owner, name = self.input.repo.owner_name
        task_id = f"{owner}__{name}-{pr.number}"

        resolved_test_cmds = targeted_test_cmds_for_pr(
            normalize_test_cmds_for_runtime(self.bootstrap.test_cmds),
            _files_in_patch(test_patch),
        )
        eval_script = build_eval_script(
            base_commit=pr.base_sha,
            test_patch=test_patch,
            test_cmds=resolved_test_cmds,
            language=self.bootstrap.language.value,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        )
        # Use image_tag for local-only bootstraps (Docker can resolve locally),
        # image_digest only when the image is in a registry that BuildKit can
        # actually pull from.
        image_ref = (
            self.bootstrap.image_digest
            if self.bootstrap.pushed_to_registry
            else self.bootstrap.image_tag
        )
        dockerfile = build_environment_dockerfile(
            bootstrap_image=image_ref,
            base_commit=pr.base_sha,
        )

        repo2env = {
            "pipeline": "pr_runtime",
            "pipeline_version": "0.3.0",
            "repo": f"{owner}/{name}",
            "ref": pr.base_sha,
            "reference": pr.url,
            "source_access": self.input.repo.access,
            "built_at": datetime.now(UTC).isoformat(),
            **({"synthesis_llm": self.input.llm.qualified_name} if self.input.llm else {}),
            "reward_kinds": ["test_execution", "diff_similarity"],
            "pr_runtime": {
                "pr_url": pr.url,
                "pr_merged_at": pr.merged_at,
                "base_commit": pr.base_sha,
                "fail_to_pass": fail_to_pass,
                "pass_to_pass": pass_to_pass,
                "validation_status": validation_status,
                "bootstrap_image": self.bootstrap.image_digest,
                # Graded reward: reward.txt carries f2p_rate*p2p_rate (training
                # signal); the strict SWE-bench `resolved` bool + full breakdown
                # go to reward-details.json (eval signal — Harbor's reward.json
                # schema is flat-numeric-only). See _pr_runtime_verifier.py.
                "reward_mode": "graded",
            },
            # Difficulty + coverage metadata so consumers can slice train/eval
            # by hardness and judge regression-guard strength (p2p_count == 0
            # means no P2P regression guard — weaker eval; see UTBoost).
            "reward_calibration": {
                "f2p_count": len(fail_to_pass),
                "p2p_count": len(pass_to_pass),
                "source_files": len(_files_in_patch(patch)),
                "loc_changed": _diff_loc_changed(patch),
                "difficulty": _difficulty_bucket(len(fail_to_pass), _diff_loc_changed(patch)),
            },
        }

        return HarborTask(
            name=task_id,
            org=self.input.output.org,
            description=pr.title or task_id,
            instruction=_build_instruction(
                pr,
                owner,
                name,
                token=getattr(self, "_token", None),
                provider=getattr(self, "_provider", github),
            ),
            oracle_diff=patch,
            repo2env=repo2env,
            difficulty="medium",
            category="bugfix",
            keywords=[name, "pr_runtime"],
            environment_dockerfile=dockerfile,
            test_script=eval_script,
            # Ship verifier.py + f2p.json + p2p.json as plain, inspectable task
            # artifacts that test.sh reads from /tests (no base64 in test.sh).
            # The egress guard blackholes the hosts that serve this PR's merged
            # diff so the agent cannot fetch the gold patch at run time.
            aux_files={
                **(_runtime_aux_files(fail_to_pass, pass_to_pass) if fail_to_pass else {}),
                "environment/docker-compose.yaml": egress_guard_compose(
                    fix_source_hosts(self.input.repo.source_kind)
                ),
            },
        )
