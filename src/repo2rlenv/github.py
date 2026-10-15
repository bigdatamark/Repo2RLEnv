# [MODULE DOC] Thin GitHub API client that shells out to the `gh` CLI (subprocess) for every call.
# Provides PR listing, PR/commit diff fetch, issue/file/language/parent-commit lookup, and the
# PullRequestSummary record shared with gitlab.py. Auth is a token passed as GH_TOKEN env var to gh,
# or gh's own login. Requires `gh` on PATH; no urllib fallback exists despite the docstring wording.
# Logging via the module logger.
"""Thin GitHub client built on the `gh` CLI for auth simplicity.

We deliberately shell out to `gh` rather than depend on PyGithub:
  - `gh auth token` already gives us auth resolution for free
  - `gh api graphql` is easier than maintaining REST pagination logic
  - one less Python dep

If `gh` is not installed, we fall back to plain `curl`-style requests via
`urllib`. For v0.1 we only support the `gh`-installed path.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)


class GitHubError(RuntimeError):
    # [CLASS DOC] Raised when the gh CLI is missing or a gh invocation exits non-zero. Subclass of
    # RuntimeError; no extra fields.
    pass


@dataclass(slots=True)
class PullRequestSummary:
    # [CLASS DOC] Slotted dataclass describing one merged PR (or GitLab MR). Fields: number, title,
    # body, state, merged_at (ISO string or None), base_ref (branch name), base_sha (base commit),
    # head_sha, is_draft, url, changed_files (list of paths). Produced by list_merged_prs here and
    # in gitlab.py.
    number: int
    title: str
    body: str
    state: str
    merged_at: str | None
    base_ref: str
    base_sha: str
    head_sha: str
    is_draft: bool
    url: str
    changed_files: list[str]


def _run_gh(args: list[str], token: str | None = None) -> str:
    # [FUNCTION DOC] HELPER: Run `gh <args>` and return its stdout text.
    # Inputs: args (gh argument list); token (optional; when set, the subprocess env is os.environ
    # plus GH_TOKEN=token). Classifies the call as read-only when it is `gh api` with no
    # -X/-f/-F/--method/ --raw-field/--field/--input flags, or `gh pr list|view|diff`; read-only
    # calls get up to 3 attempts on subprocess timeout (60 s each) with exponential backoff sleeps
    # of 0.5 s and 1 s; mutating calls get 1.
    # Raises GitHubError if gh is not on PATH or exits non-zero (message includes stderr); re-raises
    # subprocess.TimeoutExpired after the final attempt. Side effects: subprocess, network.
    if not shutil.which("gh"):
        raise GitHubError("gh CLI not found on PATH; install it or use a different auth path")
    env = None
    if token:
        import os

        env = {**os.environ, "GH_TOKEN": token}
    read_only = bool(args) and (
        (
            args[0] == "api"
            and not any(
                arg.startswith(("-X", "-f", "-F"))
                or arg.split("=", 1)[0] in {"--method", "--raw-field", "--field", "--input"}
                for arg in args[1:]
            )
        )
        or args[:2] in (["pr", "list"], ["pr", "view"], ["pr", "diff"])
    )
    for attempt in range(3 if read_only else 1):
        try:
            proc = subprocess.run(
                ["gh", *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60,
                env=env,
                check=False,
            )
            break
        except subprocess.TimeoutExpired:
            if not read_only or attempt == 2:
                raise
            time.sleep(0.5 * 2**attempt)
    if proc.returncode != 0:
        raise GitHubError(f"gh {' '.join(args)!r} failed: {proc.stderr.strip()}")
    return proc.stdout


def _fetch_base_sha(owner: str, name: str, number: int, *, token: str | None = None) -> str | None:
    # [FUNCTION DOC] HELPER: Fetch a PR's base commit SHA via `gh api
    # repos/{owner}/{name}/pulls/{n}` with jq `.base.sha`. Returns the stripped SHA, or None (with a
    # logged warning) when the gh call fails or returns empty. Never raises GitHubError.
    """
    Return a PR's base-branch commit SHA via the REST pulls endpoint, or None on failure.
    """
    try:
        raw = _run_gh(
            ["api", f"repos/{owner}/{name}/pulls/{number}", "--jq", ".base.sha"],
            token=token,
        )
    except GitHubError as exc:
        logger.warning("PR #%d: base_sha fetch failed, dropping: %s", number, exc)
        return None
    sha = raw.strip()
    if not sha:
        logger.warning("PR #%d: base_sha fetch returned empty, dropping", number)
        return None
    return sha


def list_merged_prs(
    owner: str,
    name: str,
    *,
    limit: int = 50,
    since: date | None = None,
    until: date | None = None,
    skip_drafts: bool = True,
    token: str | None = None,
) -> list[PullRequestSummary]:
    # [FUNCTION DOC] List merged PRs newest-first via `gh pr list --state merged --json ...`.
    # Inputs: owner, name; limit (max summaries returned, default 50; over-fetches min(limit*3,
    # 1000) rows); since/until (optional dates, compared against the first 10 chars of mergedAt,
    # inclusive); skip_drafts (default True); token. Each kept row costs an extra API call for
    # base_sha (_fetch_base_sha); rows whose base_sha cannot be fetched are dropped. changed_files
    # comes from the `files` JSON field. Output: list[PullRequestSummary], at most limit long.
    # Raises GitHubError if the listing call fails; json errors propagate.
    """List recently merged PRs ordered newest-first.

    Uses `gh pr list` (REST under the hood). Filters by date client-side.
    """
    args = [
        "pr",
        "list",
        "--repo",
        f"{owner}/{name}",
        "--state",
        "merged",
        "--limit",
        str(min(limit * 3, 1000)),  # over-fetch to allow client-side filtering
        "--json",
        "number,title,body,state,mergedAt,baseRefName,headRefOid,isDraft,url,files",
    ]
    raw = _run_gh(args, token=token)
    rows = json.loads(raw)

    summaries: list[PullRequestSummary] = []
    for r in rows:
        if skip_drafts and r.get("isDraft"):
            continue
        merged_at = r.get("mergedAt")
        if since and merged_at and merged_at[:10] < since.isoformat():
            continue
        if until and merged_at and merged_at[:10] > until.isoformat():
            continue
        base_sha = _fetch_base_sha(owner, name, r["number"], token=token)
        if base_sha is None:
            continue
        files = [f["path"] for f in (r.get("files") or [])]
        summaries.append(
            PullRequestSummary(
                number=r["number"],
                title=r["title"] or "",
                body=r.get("body") or "",
                state=r["state"],
                merged_at=merged_at,
                base_ref=r.get("baseRefName") or "",
                base_sha=base_sha,
                head_sha=r.get("headRefOid") or "",
                is_draft=bool(r.get("isDraft")),
                url=r["url"],
                changed_files=files,
            )
        )
        if len(summaries) >= limit:
            break
    return summaries


def fetch_pr_diff(owner: str, name: str, number: int, *, token: str | None = None) -> str:
    # [FUNCTION DOC] Return a PR's unified diff text via `gh pr diff <n> --repo owner/name`.
    # Raises GitHubError on failure. Network via subprocess.
    """Return the unified diff for a PR via `gh pr diff`."""
    return _run_gh(
        ["pr", "diff", str(number), "--repo", f"{owner}/{name}"],
        token=token,
    )


def fetch_issue(
    owner: str, name: str, number: int, *, token: str | None = None
) -> tuple[str, str] | None:
    # [FUNCTION DOC] Return (title, body) of an issue via `gh issue view --json title,body`, both
    # stripped. Used to source problem statements from the bug report rather than the PR body.
    # Returns None when the gh call fails, the output is not JSON, or both title and body are empty.
    # Issue numbers that are actually PRs are accepted as-is.
    """Return (title, body) for an issue, or None if it can't be fetched.

    Used by `pr_runtime` to source the problem statement from the linked
    issue (the bug *report*) rather than the PR body (the *fix* description,
    which routinely leaks the solution — commit SHAs, the approach, even the
    grading test names). Mirrors SWE-bench, which builds problem statements
    from issue text.

    `gh issue view` also resolves issue numbers that are actually PRs on the
    same repo; we tolerate that and just return whatever title/body comes
    back. Returns None on any error (issue deleted, cross-repo ref, etc.) so
    the caller can fall back to the PR body.
    """
    import json as _json

    try:
        raw = _run_gh(
            ["issue", "view", str(number), "--repo", f"{owner}/{name}", "--json", "title,body"],
            token=token,
        )
        data = _json.loads(raw)
    except (GitHubError, _json.JSONDecodeError):
        return None
    title = (data.get("title") or "").strip()
    body = (data.get("body") or "").strip()
    if not title and not body:
        return None
    return title, body


def get_primary_language(owner: str, name: str, *, token: str | None = None) -> str | None:
    # [FUNCTION DOC] Return GitHub Linguist's primary language for the repo (e.g. "Python") via `gh
    # api repos/{owner}/{name} --jq .language`. Returns None on GitHubError, empty output or the
    # literal "null"; unwraps a JSON-quoted string if present, else returns the raw text.
    """Return GitHub's primary language string for a repo, or None on failure.

    Used by the pipeline-language compatibility pre-flight check so we can
    fail fast (before bootstrap) if a Python-only pipeline is pointed at a
    Go / Rust / etc. repo. The result is GitHub Linguist's classification
    (e.g. "Python", "Go", "TypeScript"); use
    `bootstrap.language.language_from_github_name` to map it to LanguageHint.
    """
    import json as _json

    try:
        raw = _run_gh(
            ["api", f"repos/{owner}/{name}", "--jq", ".language"],
            token=token,
        ).strip()
    except GitHubError:
        return None
    if not raw or raw == "null":
        return None
    # `gh api --jq` strips quotes, but unwrap if present
    try:
        return _json.loads(raw) if raw.startswith('"') else raw
    except _json.JSONDecodeError:
        return raw


def fetch_commit_diff(owner: str, name: str, sha: str, *, token: str | None = None) -> str:
    # [FUNCTION DOC] Return the unified diff of one commit via `gh api repos/.../commits/{sha}` with
    # the `application/vnd.github.v3.diff` Accept header (same shape as `git show --format= <sha>`).
    # Raises GitHubError on failure.
    """Return the unified diff for a single commit via `gh api`.

    Hits `GET /repos/{owner}/{repo}/commits/{sha}` with the `diff` media
    type — same shape as `git show --format= <sha>` output.
    """
    return _run_gh(
        [
            "api",
            f"repos/{owner}/{name}/commits/{sha}",
            "-H",
            "Accept: application/vnd.github.v3.diff",
        ],
        token=token,
    )


def fetch_commit_parent(owner: str, name: str, sha: str, *, token: str | None = None) -> str:
    # [FUNCTION DOC] Return the first parent SHA of a commit from `gh api repos/.../commits/{sha}`.
    # Returns "" for a root commit or on any GitHubError / JSON decode error.
    """Return the first parent SHA of `sha` via `gh api`.

    Returns "" if the commit has no parents (root commit) or on any error.
    """
    import json as _json

    try:
        raw = _run_gh(
            ["api", f"repos/{owner}/{name}/commits/{sha}"],
            token=token,
        )
        data = _json.loads(raw)
        parents = data.get("parents", []) or []
        if not parents:
            return ""
        return parents[0].get("sha", "") or ""
    except (GitHubError, _json.JSONDecodeError):
        return ""


def fetch_file_at_ref(
    owner: str, name: str, path: str, ref: str, *, token: str | None = None
) -> str | None:
    # [FUNCTION DOC] Return a file's raw text at a git ref via the contents API
    # (`repos/.../contents/{path}?ref={ref}`, Accept `application/vnd.github.raw`). Returns None on
    # GitHubError. path and ref are interpolated unescaped into the URL.
    """Return a file's raw text content at a given ref, or None on failure.

    Hits `GET /repos/{owner}/{repo}/contents/{path}?ref={ref}` with the raw
    media type. Used to give an LLM the full pre-fix source when synthesizing
    a regression test (the diff alone lacks imports + surrounding code).
    """
    try:
        return _run_gh(
            [
                "api",
                f"repos/{owner}/{name}/contents/{path}?ref={ref}",
                "-H",
                "Accept: application/vnd.github.raw",
            ],
            token=token,
        )
    except GitHubError:
        return None
