# [MODULE DOC] Thin subprocess wrappers around the local git CLI used by commit_runtime to walk a
# clone's history: list commits with parsed metadata, show a commit's diff and its changed files.
# Uses the git CLI rather than a Python git library.
"""Thin subprocess wrappers around the local `git` CLI.

Used by `commit_runtime` to walk a local clone's history. Bootstrap clones
at `--depth=1`; commit_runtime needs deeper history (`--depth=200` by
default) so `git log` can return real candidates.

We deliberately don't use a Python git library (GitPython, dulwich) — the
`git` CLI is the source of truth, stable across versions, and matches what
the bootstrap container has access to.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


class GitError(RuntimeError):
    # [CLASS DOC] RuntimeError raised when a git subprocess exits non-zero.
    """Raised when a `git` subprocess returns non-zero."""


@dataclass(slots=True)
class CommitInfo:
    # [CLASS DOC] Slotted dataclass of one commit's metadata: sha, parent_sha (first parent or ""),
    # parents (all parent SHAs), author_name, author_email, authored_at (strict ISO-8601), subject
    # and body.
    """One commit's metadata, parsed from `git log --format=...`."""

    sha: str  # full 40-char SHA
    parent_sha: str  # first parent (empty string for root commits)
    parents: list[str]  # full list (>1 means merge commit)
    author_name: str
    author_email: str
    authored_at: str  # ISO8601 (e.g. "2026-03-12T08:15:22Z")
    subject: str
    body: str

    @property
    def is_merge(self) -> bool:
        # [FUNCTION DOC] Property: True when the commit has more than one parent.
        return len(self.parents) > 1

    @property
    def message(self) -> str:
        # [FUNCTION DOC] Property: subject alone if body is empty, else subject, blank line, body.
        """Subject + blank line + body, like `git show -s`."""
        if not self.body:
            return self.subject
        return f"{self.subject}\n\n{self.body}"


# Field delimiter unlikely to appear in commit data
_FIELD_SEP = "\x1f"  # ASCII unit separator
_RECORD_SEP = "\x1e"  # ASCII record separator

# git --format placeholders — see `git help log` § PRETTY FORMATS
_LOG_FORMAT = (
    _FIELD_SEP.join(
        [
            "%H",  # full SHA
            "%P",  # parent SHAs (space-separated; first is %p)
            "%an",  # author name
            "%ae",  # author email
            "%aI",  # author date, ISO8601 strict
            "%s",  # subject
            "%b",  # body
        ]
    )
    + _RECORD_SEP
)


def _run_git(args: list[str], cwd: Path, *, timeout: int = 60) -> str:
    # [FUNCTION DOC] HELPER: Runs git with args in cwd (timeout default 60s) and returns stdout.
    # Raises GitError with the command, exit code and first 400 chars of stderr on non-zero exit;
    # TimeoutExpired on timeout.
    """Run `git ...` and return stdout. Raises GitError on non-zero exit."""
    r = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )
    if r.returncode != 0:
        raise GitError(
            f"git {' '.join(args)!r} failed (exit {r.returncode}): {r.stderr.strip()[:400]}"
        )
    return r.stdout


def list_commits(
    clone_dir: Path,
    *,
    since: date | None = None,
    until: date | None = None,
    limit: int = 50,
    branch: str = "HEAD",
) -> list[CommitInfo]:
    # [FUNCTION DOC] Returns up to limit (default 50) commits reachable from branch (default HEAD),
    # newest first, optionally filtered by since/until dates, using git log with a custom separator
    # format (120s timeout) parsed by _parse_log_output. Raises GitError on git failure.
    # KNOWN ISSUE: the docstring says a non-existent branch yields an empty list, but git log on an
    # unknown revision exits non-zero, so GitError is raised instead.
    """Return commits on `branch` between `since` and `until`, newest first.

    Filters happen at git's layer (`--since`, `--until`, `-n <limit>`) so
    we don't have to walk the full history in Python.

    Empty list if the range produces no commits or the branch doesn't exist.
    """
    args = [
        "log",
        f"--max-count={limit}",
        f"--format={_LOG_FORMAT}",
        "--no-decorate",
    ]
    if since is not None:
        args.append(f"--since={since.isoformat()}")
    if until is not None:
        args.append(f"--until={until.isoformat()}")
    args.append(branch)

    raw = _run_git(args, clone_dir, timeout=120)
    return [c for c in _parse_log_output(raw)]


def _parse_log_output(raw: str) -> list[CommitInfo]:
    # [FUNCTION DOC] HELPER: Splits raw git log output on the record separator (0x1e) and fields on
    # the unit separator (0x1f) into CommitInfo objects, skipping (debug-logging) records with fewer
    # than 7 fields; parents split on whitespace and body stripped.
    """Parse the `--format=%H\x1f%P\x1f...\x1e` stream into CommitInfo list."""
    out: list[CommitInfo] = []
    # Records are RS-separated; the last record may have a trailing newline
    records = [r for r in raw.split(_RECORD_SEP) if r.strip()]
    for rec in records:
        fields = rec.lstrip("\n").split(_FIELD_SEP)
        if len(fields) < 7:
            logger.debug("skipping malformed git log record: %r", rec[:100])
            continue
        sha, parents_str, author_name, author_email, authored_at, subject, body = (
            fields[0],
            fields[1],
            fields[2],
            fields[3],
            fields[4],
            fields[5],
            fields[6],
        )
        parents = parents_str.split() if parents_str else []
        out.append(
            CommitInfo(
                sha=sha,
                parent_sha=parents[0] if parents else "",
                parents=parents,
                author_name=author_name,
                author_email=author_email,
                authored_at=authored_at,
                subject=subject,
                body=body.strip(),
            )
        )
    return out


def show_diff(clone_dir: Path, commit_sha: str) -> str:
    # [FUNCTION DOC] Returns the unified diff introduced by commit_sha (git show --format= --patch
    # --no-color, 60s), without the commit header. Raises GitError.
    """Return the unified diff introduced by `commit_sha` (no commit message header).

    Uses `git show --format= --patch` which suppresses the commit-info
    block at the top; we get the raw `diff --git a/X b/Y` sequence the
    same shape as `gh pr diff` output, so the existing
    `split_patch_and_test_patch()` parser works unchanged.
    """
    return _run_git(
        ["show", "--format=", "--patch", "--no-color", commit_sha],
        clone_dir,
        timeout=60,
    )


def changed_files(clone_dir: Path, commit_sha: str) -> list[str]:
    # [FUNCTION DOC] Returns non-empty path lines from git show --name-only for commit_sha (30s).
    # Raises GitError.
    """Return the list of paths touched by `commit_sha` (relative to repo root)."""
    raw = _run_git(
        ["show", "--no-color", "--format=", "--name-only", commit_sha],
        clone_dir,
        timeout=30,
    )
    return [line for line in raw.splitlines() if line.strip()]
