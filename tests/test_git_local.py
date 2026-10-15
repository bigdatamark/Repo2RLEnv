# [MODULE DOC] Tests for git_local.py (list_commits, show_diff, changed_files, CommitInfo,
# GitError) against real temporary git repos built with the git CLI. No network. Skipped when git
# is not installed.
"""`git_local` helpers — tested against a tmp in-tree git repo.

We create a tiny repo with 3 commits + a merge in a temp directory and
exercise list_commits / show_diff / changed_files against it. No network,
no subprocess mocking — uses the real `git` CLI which our caller does too.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from repo2rlenv.git_local import (
    CommitInfo,
    GitError,
    changed_files,
    list_commits,
    show_diff,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git CLI not available")


def _git(*args: str, cwd: Path) -> None:
    # [FUNCTION DOC] HELPER: Runs 'git <args>' in cwd with a fixed author/committer identity,
    # GIT_CONFIG_NOSYSTEM, HOME set to cwd (ignoring the user's gitconfig) and a minimal PATH.
    # Inputs: args (str varargs), cwd (Path, keyword). Output: None. Raises AssertionError with
    # stderr on nonzero exit. Side effects: subprocess.
    """Helper to run git with sensible defaults for tests."""
    env = {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": str(cwd),  # avoid user's ~/.gitconfig
        "PATH": "/usr/bin:/usr/local/bin:/bin",
    }
    r = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {r.stderr}")


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    # [FUNCTION DOC] Fixture: creates tmp_path/tiny on branch main with three commits (add foo.txt,
    # add bar.txt, modify foo.txt with a subject+body message). Input: tmp_path. Output: Path to
    # the repo.
    """Build a tiny repo with: init → 1st commit → 2nd commit → 3rd commit."""
    repo = tmp_path / "tiny"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)

    (repo / "foo.txt").write_text("hello\n")
    _git("add", "foo.txt", cwd=repo)
    _git("commit", "-q", "-m", "initial commit", cwd=repo)

    (repo / "bar.txt").write_text("hello bar\n")
    _git("add", "bar.txt", cwd=repo)
    _git("commit", "-q", "-m", "add bar with content", cwd=repo)

    (repo / "foo.txt").write_text("hello world\n")
    _git("add", "foo.txt", cwd=repo)
    _git(
        "commit",
        "-q",
        "-m",
        "fix: greet the world properly\n\nUpdates foo.txt to say world.",
        cwd=repo,
    )
    return repo


# -------------------------- list_commits --------------------------------------


def test_list_commits_returns_newest_first(tiny_repo: Path):
    # [FUNCTION DOC] Asserts list_commits returns 3 commits, newest (fix: greet) first and initial
    # commit last.
    commits = list_commits(tiny_repo, limit=10)
    assert len(commits) == 3
    # Most recent first (the "fix: greet" commit)
    assert commits[0].subject.startswith("fix: greet the world")
    # Oldest last
    assert commits[-1].subject == "initial commit"


def test_list_commits_respects_limit(tiny_repo: Path):
    # [FUNCTION DOC] Asserts limit=2 returns two commits.
    commits = list_commits(tiny_repo, limit=2)
    assert len(commits) == 2


def test_list_commits_parses_parent_sha(tiny_repo: Path):
    # [FUNCTION DOC] Asserts parent_sha links to the previous commit and the root has empty
    # parent_sha and parents.
    commits = list_commits(tiny_repo, limit=10)
    # 3rd commit's parent is the 2nd commit's SHA
    assert commits[0].parent_sha == commits[1].sha
    # First commit (root) has no parent
    assert commits[-1].parent_sha == ""
    assert commits[-1].parents == []


def test_list_commits_parses_subject_and_body(tiny_repo: Path):
    # [FUNCTION DOC] Asserts subject and body of the newest commit are split correctly.
    commits = list_commits(tiny_repo, limit=10)
    top = commits[0]
    assert top.subject == "fix: greet the world properly"
    assert top.body == "Updates foo.txt to say world."


def test_list_commits_parses_author(tiny_repo: Path):
    # [FUNCTION DOC] Asserts author name and email are parsed.
    commits = list_commits(tiny_repo, limit=10)
    assert commits[0].author_name == "Test"
    assert commits[0].author_email == "test@example.com"


def test_list_commits_authored_at_is_iso8601(tiny_repo: Path):
    # [FUNCTION DOC] Asserts authored_at begins with a 4-digit year followed by '-'.
    commits = list_commits(tiny_repo, limit=10)
    # ISO8601 strict: starts with year + "-" + month
    assert commits[0].authored_at[:4].isdigit()
    assert commits[0].authored_at[4] == "-"


def test_list_commits_is_merge_detection(tmp_path: Path):
    # [FUNCTION DOC] Builds a repo with a --no-ff merge of a feature branch and asserts the newest
    # commit is_merge with two parents. Side effects: git subprocesses in tmp_path.
    """Construct a merge commit + verify CommitInfo.is_merge."""
    repo = tmp_path / "merger"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / "a.txt").write_text("a\n")
    _git("add", "a.txt", cwd=repo)
    _git("commit", "-q", "-m", "main: initial", cwd=repo)
    _git("checkout", "-q", "-b", "feature", cwd=repo)
    (repo / "b.txt").write_text("b\n")
    _git("add", "b.txt", cwd=repo)
    _git("commit", "-q", "-m", "feature: add b", cwd=repo)
    _git("checkout", "-q", "main", cwd=repo)
    # Avoid fast-forward — force a merge commit
    _git("merge", "--no-ff", "-q", "-m", "Merge feature into main", "feature", cwd=repo)
    commits = list_commits(repo, limit=10)
    # The merge commit is newest
    merge = commits[0]
    assert merge.is_merge
    assert len(merge.parents) == 2


def test_list_commits_empty_when_outside_range(tiny_repo: Path):
    # [FUNCTION DOC] Asserts since=2030-01-01 returns no commits.
    from datetime import date

    commits = list_commits(tiny_repo, since=date(2030, 1, 1), limit=10)
    assert commits == []


# -------------------------- show_diff -----------------------------------------


def test_show_diff_returns_unified_diff(tiny_repo: Path):
    # [FUNCTION DOC] Asserts show_diff of the newest commit is a git diff for foo.txt without a
    # commit header.
    commits = list_commits(tiny_repo, limit=10)
    top_sha = commits[0].sha
    diff = show_diff(tiny_repo, top_sha)
    # `git show --format= --patch` should produce the diff hunk for foo.txt
    assert "diff --git" in diff
    assert "foo.txt" in diff
    assert "+hello world" in diff
    assert "-hello\n" in diff or "-hello" in diff
    # No commit-info header (we asked for --format=)
    assert not diff.startswith("commit ")


def test_show_diff_handles_initial_commit(tiny_repo: Path):
    # [FUNCTION DOC] Asserts show_diff of the root commit shows the added line.
    commits = list_commits(tiny_repo, limit=10)
    root_sha = commits[-1].sha
    # The initial commit has all-additions
    diff = show_diff(tiny_repo, root_sha)
    assert "+hello" in diff


# -------------------------- changed_files -------------------------------------


def test_changed_files_one_per_line(tiny_repo: Path):
    # [FUNCTION DOC] Asserts changed_files of the newest commit is ['foo.txt'].
    commits = list_commits(tiny_repo, limit=10)
    files = changed_files(tiny_repo, commits[0].sha)
    assert files == ["foo.txt"]


def test_changed_files_multi_file(tmp_path: Path):
    # [FUNCTION DOC] Asserts a commit adding a.txt and b.txt lists both files.
    """A commit that touches 2 files should list both."""
    repo = tmp_path / "multi"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / "a.txt").write_text("a\n")
    (repo / "b.txt").write_text("b\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "add a and b", cwd=repo)
    commits = list_commits(repo, limit=10)
    files = changed_files(repo, commits[0].sha)
    assert sorted(files) == ["a.txt", "b.txt"]


# -------------------------- errors --------------------------------------------


def test_show_diff_unknown_sha_raises(tiny_repo: Path):
    # [FUNCTION DOC] Asserts show_diff with an all-zero SHA raises GitError.
    with pytest.raises(GitError):
        show_diff(tiny_repo, "0" * 40)


def test_commit_info_message_property():
    # [FUNCTION DOC] Asserts CommitInfo.message joins subject and body with a blank line.
    """`.message` joins subject + body the way `git show -s` would."""
    ci = CommitInfo(
        sha="x",
        parent_sha="y",
        parents=["y"],
        author_name="A",
        author_email="a@b",
        authored_at="2026-01-01T00:00:00Z",
        subject="fix: thing",
        body="More detail.",
    )
    assert ci.message == "fix: thing\n\nMore detail."


def test_commit_info_message_no_body():
    # [FUNCTION DOC] Asserts CommitInfo.message is just the subject when body is empty.
    ci = CommitInfo(
        sha="x",
        parent_sha="",
        parents=[],
        author_name="A",
        author_email="a@b",
        authored_at="2026-01-01T00:00:00Z",
        subject="subject only",
        body="",
    )
    assert ci.message == "subject only"
