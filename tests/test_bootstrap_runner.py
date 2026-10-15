# [MODULE DOC] Unit tests for private helpers in bootstrap/runner.py: token scrubbing, shallow
# clone
# at a ref (HEAD / branch / SHA fallback), RepoDigests resolution, missing user_dockerfile
# error, Dockerfile reconstruction from agent turns, and removal of clone credentials from
# .git/config. subprocess.run and runner._run are mocked except in the credential-scrub test,
# which runs real 'git' in tmp_path.
"""Runner-level fixes from PR #2 review (codex P1/P2)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from repo2rlenv.bootstrap import runner
from repo2rlenv.bootstrap.runner import (
    BootstrapError,
    _resolve_repo_digest,
    _scrub_token,
    _shallow_clone_at_ref,
)


def test_scrub_token_replaces_secret():
    # [FUNCTION DOC] _scrub_token must remove the token from the message and substitute '***'.
    # Inputs: none. Output: none. Raises: AssertionError on failure.
    # Deliberately do NOT use the canonical "user:pass@host" Basic Auth shape here:
    # GitGuardian's secret-pattern detector matches that exact form even when the
    # value is obviously a fake placeholder, which blocks CI for no real reason.
    # The function we're testing is content-agnostic — it does string replace.
    fake_token = "PLACEHOLDER_TEST_TOKEN_VALUE"
    msg = f"fatal: git operation failed (token redacted: {fake_token})"
    assert fake_token not in _scrub_token(msg, fake_token)
    assert "***" in _scrub_token(msg, fake_token)


def test_scrub_token_passthrough_when_no_token():
    # [FUNCTION DOC] With token=None, _scrub_token returns the message unchanged. Inputs:
    # none. Output: none. Raises: AssertionError on failure.
    msg = "fatal: nothing"
    assert _scrub_token(msg, None) == msg


def test_shallow_clone_head_uses_plain_clone(tmp_path: Path):
    # [FUNCTION DOC] ref='HEAD' must clone without '--branch'. Inputs: tmp_path fixture.
    # Patches
    # subprocess.run to succeed and inspects the first call's argv. Raises AssertionError on
    # failure.
    """ref='HEAD' must NOT pass --branch (it'd be interpreted as a branch named 'HEAD')."""
    with mock.patch("subprocess.run") as run:
        run.return_value = mock.Mock(returncode=0, stderr="", stdout="")
        _shallow_clone_at_ref(
            "https://github.com/owner/name",
            "HEAD",
            None,
            tmp_path / "out",
        )
        args = run.call_args_list[0].args[0]
        assert "--branch" not in args


def test_shallow_clone_branch_tries_clone_branch_first(tmp_path: Path):
    # [FUNCTION DOC] A branch name ref must be passed as '--branch <ref>' in the first clone
    # call. Inputs:
    # tmp_path. subprocess.run patched. Raises AssertionError on failure.
    with mock.patch("subprocess.run") as run:
        run.return_value = mock.Mock(returncode=0, stderr="", stdout="")
        _shallow_clone_at_ref(
            "https://github.com/owner/name",
            "release-1.0",
            None,
            tmp_path / "out",
        )
        args = run.call_args_list[0].args[0]
        assert "--branch" in args
        idx = args.index("--branch")
        assert args[idx + 1] == "release-1.0"


def test_shallow_clone_falls_back_to_fetch_on_sha(tmp_path: Path):
    # [FUNCTION DOC] When 'clone --branch <sha>' fails (rc 128), the helper must fall back to
    # a sequence
    # including 'fetch' and 'checkout'. Records every subprocess call. Inputs: tmp_path.
    # Raises AssertionError on failure.
    """When --branch <sha> fails, fall back to clone-no-checkout + fetch + checkout."""
    call_sequence = []

    def fake_run(cmd, **kwargs):
        # [FUNCTION DOC] HELPER: subprocess.run stub; records cmd, returns a Mock with
        # returncode 128 for commands
        # containing '--branch', else returncode 0.
        call_sequence.append(cmd)
        # First call (clone --branch) fails with 128 (typical for SHA)
        if "--branch" in cmd:
            return mock.Mock(returncode=128, stderr="not found", stdout="")
        # Subsequent calls succeed
        return mock.Mock(returncode=0, stderr="", stdout="")

    with mock.patch("subprocess.run", side_effect=fake_run):
        _shallow_clone_at_ref(
            "https://github.com/owner/name",
            "a1b2c3d4e5f6",
            None,
            tmp_path / "out",
        )
    # Should have: clone --branch (failed), clone --no-checkout, fetch, checkout
    all_args = [arg for cmd in call_sequence for arg in cmd]
    assert "--branch" in all_args, "should have attempted clone --branch first"
    assert "fetch" in all_args, "fallback should `git fetch origin <ref>`"
    assert "checkout" in all_args, "fallback should `git checkout <ref>`"


def test_resolve_repo_digest_parses_inspect_output():
    # [FUNCTION DOC] Given inspect stdout with a JSON list of RepoDigests,
    # _resolve_repo_digest returns the first entry. Inputs: none. Output: none. Raises:
    # AssertionError on failure.
    """Should return the first RepoDigests entry post-push."""
    inspect_out = '["ghcr.io/owner/foo@sha256:abc123"]'
    with mock.patch.object(runner, "_run") as run:
        run.return_value = mock.Mock(ok=True, stdout=inspect_out)
        digest = _resolve_repo_digest("ghcr.io/owner/foo:abc")
    assert digest == "ghcr.io/owner/foo@sha256:abc123"


def test_resolve_repo_digest_returns_none_when_unpushed():
    # [FUNCTION DOC] An empty RepoDigests list ('[]') yields None. Inputs: none. Output: none.
    # Raises: AssertionError on failure.
    """No RepoDigests yet → returns None so caller keeps the local Id."""
    with mock.patch.object(runner, "_run") as run:
        run.return_value = mock.Mock(ok=True, stdout="[]")
        assert _resolve_repo_digest("local/foo:bar") is None


def test_resolve_repo_digest_returns_none_on_inspect_fail():
    # [FUNCTION DOC] A failed inspect (ok=False) yields None. Inputs: none. Output: none.
    # Raises: AssertionError on failure.
    with mock.patch.object(runner, "_run") as run:
        run.return_value = mock.Mock(ok=False, stdout="")
        assert _resolve_repo_digest("missing:tag") is None


def test_user_dockerfile_missing_path_raises(tmp_path: Path):
    # [FUNCTION DOC] ensure_bootstrap must raise BootstrapError('user_dockerfile not found')
    # when
    # BootstrapSpec.user_dockerfile points at a nonexistent file. Docker availability, clone
    # and
    # HEAD resolution are patched out. Inputs: tmp_path. Raises AssertionError on failure.
    """Pointing user_dockerfile at a non-existent file is a clear error."""
    from repo2rlenv.spec.input import AuthSpec, BootstrapSpec, LLMSpec, RepoSpec

    repo = RepoSpec(url="owner/name", access="public")
    spec = BootstrapSpec(user_dockerfile=tmp_path / "does-not-exist.Dockerfile")
    llm = LLMSpec(provider="anthropic", model="claude-sonnet-4-6")

    # Stub out the bits that would otherwise fail before we hit the dockerfile check
    with (
        mock.patch.object(runner, "is_docker_available", return_value=True),
        mock.patch.object(runner, "_shallow_clone_at_ref"),
        mock.patch.object(runner, "_resolve_head_sha", return_value="a" * 40),
    ):
        with pytest.raises(BootstrapError, match="user_dockerfile not found"):
            runner.ensure_bootstrap(repo, spec, llm, AuthSpec())


def test_reconstructed_dockerfile_is_rebuildable():
    # [FUNCTION DOC] _reconstruct_dockerfile must emit FROM <base>, then WORKDIR /workspace
    # and COPY . /workspace
    # before the first RUN, and only turn BASH actions into RUN lines (READ_FILE is skipped).
    # Inputs: none. Output: none. Raises: AssertionError on failure.
    """Reconstructed Dockerfile must COPY the repo before agent RUNs.

    The agent runs commands inside /workspace where the repo already exists;
    `pip install -e .` etc. assume repo files in CWD. A Dockerfile that only
    replays RUN lines without COPY would fail immediately on rebuild.
    """
    from repo2rlenv.bootstrap.runner import _reconstruct_dockerfile

    class FakeAction:
        # [CLASS DOC] HELPER: Minimal agent action stand-in with name and input
        # attributes.
        def __init__(self, name, input):
            # [FUNCTION DOC] HELPER: Stores name (action kind, e.g. 'BASH') and input
            # (command text). Output: None.
            self.name = name
            self.input = input

    class FakeTurn:
        # [CLASS DOC] HELPER: Minimal agent turn stand-in holding an action.
        def __init__(self, action):
            # [FUNCTION DOC] HELPER: Stores the given action on self.action. Output:
            # None.
            self.action = action

    turns = [
        FakeTurn(FakeAction("BASH", "pip install -e .")),
        FakeTurn(FakeAction("BASH", "pytest --collect-only")),
        FakeTurn(FakeAction("READ_FILE", "/workspace/setup.py")),  # non-BASH skipped
    ]
    dockerfile = _reconstruct_dockerfile("python:3.12-slim", turns)
    assert "FROM python:3.12-slim" in dockerfile
    workdir_idx = dockerfile.index("WORKDIR /workspace")
    copy_idx = dockerfile.index("COPY . /workspace")
    first_run_idx = dockerfile.index("RUN pip install")
    assert workdir_idx < first_run_idx and copy_idx < first_run_idx, (
        "WORKDIR + COPY must precede any RUN line"
    )
    # Non-BASH actions should not become RUN lines
    assert "READ_FILE" not in dockerfile


def test_scrub_clone_credentials_strips_token(tmp_path: Path):
    # [FUNCTION DOC] Creates a real git repo in tmp_path with a token-bearing origin URL, then
    # calls
    # _scrub_clone_credentials and asserts .git/config no longer contains the token and does
    # contain the bare URL. Side effects: runs 'git init' / 'git remote add' subprocesses.
    # Inputs: tmp_path. Raises AssertionError or CalledProcessError on failure.
    """After cloning with a token, .git/config should not retain the token."""
    import subprocess as sp

    from repo2rlenv.bootstrap.runner import _scrub_clone_credentials

    fake_token = "PLACEHOLDER_TEST_TOKEN_VALUE"
    bare_url = "https://github.com/owner/name"
    auth_url = f"https://x-access-token:{fake_token}@github.com/owner/name"

    clone_dir = tmp_path / "repo"
    clone_dir.mkdir()
    sp.run(["git", "init", "-q"], cwd=clone_dir, check=True)
    sp.run(["git", "remote", "add", "origin", auth_url], cwd=clone_dir, check=True)

    config = (clone_dir / ".git" / "config").read_text()
    assert fake_token in config

    _scrub_clone_credentials(clone_dir, bare_url)

    config = (clone_dir / ".git" / "config").read_text()
    assert fake_token not in config, "scrub must remove the embedded token"
    assert bare_url in config, "remote URL should now be the clean form"
