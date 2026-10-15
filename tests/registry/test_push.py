# [MODULE DOC] Tests for registry.push, the docker CLI wrapper: manifest_exists,
# _resolve_repo_digest (host filtering, Docker Hub bare digests, skipping local/ digests) and
# push_image (missing docker, missing local image, idempotent skip, happy path, failure stderr,
# skip_if_exists=False). subprocess.run is replaced with a scripted prefix-matching fake; no real
# docker calls.
"""Tests for `registry.push` — docker push subprocess wrapper."""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

import pytest

from repo2rlenv.registry import push as push_mod
from repo2rlenv.registry.push import (
    PushError,
    _resolve_repo_digest,
    manifest_exists,
    push_image,
)


def _stub_run(
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[tuple[str, ...], tuple[int, str, str]],
) -> list[list[str]]:
    # [FUNCTION DOC] HELPER: Monkeypatches subprocess.run with a fake that records each argv and
    # returns a MagicMock(returncode, stdout, stderr) from responses, choosing the longest key that
    # is a prefix of argv; unmatched commands return returncode 127. Returns the list of recorded
    # argv lists.
    """Record subprocess.run invocations and return scripted responses.

    `responses` maps (docker, subcommand, ...) prefixes to (returncode, stdout, stderr).
    The most specific matching prefix wins.
    """
    called: list[list[str]] = []

    def fake(args: list[str], **kwargs: Any) -> mock.MagicMock:
        # [FUNCTION DOC] HELPER: The fake subprocess.run: appends args to called, finds the longest
        # matching prefix in responses, returns the scripted MagicMock or a 127 'no script' result.
        called.append(args)
        # Find the longest prefix match
        match: tuple[int, str, str] | None = None
        match_len = -1
        for key, value in responses.items():
            if len(key) > match_len and tuple(args[: len(key)]) == key:
                match = value
                match_len = len(key)
        if match is None:
            return mock.MagicMock(returncode=127, stdout="", stderr=f"no script: {args}")
        rc, stdout, stderr = match
        return mock.MagicMock(returncode=rc, stdout=stdout, stderr=stderr)

    monkeypatch.setattr("subprocess.run", fake)
    return called


class TestManifestExists:
    # [CLASS DOC] Tests for manifest_exists.
    def test_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] docker manifest inspect exit 0 -> True.
        monkeypatch.setattr(push_mod, "_docker_available", lambda: True)
        _stub_run(monkeypatch, {("docker", "manifest", "inspect"): (0, "{}", "")})
        assert manifest_exists("ghcr.io/foo/bar:tag") is True

    def test_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Non-zero exit -> False.
        monkeypatch.setattr(push_mod, "_docker_available", lambda: True)
        _stub_run(monkeypatch, {("docker", "manifest", "inspect"): (1, "", "manifest unknown")})
        assert manifest_exists("ghcr.io/foo/bar:tag") is False

    def test_no_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Docker unavailable -> False without running anything.
        monkeypatch.setattr(push_mod, "_docker_available", lambda: False)
        assert manifest_exists("ghcr.io/foo/bar:tag") is False


class TestResolveRepoDigest:
    # [CLASS DOC] Tests for _resolve_repo_digest over 'docker image inspect' RepoDigests.
    def test_filters_by_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Returns the digest matching the remote ref's host.
        digests = json.dumps(
            [
                "docker.io/foo/bar@sha256:111",
                "ghcr.io/foo/bar@sha256:222",
            ]
        )
        _stub_run(monkeypatch, {("docker", "image", "inspect"): (0, digests, "")})
        result = _resolve_repo_digest("ghcr.io/foo/bar:tag")
        assert result == "ghcr.io/foo/bar@sha256:222"

    def test_docker_hub_bare_digest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Bare Docker Hub digests get the index.docker.io host re-attached; local/
        # entries are skipped.
        """Docker Hub returns digests without the host prefix; we re-attach it."""
        digests = json.dumps(
            [
                "local/r2e-bootstrap/pallets__click@sha256:abc",  # local, must skip
                "myuser/r2e-bootstrap-pallets-click@sha256:cafe",  # bare Docker Hub
            ]
        )
        _stub_run(monkeypatch, {("docker", "image", "inspect"): (0, digests, "")})
        result = _resolve_repo_digest(
            "index.docker.io/myuser/r2e-bootstrap-pallets-click:fc6c7c47edd6"
        )
        # Result MUST carry the canonical host so docker pull resolves it
        assert result == "index.docker.io/myuser/r2e-bootstrap-pallets-click@sha256:cafe"

    def test_skips_local_digests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] A digest list with only local/ entries yields None.
        """`local/...@sha256:...` entries are NOT pullable — must never be returned."""
        digests = json.dumps(["local/r2e-bootstrap/pallets__click@sha256:abc"])
        _stub_run(monkeypatch, {("docker", "image", "inspect"): (0, digests, "")})
        # Even with no other candidate, we don't return a local-only digest
        assert _resolve_repo_digest("ghcr.io/foo/bar:tag") is None

    def test_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Empty RepoDigests -> None.
        _stub_run(monkeypatch, {("docker", "image", "inspect"): (0, "[]", "")})
        assert _resolve_repo_digest("ghcr.io/foo/bar:tag") is None

    def test_inspect_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Inspect failure -> None.
        _stub_run(monkeypatch, {("docker", "image", "inspect"): (1, "", "not found")})
        assert _resolve_repo_digest("ghcr.io/foo/bar:tag") is None


class TestPushImage:
    # [CLASS DOC] Tests for push_image.
    def test_no_docker_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Docker unavailable raises PushError ('not available').
        monkeypatch.setattr(push_mod, "_docker_available", lambda: False)
        with pytest.raises(PushError, match="not available"):
            push_image("local:tag", "ghcr.io/foo/bar:tag")

    def test_missing_local_image_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Failed local image inspect raises PushError ('not found').
        monkeypatch.setattr(push_mod, "_docker_available", lambda: True)
        _stub_run(monkeypatch, {("docker", "image", "inspect"): (1, "", "no such image")})
        with pytest.raises(PushError, match="not found"):
            push_image("local:tag", "ghcr.io/foo/bar:tag")

    def test_idempotent_skip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] An existing remote manifest skips tag/push and returns pushed False with
        # the resolved digest.
        """Manifest already at registry → skip the push step entirely."""
        monkeypatch.setattr(push_mod, "_docker_available", lambda: True)
        digest_json = json.dumps(["ghcr.io/foo/bar@sha256:cafe"])
        called = _stub_run(
            monkeypatch,
            {
                # image inspect → exists
                ("docker", "image", "inspect", "local:tag"): (0, "sha256:abc", ""),
                # manifest exists → 0
                ("docker", "manifest", "inspect"): (0, "{}", ""),
                # repo digest lookup
                ("docker", "image", "inspect", "ghcr.io/foo/bar:tag"): (0, digest_json, ""),
            },
        )
        result = push_image("local:tag", "ghcr.io/foo/bar:tag")
        assert result.pushed is False
        assert result.digest == "ghcr.io/foo/bar@sha256:cafe"
        # No `docker tag` or `docker push` should have been issued
        cmds = [tuple(c[:2]) for c in called]
        assert ("docker", "tag") not in cmds
        assert ("docker", "push") not in cmds

    def test_happy_push(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Missing remote manifest -> tag and push run, pushed True with the resolved
        # digest.
        monkeypatch.setattr(push_mod, "_docker_available", lambda: True)
        digest_json = json.dumps(["ghcr.io/foo/bar@sha256:beef"])
        called = _stub_run(
            monkeypatch,
            {
                ("docker", "image", "inspect", "local:tag"): (0, "sha256:abc", ""),
                ("docker", "manifest", "inspect"): (1, "", "no such manifest"),
                ("docker", "tag"): (0, "", ""),
                ("docker", "push"): (0, "pushed", ""),
                ("docker", "image", "inspect", "ghcr.io/foo/bar:tag"): (0, digest_json, ""),
            },
        )
        result = push_image("local:tag", "ghcr.io/foo/bar:tag")
        assert result.pushed is True
        assert result.digest == "ghcr.io/foo/bar@sha256:beef"
        # Verify the sequence: inspect → manifest inspect → tag → push → inspect
        cmds = [tuple(c[:2]) for c in called]
        assert cmds[0] == ("docker", "image")
        assert ("docker", "tag") in cmds
        assert ("docker", "push") in cmds

    def test_push_failure_surfaces_stderr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] A failing docker push raises PushError containing the stderr text.
        monkeypatch.setattr(push_mod, "_docker_available", lambda: True)
        _stub_run(
            monkeypatch,
            {
                ("docker", "image", "inspect", "local:tag"): (0, "sha256:abc", ""),
                ("docker", "manifest", "inspect"): (1, "", ""),
                ("docker", "tag"): (0, "", ""),
                ("docker", "push"): (1, "", "denied: permission_denied"),
            },
        )
        with pytest.raises(PushError, match="denied"):
            push_image("local:tag", "ghcr.io/foo/bar:tag")

    def test_skip_if_exists_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] skip_if_exists=False pushes even when the manifest exists.
        """When skip_if_exists=False, push proceeds even if manifest is present."""
        monkeypatch.setattr(push_mod, "_docker_available", lambda: True)
        digest_json = json.dumps(["ghcr.io/foo/bar@sha256:new"])
        called = _stub_run(
            monkeypatch,
            {
                ("docker", "image", "inspect", "local:tag"): (0, "sha256:abc", ""),
                ("docker", "manifest", "inspect"): (0, "{}", ""),  # exists
                ("docker", "tag"): (0, "", ""),
                ("docker", "push"): (0, "pushed", ""),
                ("docker", "image", "inspect", "ghcr.io/foo/bar:tag"): (0, digest_json, ""),
            },
        )
        result = push_image("local:tag", "ghcr.io/foo/bar:tag", skip_if_exists=False)
        assert result.pushed is True
        cmds = [tuple(c[:2]) for c in called]
        assert ("docker", "push") in cmds
