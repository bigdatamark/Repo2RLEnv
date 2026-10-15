# [MODULE DOC] Tests for registry.visibility: _parse_ghcr_package and ensure_ghcr_visibility (non-
# GHCR refs, missing gh CLI, user vs org API routing, API failure with manual URL). gh availability,
# gh user lookup and subprocess.run are monkeypatched; no network.
"""Tests for `registry.visibility` — GHCR package visibility flips."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from repo2rlenv.registry import visibility as vis_mod
from repo2rlenv.registry.visibility import (
    _parse_ghcr_package,
    ensure_ghcr_visibility,
)


class TestParseGhcrPackage:
    # [CLASS DOC] Tests for _parse_ghcr_package.
    def test_tag_ref(self) -> None:
        # [FUNCTION DOC] Tag ref -> (owner, package).
        assert _parse_ghcr_package("ghcr.io/huggingface/r2e-bootstrap:tag") == (
            "huggingface",
            "r2e-bootstrap",
        )

    def test_digest_ref(self) -> None:
        # [FUNCTION DOC] Digest ref -> (owner, package).
        assert _parse_ghcr_package("ghcr.io/u/n@sha256:abc") == ("u", "n")

    def test_not_ghcr_returns_none(self) -> None:
        # [FUNCTION DOC] Non-ghcr.io host -> None.
        assert _parse_ghcr_package("docker.io/u/n:tag") is None

    def test_malformed_returns_none(self) -> None:
        # [FUNCTION DOC] Single path segment -> None.
        assert _parse_ghcr_package("ghcr.io/just-one-segment:tag") is None


class TestEnsureGhcrVisibility:
    # [CLASS DOC] Tests for ensure_ghcr_visibility.
    def test_not_a_ghcr_ref(self) -> None:
        # [FUNCTION DOC] Non-GHCR ref -> failure with 'not a ghcr.io ref'.
        result = ensure_ghcr_visibility("docker.io/u/n:tag")
        assert result.success is False
        assert "not a ghcr.io ref" in (result.error or "")

    def test_no_gh_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Missing gh -> failure carrying a github.com manual URL.
        monkeypatch.setattr(vis_mod, "_gh_available", lambda: False)
        result = ensure_ghcr_visibility("ghcr.io/u/n:tag")
        assert result.success is False
        assert result.manual_url and "github.com" in result.manual_url

    def test_user_namespace_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Owner equal to the gh user hits the /user/packages endpoint with
        # visibility=public.
        monkeypatch.setattr(vis_mod, "_gh_available", lambda: True)
        monkeypatch.setattr(vis_mod, "_gh_user", lambda: "alice")

        captured: list[list[str]] = []

        def fake_run(args: list[str], **kwargs: Any) -> mock.MagicMock:
            # [FUNCTION DOC] HELPER: Records argv and returns success.
            captured.append(args)
            return mock.MagicMock(returncode=0, stdout="{}", stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)
        result = ensure_ghcr_visibility("ghcr.io/alice/myimg:tag", target="public")
        assert result.success
        # Should hit /user/... endpoint, not /orgs/...
        assert any("/user/packages/container/myimg" in arg for arg in captured[0])
        assert any("visibility=public" in arg for arg in captured[0])

    def test_org_namespace_routing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] A different owner routes to /orgs/<org>/packages.
        monkeypatch.setattr(vis_mod, "_gh_available", lambda: True)
        monkeypatch.setattr(vis_mod, "_gh_user", lambda: "alice")

        captured: list[list[str]] = []

        def fake_run(args: list[str], **kwargs: Any) -> mock.MagicMock:
            # [FUNCTION DOC] HELPER: Records argv and returns success.
            captured.append(args)
            return mock.MagicMock(returncode=0, stdout="{}", stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)
        result = ensure_ghcr_visibility("ghcr.io/bigorg/myimg:tag")
        assert result.success
        assert any("/orgs/bigorg/packages/container/myimg" in arg for arg in captured[0])

    def test_api_failure_includes_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] A failing gh api call yields failure whose error contains 'package not
        # found' plus a manual URL. KNOWN ISSUE: the 'parsed message or raw stderr' assertion has
        # two identical operands, so it only checks one condition.
        monkeypatch.setattr(vis_mod, "_gh_available", lambda: True)
        monkeypatch.setattr(vis_mod, "_gh_user", lambda: "alice")

        def fake_run(args: list[str], **kwargs: Any) -> mock.MagicMock:
            # [FUNCTION DOC] HELPER: Returns exit 1 with a JSON stderr message 'package not found'.
            return mock.MagicMock(
                returncode=1,
                stdout="",
                stderr='{"message": "package not found"}',
            )

        monkeypatch.setattr("subprocess.run", fake_run)
        result = ensure_ghcr_visibility("ghcr.io/bigorg/myimg:tag")
        assert result.success is False
        # Either the parsed message or the raw stderr — both fine
        assert "package not found" in (result.error or "") or "package not found" in (
            result.error or ""
        )
        assert result.manual_url and "github.com" in result.manual_url
