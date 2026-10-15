# [MODULE DOC] Tests for repo2rlenv.registry.auth: host classification into RegistryKind, discovery
# of logged-in registries from a Docker config.json (inline auth, credHelpers, credsStore, empty
# entries, Docker Hub v1 URL normalisation) and filtering out unknown registry kinds. Uses tmp_path
# config files only.
"""Tests for `registry.auth` — file-level Docker login discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repo2rlenv.registry.auth import (
    CredentialSource,
    RegistryAuth,
    RegistryKind,
    classify_host,
    discover_logged_in_registries,
    filter_known_registries,
)


class TestClassifyHost:
    @pytest.mark.parametrize(
        "host,expected",
        [
            ("ghcr.io", RegistryKind.GHCR),
            ("https://ghcr.io/", RegistryKind.GHCR),
            ("public.ecr.aws", RegistryKind.ECR_PUBLIC),
            ("123456789012.dkr.ecr.us-east-1.amazonaws.com", RegistryKind.ECR_PRIVATE),
            ("999.dkr.ecr.eu-west-2.amazonaws.com", RegistryKind.ECR_PRIVATE),
            ("myregistry.azurecr.io", RegistryKind.ACR),
            ("us-central1-docker.pkg.dev", RegistryKind.GCP_AR),
            ("asia-southeast1-docker.pkg.dev", RegistryKind.GCP_AR),
            ("index.docker.io", RegistryKind.DOCKER_HUB),
            ("registry-1.docker.io", RegistryKind.DOCKER_HUB),
            ("docker.io", RegistryKind.DOCKER_HUB),
            ("https://index.docker.io/v1/", RegistryKind.DOCKER_HUB),
            ("localhost:5000", RegistryKind.LOCAL),
            ("127.0.0.1:5000", RegistryKind.LOCAL),
            ("quay.io", RegistryKind.OTHER),
            ("", RegistryKind.OTHER),
        ],
    )
    # [CLASS DOC] Parametrized checks that classify_host maps hosts/URLs to the right RegistryKind.
    def test_classification(self, host: str, expected: RegistryKind) -> None:
        # [FUNCTION DOC] Asserts classify_host(host) is the expected RegistryKind for GHCR, ECR
        # public/private, ACR, GCP AR, Docker Hub variants, localhost/127.0.0.1, quay.io and the
        # empty string (OTHER).
        assert classify_host(host) is expected


class TestDiscover:
    # [CLASS DOC] Tests for discover_logged_in_registries reading a Docker config file.
    def test_missing_config(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A non-existent config path yields an empty list.
        assert discover_logged_in_registries(tmp_path / "nope.json") == []

    def test_malformed_config(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A non-JSON config file yields an empty list.
        cfg = tmp_path / "config.json"
        cfg.write_text("not json", encoding="utf-8")
        assert discover_logged_in_registries(cfg) == []

    def test_inline_auth(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] An 'auth' entry for ghcr.io produces one RegistryAuth with kind GHCR,
        # source INLINE and no helper.
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps(
                {
                    "auths": {
                        "ghcr.io": {"auth": "dXNlcjpwYXNz"},  # user:pass
                    }
                }
            ),
            encoding="utf-8",
        )
        out = discover_logged_in_registries(cfg)
        assert out == [
            RegistryAuth(
                host="ghcr.io",
                kind=RegistryKind.GHCR,
                cred_source=CredentialSource.INLINE,
                helper=None,
            )
        ]

    def test_credhelper_takes_precedence(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A per-host credHelpers entry beats the global credsStore: ECR private host
        # reports CREDHELPER with helper 'ecr-login'.
        """Per-host credHelpers override global credsStore."""
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps(
                {
                    "credsStore": "desktop",
                    "credHelpers": {
                        "123456789012.dkr.ecr.us-east-1.amazonaws.com": "ecr-login",
                    },
                    "auths": {
                        "123456789012.dkr.ecr.us-east-1.amazonaws.com": {},
                    },
                }
            ),
            encoding="utf-8",
        )
        out = discover_logged_in_registries(cfg)
        assert len(out) == 1
        assert out[0].kind is RegistryKind.ECR_PRIVATE
        assert out[0].cred_source is CredentialSource.CREDHELPER
        assert out[0].helper == "ecr-login"

    def test_credstore_with_empty_auths(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Empty {} auth entries with a credsStore are classified as CREDSTORE for
        # both GHCR and Docker Hub.
        """Docker Desktop populates empty {} entries; classify as CREDSTORE."""
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps(
                {
                    "credsStore": "desktop",
                    "auths": {
                        "ghcr.io": {},
                        "index.docker.io": {},
                    },
                }
            ),
            encoding="utf-8",
        )
        out = discover_logged_in_registries(cfg)
        kinds = {a.kind for a in out}
        sources = {a.cred_source for a in out}
        assert kinds == {RegistryKind.GHCR, RegistryKind.DOCKER_HUB}
        assert sources == {CredentialSource.CREDSTORE}

    def test_empty_auths_without_credstore(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] An empty {} entry with no credsStore is reported with source EMPTY.
        """Bare {} with no credsStore → EMPTY (still a logged-in signal)."""
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({"auths": {"ghcr.io": {}}}),
            encoding="utf-8",
        )
        out = discover_logged_in_registries(cfg)
        assert len(out) == 1
        assert out[0].cred_source is CredentialSource.EMPTY

    def test_mixed_sources(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Inline, credsStore and credHelper sources coexist and are attributed per
        # host, including the helper name 'gcloud'.
        """credHelper + credsStore + inline all coexist."""
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps(
                {
                    "credsStore": "desktop",
                    "credHelpers": {
                        "us-central1-docker.pkg.dev": "gcloud",
                    },
                    "auths": {
                        "ghcr.io": {"auth": "dXNlcjpwYXNz"},
                        "myregistry.azurecr.io": {},
                        "us-central1-docker.pkg.dev": {},
                    },
                }
            ),
            encoding="utf-8",
        )
        out = discover_logged_in_registries(cfg)
        by_host = {a.host: a for a in out}
        assert by_host["ghcr.io"].cred_source is CredentialSource.INLINE
        assert by_host["myregistry.azurecr.io"].cred_source is CredentialSource.CREDSTORE
        assert by_host["us-central1-docker.pkg.dev"].cred_source is CredentialSource.CREDHELPER
        assert by_host["us-central1-docker.pkg.dev"].helper == "gcloud"

    def test_normalizes_docker_hub_v1_url(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] The 'https://index.docker.io/v1/' key is normalised to host
        # 'index.docker.io' with kind DOCKER_HUB.
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({"auths": {"https://index.docker.io/v1/": {"auth": "dXg="}}}),
            encoding="utf-8",
        )
        out = discover_logged_in_registries(cfg)
        assert len(out) == 1
        assert out[0].host == "index.docker.io"
        assert out[0].kind is RegistryKind.DOCKER_HUB


class TestFilterKnown:
    # [CLASS DOC] Tests for filter_known_registries.
    def test_drops_other_kind(self) -> None:
        # [FUNCTION DOC] OTHER-kind entries are dropped while GHCR and LOCAL are kept.
        auths = [
            RegistryAuth("ghcr.io", RegistryKind.GHCR, CredentialSource.INLINE),
            RegistryAuth("quay.io", RegistryKind.OTHER, CredentialSource.INLINE),
            RegistryAuth("localhost:5000", RegistryKind.LOCAL, CredentialSource.EMPTY),
        ]
        out = filter_known_registries(auths)
        kinds = {a.kind for a in out}
        assert RegistryKind.OTHER not in kinds
        assert RegistryKind.GHCR in kinds
        assert RegistryKind.LOCAL in kinds
