# [MODULE DOC] Tests for registry.naming: slugify_repo (OCI-safe 'r2e-bootstrap-owner-name' slugs),
# build_image_ref (prefix/slug:sha12[-opts8] refs, length bounds, validation) and split_ref (round-
# trip and error cases). Module-level regexes encode the OCI name and tag grammars. Pure, no I/O.
"""Tests for `registry.naming` — slug + tag construction."""

from __future__ import annotations

import re

import pytest

from repo2rlenv.registry.naming import (
    DEFAULT_BOOTSTRAP_PREFIX,
    build_image_ref,
    slugify_repo,
    split_ref,
)

_OCI_NAME = re.compile(r"^[a-z0-9]+((?:[._]|__|-+)[a-z0-9]+)*$")
_OCI_TAG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


class TestSlugifyRepo:
    @pytest.mark.parametrize(
        "owner,name,expected",
        [
            ("pallets", "click", "r2e-bootstrap-pallets-click"),
            ("Pallets", "Click", "r2e-bootstrap-pallets-click"),
            ("huggingface", "trl", "r2e-bootstrap-huggingface-trl"),
            ("scikit-learn", "scikit-learn", "r2e-bootstrap-scikit-learn-scikit-learn"),
        ],
    )
    # [CLASS DOC] Tests for slugify_repo.
    def test_basic(self, owner: str, name: str, expected: str) -> None:
        # [FUNCTION DOC] Parametrized: owner/name are lowercased and joined under the default
        # prefix.
        assert slugify_repo(owner, name) == expected

    def test_collapses_special_chars(self) -> None:
        # [FUNCTION DOC] Dots, underscores and '@' collapse to hyphens and the result matches the
        # OCI name grammar.
        out = slugify_repo("a.b_c", "d-e@f")
        assert _OCI_NAME.match(out)
        assert "@" not in out
        # The components are still recognisable
        assert "a-b-c" in out
        assert "d-e-f" in out

    def test_custom_prefix(self) -> None:
        # [FUNCTION DOC] A custom prefix replaces the default.
        out = slugify_repo("foo", "bar", prefix="custom")
        assert out == "custom-foo-bar"

    def test_truncates_long_input(self) -> None:
        # [FUNCTION DOC] Very long names are truncated to at most 100 chars and stay OCI-valid.
        long_name = "x" * 200
        out = slugify_repo("foo", long_name)
        assert len(out) <= 100
        assert _OCI_NAME.match(out)

    def test_empty_inputs_raise(self) -> None:
        # [FUNCTION DOC] Empty owner or name raises ValueError.
        with pytest.raises(ValueError):
            slugify_repo("", "bar")
        with pytest.raises(ValueError):
            slugify_repo("foo", "")


class TestBuildImageRef:
    # [CLASS DOC] Tests for build_image_ref.
    def test_default_ghcr_shape(self) -> None:
        # [FUNCTION DOC] Produces '<prefix>/r2e-bootstrap-owner-name:<sha12>' with an OCI-valid tag.
        ref = build_image_ref(
            registry_prefix="ghcr.io/huggingface",
            owner="pallets",
            name="click",
            commit_sha="a1b2c3d4e5f67890",
        )
        assert ref == "ghcr.io/huggingface/r2e-bootstrap-pallets-click:a1b2c3d4e5f6"
        assert _OCI_TAG.match(ref.rpartition(":")[2])

    def test_with_options_hash(self) -> None:
        # [FUNCTION DOC] An options hash appends '-<first 8 chars>' to the tag.
        ref = build_image_ref(
            registry_prefix="ghcr.io/huggingface",
            owner="pallets",
            name="click",
            commit_sha="a1b2c3d4e5f6",
            options_hash="7d8e9f0123456789",
        )
        assert ref.endswith(":a1b2c3d4e5f6-7d8e9f01")

    def test_ecr_private_prefix(self) -> None:
        # [FUNCTION DOC] ECR private prefixes are preserved at the start of the ref.
        ref = build_image_ref(
            registry_prefix="123456789.dkr.ecr.us-east-1.amazonaws.com/r2e",
            owner="pallets",
            name="click",
            commit_sha="abcdef123456",
        )
        assert ref.startswith("123456789.dkr.ecr.us-east-1.amazonaws.com/r2e/")

    def test_localhost_registry(self) -> None:
        # [FUNCTION DOC] localhost:5000 prefixes are preserved.
        ref = build_image_ref(
            registry_prefix="localhost:5000",
            owner="foo",
            name="bar",
            commit_sha="0123456789ab",
        )
        assert ref.startswith("localhost:5000/")

    def test_truncates_sha_to_12(self) -> None:
        # [FUNCTION DOC] The commit sha is lowercased and truncated to 12 chars.
        ref = build_image_ref(
            registry_prefix="ghcr.io/u",
            owner="o",
            name="n",
            commit_sha="A1B2C3D4E5F60123456789",
        )
        # truncated AND lowercased
        assert ref.endswith(":a1b2c3d4e5f6")

    def test_total_length_bound(self) -> None:
        # [FUNCTION DOC] Long inputs still yield a ref of at most 255 chars.
        ref = build_image_ref(
            registry_prefix="ghcr.io/very-long-org-name-that-is-still-realistic",
            owner="x" * 30,
            name="y" * 30,
            commit_sha="0123456789abcdef",
            options_hash="01234567",
        )
        assert len(ref) <= 255

    def test_empty_prefix_raises(self) -> None:
        # [FUNCTION DOC] An empty registry prefix raises ValueError.
        with pytest.raises(ValueError):
            build_image_ref(
                registry_prefix="",
                owner="foo",
                name="bar",
                commit_sha="abc",
            )

    def test_empty_sha_raises(self) -> None:
        # [FUNCTION DOC] An empty commit sha raises ValueError.
        with pytest.raises(ValueError):
            build_image_ref(
                registry_prefix="ghcr.io/u",
                owner="foo",
                name="bar",
                commit_sha="",
            )

    def test_default_prefix_used(self) -> None:
        # [FUNCTION DOC] DEFAULT_BOOTSTRAP_PREFIX appears in the ref.
        ref = build_image_ref(
            registry_prefix="ghcr.io/u",
            owner="foo",
            name="bar",
            commit_sha="abc123def456",
        )
        assert DEFAULT_BOOTSTRAP_PREFIX in ref


class TestSplitRef:
    # [CLASS DOC] Tests for split_ref.
    def test_round_trip(self) -> None:
        # [FUNCTION DOC] split_ref inverts build_image_ref into (prefix, image, tag).
        ref = build_image_ref(
            registry_prefix="ghcr.io/huggingface",
            owner="pallets",
            name="click",
            commit_sha="a1b2c3d4e5f6",
        )
        prefix, image, tag = split_ref(ref)
        assert prefix == "ghcr.io/huggingface"
        assert image == "r2e-bootstrap-pallets-click"
        assert tag == "a1b2c3d4e5f6"

    def test_missing_tag_raises(self) -> None:
        # [FUNCTION DOC] A ref without a tag raises ValueError.
        with pytest.raises(ValueError):
            split_ref("ghcr.io/foo/bar")

    def test_missing_host_raises(self) -> None:
        # [FUNCTION DOC] A ref without a host/prefix raises ValueError.
        with pytest.raises(ValueError):
            split_ref("foo:tag")
