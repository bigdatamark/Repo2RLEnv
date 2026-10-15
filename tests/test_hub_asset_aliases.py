# [MODULE DOC] Tests for execution/hub_asset_alias.py:install_aliases and HubAsset cache_aliases
# validation: aliases are symlinks in a Hugging Face cache (models--<alias> ->
# models--<org>--<name>) proven offline to resolve the same pinned files, written idempotently with
# a tasksmith-asset-aliases.json manifest, and never overwriting or partially applying. Also checks
# python_build.dependency_recipe stays prefix-stable. Uses synthetic cache files and a mocked
# huggingface_hub.
# KNOWN ISSUE: the module lacks 'from __future__ import annotations', violating the repo invariant
# that every module starts with it.
"""Cache alias preparation only uses synthetic files and mocked Hub metadata."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repo2rlenv.execution.hub_asset_alias import install_aliases
from repo2rlenv.execution.python_build import dependency_recipe
from repo2rlenv.spec.recipe_options import HubAsset, PythonRepositoryProfile

REVISION = "a" * 40


def asset(**changes):
    # [FUNCTION DOC] HELPER: Builds a validated HubAsset for testing/tiny at REVISION with two
    # filenames, max_bytes 100 and a purpose, overridden by keyword changes. Output: HubAsset.
    # Raises ValueError (pydantic) on invalid input.
    return HubAsset.model_validate(
        {
            "repo_id": "testing/tiny",
            "revision": REVISION,
            "filenames": ["config.json", "tokenizer.json"],
            "max_bytes": 100,
            "purpose": "Exercise actual tokenizer parsing offline.",
            **changes,
        }
    )


def test_alias_does_not_invalidate_existing_canonical_download_layer():
    # [FUNCTION DOC] Asserts the dependency_recipe without aliases has a recorded sha256, and
    # adding cache_aliases only appends an install_aliases step mentioning the alias while fetching
    # assets exactly once.
    original = dependency_recipe(
        "python:3.12-slim", ["huggingface-hub==0.36.0"], hub_assets=[asset()]
    )
    # Recorded before adding cache_aliases: existing build prefixes stay reusable.
    assert (
        hashlib.sha256(original.encode()).hexdigest()
        == "a2f650069b34df0637da263a4a9420f2968bb4b6ac12f3dda5dbd9c0bd0e48e0"
    )
    updated = dependency_recipe(
        "python:3.12-slim", ["huggingface-hub==0.36.0"], hub_assets=[asset(cache_aliases=["tiny"])]
    )
    assert updated.startswith(original)
    assert "install_aliases(json.loads(" in updated[len(original) :]
    assert "tiny" in updated[len(original) :]
    assert updated.count("fetch_assets(json.loads(") == 1


@pytest.mark.parametrize(
    "aliases",
    [
        ["tiny", "tiny"],
        ["../tiny"],
        ["owner/../tiny"],
        ["tiny--copy"],
        ["tiny..copy"],
        ["https://example.com/tiny"],
        ["tiny.git"],
        ["tiny/"],
    ],
)
def test_alias_rejects_ambiguous_or_invalid_names(aliases):
    # [FUNCTION DOC] Parametrized over duplicate, traversal, '--', '..', URL, '.git' and trailing-
    # slash aliases; asserts HubAsset validation raises ValueError.
    with pytest.raises(ValueError):
        asset(cache_aliases=aliases)


@pytest.mark.parametrize(
    "assets",
    [
        [asset(cache_aliases=["testing/tiny"])],
        [asset(cache_aliases=["other/tiny"]), asset(repo_id="other/tiny")],
        [asset(cache_aliases=["tiny"]), asset(repo_id="other/tiny", cache_aliases=["tiny"])],
    ],
)
def test_alias_collisions_rejected_in_profile(assets):
    # [FUNCTION DOC] Parametrized over an alias equal to its own repo, to another asset's repo, or
    # shared by two assets; asserts PythonRepositoryProfile raises ValueError 'must not collide'.
    with pytest.raises(ValueError, match="must not collide"):
        PythonRepositoryProfile(
            source_paths=["src"],
            test_paths=["tests"],
            dependencies=["huggingface_hub==0.36.2"],
            hub_assets=assets,
        )


@pytest.fixture
def cache(tmp_path):
    # [FUNCTION DOC] Fixture: builds a synthetic HF cache under tmp_path/cache with snapshot files
    # for testing/tiny, a refs/main pointing at REVISION, and a tasksmith-assets.json recording
    # file sizes and sha256s. Output: cache root Path.
    root = tmp_path / "cache"
    snapshot = root / "models--testing--tiny" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    records = []
    for name in ["config.json", "tokenizer.json"]:
        (snapshot / name).write_text("{}")
        records.append({"filename": name, "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()})
    refs = root / "models--testing--tiny" / "refs"
    refs.mkdir()
    (refs / "main").write_text(REVISION)
    (root / "tasksmith-assets.json").write_text(
        json.dumps(
            [
                {
                    "repo_id": "testing/tiny",
                    "revision": REVISION,
                    "offline_main": REVISION,
                    "files": records,
                }
            ]
        )
    )
    return root


@pytest.fixture
def hub(monkeypatch):
    # [FUNCTION DOC] Fixture: monkeypatches huggingface_hub.HfApi (model_info) and hf_hub_download
    # with fakes that assert pinned, token-free, offline use and record calls. Output: the shared
    # calls list.
    import huggingface_hub

    calls = []

    def model_info(repo_id, *, revision, token, timeout):
        # [FUNCTION DOC] HELPER: Fake HfApi.model_info; asserts revision REVISION, token False and
        # timeout 30, records a metadata call and returns id testing/tiny with sha REVISION.
        assert revision == REVISION and token is False and timeout == 30
        calls.append(("metadata", repo_id))
        return SimpleNamespace(id="testing/tiny", sha=REVISION)

    def download(**kwargs):
        # [FUNCTION DOC] HELPER: Fake hf_hub_download; asserts local_files_only and token False,
        # records an offline call, and returns the snapshot path for the requested file, resolving
        # revision 'main' via refs/main.
        assert kwargs["local_files_only"] is True and kwargs["token"] is False
        calls.append(("offline", kwargs["repo_id"], kwargs["revision"]))
        root = Path(kwargs["cache_dir"]) / ("models--" + kwargs["repo_id"].replace("/", "--"))
        revision = (
            (root / "refs/main").read_text() if kwargs["revision"] == "main" else kwargs["revision"]
        )
        return str(root / "snapshots" / revision / kwargs["filename"])

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=model_info))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls


def request(aliases=("tiny",)):
    # [FUNCTION DOC] HELPER: Returns a one-item alias request list for testing/tiny at REVISION
    # with the given aliases (default ('tiny',)).
    return [{"repo_id": "testing/tiny", "revision": REVISION, "cache_aliases": list(aliases)}]


def test_alias_resolves_same_pinned_files_offline_and_is_idempotent(cache, hub):
    # [FUNCTION DOC] Asserts install_aliases creates a relative models--tiny symlink, makes 5 hub
    # calls (1 metadata, 2 per file for main and the pinned revision), and that a second run leaves
    # the manifest byte-identical with the pinned revision.
    install_aliases(request(), cache)
    target = cache / "models--tiny"
    assert target.is_symlink() and target.readlink() == Path("models--testing--tiny")
    assert len(hub) == 5  # one metadata request and two local lookups per file
    assert {call[2] for call in hub if call[0] == "offline"} == {"main", REVISION}
    manifest = cache / "tasksmith-asset-aliases.json"
    before = manifest.read_bytes()
    install_aliases(request(), cache)
    assert manifest.read_bytes() == before
    assert json.loads(before)["aliases"][0]["revision"] == REVISION


def test_sdk_offline_lookup_accepts_alias_with_snapshot_blob_links(cache, hub, monkeypatch):
    # [FUNCTION DOC] Converts snapshot files into blob symlinks, uses the real hf_hub_download, and
    # asserts the manifest lists both files and only the metadata call was faked.
    import huggingface_hub
    from huggingface_hub.file_download import hf_hub_download

    root = cache / "models--testing--tiny"
    blobs = root / "blobs"
    blobs.mkdir()
    blob = blobs / hashlib.sha256(b"{}").hexdigest()
    blob.write_text("{}")
    for path in (root / "snapshots" / REVISION).iterdir():
        path.unlink()
        path.symlink_to(Path("../../blobs") / blob.name)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    install_aliases(request(), cache)
    manifest = json.loads((cache / "tasksmith-asset-aliases.json").read_text())
    assert len(manifest["aliases"][0]["files"]) == 2
    assert hub == [("metadata", "tiny")]


@pytest.mark.parametrize("mismatch", ["identity", "revision"])
def test_wrong_hub_identity_rejected_before_any_alias_mutation(cache, hub, monkeypatch, mismatch):
    # [FUNCTION DOC] Parametrized identity/revision mismatch for the second alias; asserts
    # ValueError 'canonical repository and pinned revision' and that neither the first alias link
    # nor the manifest was created.
    import huggingface_hub

    def model_info(repo_id, **kwargs):
        # [FUNCTION DOC] HELPER: Fake model_info; correct for 'tiny', otherwise returns a wrong id
        # or wrong sha per mismatch.
        # First alias is valid; all aliases must pass before even that link exists.
        if repo_id == "tiny":
            return SimpleNamespace(id="testing/tiny", sha=REVISION)
        return SimpleNamespace(
            id="wrong/tiny" if mismatch == "identity" else "testing/tiny",
            sha="b" * 40 if mismatch == "revision" else REVISION,
        )

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=model_info))
    with pytest.raises(ValueError, match="canonical repository and pinned revision"):
        install_aliases(request(("tiny", "second")), cache)
    assert not (cache / "models--tiny").is_symlink()
    assert not (cache / "tasksmith-asset-aliases.json").exists()


@pytest.mark.parametrize("kind", ["directory", "absolute_link", "other_link"])
def test_existing_cache_entries_are_never_overwritten(cache, hub, kind):
    # [FUNCTION DOC] Parametrized over an existing directory, absolute symlink or foreign symlink
    # at models--tiny; asserts ValueError, the entry is left in place, and no hub calls happen.
    target = cache / "models--tiny"
    if kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(
            cache / "models--testing--tiny" if kind == "absolute_link" else "models--other"
        )
    with pytest.raises(ValueError, match=r"Existing cache alias|overwrite an existing"):
        install_aliases(request(), cache)
    assert target.exists() or target.is_symlink()
    assert not hub


def test_modified_asset_bytes_rejected_before_identity_lookup(cache, hub):
    # [FUNCTION DOC] Changes a snapshot file; asserts ValueError 'bytes differ' before any hub call
    # and no alias created.
    (cache / "models--testing--tiny/snapshots" / REVISION / "config.json").write_text("changed")
    with pytest.raises(ValueError, match="bytes differ"):
        install_aliases(request(), cache)
    assert not hub
    assert not (cache / "models--tiny").is_symlink()


def test_failed_offline_proof_rolls_back_new_aliases(cache, hub, monkeypatch):
    # [FUNCTION DOC] Makes hf_hub_download return a wrong file; asserts ValueError 'did not return
    # the pinned' and that the new alias and manifest are rolled back.
    import huggingface_hub

    wrong = cache / "wrong.json"
    wrong.write_text("{}")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **_: str(wrong))
    with pytest.raises(ValueError, match="did not return the pinned"):
        install_aliases(request(), cache)
    assert not (cache / "models--tiny").is_symlink()
    assert not (cache / "tasksmith-asset-aliases.json").exists()


def test_failed_offline_proof_preserves_existing_correct_alias(cache, hub, monkeypatch):
    # [FUNCTION DOC] With a correct alias already present, makes downloads raise FileNotFoundError;
    # asserts the error propagates, the existing link is preserved and no manifest is written.
    import huggingface_hub

    target = cache / "models--tiny"
    target.symlink_to("models--testing--tiny", target_is_directory=True)

    def missing(**kwargs):
        # [FUNCTION DOC] HELPER: Fake hf_hub_download that raises FileNotFoundError('synthetic
        # missing asset').
        raise FileNotFoundError("synthetic missing asset")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    with pytest.raises(FileNotFoundError, match="synthetic missing"):
        install_aliases(request(), cache)
    assert target.is_symlink() and target.readlink() == Path("models--testing--tiny")
    assert not (cache / "tasksmith-asset-aliases.json").exists()


def test_existing_alias_manifest_conflict_fails_before_mutation(cache, hub):
    # [FUNCTION DOC] Pre-writes a different manifest; asserts ValueError 'manifest differs', the
    # file is untouched and no alias is created.
    output = cache / "tasksmith-asset-aliases.json"
    output.write_text('{"previous": "preserve"}')
    with pytest.raises(ValueError, match="manifest differs"):
        install_aliases(request(), cache)
    assert output.read_text() == '{"previous": "preserve"}'
    assert not (cache / "models--tiny").is_symlink()
