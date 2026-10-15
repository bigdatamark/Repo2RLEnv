# [MODULE DOC] Tests for pinned Hugging Face Hub asset preparation: the private grader environment keeps HF
# cache/offline vars but drops credentials and PYTHONPATH; HubAsset validation (safe filenames,
# 40-char revision, huggingface-hub pin required); the asset-fetch layer precedes source COPY in
# the Dockerfile and only changes with revision; fetch_assets checks metadata size and revision
# before downloading and writes an offline refs alias plus manifest. Hub calls are monkeypatched.
"""Pinned asset preparation uses the remote image path and fails before oversize downloads."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from repo2rlenv.execution.hub_asset_fetch import fetch_assets
from repo2rlenv.execution.python_build import dependency_recipe
from repo2rlenv.pipelines.recipes.repository.export import repository_build
from repo2rlenv.spec.recipe_options import HubAsset, PythonRepositoryProfile


def test_private_grader_keeps_asset_cache_without_forwarding_credentials(monkeypatch):
    # [FUNCTION DOC] Sets HF cache/offline and CUDA env vars plus credential-like vars, then asserts
    # child_environment keeps the prepared values, sets HOME to the given path and forwards none
    # of the credential values.
    from repo2rlenv.pipelines.recipes.swe_smith.grade import child_environment

    prepared = {
        "HF_HOME": "/opt/tasksmith-hf",
        "HF_HUB_CACHE": "/opt/tasksmith-hf/hub",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    for name, value in prepared.items():
        monkeypatch.setenv(name, value)
    for name in ("HF_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PYTHONPATH"):
        monkeypatch.setenv(name, "must-not-be-forwarded")
    environment = child_environment("/tmp/private-grader")
    assert all(environment[name] == value for name, value in prepared.items())
    assert environment["HOME"] == "/tmp/private-grader"
    assert "must-not-be-forwarded" not in environment.values()


def asset(**changes):
    # [FUNCTION DOC] HELPER: Builds a validated HubAsset with defaults (repo testing/tiny, 40-char revision
    # 'a'*40, two json filenames, max_bytes 100, a purpose), overridden by keyword changes. Raises
    # pydantic ValueError on invalid input.
    return HubAsset.model_validate(
        {
            "repo_id": "testing/tiny",
            "revision": "a" * 40,
            "filenames": ["config.json", "tokenizer.json"],
            "max_bytes": 100,
            "purpose": "Exercise actual tokenizer parsing offline.",
            **changes,
        }
    )


def test_asset_allowlist_and_dependency_contract():
    # [FUNCTION DOC] Expects ValueError for traversal, absolute, .py and glob filenames, for a non-commit
    # revision, and 'version pin' when a profile has hub_assets but no pinned huggingface-hub
    # dependency.
    for filename in ["../config.json", "/config.json", "model.py", "*.json"]:
        with pytest.raises(ValueError):
            asset(filenames=[filename])
    with pytest.raises(ValueError):
        asset(revision="main")
    with pytest.raises(ValueError, match="version pin"):
        PythonRepositoryProfile(source_paths=["src"], test_paths=["tests"], hub_assets=[asset()])


def test_asset_layer_precedes_source_and_changes_with_revision():
    # [FUNCTION DOC] Asserts repository_build starts with the dependency_recipe prefix, fetch_assets appears
    # before COPY source, HF_HUB_OFFLINE=1 is set, changing only purpose leaves the prefix
    # unchanged and changing revision changes it.
    options = PythonRepositoryProfile(
        source_paths=["src"],
        test_paths=["tests"],
        dependencies=["huggingface-hub==0.36.0"],
        hub_assets=[asset()],
    )
    recipe = repository_build(options)
    prefix = dependency_recipe(
        options.base_image, options.dependencies, hub_assets=options.hub_assets
    )
    assert recipe.startswith(prefix)
    assert recipe.index("fetch_assets") < recipe.index("COPY source")
    assert "HF_HUB_OFFLINE=1" in recipe
    assert prefix == dependency_recipe(
        options.base_image,
        options.dependencies,
        hub_assets=[asset(purpose="A clearer description of the same required tokenizer files.")],
    )
    changed = dependency_recipe(
        options.base_image, options.dependencies, hub_assets=[asset(revision="b" * 40)]
    )
    assert changed != prefix


def test_pinned_files_and_offline_alias_are_recorded(monkeypatch, tmp_path):
    # [FUNCTION DOC] Patches huggingface_hub metadata/download. Asserts fetch_assets downloads both files
    # anonymously at the pinned revision, writes refs/main alias and a tasksmith-assets.json
    # manifest with sha256s; then raises ValueError 'allowance' when max_bytes is too small and
    # 'pinned revision' when metadata reports another commit, both before downloading.
    import huggingface_hub as hub

    calls = []

    def metadata(url, *, token):
        # [FUNCTION DOC] HELPER: fake get_hf_file_metadata asserting token False and the pinned revision in url;
        # returns commit_hash and size 2.
        assert token is False and "a" * 40 in url
        return SimpleNamespace(commit_hash="a" * 40, size=2)

    def download(**kwargs):
        # [FUNCTION DOC] HELPER: fake hf_hub_download recording kwargs, asserting anonymous pinned access,
        # writing '{}' to a tmp file and returning its path.
        calls.append(kwargs)
        assert kwargs["token"] is False and kwargs["revision"] == "a" * 40
        file = tmp_path / "files" / kwargs["filename"]
        file.parent.mkdir(exist_ok=True)
        file.write_text("{}")
        return str(file)

    monkeypatch.setattr(hub, "get_hf_file_metadata", metadata)
    monkeypatch.setattr(hub, "hf_hub_download", download)
    cache = tmp_path / "cache"
    fetch_assets([asset().model_dump()], cache)
    assert len(calls) == 2
    assert (cache / "models--testing--tiny/refs/main").read_text() == "a" * 40
    manifest = json.loads((cache / "tasksmith-assets.json").read_text())
    assert manifest[0]["files"][0]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    calls.clear()
    with pytest.raises(ValueError, match="allowance"):
        fetch_assets([asset(max_bytes=1).model_dump()], cache)
    assert not calls
    monkeypatch.setattr(
        hub, "get_hf_file_metadata", lambda *a, **kw: SimpleNamespace(commit_hash="b" * 40, size=2)
    )
    with pytest.raises(ValueError, match="pinned revision"):
        fetch_assets([asset().model_dump()], cache)
    assert not calls
