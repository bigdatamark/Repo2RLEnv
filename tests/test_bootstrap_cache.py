# [MODULE DOC] Tests for bootstrap.cache: on-disk layout (<root>/<owner>__<name>/<sha12>[opts] with
# bootstrap.json and Dockerfile), cache miss, save/load round-trip, tolerance of unknown JSON
# fields, and option-aware cache keys (defaults/None/empty options map to the legacy path; platform
# and base_image produce distinct slots). Uses tmp_path only.
"""Cache load/save round-trip."""

from __future__ import annotations

from pathlib import Path

from repo2rlenv.bootstrap import cache as cache_mod
from repo2rlenv.bootstrap.spec import BootstrapResult, LanguageHint


def _sample_result(
    repo: str = "huggingface/trl", ref: str = "a1b2c3d4e5f6abcdef"
) -> BootstrapResult:
    # [FUNCTION DOC] HELPER: Builds a representative Python BootstrapResult for repo (default
    # huggingface/trl) and ref (default a1b2c3d4e5f6abcdef) with fixed digest, commands and a
    # Dockerfile reconstruction.
    return BootstrapResult(
        image_digest="sha256:" + "a" * 64,
        image_tag="local/r2e/trl:a1b2c3d4e5f6",
        language=LanguageHint.PYTHON,
        repo=repo,
        ref=ref,
        rebuild_cmds=["pip install -e ."],
        test_cmds=["pytest -x"],
        smoke_passed=True,
        iterations=4,
        build_time_sec=120.5,
        llm_provider="anthropic/claude-sonnet-4-6",
        dockerfile_reconstruction="FROM python:3.12-slim\nRUN pip install -e .\n",
    )


def test_save_creates_expected_layout(tmp_path: Path):
    # [FUNCTION DOC] save() returns <root>/huggingface__trl/a1b2c3d4e5f6 containing bootstrap.json
    # and Dockerfile.
    result = _sample_result()
    slot = cache_mod.save(result, tmp_path)
    assert slot == tmp_path / "huggingface__trl" / "a1b2c3d4e5f6"
    assert (slot / "bootstrap.json").is_file()
    assert (slot / "Dockerfile").is_file()


def test_load_returns_none_on_cache_miss(tmp_path: Path):
    # [FUNCTION DOC] load() of an absent entry returns None.
    assert cache_mod.load("foo/bar", "deadbeef", tmp_path) is None


def test_save_then_load_roundtrip(tmp_path: Path):
    # [FUNCTION DOC] Saved fields (digest, language, commands, iterations) load back equal.
    original = _sample_result()
    cache_mod.save(original, tmp_path)
    loaded = cache_mod.load(original.repo, original.ref, tmp_path)
    assert loaded is not None
    assert loaded.image_digest == original.image_digest
    assert loaded.language == LanguageHint.PYTHON
    assert loaded.rebuild_cmds == original.rebuild_cmds
    assert loaded.test_cmds == original.test_cmds
    assert loaded.iterations == original.iterations


def test_load_tolerates_unknown_fields(tmp_path: Path):
    # [FUNCTION DOC] An extra field injected into bootstrap.json does not break load().
    """Cache loads should not break if BootstrapResult adds new fields later."""
    result = _sample_result()
    slot = cache_mod.save(result, tmp_path)
    # Inject a future-version field
    data = (slot / "bootstrap.json").read_text()
    import json

    payload = json.loads(data)
    payload["_future_field"] = "should be ignored"
    (slot / "bootstrap.json").write_text(json.dumps(payload))

    loaded = cache_mod.load(result.repo, result.ref, tmp_path)
    assert loaded is not None
    assert loaded.image_digest == result.image_digest


def test_dockerfile_written_when_present(tmp_path: Path):
    # [FUNCTION DOC] The Dockerfile reconstruction is written to the slot.
    result = _sample_result()
    slot = cache_mod.save(result, tmp_path)
    assert (slot / "Dockerfile").read_text().startswith("FROM python:")


def test_cache_key_no_options_is_backwards_compatible(tmp_path: Path):
    # [FUNCTION DOC] No/None/empty/all-None options give the same unsuffixed path named by the sha.
    """Default-only spec should produce the same v0.2 path (no opts hash suffix)."""
    p1 = cache_mod.cache_key("foo/bar", "abc123def456", tmp_path)
    p2 = cache_mod.cache_key("foo/bar", "abc123def456", tmp_path, options=None)
    p3 = cache_mod.cache_key("foo/bar", "abc123def456", tmp_path, options={})
    p4 = cache_mod.cache_key(
        "foo/bar", "abc123def456", tmp_path, options={"base_image": None, "platform": None}
    )
    assert p1 == p2 == p3 == p4
    assert p1.name == "abc123def456"


def test_cache_key_differs_when_options_differ(tmp_path: Path):
    # [FUNCTION DOC] Different platform or base_image produce different cache paths.
    """Different platforms / base images must map to different cache slots."""
    amd64 = cache_mod.cache_key("foo/bar", "abc", tmp_path, options={"platform": "linux/amd64"})
    arm64 = cache_mod.cache_key("foo/bar", "abc", tmp_path, options={"platform": "linux/arm64"})
    custom_base = cache_mod.cache_key(
        "foo/bar",
        "abc",
        tmp_path,
        options={"platform": "linux/amd64", "base_image": "ubuntu:24.04"},
    )
    assert amd64 != arm64, "platform must affect cache slot"
    assert amd64 != custom_base, "base_image must affect cache slot"
    assert arm64 != custom_base


def test_cache_save_and_load_with_options(tmp_path: Path):
    # [FUNCTION DOC] Matching options round-trip; different options miss.
    """save/load with matching options round-trips; mismatched options miss."""
    result = _sample_result()
    opts = {"platform": "linux/arm64", "base_image": "ubuntu:24.04"}
    cache_mod.save(result, tmp_path, options=opts)
    assert cache_mod.load(result.repo, result.ref, tmp_path, options=opts) is not None
    # Different options → cache miss
    assert (
        cache_mod.load(result.repo, result.ref, tmp_path, options={"platform": "linux/amd64"})
        is None
    )
