# [MODULE DOC] Filesystem cache for BootstrapResult records. Each (repo, ref, options) maps to the
# slot <cache_dir>/<owner>__<name>/<ref[:12]>[__<opts_hash>]/ holding bootstrap.json, an optional
# reconstructed Dockerfile, and (written by runner) transcript.jsonl and build logs. The 8-char
# options hash distinguishes image-identity options (platform, base image, user Dockerfile,
# registry, language hint) so e.g. amd64 and arm64 builds of one SHA do not collide.
# KNOWN ISSUE: the module docstring says cache_dir defaults to ./envs; the actual default
# (BootstrapSpec.cache_dir) is $R2E_CACHE_DIR or ./workspace/bootstrap.
"""Filesystem cache for bootstrap results.

Layout under cache_dir (defaults to ./envs):

  <cache_dir>/<owner>__<name>/<short_commit>[__<opts_hash>]/
    bootstrap.json         # BootstrapResult, serialized
    Dockerfile             # reconstructed from agent commands
    transcript.jsonl       # full agent trace

The opts_hash is appended when the spec deviates from defaults along axes
that change image identity (platform, base_image, user_dockerfile,
image_registry). Without it, a prior `linux/amd64` build would silently
satisfy a later `--platform linux/arm64` request from the same SHA.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

from repo2rlenv.bootstrap.spec import BootstrapResult, LanguageHint

logger = logging.getLogger(__name__)


def _options_hash(opts: dict[str, Any] | None) -> str:
    # [FUNCTION DOC] HELPER: Returns the first 8 hex chars of SHA-256 over the sort_keys JSON
    # (default=str) of the options with None values dropped. Returns "" when opts is None/empty or
    # all values are None, keeping legacy short-commit-only slot paths. Output: str.
    """Stable 8-char hash of spec options that affect image identity.

    None / empty → returns "" so existing single-config caches keep their
    short-commit-only path (backwards compatible with v0.2 caches).
    """
    if not opts:
        return ""
    # Sort + JSON-serialize for stability; ignore None values so a
    # default-everywhere spec hashes the same as one with explicit defaults.
    filtered = {k: v for k, v in opts.items() if v is not None}
    if not filtered:
        return ""
    payload = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def cache_key(
    repo: str,
    ref: str,
    cache_dir: Path,
    *,
    options: dict[str, Any] | None = None,
) -> Path:
    # [FUNCTION DOC] Computes the cache slot directory (does not create it). Inputs: repo
    # "owner/name" (a bare name becomes owner "_"), ref (first 12 chars; empty means "head"),
    # cache_dir, options (keyword). Output: Path cache_dir/owner__name/short[__hash].
    """Return the cache directory for a (repo, ref, options) tuple."""
    owner, _, name = repo.partition("/")
    if not name:
        name = owner
        owner = "_"
    short = (ref or "head")[:12]
    opts_hash = _options_hash(options)
    slug = f"{short}__{opts_hash}" if opts_hash else short
    return cache_dir / f"{owner}__{name}" / slug


def save(
    result: BootstrapResult, cache_dir: Path, *, options: dict[str, Any] | None = None
) -> Path:
    # [FUNCTION DOC] Writes result to its slot (creating directories): bootstrap.json with the
    # dataclass as indented JSON (language as its string value, transcript_path as str) and, when
    # non-empty, the Dockerfile reconstruction. Inputs: result, cache_dir, options (must match those
    # used for load). Output: slot Path. Side effects: filesystem writes, overwrites existing files.
    """Write a BootstrapResult to its cache slot. Returns the dir."""
    slot = cache_key(result.repo, result.ref, cache_dir, options=options)
    slot.mkdir(parents=True, exist_ok=True)

    payload = asdict(result)
    # Pathlib + enum aren't JSON-serializable by default
    payload["language"] = result.language.value
    if result.transcript_path is not None:
        payload["transcript_path"] = str(result.transcript_path)
    (slot / "bootstrap.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if result.dockerfile_reconstruction:
        (slot / "Dockerfile").write_text(result.dockerfile_reconstruction, encoding="utf-8")

    return slot


def load(
    repo: str,
    ref: str,
    cache_dir: Path,
    *,
    options: dict[str, Any] | None = None,
) -> BootstrapResult | None:
    # [FUNCTION DOC] Reads a cached BootstrapResult for (repo, ref, options). Returns None if
    # bootstrap.json is missing or unreadable/invalid JSON (logs a warning). Coerces language back
    # to LanguageHint (unknown strings -> UNKNOWN), transcript_path to Path, and drops keys that are
    # not dataclass fields for forward compatibility. Raises TypeError if required fields are
    # missing from the JSON.
    """Return a cached BootstrapResult, or None if not present / unparseable."""
    slot = cache_key(repo, ref, cache_dir, options=options)
    f = slot / "bootstrap.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("cache load failed for %s: %s", f, exc)
        return None

    # Coerce LanguageHint back from string
    if isinstance(data.get("language"), str):
        try:
            data["language"] = LanguageHint(data["language"])
        except ValueError:
            data["language"] = LanguageHint.UNKNOWN

    if isinstance(data.get("transcript_path"), str):
        data["transcript_path"] = Path(data["transcript_path"])

    # Filter to known fields so future BootstrapResult additions don't break cached loads
    if is_dataclass(BootstrapResult):
        known = {f.name for f in fields(BootstrapResult)}
        data = {k: v for k, v in data.items() if k in known}
    return BootstrapResult(**data)
