# [MODULE DOC] Builds and parses OCI image references for bootstrap images pushed to a registry.
# Shape: <registry_prefix>/<prefix>-<owner>-<repo slug>:<sha12>[-<opts8>], where opts8 is the first
# 8 chars of the bootstrap cache options hash (empty hash => no suffix). Enforces OCI rules:
# lowercase name components matching the OCI path-component regex, tag <=128 chars, full ref <=255
# chars. Pure string logic: no I/O, no global state. Constants: DEFAULT_BOOTSTRAP_PREFIX
# ("r2e-bootstrap").
"""Slug + tag construction for the bootstrap image reference.

Final shape (default, when GHCR is the chosen registry):

    ghcr.io/<owner>/r2e-bootstrap-<repo-slug>:<sha12>[-<opts8>]

`<opts8>` is the same `cache._options_hash` output used in `bootstrap/cache.py`
— reuse, don't reinvent. When the bootstrap was built with default options
(empty opts hash), the tag drops the suffix.

OCI Distribution Spec ref:
- Lowercase enforced.
- Path components match `[a-z0-9]+((\\.|_|__|-+)[a-z0-9]+)*`.
- Total ref ≤ 255 chars.
- Tag ≤ 128 chars.
"""

from __future__ import annotations

import re

DEFAULT_BOOTSTRAP_PREFIX = "r2e-bootstrap"

# OCI path component: lowercase letters, digits, separator runs of . _ __ -+
_OCI_NAME_RE = re.compile(r"^[a-z0-9]+((?:[._]|__|-+)[a-z0-9]+)*$")
_OCI_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


def slugify_repo(owner: str, name: str, *, prefix: str = DEFAULT_BOOTSTRAP_PREFIX) -> str:
    # [FUNCTION DOC] Builds the image-name component "<prefix>-<owner>-<name>" in OCI-safe form.
    # Lowercases, collapses every run of characters outside [a-z0-9] into one "-", strips
    # leading/trailing "-", "." and "_", truncates to 100 chars (re-stripping trailing separators).
    # Inputs: owner, name (non-empty str); prefix (keyword, default DEFAULT_BOOTSTRAP_PREFIX).
    # Output: slug str. Raises ValueError if owner/name empty or the slug fails the OCI name regex.
    """Build the image *name* portion: `<prefix>-<owner>-<repo>`, slugified.

    Lowercase, non-`[a-z0-9-]` collapsed to a single `-`, trimmed.
    Output guaranteed to pass `_OCI_NAME_RE`.
    """
    if not owner or not name:
        raise ValueError("owner and name must both be non-empty")
    raw = f"{prefix}-{owner}-{name}".lower()
    # Collapse runs of non-[a-z0-9] to a single `-`
    slug = re.sub(r"[^a-z0-9]+", "-", raw)
    slug = slug.strip("-._")
    if len(slug) > 100:
        slug = slug[:100].rstrip("-._")
    if not _OCI_NAME_RE.match(slug):
        raise ValueError(f"slugify produced invalid OCI name component: {slug!r}")
    return slug


def build_image_ref(
    *,
    registry_prefix: str,
    owner: str,
    name: str,
    commit_sha: str,
    options_hash: str = "",
    prefix: str = DEFAULT_BOOTSTRAP_PREFIX,
) -> str:
    # [FUNCTION DOC] Composes the full reference "<registry_prefix>/<slug>:<tag>". registry_prefix
    # is a host or host/namespace (whitespace and trailing "/" trimmed). Tag is the first 12 chars
    # of commit_sha (stripped, lowercased), plus "-" and the first 8 chars of options_hash when that
    # is non-empty. Slug comes from slugify_repo(owner, name, prefix=prefix). All args keyword-only.
    # Output: ref str. Raises ValueError if registry_prefix or commit_sha is empty, the tag fails
    # the OCI tag regex, the ref exceeds 255 chars, or slugify_repo fails. Side effects: none.
    """Compose the full `host[/namespace]/<image-name>:<tag>` reference.

    `registry_prefix` is a registry-with-namespace string like:
        ghcr.io/huggingface
        123456789.dkr.ecr.us-east-1.amazonaws.com/r2e
        public.ecr.aws/myalias
        myregistry.azurecr.io/r2e
        us-central1-docker.pkg.dev/myproject/r2e
        index.docker.io/myuser
        localhost:5000

    `commit_sha` is truncated to 12 chars in the tag. `options_hash` is
    appended after `-` when non-empty (matches `cache._options_hash` semantics).
    """
    if not registry_prefix:
        raise ValueError("registry_prefix is required (host or host/namespace)")
    if not commit_sha:
        raise ValueError("commit_sha is required")
    image_name = slugify_repo(owner, name, prefix=prefix)
    prefix_trimmed = registry_prefix.strip().rstrip("/")
    sha12 = commit_sha.strip()[:12].lower()
    tag = f"{sha12}-{options_hash[:8]}" if options_hash else sha12
    if not _OCI_TAG_RE.match(tag):
        raise ValueError(f"invalid OCI tag produced: {tag!r}")
    ref = f"{prefix_trimmed}/{image_name}:{tag}"
    if len(ref) > 255:
        raise ValueError(f"OCI ref exceeds 255 chars ({len(ref)}): {ref!r}")
    return ref


def split_ref(ref: str) -> tuple[str, str, str]:
    # [FUNCTION DOC] Splits "host[/namespace]/<image>:<tag>" into (registry_prefix, image, tag).
    # Partitions on the LAST ":" for the tag, then on the last "/" of the remainder for the image
    # name. Raises ValueError if there is no ":" or the part before the tag has no "/". Does not
    # handle digest ("@sha256:") references specially. Side effects: none.
    """Split `host[/namespace]/<image>:<tag>` into (registry_prefix, image, tag).

    Inverse of `build_image_ref` minus the slugify step. Useful for the
    push-time rewrite that needs to look up the source `(repo, commit)` for
    a manifest-already-exists check.
    """
    if ":" not in ref:
        raise ValueError(f"ref missing tag: {ref!r}")
    base, _, tag = ref.rpartition(":")
    # registry_prefix = everything before the final path component
    if "/" not in base:
        raise ValueError(f"ref missing host or namespace: {ref!r}")
    registry_prefix, _, image = base.rpartition("/")
    return registry_prefix, image, tag
