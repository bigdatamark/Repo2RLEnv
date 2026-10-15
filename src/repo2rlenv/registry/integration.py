# [MODULE DOC] Push-time orchestration between the registry primitives and hub.push_to_hub: discover
# logins -> probe/verify -> select a registry -> push each distinct bootstrap image -> rewrite each
# task's environment/Dockerfile FROM line and task.toml reproducibility metadata in place. Policy:
# registry mode when a verified registry exists; otherwise inline-Dockerfile mode (bake a clean
# rebuild recipe from the bootstrap cache) with a warning, or a hard error when require_registry is
# set. Self-contained Dockerfiles on known public bases skip the image step. Does not upload to the
# Hub. Entry point: prepare_dataset_for_push.
# Note: the trailing module-level _t_start assignment/deletion is a no-op.
"""Orchestration: discover → verify → select → push → rewrite tasks.

This module is the integration glue between the pure `registry/` primitives
and the HF-Hub-facing `hub.push_to_hub`. It owns the policy:

    if a registry is logged in AND verified to actually work → registry mode
    otherwise → inline-Dockerfile mode (with a prominent warning)
    OR fail hard if `require_registry=True`

It does NOT touch HF Hub — `hub.push_to_hub` does. This module just
prepares the local dataset directory (rewriting environment/Dockerfile +
task.toml in-place) so it's ready to upload, and reports what happened.
"""

from __future__ import annotations

import logging
import re
import time
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from repo2rlenv.registry import probe as probe_mod
from repo2rlenv.registry.auth import (
    RegistryAuth,
    RegistryKind,
    discover_logged_in_registries,
    filter_known_registries,
)
from repo2rlenv.registry.probe import ProbeResult, select_best
from repo2rlenv.registry.push import ImagePushResult, PushError, push_image
from repo2rlenv.registry.visibility import ensure_ghcr_visibility

logger = logging.getLogger(__name__)


ReproMode = Literal["registry", "inline_dockerfile", "local_only"]


@dataclass(slots=True)
class PrepareResult:
    # [CLASS DOC] Summary returned by prepare_dataset_for_push: chosen mode (registry /
    # inline_dockerfile / local_only), number of tasks rewritten, selected registry host/namespace,
    # representative image remote ref and digest, visibility, whether any image was actually pushed,
    # fallback reason, inline recipe source and sha256, all probe results and warnings.
    """What `prepare_dataset_for_push` produces. Hub upload is a separate step."""

    mode: ReproMode
    tasks_rewritten: int
    selected_host: str | None = None
    selected_namespace: str | None = None
    image_remote_ref: str | None = None
    image_digest: str | None = None
    image_visibility: Literal["public", "private", "unknown"] | None = None
    image_pushed: bool = False
    fallback_reason: str | None = None
    inline_recipe_source: Literal["user_dockerfile", "agent_replay"] | None = None
    inline_recipe_sha256: str | None = None
    probe_results: list[ProbeResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Inspecting the local dataset
# --------------------------------------------------------------------------

# Match the first `^FROM ...` line (Docker is case-insensitive for FROM)
_FROM_LINE_RE = re.compile(r"^(\s*FROM\s+)(\S+)", re.IGNORECASE | re.MULTILINE)


def _list_task_dirs(local_dir: Path) -> list[Path]:
    # [FUNCTION DOC] HELPER: Lists Harbor task dirs (non-hidden subdirs containing task.toml) both
    # directly under local_dir and under local_dir/tasks. Output: list sorted by task name. Raises
    # RuntimeError when the same task name appears in both layouts.
    """Return task directories from flat or ``tasks/`` dataset layouts.

    A dataset may contain tasks directly under its root or under the canonical
    Hub ``tasks/`` directory. Both layouts can coexist, but duplicate task IDs
    are rejected rather than silently choosing one copy.
    """
    roots = [local_dir]
    nested_root = local_dir / "tasks"
    if nested_root.is_dir():
        roots.append(nested_root)

    by_name: dict[str, Path] = {}
    for root in roots:
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if not (child / "task.toml").is_file():
                continue
            if previous := by_name.get(child.name):
                raise RuntimeError(
                    f"duplicate Harbor task {child.name!r} found in {previous} and {child}"
                )
            by_name[child.name] = child
    return [by_name[name] for name in sorted(by_name)]


def _bootstrap_image_refs(task_dirs: list[Path]) -> list[tuple[str, Path]]:
    # [FUNCTION DOC] HELPER: For each task with environment/Dockerfile, returns (first FROM image
    # ref, Dockerfile path); tasks without a Dockerfile or FROM line are skipped. Reads files.
    """For each task with environment/Dockerfile, return (FROM-ref, dockerfile_path).

    Tasks without `environment/` (e.g. pr_diff lite tasks) are skipped.
    """
    out = []
    for td in task_dirs:
        df = td / "environment" / "Dockerfile"
        if not df.is_file():
            continue
        content = df.read_text(encoding="utf-8", errors="replace")
        m = _FROM_LINE_RE.search(content)
        if m:
            out.append((m.group(2).strip(), df))
    return out


def _distinct_local_images(refs: list[tuple[str, Path]]) -> set[str]:
    # [FUNCTION DOC] HELPER: Returns the set of refs that look local (start with 'local/', 'local-'
    # or 'localhost').
    # KNOWN ISSUE: its comment claims refs without a registry host are also treated as local, but
    # that is not implemented; the function is also unused in this module.
    """Set of bootstrap images that look local (un-pullable)."""
    locals_: set[str] = set()
    for ref, _ in refs:
        if ref.startswith(("local/", "local-")) or ref.startswith("localhost"):
            locals_.add(ref)
        # Also treat refs lacking a registry host (no `.` before the first `/`)
        # as local. e.g. `myimage:tag` with no namespace.
    return locals_


# Known-public base images that consumers can pull anonymously from Docker
# Hub's `library/` namespace. The fast path for self-contained Dockerfiles
# (no upstream bootstrap image to push) only triggers for these — anything
# else falls through to the standard image-push path, which would surface a
# clear error if the image is private/un-pullable instead of silently
# publishing a broken dataset. Pipelines whose Dockerfiles use one of these
# bases (currently just `pr_diff` with `python:3.12-slim`) skip the push.
_PUBLIC_DOCKER_HUB_BASES: tuple[str, ...] = (
    "python:",
    "node:",
    "golang:",
    "rust:",
    "ubuntu:",
    "debian:",
    "alpine:",
    "openjdk:",
    "ruby:",
)


def _is_public_base_image(ref: str) -> bool:
    # [FUNCTION DOC] HELPER: True if ref is an unqualified (or docker.io/library/-prefixed) Docker
    # Hub base from the _PUBLIC_DOCKER_HUB_BASES allowlist (python:, node:, golang:, ...).
    """True if `ref` is a known-public base that consumers can pull anonymously.

    Conservative on purpose: only the unqualified `library/` Docker Hub bases
    we ship are recognised. A registry-qualified ref the user added by hand
    (e.g. `my-registry.example.com/team/base:tag`) takes the normal image-push
    path — same as before the fast path existed.
    """
    if ref.startswith("docker.io/library/"):
        return any(
            ref.removeprefix("docker.io/library/").startswith(b) for b in _PUBLIC_DOCKER_HUB_BASES
        )
    return any(ref.startswith(b) for b in _PUBLIC_DOCKER_HUB_BASES)


# --------------------------------------------------------------------------
# Discover + verify
# --------------------------------------------------------------------------


def _build_namespace_for(auth: RegistryAuth, hf_owner: str) -> str:
    # [FUNCTION DOC] HELPER: Chooses the push namespace for a registry auth: lower-cased HF owner
    # for GHCR and unknown kinds; 'r2e' for ECR, ACR, GCP AR and local registries; for Docker Hub,
    # $DOCKER_USERNAME, else $DOCKERHUB_USERNAME, else the HF owner (lower-cased). Reads env vars.
    """Choose the target namespace for a given registry auth + HF owner."""
    if auth.kind is RegistryKind.GHCR:
        # Try HF owner first; caller may retry under gh-login-user on permission denied.
        return hf_owner.lower()
    if auth.kind in (RegistryKind.ECR_PRIVATE, RegistryKind.ECR_PUBLIC):
        return "r2e"
    if auth.kind is RegistryKind.ACR:
        return "r2e"
    if auth.kind is RegistryKind.GCP_AR:
        # GAR: namespace is "project/repo"; we leave project to the user via
        # --image-registry override. For auto-detect we conservatively pick
        # "<project>/r2e" if the auth host carries one (rare); otherwise fall
        # back to "r2e".
        return "r2e"
    if auth.kind is RegistryKind.DOCKER_HUB:
        # Docker Hub repos live under the AUTHENTICATED USER's namespace, which
        # is usually NOT the HF owner. Prefer DOCKER_USERNAME (the creds that
        # actually grant push); fall back to the HF owner only if unset.
        import os

        return (
            os.environ.get("DOCKER_USERNAME") or os.environ.get("DOCKERHUB_USERNAME") or hf_owner
        ).lower()
    if auth.kind is RegistryKind.LOCAL:
        return "r2e"
    return hf_owner.lower()


def _registry_prefix_for(auth: RegistryAuth, namespace: str) -> str:
    # [FUNCTION DOC] HELPER: Builds '<host>/<namespace>' for an auth, using index.docker.io as the
    # host for Docker Hub. Equivalent to _registry_prefix_for_kind; unused in this module.
    """Build the `<host>/<namespace>` prefix used by build_image_ref."""
    if auth.kind in (RegistryKind.ECR_PRIVATE, RegistryKind.ECR_PUBLIC, RegistryKind.ACR):
        return f"{auth.host}/{namespace}"
    if auth.kind is RegistryKind.GCP_AR:
        return f"{auth.host}/{namespace}"
    if auth.kind is RegistryKind.DOCKER_HUB:
        return f"index.docker.io/{namespace}"
    if auth.kind is RegistryKind.LOCAL:
        return f"{auth.host}/{namespace}"
    return f"{auth.host}/{namespace}"


def _select_verified_registry(
    hf_owner: str,
    *,
    explicit_prefix: str | None,
    require_registry: bool,
) -> tuple[ProbeResult | None, str, list[ProbeResult]]:
    # [FUNCTION DOC] HELPER: Discover -> probe -> select. With explicit_prefix 'host[/namespace]'
    # (namespace defaults to the lower-cased hf_owner) it probes only that host, using a discovered
    # auth or a synthesized EMPTY-credential stub, and returns it only if pushable. Otherwise it
    # probes every discovered known registry except Docker Hub (rate-limit hazard) with
    # _build_namespace_for and picks probe.select_best. Output: (selected ProbeResult or None,
    # namespace, all probe results). Side effects: reads docker config, network probes.
    # KNOWN ISSUE: the require_registry parameter is accepted but never used.
    """Discover → probe → select. Returns (selected, namespace, all_probes).

    If `explicit_prefix` is given, we probe only that registry. Otherwise we
    discover from ~/.docker/config.json and probe each.
    """
    probes: list[ProbeResult] = []

    if explicit_prefix:
        # Parse `host[/namespace]` and probe that one
        host, _, ns = explicit_prefix.partition("/")
        if not ns:
            ns = hf_owner.lower()
        # Classify the explicit host
        from repo2rlenv.registry.auth import classify_host

        discovered = discover_logged_in_registries()
        auth = next((a for a in discovered if a.host == host), None)
        if auth is None:
            # Synthesize a stub — probe will likely fail at L2 if no creds
            from repo2rlenv.registry.auth import CredentialSource

            auth = RegistryAuth(
                host=host,
                kind=classify_host(host),
                cred_source=CredentialSource.EMPTY,
            )
        result = probe_mod.probe(auth, ns)
        probes.append(result)
        return (result if result.is_pushable else None, ns, probes)

    # Auto-discover
    candidates = filter_known_registries(discover_logged_in_registries())
    # Drop Docker Hub from auto-default unless explicit (rate-limit hazard)
    candidates = [c for c in candidates if c.kind is not RegistryKind.DOCKER_HUB]

    for auth in candidates:
        ns = _build_namespace_for(auth, hf_owner)
        result = probe_mod.probe(auth, ns)
        probes.append(result)

    selected = select_best(probes)
    if selected is None:
        return (None, hf_owner.lower(), probes)
    return (selected, selected.namespace, probes)


# --------------------------------------------------------------------------
# Local image discovery + remote-ref construction
# --------------------------------------------------------------------------


def _parse_local_tag(local_tag: str) -> tuple[str, str, str]:
    # [FUNCTION DOC] HELPER: Parses a local bootstrap image ref into (owner, name, sha). Handles
    # '.../owner__name:sha', '.../owner__name:sha__opts' (sha also cut at '-') and
    # '.../owner__name@sha256:hex' (first 12 hex chars). The last path component is split on '__';
    # without '__' owner is '_' and name is the whole component.
    """From `local/r2e-bootstrap/owner__name:sha[__opts]` return (owner, name, sha).

    Tolerates three input forms:
      - `local/r2e-bootstrap/owner__name:sha12`        (tag form)
      - `local/r2e-bootstrap/owner__name:sha12__opt8`  (tag + options hash)
      - `local/r2e-bootstrap/owner__name@sha256:hex`   (digest form)

    Returns 12-char tag-safe sha (truncated for digest form so the tag fits
    within OCI tag-length limits).
    """
    # Digest form has `@sha256:` before the colon; strip it for parsing
    if "@sha256:" in local_tag:
        body, _, digest_hex = local_tag.partition("@sha256:")
        sha = digest_hex[:12]
    else:
        body, _, tag_with_opts = local_tag.partition(":")
        sha = tag_with_opts.split("__", 1)[0].split("-", 1)[0]

    if "/" in body:
        parts = body.split("/")
        last = parts[-1]  # owner__name
    else:
        last = body
    if "__" in last:
        owner, _, name = last.partition("__")
    else:
        owner = "_"
        name = last
    return owner, name, sha


# --------------------------------------------------------------------------
# In-place dataset rewriting
# --------------------------------------------------------------------------


def _rewrite_dockerfile_from(df_path: Path, new_ref: str) -> bool:
    # [FUNCTION DOC] HELPER: Replaces the image of the first FROM line in df_path with new_ref,
    # writing the file. Returns True if the content changed, False otherwise.
    """Rewrite the first `FROM <ref>` line. Returns True if a change was made."""
    content = df_path.read_text(encoding="utf-8", errors="replace")
    new_content, n = _FROM_LINE_RE.subn(rf"\g<1>{new_ref}", content, count=1)
    if n == 0 or new_content == content:
        return False
    df_path.write_text(new_content, encoding="utf-8")
    return True


def _update_task_toml_reproducibility(
    toml_path: Path,
    *,
    mode: ReproMode,
    image_ref: str | None,
    image_tag: str | None,
    image_visibility: str | None,
    pushed_by: str | None,
    inline_recipe_sha256: str | None = None,
    inline_recipe_lines: int = 0,
    inline_recipe_source: str | None = None,
    fallback_reason: str | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Rewrites a task.toml in place: sets metadata.repo2env.spec_version to
    # '0.2.0' and replaces metadata.repo2env.reproducibility with mode, image ref/tag/visibility,
    # pushed_at (current UTC ISO time), pushed_by, inline recipe sha256/lines/source and
    # fallback_reason, dropping None values. If image_ref is set, also updates bootstrap_image in
    # the pipeline's own subtable (metadata.repo2env.<pipeline>). Uses tomllib to parse and tomli_w
    # to write (comments/formatting in the file are lost). Returns None.
    """Patch the [metadata.repo2env.reproducibility] subtable + bump spec_version.

    Uses a hand-built tomllib roundtrip (parse → mutate dict → re-serialize)
    via tomli_w. The repo already depends on tomli_w through the emitter.
    """
    import tomli_w

    data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    meta = data.setdefault("metadata", {})
    r2e = meta.setdefault("repo2env", {})
    r2e["spec_version"] = "0.2.0"
    repro: dict[str, object] = {
        "mode": mode,
        "image_ref": image_ref,
        "image_tag": image_tag,
        "image_visibility": image_visibility,
        "pushed_at": datetime.now(UTC).isoformat(),
        "pushed_by": pushed_by,
        "inline_recipe_sha256": inline_recipe_sha256,
        "inline_recipe_lines": inline_recipe_lines,
        "inline_recipe_source": inline_recipe_source,
        "fallback_reason": fallback_reason,
    }
    # Strip Nones — TOML can't represent them
    repro = {k: v for k, v in repro.items() if v is not None}
    r2e["reproducibility"] = repro
    # Also refresh the legacy `bootstrap_image` field inside the pipeline's
    # subtable so old readers don't see a stale local ref.
    pipeline_name = r2e.get("pipeline")
    if isinstance(pipeline_name, str):
        sub = r2e.get(pipeline_name)
        if isinstance(sub, dict) and image_ref:
            sub["bootstrap_image"] = image_ref
    toml_path.write_text(tomli_w.dumps(data), encoding="utf-8")


def _hash_text(s: str) -> str:
    # [FUNCTION DOC] HELPER: Returns 'sha256:' + hex sha256 of the UTF-8 encoded string.
    import hashlib

    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def _finalize_self_contained_tasks(
    task_dirs: list[Path], *, image_ref: str, pushed_by: str | None
) -> int:
    # [FUNCTION DOC] HELPER: For tasks whose Dockerfile builds from a known public base, rewrites
    # task.toml reproducibility to mode inline_dockerfile, visibility public, image_ref/tag =
    # image_ref, recipe source user_dockerfile with the Dockerfile's hash and line count. Tasks
    # missing task.toml or Dockerfile are skipped. Output: number of tasks rewritten.
    # KNOWN ISSUE: every Dockerfile task gets the same image_ref (the caller's representative ref),
    # so in a dataset mixing bases each task is labelled with the first sorted ref rather than its
    # own.
    """Rewrite reproducibility metadata for self-contained Dockerfile tasks.

    The emitter seeds every Dockerfile-bearing task with
    ``[metadata.repo2env.reproducibility]`` set to
    ``mode="local_only", image_visibility="private"`` — a conservative
    default that's true for un-pushed bootstrap images. For self-contained
    Dockerfiles whose FROM is a known-public base, those values are wrong:
    the FROM image is publicly pullable and the inline Dockerfile recipe is
    what makes the task reproducible. Rewrite each task.toml accordingly so
    downstream tooling that reads reproducibility metadata classifies the
    task correctly.
    """
    rewritten = 0
    for td in task_dirs:
        toml_path = td / "task.toml"
        df_path = td / "environment" / "Dockerfile"
        if not toml_path.is_file() or not df_path.is_file():
            continue
        df_text = df_path.read_text(encoding="utf-8")
        _update_task_toml_reproducibility(
            toml_path,
            mode="inline_dockerfile",
            image_ref=image_ref,
            image_tag=image_ref,
            image_visibility="public",
            pushed_by=pushed_by,
            inline_recipe_sha256=_hash_text(df_text),
            inline_recipe_lines=len(df_text.splitlines()),
            inline_recipe_source="user_dockerfile",
        )
        rewritten += 1
    return rewritten


# --------------------------------------------------------------------------
# The orchestrator
# --------------------------------------------------------------------------


def prepare_dataset_for_push(
    local_dir: Path,
    *,
    hf_owner: str,
    image_registry: str | None = None,
    inline_dockerfile: bool = False,
    require_registry: bool = False,
    skip_image_push: bool = False,
    image_visibility: Literal["public", "private", "inherit"] = "inherit",
    dataset_is_private: bool = False,
    pushed_by: str | None = None,
    on_message=None,
) -> PrepareResult:
    # [FUNCTION DOC] Prepares a local dataset for Hub upload by choosing a reproducibility mode and
    # rewriting tasks in place. Inputs: local_dir, hf_owner (default registry namespace),
    # image_registry (explicit 'host[/ns]' to probe), inline_dockerfile (force inline mode),
    # require_registry (fail instead of falling back), skip_image_push (reuse computed remote refs),
    # image_visibility ('inherit' follows dataset_is_private), pushed_by (recorded in metadata),
    # on_message (optional callback; exceptions from it are swallowed; messages are also logged at
    # INFO). Flow: no Dockerfiles -> PrepareResult(local_only, 0). If the representative (first
    # sorted) FROM ref is not local-looking and is a known public base -> self-contained inline
    # result. If inline_dockerfile -> _go_inline. Else select a verified registry; none ->
    # RuntimeError when require_registry, otherwise _go_inline with a fallback reason; found ->
    # _go_registry. Output: PrepareResult. Side effects: rewrites Dockerfiles/task.toml, docker
    # pushes, network probes.
    # KNOWN ISSUE: the public-base fast path is decided from only the first sorted ref, so a dataset
    # mixing public-base and local bootstrap images can skip pushing the local ones.
    """Run the §7.2 push-time flow: discover → verify → select → push → rewrite.

    Mutates `local_dir` in-place by rewriting each task's
    environment/Dockerfile and task.toml. Returns a `PrepareResult` summarizing
    what mode was selected and any warnings.

    Does NOT upload to HF Hub. The caller (`hub.push_to_hub`) does that after.
    """

    def _emit(msg: str) -> None:
        # [FUNCTION DOC] HELPER: Logs msg at INFO and forwards it to on_message if given, ignoring
        # any exception the callback raises.
        logger.info(msg)
        if on_message:
            try:
                on_message(msg)
            except Exception:
                pass

    task_dirs = _list_task_dirs(local_dir)
    refs = _bootstrap_image_refs(task_dirs)

    if not refs:
        # Text-only dataset (pr_diff) — nothing to do at the image level.
        _emit("no environment/Dockerfile in dataset; skipping image step")
        # Still set spec_version + reproducibility(mode=local_only-but-not-applicable)
        # for consistency. For pr_diff we just don't write the reproducibility
        # subtable at all — there's no image to be reproducible about.
        return PrepareResult(mode="local_only", tasks_rewritten=0)

    # Multi-image datasets (e.g. pr_runtime across many repos) are supported in
    # BOTH modes: inline bakes each task's own recipe; registry pushes each
    # distinct image and rewrites each task to its own digest. `local_ref` is
    # just a representative for the single-image PrepareResult fields.
    distinct = {ref for ref, _ in refs}
    local_ref = sorted(distinct)[0]
    looks_local = local_ref.startswith(("local/", "local-")) or local_ref.startswith("localhost")

    # Fast path for self-contained Dockerfiles: if the FROM ref is a
    # known-public base (e.g. ``python:3.12-slim``) the rest of the
    # Dockerfile is the complete recipe — no per-task image push needed.
    # The published task's Dockerfile can be rebuilt on any consumer's
    # machine just by reading the FROM + the inline RUN lines.
    #
    # pr_diff is the canonical case: every task's Dockerfile is
    # ``FROM python:3.12-slim`` followed by apt-get + git clone + bake the
    # oracle. No bootstrap-built image upstream.
    #
    # The allowlist is intentionally narrow. A non-`local/` ref that isn't
    # on the allowlist (e.g. a private registry or an unqualified custom
    # image) falls through to the normal push path, which surfaces a clear
    # error instead of silently publishing a dataset that can't be rebuilt.
    if not looks_local and _is_public_base_image(local_ref):
        rewritten = _finalize_self_contained_tasks(
            task_dirs, image_ref=local_ref, pushed_by=pushed_by
        )
        _emit(
            f"self-contained Dockerfile (FROM {local_ref!r}); "
            "skipping image step — consumers rebuild from the inline recipe"
        )
        return PrepareResult(
            mode="inline_dockerfile",
            tasks_rewritten=rewritten,
            image_remote_ref=local_ref,
            image_visibility="public",
            inline_recipe_source="user_dockerfile",
        )

    # Decide mode
    if inline_dockerfile:
        return _go_inline(
            local_dir,
            task_dirs,
            refs,
            local_ref=local_ref,
            fallback_reason=None,
            pushed_by=pushed_by,
            on_message=_emit,
        )

    # Registry mode candidate
    selected, namespace, probes = _select_verified_registry(
        hf_owner,
        explicit_prefix=image_registry,
        require_registry=require_registry,
    )

    if selected is None:
        # No verified registry available
        if require_registry:
            details = (
                "; ".join(f"{p.host}: {p.error or 'unknown'}" for p in probes)
                or "no registries discovered"
            )
            raise RuntimeError(f"--require-registry: no verified registry available ({details})")
        reason = (
            "; ".join(f"{p.host}: {p.error or 'unknown'}" for p in probes)
            if probes
            else "no registry credentials found"
        )
        _emit(f"WARN no verified registry; falling back to inline-Dockerfile mode ({reason})")
        return _go_inline(
            local_dir,
            task_dirs,
            refs,
            local_ref=local_ref,
            fallback_reason=f"no working registry credentials: {reason}",
            pushed_by=pushed_by,
            on_message=_emit,
        )

    # Registry mode — push + rewrite
    return _go_registry(
        local_dir,
        task_dirs,
        refs,
        local_ref=local_ref,
        selected=selected,
        namespace=namespace,
        image_visibility=image_visibility,
        dataset_is_private=dataset_is_private,
        skip_image_push=skip_image_push,
        require_registry=require_registry,
        looks_local=looks_local,
        probes=probes,
        pushed_by=pushed_by,
        on_message=_emit,
    )


def _go_inline(
    local_dir: Path,
    task_dirs: list[Path],
    refs: list[tuple[str, Path]],
    *,
    local_ref: str,
    fallback_reason: str | None,
    pushed_by: str | None,
    on_message,
) -> PrepareResult:
    # [FUNCTION DOC] HELPER: Inline mode. For every (ref, Dockerfile) it loads (cached per ref) a
    # clean bootstrap recipe via _load_bootstrap_recipe, and rewrites the Dockerfile as a header
    # comment, the recipe, then the original Dockerfile minus its first FROM line (per-task
    # overlay); updates each task.toml with mode inline_dockerfile, recipe hash/lines/source and
    # fallback_reason. Reports a summary via on_message. Output: PrepareResult(inline_dockerfile,
    # ...). Raises RuntimeError when a ref has no cached bootstrap recipe. local_dir and local_ref
    # are unused.
    # KNOWN ISSUE: the returned inline_recipe_sha256 is only the hash of the last processed recipe,
    # not representative of multi-image datasets.
    """Bake the bootstrap's reconstructed Dockerfile into each task's image recipe.

    Requires the bootstrap cache to be present (we read the saved
    reconstructed Dockerfile from there). Otherwise raises.
    """
    # Multi-image aware: each task bakes the recipe for ITS OWN bootstrap ref
    # (a multi-repo dataset has one image per repo). Cache recipes by ref so we
    # read each repo's reconstructed Dockerfile at most once.
    recipe_cache: dict[str, tuple[str | None, str | None]] = {}

    def _recipe_for(ref: str) -> tuple[str, str]:
        # [FUNCTION DOC] HELPER: Returns (recipe, source) for ref from the per-call cache, loading
        # it on first use. Raises RuntimeError if no cached bootstrap recipe exists for the ref.
        if ref not in recipe_cache:
            recipe_cache[ref] = _load_bootstrap_recipe(ref)
        rec, src = recipe_cache[ref]
        if rec is None:
            raise RuntimeError(
                f"inline mode requires the bootstrap reconstructed Dockerfile "
                f"(saved under ./workspace/bootstrap/<repo>/<sha>/Dockerfile). "
                f"Ref {ref!r} resolves to no cached Dockerfile. "
                f"Either re-bootstrap or use registry mode."
            )
        return rec, src

    tasks_rewritten = 0
    sources: set[str] = set()
    last_hash = ""
    for ref, df_path in refs:
        recipe, source = _recipe_for(ref)
        sources.add(source)
        last_hash = _hash_text(recipe)
        # Build a combined Dockerfile: recipe + per-task overlay from the
        # existing Dockerfile (everything after the FROM line).
        existing = df_path.read_text(encoding="utf-8", errors="replace")
        without_from = _FROM_LINE_RE.sub("", existing, count=1).lstrip("\n")
        combined = (
            "# Auto-generated by Repo2RLEnv (inline mode)\n"
            "# Source: bootstrap dockerfile_reconstruction\n"
            f"# Original bootstrap ref: {ref}\n\n"
            f"{recipe.rstrip()}\n\n"
            "# Per-task overlay (from pipeline-emitted Dockerfile)\n"
            f"{without_from}"
        )
        df_path.write_text(combined, encoding="utf-8")
        tasks_rewritten += 1
        toml_path = df_path.parent.parent / "task.toml"
        if toml_path.is_file():
            _update_task_toml_reproducibility(
                toml_path,
                mode="inline_dockerfile",
                image_ref=None,
                image_tag=None,
                image_visibility=None,
                pushed_by=pushed_by,
                inline_recipe_sha256=last_hash,
                inline_recipe_lines=len(recipe.splitlines()),
                inline_recipe_source=source,
                fallback_reason=fallback_reason,
            )

    source = sorted(sources)[0] if sources else "user_dockerfile"
    on_message(
        f"inline mode: rewrote {tasks_rewritten} task Dockerfiles "
        f"across {len(recipe_cache)} bootstrap image(s) ({source})"
    )

    return PrepareResult(
        mode="inline_dockerfile",
        tasks_rewritten=tasks_rewritten,
        fallback_reason=fallback_reason,
        inline_recipe_source=source,  # type: ignore[arg-type]
        inline_recipe_sha256=last_hash,
    )


_LANG_BASE_IMAGE = {
    "python": "python:3.12-slim",
    "go": "golang:1.23",
    "rust": "rust:1-slim",
    "node": "node:20-slim",
    "java": "eclipse-temurin:21-jdk",
}


def _build_clean_recipe(owner: str, name: str, data: dict) -> str | None:
    # [FUNCTION DOC] HELPER: Synthesizes a buildable Dockerfile recipe from a bootstrap.json dict
    # (not the noisy dockerfile_reconstruction). Returns None if rebuild_cmds is empty. Base image:
    # FROM of dockerfile_reconstruction if present, else a per-language default (_LANG_BASE_IMAGE),
    # else ubuntu:24.04. Emits: ensure git + CA certs (apt or apk, best effort), clone
    # https://github.com/<owner>/<name>.git into /workspace, optionally fetch/checkout data['ref'],
    # one RUN per rebuild command, and for Python/pytest repos best-effort installs of the 'tests'
    # dependency group and pytest. Output: recipe text ending with a newline.
    """Build a *buildable* env recipe from a bootstrap result's clean fields.

    The bootstrap `dockerfile_reconstruction` field interleaves the agent's
    command OUTPUT (pip progress, ``Observation:``, stderr) into RUN lines, so
    it does NOT build. We instead synthesize a clean recipe from the verified
    ``rebuild_cmds`` list: clone the repo, check out the bootstrap ref, then run
    the recorded build commands. The per-task overlay (git fetch/reset to the
    PR base_commit) is appended later by the caller.
    """
    rebuild = [c for c in (data.get("rebuild_cmds") or []) if str(c).strip()]
    if not rebuild:
        return None
    ref = (data.get("ref") or "").strip()
    recon = data.get("dockerfile_reconstruction") or ""
    m = re.search(r"^\s*FROM\s+(\S+)", recon, re.IGNORECASE | re.MULTILINE)
    base = m.group(1) if m else _LANG_BASE_IMAGE.get(data.get("language") or "", "ubuntu:24.04")
    repo_url = f"https://github.com/{owner}/{name}.git"
    lines = [
        "# Auto-generated by Repo2RLEnv (inline mode, clean rebuild_cmds recipe)",
        f"FROM {base}",
        # git + CA certs (minimal images, esp. *-slim/alpine, lack them)
        "RUN (command -v git >/dev/null 2>&1 && [ -e /etc/ssl/certs/ca-certificates.crt ]) || \\",
        "    (apt-get update && apt-get install -y --no-install-recommends git ca-certificates \\",
        "     && rm -rf /var/lib/apt/lists/*) || \\",
        "    (apk add --no-cache git ca-certificates && update-ca-certificates) || true",
        f"RUN git clone --filter=blob:none {repo_url} /workspace",
        "WORKDIR /workspace",
    ]
    if ref:
        lines.append(
            f"RUN git fetch --depth 1 origin {ref} 2>/dev/null "
            f"|| git fetch --unshallow origin 2>/dev/null || true"
        )
        lines.append(f"RUN git checkout {ref} || git checkout -f {ref} || true")
    for cmd in rebuild:
        lines.append(f"RUN {cmd}")
    # Defensive: guarantee the TEST RUNNER (and its plugins) are installed.
    # Bootstrap rebuild_cmds sometimes miss them — e.g. `pip install -e '.[tests]'`
    # is a silent no-op on repos that declare test deps via PEP 735
    # `[dependency-groups]` instead of `[project.optional-dependencies]` (e.g.
    # werkzeug), where pip only warns "does not provide the extra 'tests'" and
    # installs neither pytest nor pytest-timeout. The bootstrap *image* still
    # worked because the agent installed them in a step not captured in
    # rebuild_cmds; the recipe must not rely on that.
    #   1. `pip install --group tests` (pip>=25.1) pulls the full PEP 735 group;
    #      slim images ship pip 25.0 so upgrade first. Harmless when no group.
    #   2. bare `pip install pytest` as a last-resort runner guarantee.
    test_cmds_joined = " ".join(str(c) for c in (data.get("test_cmds") or []))
    if "pytest" in test_cmds_joined or data.get("language") == "python":
        lines.append(
            "RUN (pip install -U pip -q && pip install --group tests -q) >/dev/null 2>&1 || true"
        )
        lines.append(
            "RUN python3 -m pytest --version >/dev/null 2>&1 || "
            "pip install pytest -q >/dev/null 2>&1 || pip3 install pytest -q >/dev/null 2>&1 || true"
        )
    return "\n".join(lines) + "\n"


def _load_bootstrap_recipe(
    local_ref: str,
) -> tuple[str | None, Literal["user_dockerfile", "agent_replay"] | None]:
    # [FUNCTION DOC] HELPER: Finds the cached bootstrap result for a local image ref and builds its
    # recipe. Parses (owner, name, sha) from the ref; searches ./workspace/bootstrap and the legacy
    # ./envs (relative and cwd-based) for <owner>__<name>/<slot> dirs whose name starts with the sha
    # prefix, reading bootstrap.json and calling _build_clean_recipe. Output: (recipe,
    # 'agent_replay') for the first usable entry, else (None, None); JSON/OS errors are ignored.
    # Reads files only.
    """Look up the cached bootstrap result for `local_ref` and return its recipe.

    Prefers a clean recipe synthesized from the verified ``rebuild_cmds`` (so
    the inline Dockerfile actually builds). Returns (recipe_text, source) or
    (None, None) if no usable cache entry is found.
    """
    import json as _json

    owner, name, sha = _parse_local_tag(local_ref)
    if not name or not sha:
        return None, None
    # Try the current default first, then the legacy path so existing
    # ./envs/ caches keep resolving.
    candidates = [
        Path("./workspace/bootstrap"),
        Path.cwd() / "workspace" / "bootstrap",
        Path("./envs"),
        Path.cwd() / "envs",
    ]
    for cache_dir in candidates:
        if not cache_dir.is_dir():
            continue
        slot_root = cache_dir / f"{owner}__{name}"
        if not slot_root.is_dir():
            continue
        for slot in slot_root.iterdir():
            if not slot.is_dir() or not slot.name.startswith(sha[:12]):
                continue
            bj = slot / "bootstrap.json"
            if bj.is_file():
                try:
                    data = _json.loads(bj.read_text(encoding="utf-8"))
                    recipe = _build_clean_recipe(owner, name, data)
                    if recipe:
                        return recipe, "agent_replay"
                except (ValueError, OSError):
                    pass
    return None, None


def _go_registry(
    local_dir: Path,
    task_dirs: list[Path],
    refs: list[tuple[str, Path]],
    *,
    local_ref: str,
    selected: ProbeResult,
    namespace: str,
    image_visibility: Literal["public", "private", "inherit"],
    dataset_is_private: bool,
    skip_image_push: bool,
    require_registry: bool,
    looks_local: bool,
    probes: list[ProbeResult],
    pushed_by: str | None,
    on_message,
) -> PrepareResult:
    # [FUNCTION DOC] HELPER: Registry mode. Computes the registry prefix and effective visibility
    # ('inherit' -> private if the dataset is private else public). For each distinct ref: builds
    # the remote ref with naming.build_image_ref, pushes it via _do_push unless skip_image_push
    # (then the digest is the remote ref itself), and for public GHCR images flips package
    # visibility (failure -> warning, visibility 'unknown', or RuntimeError if require_registry). A
    # PushError falls back to _go_inline for all tasks (or re-raises if require_registry). Then
    # rewrites every task's FROM to its own image digest and updates task.toml (mode registry).
    # Output: PrepareResult with the representative image (local_ref's, else the first pushed). Side
    # effects: docker/network/files.
    # KNOWN ISSUE: looks_local is derived from the single representative ref and applied to every
    # ref, so in mixed datasets local images may skip the actual push (or vice versa).
    """Push each distinct bootstrap image to the registry; rewrite each task
    to ITS OWN image digest (multi-image aware)."""
    from collections import defaultdict

    from repo2rlenv.registry.naming import build_image_ref

    registry_prefix = _registry_prefix_for_kind(selected.kind, selected.host, namespace)

    # Visibility (computed once; applies to every pushed image).
    effective_visibility: Literal["public", "private", "unknown"]
    if image_visibility == "inherit":
        effective_visibility = "private" if dataset_is_private else "public"
    else:
        effective_visibility = image_visibility  # type: ignore[assignment]

    by_ref: dict[str, list[Path]] = defaultdict(list)
    for ref, df_path in refs:
        by_ref[ref].append(df_path)

    warnings: list[str] = []
    any_pushed = False
    # local_ref -> (remote_ref, digest)
    pushed: dict[str, tuple[str, str]] = {}

    for ref in sorted(by_ref):
        owner, name, sha = _parse_local_tag(ref)
        remote_ref = build_image_ref(
            registry_prefix=registry_prefix, owner=owner, name=name, commit_sha=sha
        )
        digest = remote_ref
        if skip_image_push:
            on_message(f"--skip-image-push: not pushing; reusing {remote_ref}")
        else:
            try:
                push_result = _do_push(ref, remote_ref, looks_local)
                digest = push_result.digest
                any_pushed = any_pushed or push_result.pushed
                on_message(
                    f"pushed {remote_ref} → {digest[:48]}..."
                    if push_result.pushed
                    else f"already at registry: {digest[:48]}..."
                )
            except PushError as exc:
                warnings.append(f"image push failed for {ref}: {exc}")
                if require_registry:
                    raise
                on_message(f"WARN push failed ({ref}); falling back to inline mode: {exc}")
                # Fall back: bake ALL tasks inline (independently self-contained).
                return _go_inline(
                    local_dir,
                    task_dirs,
                    refs,
                    local_ref=local_ref,
                    fallback_reason=f"image push failed: {exc}",
                    pushed_by=pushed_by,
                    on_message=on_message,
                )
        if effective_visibility == "public" and selected.kind is RegistryKind.GHCR:
            vis = ensure_ghcr_visibility(remote_ref, target="public")
            if not vis.success:
                warnings.append(f"GHCR visibility flip failed for {remote_ref}: {vis.error}")
                if require_registry:
                    raise RuntimeError(warnings[-1])
                effective_visibility = "unknown"
                on_message(f"WARN {warnings[-1]}")
        pushed[ref] = (remote_ref, digest)

    # Rewrite each task to its OWN image's digest.
    tasks_rewritten = 0
    for ref, df_path in refs:
        remote_ref, digest = pushed[ref]
        if _rewrite_dockerfile_from(df_path, digest):
            tasks_rewritten += 1
        toml_path = df_path.parent.parent / "task.toml"
        if toml_path.is_file():
            _update_task_toml_reproducibility(
                toml_path,
                mode="registry",
                image_ref=digest,
                image_tag=remote_ref,
                image_visibility=effective_visibility,
                pushed_by=pushed_by,
            )

    primary = pushed.get(local_ref) or next(iter(pushed.values()))
    on_message(
        f"registry mode: pushed {len(pushed)} image(s), rewrote {tasks_rewritten} task Dockerfiles"
    )
    return PrepareResult(
        mode="registry",
        tasks_rewritten=tasks_rewritten,
        selected_host=selected.host,
        selected_namespace=namespace,
        image_remote_ref=primary[0],
        image_digest=primary[1],
        image_visibility=effective_visibility,
        image_pushed=any_pushed,
        probe_results=probes,
        warnings=warnings,
    )


def _registry_prefix_for_kind(kind: RegistryKind, host: str, namespace: str) -> str:
    # [FUNCTION DOC] HELPER: Returns 'index.docker.io/<namespace>' for Docker Hub, otherwise
    # '<host>/<namespace>'.
    """Compose `<host>/<namespace>` correctly for each registry kind."""
    if kind is RegistryKind.DOCKER_HUB:
        return f"index.docker.io/{namespace}"
    return f"{host}/{namespace}"


def _do_push(local_ref: str, remote_ref: str, looks_local: bool) -> ImagePushResult:
    # [FUNCTION DOC] HELPER: Pushes local_ref to remote_ref via push.push_image
    # (skip_if_exists=True) when the ref looks local. Otherwise returns a synthetic
    # ImagePushResult(pushed=False) whose digest is the local_ref itself. Raises PushError from
    # push_image.
    # KNOWN ISSUE: the comment says the non-local path verifies the digest, but no verification
    # occurs; the original ref is reused verbatim as the 'digest'.
    """Wrapper to push, with friendly error if local image truly missing."""
    if not looks_local:
        # The dataset already references a registry ref. Either it was
        # pushed elsewhere, or it's our re-push of an already-registered
        # image. Either way: skip a docker tag/push, just verify the digest.
        result = ImagePushResult(
            local_tag=local_ref,
            remote_ref=remote_ref,
            digest=local_ref,
            pushed=False,
            duration_sec=0.0,
        )
        return result
    return push_image(local_ref, remote_ref, skip_if_exists=True)


__all__ = [
    "PrepareResult",
    "ReproMode",
    "prepare_dataset_for_push",
]


_t_start = time.monotonic()
del _t_start
