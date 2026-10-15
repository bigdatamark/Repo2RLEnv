# [MODULE DOC] Pushes a local Docker image to a remote OCI registry by shelling out to the docker
# CLI (no docker-py dependency) and resolves the pushed image's registry-qualified digest from
# RepoDigests. Idempotent: optionally skips the push when "docker manifest inspect" finds the ref
# already exists. Public API: push_image, manifest_exists, ImagePushResult, PushError. Side effects:
# docker subprocesses, network traffic to the registry, logging.
"""Push a local docker image to a remote OCI registry.

Wraps the `docker` CLI rather than docker-py because:

  1. docker-py is an extra runtime dep we don't otherwise need.
  2. The push wire-format is identical across all OCI-compliant registries;
     we just need to invoke `docker push <ref>` and parse RepoDigests.

The push step is idempotent at the registry level: re-pushing the same
local image to the same tag is a no-op for the layers that already exist,
and `RepoDigests` resolves to the same `sha256` content hash.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PushError(Exception):
    # [CLASS DOC] Exception raised by push_image when docker is unavailable, the local image is
    # missing, or docker tag/push fails. The message includes truncated docker stderr.
    """Raised when `docker push` fails irrecoverably."""


@dataclass(slots=True)
class ImagePushResult:
    # [CLASS DOC] Dataclass (slots) describing a push outcome: local_tag; remote_ref (target tag
    # ref); digest (registry-qualified "host/path@sha256:..." or, if unresolvable, the tag ref
    # itself); pushed (False when skipped because the manifest already existed); duration_sec
    # (rounded to 0.01s).
    local_tag: str
    remote_ref: str
    digest: str  # full `host/path@sha256:...`
    pushed: bool  # False if "already at registry, skipped"
    duration_sec: float


def _docker_available() -> bool:
    # [FUNCTION DOC] HELPER: True if a "docker" executable is on PATH.
    return shutil.which("docker") is not None


def _run(args: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    # [FUNCTION DOC] HELPER: Runs a command (list of args) via subprocess with captured UTF-8 text
    # output, timeout seconds (default 1800), never raising on non-zero exit. Output:
    # CompletedProcess. Raises subprocess.TimeoutExpired on timeout.
    """Run a docker subcommand, capturing output and never raising on nonzero."""
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )


def manifest_exists(remote_ref: str) -> bool:
    # [FUNCTION DOC] Returns True iff "docker manifest inspect <remote_ref>" exits 0 (60s timeout),
    # i.e. the registry already has that manifest. Returns False when docker is not on PATH. Side
    # effects: docker subprocess / registry network call.
    """`docker manifest inspect <ref>` returns 0 iff the registry already has it.

    Used for the idempotent-re-push optimization: skip the push entirely if
    the image is already at the target.
    """
    if not _docker_available():
        return False
    proc = _run(
        ["docker", "manifest", "inspect", remote_ref],
        timeout=60,
    )
    return proc.returncode == 0


def _resolve_repo_digest(remote_ref: str) -> str | None:
    # [FUNCTION DOC] HELPER: Finds the registry-qualified digest for remote_ref via "docker image
    # inspect <ref> --format {{json .RepoDigests}}". Ignores entries starting with "local/" or
    # "local-". Returns the first digest prefixed by remote_ref's host component; otherwise the
    # first digest whose bare path (remote_ref minus host, minus tag) matches "<path>@", prefixed
    # with the host when the host is a Docker Hub alias (index.docker.io, registry-1.docker.io,
    # docker.io). Output: str or None (inspect failure, invalid JSON, empty or no match). Side
    # effect: docker subprocess.
    # KNOWN ISSUE: the path/tag split uses the first "/" and first ":" after it, so a ref without a
    # host (e.g. "user/img:tag") treats the user as the host, and the Docker Hub branch only applies
    # when the host is literally one of the listed aliases.
    """Return the registry-qualified digest for `remote_ref`, or None.

    `docker image inspect <ref>` exposes RepoDigests AFTER a push (or pull).
    We pick the digest whose path component matches `remote_ref`'s path,
    not the host — Docker Hub returns digests as `<user>/<image>@sha256:...`
    (no `index.docker.io/` prefix) since it's the default registry, while
    GHCR / ECR / etc. return the full `<host>/<path>@...` form.
    """
    proc = _run(
        ["docker", "image", "inspect", remote_ref, "--format", "{{json .RepoDigests}}"],
        timeout=30,
    )
    if proc.returncode != 0:
        return None
    try:
        digests = json.loads(proc.stdout.strip() or "[]")
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(digests, list) or not digests:
        return None

    # The path component of remote_ref (everything after the host) — this is
    # what Docker Hub digests use as the full identifier.
    remote_path = remote_ref.split("/", 1)[1] if "/" in remote_ref else remote_ref
    remote_path_no_tag = remote_path.split(":", 1)[0]
    host_prefix = remote_ref.split("/", 1)[0]

    # Build expected match prefixes: try host-qualified first, then bare path
    # (Docker Hub). Skip `local/...` entries — those are un-pullable.
    candidates: list[str] = []
    for d in digests:
        if not isinstance(d, str):
            continue
        if d.startswith("local/") or d.startswith("local-"):
            continue
        if d.startswith(f"{host_prefix}/"):
            return d  # exact host match wins
        if d.startswith(f"{remote_path_no_tag}@"):
            candidates.append(d)  # bare-path match (Docker Hub style)

    if candidates:
        # For Docker Hub, prefix with the canonical host so the consumer's
        # `docker pull` resolves unambiguously.
        d = candidates[0]
        if host_prefix in ("index.docker.io", "registry-1.docker.io", "docker.io"):
            return f"{host_prefix}/{d}"
        return d
    return None


def push_image(
    local_tag: str,
    remote_ref: str,
    *,
    timeout: int = 1800,
    skip_if_exists: bool = True,
) -> ImagePushResult:
    # [FUNCTION DOC] Tags local_tag as remote_ref, pushes it, and resolves the digest. Inputs:
    # local_tag, remote_ref; keyword timeout (push timeout seconds, default 1800) and skip_if_exists
    # (default True). Steps: require docker on PATH; verify the local image exists via image
    # inspect; if skip_if_exists and manifest_exists, return pushed=False with the resolved digest
    # (or the ref); else docker tag, docker push, then resolve the digest, falling back to the tag
    # ref with a logged warning. Output: ImagePushResult. Raises PushError (no docker, local image
    # missing, tag or push failure) and subprocess.TimeoutExpired. Side effects: docker
    # subprocesses, registry upload, logging.
    # KNOWN ISSUE: the docstring says a non-running docker daemon raises PushError; it does, but via
    # the image-inspect failure, so the message misleadingly reports the local image as not found.
    """Tag → push → resolve digest. Returns the canonical registry-qualified ref.

    Raises `PushError` if the local image isn't present, the docker daemon
    isn't running, or the push itself fails.
    """
    import time

    if not _docker_available():
        raise PushError("docker CLI not available on PATH")

    start = time.monotonic()

    # 1. Verify the local image exists
    inspect = _run(
        ["docker", "image", "inspect", local_tag, "--format", "{{.Id}}"],
        timeout=30,
    )
    if inspect.returncode != 0:
        raise PushError(
            f"local image {local_tag!r} not found. "
            "Run `repo2rlenv bootstrap` first, or pass --inline-dockerfile."
        )

    # 2. Optionally skip if already at registry
    if skip_if_exists and manifest_exists(remote_ref):
        logger.info("manifest already at %s; skipping push", remote_ref)
        digest = _resolve_repo_digest(remote_ref) or remote_ref
        return ImagePushResult(
            local_tag=local_tag,
            remote_ref=remote_ref,
            digest=digest,
            pushed=False,
            duration_sec=round(time.monotonic() - start, 2),
        )

    # 3. Tag local → remote
    tag_proc = _run(["docker", "tag", local_tag, remote_ref], timeout=60)
    if tag_proc.returncode != 0:
        raise PushError(
            f"docker tag {local_tag} {remote_ref} failed: {tag_proc.stderr.strip()[:400]}"
        )

    # 4. Push
    push_proc = _run(["docker", "push", remote_ref], timeout=timeout)
    if push_proc.returncode != 0:
        # Surface the actual docker error for the caller to classify (auth
        # denied vs. network vs. repo-doesn't-exist for ECR/GAR).
        raise PushError(f"docker push {remote_ref} failed:\n{push_proc.stderr.strip()[:1000]}")

    # 5. Resolve the canonical digest from RepoDigests
    digest = _resolve_repo_digest(remote_ref)
    if digest is None:
        # Push succeeded but inspect didn't find a digest — degrade to the
        # tagged ref (still pullable, just not pinned).
        digest = remote_ref
        logger.warning("could not resolve RepoDigest for %s; using tag", remote_ref)

    return ImagePushResult(
        local_tag=local_tag,
        remote_ref=remote_ref,
        digest=digest,
        pushed=True,
        duration_sec=round(time.monotonic() - start, 2),
    )


__all__ = [
    "ImagePushResult",
    "PushError",
    "manifest_exists",
    "push_image",
]
