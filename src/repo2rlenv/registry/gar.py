# [MODULE DOC] GCP Artifact Registry helper that idempotently pre-creates the docker repository a
# push targets, by shelling out to `gcloud` (avoids a google-cloud SDK dependency). Mirrors ecr.py.
"""GCP Artifact Registry helper: pre-create a repository before push.

Same shape as `ecr.py` — the AR repository must exist before push will
accept blobs. We shell out to `gcloud` to avoid pulling in
google-cloud-artifactregistry as a runtime dep.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class GARError(Exception):
    # [CLASS DOC] Raised when a GAR ref cannot be parsed, gcloud is missing, or repository creation
    # fails.
    """Raised when a GCP AR API call fails irrecoverably."""


@dataclass(slots=True)
class GARRepoResult:
    # [CLASS DOC] Result of ensure_gar_repository: project, location, repo and whether this call
    # created the repository.
    project: str
    location: str
    repo: str
    created: bool


def _gcloud_available() -> bool:
    # [FUNCTION DOC] HELPER: True if a `gcloud` executable is on PATH.
    return shutil.which("gcloud") is not None


def _parse_gar_ref(remote_ref: str) -> tuple[str, str, str, str] | None:
    # [FUNCTION DOC] HELPER: Parses
    # '<location>-docker.pkg.dev/<project>/<repo>/<image>[:tag|@digest]' into (location, project,
    # repo, image). Returns None if the host lacks '-docker.pkg.dev', the location is empty, or
    # fewer than three path components follow the host.
    """Return (location, project, repo, image) for a GAR ref, or None.

    Form: `<location>-docker.pkg.dev/<project>/<repo>/<image>[:<tag>]`
    """
    body = remote_ref.split(":", 1)[0].split("@", 1)[0]
    if "-docker.pkg.dev/" not in body:
        return None
    host, _, rest = body.partition("/")
    location = host.removesuffix("-docker.pkg.dev")
    if not location or location == host:
        return None
    parts = rest.split("/", 2)
    if len(parts) < 3:
        return None
    project, repo, image = parts[0], parts[1], parts[2]
    return (location, project, repo, image)


def ensure_gar_repository(remote_ref: str) -> GARRepoResult:
    # [FUNCTION DOC] Ensures the GAR docker repository for remote_ref exists. Parses the ref,
    # requires gcloud, runs 'gcloud artifacts repositories describe' (30s timeout) and, on failure,
    # 'create --repository-format=docker' (120s timeout) for the parsed location/project. Output:
    # GARRepoResult (created False/True). Raises GARError on unparsable ref, missing CLI or failed
    # create; a subprocess timeout may propagate. Side effects: GCP API calls via subprocess; may
    # create a repository.
    """Idempotent: ensure the GAR docker repository exists."""
    parsed = _parse_gar_ref(remote_ref)
    if parsed is None:
        raise GARError(f"could not parse GAR ref: {remote_ref!r}")
    location, project, repo, _image = parsed

    if not _gcloud_available():
        raise GARError("gcloud CLI not available on PATH")

    describe = subprocess.run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "describe",
            repo,
            "--location",
            location,
            "--project",
            project,
            "--format",
            "value(name)",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    if describe.returncode == 0:
        return GARRepoResult(project=project, location=location, repo=repo, created=False)

    create = subprocess.run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "create",
            repo,
            "--repository-format=docker",
            "--location",
            location,
            "--project",
            project,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    if create.returncode != 0:
        raise GARError(f"GAR create failed: {create.stderr.strip() or create.stdout.strip()}")
    return GARRepoResult(project=project, location=location, repo=repo, created=True)


__all__ = [
    "GARError",
    "GARRepoResult",
    "ensure_gar_repository",
]
