# [MODULE DOC] AWS ECR helper that idempotently pre-creates the target repository before a push
# (ECR, unlike GHCR/Docker Hub/ACR, rejects pushes to non-existent repositories). Shells out to the
# `aws` CLI so auth stays with the user's AWS configuration and boto3 is not a dependency.
"""ECR-specific helper: pre-create a repository before push.

ECR (both private and public) requires the repository to exist before
`docker push` will accept blobs. Unlike GHCR / Docker Hub / ACR which
auto-create repos on first push.

We call the `aws` CLI to keep auth handling out of our codebase —
boto3 would be an extra runtime dep we don't otherwise need.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ECRError(Exception):
    # [CLASS DOC] Raised when an ECR ref cannot be parsed, the aws CLI is missing, or repository
    # creation fails.
    """Raised when an ECR API call fails irrecoverably."""


@dataclass(slots=True)
class ECRRepoResult:
    # [CLASS DOC] Result of ensure_ecr_repository: repo name, created (True if this call created it,
    # False if it already existed) and is_public (ECR Public vs private).
    repo: str
    created: bool  # True if we just created it; False if it already existed
    is_public: bool


def _aws_available() -> bool:
    # [FUNCTION DOC] HELPER: True if an `aws` executable is on PATH.
    return shutil.which("aws") is not None


def _parse_ecr_ref(remote_ref: str) -> tuple[str, str, bool] | None:
    # [FUNCTION DOC] HELPER: Parses an ECR image ref (tag after the first ':' and digest after '@'
    # removed) into (region_or_alias, repo, is_public). Public form public.ecr.aws/<alias>/<repo>
    # gives (alias, repo, True); private form <acct>.dkr.ecr.<region>.amazonaws.com/<repo> gives
    # (region, repo, False). Returns None when the shape does not match (missing repo, host parts
    # not '<acct>.dkr.ecr.<region>...').
    # KNOWN ISSUE: the first private-form branch and the trailing fallback perform identical
    # parsing, so the 'endswith(.amazonaws.com) is False' condition is effectively redundant dead
    # logic.
    """Return (region_or_alias, repo, is_public) from an ECR ref, or None.

    - Private: `<acct>.dkr.ecr.<region>.amazonaws.com/<repo>[:<tag>]`
    - Public:  `public.ecr.aws/<alias>/<repo>[:<tag>]`
    """
    body = remote_ref.split(":", 1)[0].split("@", 1)[0]
    if body.startswith("public.ecr.aws/"):
        rest = body.removeprefix("public.ecr.aws/")
        if "/" not in rest:
            return None
        alias, repo = rest.split("/", 1)
        return (alias, repo, True)
    # Private form
    if ".dkr.ecr." in body and body.endswith(".amazonaws.com") is False:
        # host part is something like 123.dkr.ecr.us-east-1.amazonaws.com/<repo>
        host, _, repo = body.partition("/")
        if not repo:
            return None
        parts = host.split(".")
        # Expect: ['123', 'dkr', 'ecr', '<region>', 'amazonaws', 'com']
        if len(parts) < 6 or parts[1] != "dkr" or parts[2] != "ecr":
            return None
        region = parts[3]
        return (region, repo, False)
    # Tolerate the alternative split (host first)
    host, _, repo = body.partition("/")
    if not repo or ".dkr.ecr." not in host:
        return None
    parts = host.split(".")
    if len(parts) < 6 or parts[1] != "dkr" or parts[2] != "ecr":
        return None
    region = parts[3]
    return (region, repo, False)


def ensure_ecr_repository(remote_ref: str) -> ECRRepoResult:
    # [FUNCTION DOC] Ensures the ECR repository for remote_ref exists. Parses the ref, requires the
    # aws CLI, runs 'aws ecr[-public] describe-repositories' (30s timeout; public always uses region
    # us-east-1, private uses the parsed region) and, if that fails, 'create-repository' (60s
    # timeout). Output: ECRRepoResult(created=False if described OK, True if created). Raises
    # ECRError on an unparsable ref, missing CLI or failed create (with stderr/stdout).
    # Subprocess.TimeoutExpired may propagate. Side effects: AWS API calls via subprocess; may
    # create a repository.
    """Idempotent: create the ECR repository for `remote_ref` if it doesn't exist."""
    parsed = _parse_ecr_ref(remote_ref)
    if parsed is None:
        raise ECRError(f"could not parse ECR ref: {remote_ref!r}")
    region_or_alias, repo, is_public = parsed

    if not _aws_available():
        raise ECRError("aws CLI not available on PATH")

    if is_public:
        describe_args = [
            "aws",
            "ecr-public",
            "describe-repositories",
            "--repository-names",
            repo,
            "--region",
            "us-east-1",  # ECR Public is single-region (us-east-1)
        ]
        create_args = [
            "aws",
            "ecr-public",
            "create-repository",
            "--repository-name",
            repo,
            "--region",
            "us-east-1",
        ]
    else:
        describe_args = [
            "aws",
            "ecr",
            "describe-repositories",
            "--repository-names",
            repo,
            "--region",
            region_or_alias,
        ]
        create_args = [
            "aws",
            "ecr",
            "create-repository",
            "--repository-name",
            repo,
            "--region",
            region_or_alias,
        ]

    describe = subprocess.run(
        describe_args, capture_output=True, text=True, encoding="utf-8", timeout=30, check=False
    )
    if describe.returncode == 0:
        return ECRRepoResult(repo=repo, created=False, is_public=is_public)

    create = subprocess.run(
        create_args, capture_output=True, text=True, encoding="utf-8", timeout=60, check=False
    )
    if create.returncode != 0:
        raise ECRError(
            f"ECR create-repository failed: {create.stderr.strip() or create.stdout.strip()}"
        )
    return ECRRepoResult(repo=repo, created=True, is_public=is_public)


__all__ = [
    "ECRError",
    "ECRRepoResult",
    "ensure_ecr_repository",
]
