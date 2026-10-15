# [MODULE DOC] File-level discovery of Docker login state from the Docker CLI config.json (honouring
# $DOCKER_CONFIG): which registry hosts appear logged in and how the credential is stored. No
# network calls; whether credentials actually work is checked by registry.probe. Also classifies
# registry hostnames into RegistryKind values via regexes/known host sets.
"""Read Docker login state from ~/.docker/config.json.

Discovery is file-level only — answers "does the user have credentials for
this host?" without making network calls. Real verification ("do those
credentials still work?") lives in `probe.py`.

Reference:
- https://docs.docker.com/reference/cli/docker/login/
- https://github.com/docker/docker-credential-helpers
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class RegistryKind(StrEnum):
    # [CLASS DOC] String enum of supported registry families: ghcr, ecr_private, ecr_public, acr,
    # gcp_ar, docker_hub, local (localhost/127.0.0.1) and other.
    GHCR = "ghcr"
    ECR_PRIVATE = "ecr_private"
    ECR_PUBLIC = "ecr_public"
    ACR = "acr"
    GCP_AR = "gcp_ar"
    DOCKER_HUB = "docker_hub"
    LOCAL = "local"
    OTHER = "other"


class CredentialSource(StrEnum):
    # [CLASS DOC] String enum describing how a host's credential is stored: INLINE (base64 auth blob
    # in auths), CREDSTORE (global credsStore helper), CREDHELPER (per-host credHelpers entry) or
    # EMPTY (an empty auths entry, typical of Docker Desktop).
    """How the credential is stored.

    INLINE   — `auths.<host>.auth` is a base64 user:pass blob in config.json
    CREDSTORE — `credsStore` resolves the host via an OS keychain helper
    CREDHELPER — per-host `credHelpers` overrides credsStore (typical for ECR)
    EMPTY    — `auths.<host>` exists but is `{}` (Docker Desktop signal)
    """

    INLINE = "auths_inline"
    CREDSTORE = "credstore"
    CREDHELPER = "credhelper"
    EMPTY = "auths_empty"


@dataclass(slots=True)
class RegistryAuth:
    # [CLASS DOC] A discovered login: normalized host, its RegistryKind, the CredentialSource and,
    # for helper-backed sources, the helper binary name (e.g. 'ecr-login', 'desktop'), else None.
    host: str
    kind: RegistryKind
    cred_source: CredentialSource
    # The credential helper binary name when cred_source is CREDHELPER or
    # CREDSTORE — e.g. "ecr-login", "gcloud", "desktop", "osxkeychain".
    helper: str | None = None


# Hostname classification patterns. Order matters: more specific first.
_ECR_PRIVATE_RE = re.compile(r"^[0-9]+\.dkr\.ecr\.[^.]+\.amazonaws\.com$")
_GCP_AR_RE = re.compile(r"^[a-z0-9-]+-docker\.pkg\.dev$")
_ACR_RE = re.compile(r"^[a-z0-9]+\.azurecr\.io$", re.IGNORECASE)
_DOCKER_HUB_HOSTS = frozenset(
    {"index.docker.io", "registry-1.docker.io", "docker.io", "https://index.docker.io/v1/"}
)


def classify_host(host: str) -> RegistryKind:
    # [FUNCTION DOC] Maps a config.json auths key / hostname to a RegistryKind. Strips whitespace,
    # scheme, trailing slashes and a trailing /v1; then checks, in order, Docker Hub aliases,
    # ghcr.io, public.ecr.aws, private ECR, ACR (*.azurecr.io, case-insensitive), GCP AR
    # (*-docker.pkg.dev) and localhost/127.0.0.1 (with optional port). Empty input or no match
    # returns OTHER.
    """Map a docker config.json `auths` key to a RegistryKind.

    Docker Hub's config.json key is sometimes the legacy v1 URL
    `https://index.docker.io/v1/`; normalize that too.
    """
    h = host.strip()
    if not h:
        return RegistryKind.OTHER
    # Strip scheme + trailing slashes for matching
    normalized = h.removeprefix("https://").removeprefix("http://").rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized.removesuffix("/v1")

    if h in _DOCKER_HUB_HOSTS or normalized in _DOCKER_HUB_HOSTS:
        return RegistryKind.DOCKER_HUB
    if normalized == "ghcr.io":
        return RegistryKind.GHCR
    if normalized == "public.ecr.aws":
        return RegistryKind.ECR_PUBLIC
    if _ECR_PRIVATE_RE.match(normalized):
        return RegistryKind.ECR_PRIVATE
    if _ACR_RE.match(normalized):
        return RegistryKind.ACR
    if _GCP_AR_RE.match(normalized):
        return RegistryKind.GCP_AR
    if normalized in {"localhost", "127.0.0.1"} or normalized.startswith(
        ("localhost:", "127.0.0.1:")
    ):
        return RegistryKind.LOCAL
    return RegistryKind.OTHER


def _config_path() -> Path:
    # [FUNCTION DOC] HELPER: Returns the Docker CLI config.json path: $DOCKER_CONFIG/config.json
    # (user-expanded) when the env var is set, else ~/.docker/config.json. Reads the environment.
    """The active Docker CLI config path. Respects $DOCKER_CONFIG."""
    if env := os.environ.get("DOCKER_CONFIG"):
        return Path(env).expanduser() / "config.json"
    return Path.home() / ".docker" / "config.json"


def _normalize_host(raw: str) -> str:
    # [FUNCTION DOC] HELPER: Normalizes an auths key to a bare host by stripping whitespace,
    # http(s):// scheme, trailing slashes and a trailing '/v1'. Output: str.
    """Normalize a config.json auths key to a bare hostname.

    The Docker CLI accepts both the v1 URL and a bare hostname as the auths
    key; we keep the original Docker Hub aliases as a single "docker_hub"
    classification but strip scheme/trailing slashes for everything else.
    """
    s = raw.strip()
    s = s.removeprefix("https://").removeprefix("http://").rstrip("/")
    s = s.removesuffix("/v1")
    return s


def discover_logged_in_registries(config_path: Path | None = None) -> list[RegistryAuth]:
    # [FUNCTION DOC] Returns the registries the user appears logged into, without validating them.
    # Input: config_path (optional; default _config_path()). Returns [] if the file is missing or
    # unreadable/invalid JSON (logged at debug). First every credHelpers entry becomes a CREDHELPER
    # auth (helper name recorded); then each auths entry not already seen becomes INLINE (non-empty
    # 'auth'), CREDSTORE (global credsStore set; helper = store name) or EMPTY. Hosts are normalized
    # and deduplicated. Output: list[RegistryAuth] in that order. Side effects: reads the file.
    """Return the list of registries the user appears logged into.

    File-level only — does not validate that credentials actually work.
    Use `registry.probe.probe()` to verify.

    Resolution:
      1. Per-host `credHelpers` entry → CREDHELPER
      2. Global `credsStore` + `auths.<host>` present → CREDSTORE
      3. `auths.<host>.auth` non-empty → INLINE
      4. `auths.<host>` is `{}` (Desktop keychain signal) → EMPTY
    """
    path = config_path or _config_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("could not read %s: %s", path, exc)
        return []

    auths = data.get("auths", {}) if isinstance(data, dict) else {}
    cred_helpers = data.get("credHelpers", {}) if isinstance(data, dict) else {}
    creds_store = data.get("credsStore") if isinstance(data, dict) else None

    results: list[RegistryAuth] = []
    seen: set[str] = set()

    # 1. Per-host credHelpers (highest precedence)
    if isinstance(cred_helpers, dict):
        for raw_host, helper in cred_helpers.items():
            host = _normalize_host(raw_host)
            if host in seen:
                continue
            seen.add(host)
            results.append(
                RegistryAuth(
                    host=host,
                    kind=classify_host(host),
                    cred_source=CredentialSource.CREDHELPER,
                    helper=str(helper) if helper else None,
                )
            )

    # 2 + 3 + 4: walk auths entries that aren't already covered by credHelpers
    if isinstance(auths, dict):
        for raw_host, entry in auths.items():
            host = _normalize_host(raw_host)
            if host in seen:
                continue
            seen.add(host)
            inline_auth = ""
            if isinstance(entry, dict):
                inline_auth = (entry.get("auth") or "").strip()
            if inline_auth:
                source = CredentialSource.INLINE
                helper = None
            elif creds_store:
                source = CredentialSource.CREDSTORE
                helper = str(creds_store)
            else:
                # Empty `{}` with no global credsStore → most plausibly Docker
                # Desktop on a host where credsStore wasn't written. Treat as
                # logged-in signal; probe will catch the case where it isn't.
                source = CredentialSource.EMPTY
                helper = None
            results.append(
                RegistryAuth(
                    host=host,
                    kind=classify_host(host),
                    cred_source=source,
                    helper=helper,
                )
            )

    return results


def filter_known_registries(auths: list[RegistryAuth]) -> list[RegistryAuth]:
    # [FUNCTION DOC] Returns the given auths without entries classified as RegistryKind.OTHER.
    """Drop entries classified as OTHER. Convenience for the push path."""
    return [a for a in auths if a.kind is not RegistryKind.OTHER]
