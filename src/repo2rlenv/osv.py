# [MODULE DOC] Client for the public OSV vulnerability API (https://api.osv.dev/v1/query, no auth),
# used by the cve_patches pipeline: query vulns per package/ecosystem, distill records into OSVVuln,
# extract GitHub fix-commit SHAs from references, guess an ecosystem from a GitHub owner, and
# compare severities. Network via urllib.
"""OSV (Open Source Vulnerabilities) API client.

OSV is Google's public vuln database (https://osv.dev) — no auth required,
covers PyPI / npm / crates.io / Go / Maven / NuGet / Debian / Alpine / etc.

We use it to power the `cve_patches` pipeline: query → extract fix commit
URLs from `references[]` → map to a (base_commit, patch) pair via the
standard `gh api repos/.../commits/<sha>` route.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


OSV_QUERY_URL = "https://api.osv.dev/v1/query"

# A `references[].url` that points at a fix commit. We accept either:
#   https://github.com/<owner>/<repo>/commit/<sha40>
#   https://github.com/<owner>/<repo>/pull/<n>/commits/<sha40>
_GH_COMMIT_RE = re.compile(
    r"https?://github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)/(?:commit|pull/\d+/commits)/(?P<sha>[a-f0-9]{7,40})\b"
)


@dataclass(slots=True)
class OSVVuln:
    # [CLASS DOC] Slotted dataclass for one OSV record: id (GHSA/PYSEC/CVE...), aliases, summary,
    # details, severity_text (upper-cased label such as HIGH, or a CVSS type when no label exists),
    # cwe_ids, published (ISO8601 string), references (raw list of {type,url} dicts).
    """One vulnerability record, distilled from the OSV JSON shape."""

    id: str  # primary identifier (GHSA-* or PYSEC-* or CVE-* etc.)
    aliases: list[str] = field(default_factory=list)  # other ids (often a CVE)
    summary: str = ""
    details: str = ""
    severity_text: str = ""  # human-readable: "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"
    cwe_ids: list[str] = field(default_factory=list)
    published: str = ""  # ISO8601
    references: list[dict[str, str]] = field(default_factory=list)  # raw

    @property
    def cve_id(self) -> str | None:
        # [FUNCTION DOC] Property: first identifier among id then aliases that starts with "CVE-"
        # (case-insensitive), else None.
        """First CVE-* identifier in id or aliases."""
        for cand in [self.id, *self.aliases]:
            if cand.upper().startswith("CVE-"):
                return cand
        return None

    def fix_commits(self, *, owner: str, repo: str) -> list[str]:
        # [FUNCTION DOC] Return de-duplicated commit SHAs (order preserved) from references whose
        # URL matches a github.com commit or pull/N/commits URL for exactly owner/repo (case-
        # insensitive). SHAs of 7-40 hex chars are accepted, so abbreviated SHAs may be returned.
        """Return commit SHAs in `references[]` that point at <owner>/<repo>.

        Filters by owner/repo so we don't accept fix URLs from forks /
        unrelated repos (OSV sometimes lists multiple linked projects).
        """
        out: list[str] = []
        for ref in self.references:
            url = ref.get("url", "")
            m = _GH_COMMIT_RE.search(url)
            if not m:
                continue
            if m.group("owner").lower() != owner.lower():
                continue
            if m.group("repo").lower() != repo.lower():
                continue
            sha = m.group("sha")
            if sha not in out:
                out.append(sha)
        return out


def _from_raw(raw: dict) -> OSVVuln:
    # [FUNCTION DOC] HELPER: Build an OSVVuln from a raw OSV JSON dict. severity_text comes from
    # database_specific.severity; if absent, from the first severity[] entry's "type" (e.g.
    # "CVSS_V3"); upper-cased. cwe_ids come from database_specific.cwe_ids. Missing fields default
    # to empty.
    # KNOWN ISSUE: the inline comment says the fallback is "UNKNOWN", but the code stores the CVSS
    # type string (e.g. "CVSS_V3"), which severity_at_least then ranks as 0.
    db = raw.get("database_specific", {}) or {}
    severity_text = db.get("severity", "") or ""
    if not severity_text:
        # Some records use `severity[]` with CVSS score arrays — fall back to "UNKNOWN"
        sev = raw.get("severity", []) or []
        if sev and isinstance(sev[0], dict):
            severity_text = sev[0].get("type", "")  # e.g. "CVSS_V3"
    return OSVVuln(
        id=raw.get("id", ""),
        aliases=list(raw.get("aliases", []) or []),
        summary=raw.get("summary", "") or "",
        details=raw.get("details", "") or "",
        severity_text=severity_text.upper(),
        cwe_ids=list(db.get("cwe_ids", []) or []),
        published=raw.get("published", "") or "",
        references=list(raw.get("references", []) or []),
    )


class OSVError(RuntimeError):
    # [CLASS DOC] Raised for OSV HTTP, network or JSON decoding failures. Subclass of RuntimeError.
    pass


def query_vulns(
    package: str,
    ecosystem: str,
    *,
    timeout: float = 30.0,
) -> list[OSVVuln]:
    # [FUNCTION DOC] POST {"package": {"name", "ecosystem"}} to the OSV query endpoint and return
    # the parsed list of OSVVuln (empty list when no "vulns"). Inputs: package; ecosystem (e.g.
    # PyPI, npm, crates.io, Go); timeout seconds (default 30). Raises OSVError on HTTPError,
    # URLError or malformed JSON. Only the first result page is read (OSV page tokens are ignored).
    """POST to OSV `/v1/query` and return the parsed list of vulns.

    `ecosystem` examples: "PyPI", "npm", "crates.io", "Go", "Maven",
    "NuGet", "Debian", "Alpine".
    """
    payload = json.dumps({"package": {"name": package, "ecosystem": ecosystem}}).encode()
    req = urllib.request.Request(
        OSV_QUERY_URL,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise OSVError(f"OSV HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise OSVError(f"OSV network error: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise OSVError(f"OSV returned malformed JSON: {exc}") from exc
    vulns = data.get("vulns") or []
    return [_from_raw(v) for v in vulns]


# Lookup tables for repo → (ecosystem, package) auto-detection.
# These are heuristics — users can always override via pipeline options.
_KNOWN_OWNER_ECOSYSTEM: dict[str, str] = {
    "pallets": "PyPI",
    "pypa": "PyPI",
    "pytest-dev": "PyPI",
    "psf": "PyPI",
    "django": "PyPI",
    "fastapi": "PyPI",
    "encode": "PyPI",
    "nodejs": "npm",
    "expressjs": "npm",
    "facebook": "npm",
    "rust-lang": "crates.io",
    "tokio-rs": "crates.io",
}


def guess_ecosystem(owner: str) -> str:
    # [FUNCTION DOC] Map a GitHub owner (case-insensitive) to an OSV ecosystem using the small
    # _KNOWN_OWNER_ECOSYSTEM table; defaults to "PyPI". Pure.
    """Best-effort: map a GitHub owner to a likely OSV ecosystem.

    Falls back to "PyPI" since that's where the bulk of CVE-attached repos
    Repo2RLEnv targets live. The user can always override via the
    `osv_ecosystem` pipeline option.
    """
    return _KNOWN_OWNER_ECOSYSTEM.get(owner.lower(), "PyPI")


_SEVERITY_RANK = {"LOW": 1, "MEDIUM": 2, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4}


def severity_at_least(vuln: OSVVuln, threshold: str) -> bool:
    # [FUNCTION DOC] True iff the vuln's severity rank >= the threshold's rank, using LOW=1,
    # MEDIUM/MODERATE=2, HIGH=3, CRITICAL=4 (case-insensitive). Unknown vuln severities rank 0
    # (below any known threshold); an unknown threshold also ranks 0, so it accepts everything.
    # Pure.
    """True iff the vuln's severity meets or exceeds `threshold` (case-insensitive).

    Unknown severities are treated as below threshold (conservative).
    """
    threshold_rank = _SEVERITY_RANK.get(threshold.upper(), 0)
    vuln_rank = _SEVERITY_RANK.get(vuln.severity_text.upper(), 0)
    return vuln_rank >= threshold_rank
