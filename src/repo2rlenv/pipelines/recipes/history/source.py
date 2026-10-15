# [MODULE DOC] Controller-side merged pull-request discovery for swe_next, using the gh CLI so
# credentials never reach the remote worker.
"""Resolve merged PR metadata on the controller without exposing credentials remotely."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from repo2rlenv.execution.lifecycle import save_record
from repo2rlenv.github import _run_gh


def merged_pulls(url: str, options, receipt: Path) -> list[dict]:
    # [FUNCTION DOC] Lists merged PRs for a GitHub repository, cached in a receipt.
    # Inputs: url (repository URL; owner/name taken from its path, .git stripped); options
    # (pr_numbers or max_prs); receipt (JSON cache path). If the receipt exists its contents are
    # returned. Otherwise fetches the listed PR numbers, or pages closed PRs (100 per page, by updated
    # desc) up to max_prs, keeps only merged PRs with a merge commit, and stores number, head
    # (merge_commit_sha), title, body (max 10000 chars), url and merged_at.
    # Output: list of dicts. Raises: errors from _run_gh/json. Side effects: gh API calls, writes
    # the receipt.
    if receipt.exists():
        return json.loads(receipt.read_text())
    path = urlparse(url).path.strip("/").removesuffix(".git")
    rows = []
    if options.pr_numbers:
        rows = [
            json.loads(_run_gh(["api", f"repos/{path}/pulls/{number}"]))
            for number in options.pr_numbers
        ]
    else:
        for page in range(1, (options.max_prs + 99) // 100 + 1):
            batch = json.loads(
                _run_gh(
                    [
                        "api",
                        f"repos/{path}/pulls?state=closed&sort=updated&direction=desc&per_page=100&page={page}",
                    ]
                )
            )
            rows.extend(batch)
            if len(batch) < 100:
                break
        rows = rows[: options.max_prs]
    selected = [
        {
            "number": row["number"],
            "head": row["merge_commit_sha"],
            "title": row["title"],
            "body": (row.get("body") or "")[:10000],
            "url": row["html_url"],
            "merged_at": row["merged_at"],
        }
        for row in rows
        if row.get("merged_at") and row.get("merge_commit_sha")
    ]
    save_record(receipt, selected)
    return selected
