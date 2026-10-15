# [MODULE DOC] Applies a recorded remote CPU bootstrap report (from bootstrap_matrix) to Tasksmith
# Options: binds the Modal worker snapshot and per-repository BootstrapHints without copying any
# target source images.
"""Consume recorded CPU bootstrap evidence without copying target source images."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from repo2rlenv.tasksmith.bootstrap_matrix import RepositoryBootstrap
from repo2rlenv.tasksmith.models import BootstrapHint, Options


def apply_cpu_bootstrap(options: Options, path: Path) -> Options:
    # [FUNCTION DOC] Reads the JSON bootstrap report at path and returns a new Options with
    # worker_snapshot set to the report's snapshot_id and bootstrap_hints merged with the hints of
    # all "ready" repositories (keyed by repository URL). Raises ValueError if the report is not a
    # completed CPU report, options.provider is not "modal", the snapshot timestamp is naive or TTL
    # not a positive int, the snapshot has expired (created + ttl <= now UTC), options already
    # select a different snapshot, a hint is duplicated or its ref differs from the source ref,
    # options carry a conflicting hint for the same URL, or no repository is ready. Pure apart from
    # reading the file.
    report = json.loads(path.read_text())
    if report.get("resource") != "cpu" or report.get("status") != "completed":
        raise ValueError("Generation requires a completed CPU bootstrap report")
    if options.provider != "modal":
        raise ValueError("CPU bootstrap snapshots currently require Modal")
    created = datetime.fromisoformat(report["snapshot_created_at"])
    ttl = report["snapshot_ttl_sec"]
    if created.tzinfo is None or type(ttl) is not int or ttl <= 0:
        raise ValueError("Bootstrap snapshot needs a timezone-aware timestamp and positive TTL")
    if created + timedelta(seconds=ttl) <= datetime.now(UTC):
        raise ValueError("Bootstrap snapshot has expired; create a new bootstrap snapshot")
    snapshot = report["snapshot_id"]
    if options.worker_snapshot and options.worker_snapshot != snapshot:
        raise ValueError("Options and bootstrap report select different worker snapshots")
    hints = dict(options.bootstrap_hints)
    seen = set()
    for row in report["repositories"]:
        if row["status"] != "ready":
            continue
        source = RepositoryBootstrap.model_validate(row["source"])
        hint = BootstrapHint.model_validate(row["hint"])
        if source.url in seen or hint.ref != source.ref:
            raise ValueError("Bootstrap hints must be unique and bound to their checked source")
        seen.add(source.url)
        if source.url in hints and hints[source.url] != hint:
            raise ValueError(f"Options and bootstrap report have different hints for {source.url}")
        hints[source.url] = hint
    if not seen:
        raise ValueError("Bootstrap report contains no ready repositories")
    return Options.model_validate(
        {
            **options.model_dump(),
            "worker_snapshot": snapshot,
            "bootstrap_hints": hints,
        }
    )
