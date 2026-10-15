# [MODULE DOC] Reuse support for Tasksmith: binds previously generated tasks (and optionally their
# execution evidence and probes) from an earlier generation run back to the frozen PR records, and
# validates externally prepared probe manifests. Every check is identity-based (task bundle hash,
# source diff SHA-256, PR URL/head/base) so stale or foreign evidence is rejected rather than
# silently reused.
"""Bind reusable generations to their original run and frozen PR evidence."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Sequence
from pathlib import Path

from repo2rlenv.quality.loop.artifacts import digest, import_trial, task_identity
from repo2rlenv.quality.loop.models import ProbeFocus, ProbeManifest, SemanticProbe, TrialRecord
from repo2rlenv.tasksmith.models import Panel


def prepared_probe_definitions(
    task: Path | None,
    manifest: ProbeManifest | None,
    *,
    max_probes: int,
    required_focus: Sequence[ProbeFocus] = (),
) -> list[dict]:
    # [FUNCTION DOC] Validates a caller-supplied ProbeManifest for a prepared task and returns its
    # probes as JSON-ready dicts.
    # Inputs: task (prepared task directory or None); manifest (ProbeManifest or None); max_probes
    # (int, keyword-only upper bound); required_focus (sequence of ProbeFocus, default empty).
    # Output: list of probe dicts (model_dump mode json); empty list if manifest is None.
    # Raises ValueError when manifest is given without task, there are more than max_probes probes,
    # probe names are not unique, manifest.bundle_hash differs from task_identity(task), or
    # required_focus is not a subset of the focus already annotated on the task (task_probe_focus).
    # The manifest is re-validated via a dump/validate round trip first. Side effects: reads the
    # task directory.
    """Validate exact prepared inputs without importing any execution evidence."""
    if manifest is None:
        return []
    if task is None:
        raise ValueError("Prepared probes require a prepared task")
    # Revalidate nested, mutable Pydantic models at both admission and execution.
    manifest = ProbeManifest.model_validate(manifest.model_dump(mode="json"))
    if len(manifest.probes) > max_probes:
        raise ValueError("Prepared probes exceed quality.max_probes")
    if len({probe.name for probe in manifest.probes}) != len(manifest.probes):
        raise ValueError("Prepared probe names must be unique")
    if manifest.bundle_hash != task_identity(task):
        raise ValueError("Prepared probes belong to a different task identity")
    if required_focus:
        from repo2rlenv.quality.loop.requirements import task_probe_focus

        if set(required_focus) - task_probe_focus(task):
            raise ValueError(
                "Prepared probes require the task to already include required_probe_focus; "
                "annotate the task first and bind the definitions to its new identity"
            )
    return [probe.model_dump(mode="json") for probe in manifest.probes]


def generation_task(
    record: dict, source: dict, directory: Path, *, reuse_evidence: bool = False
) -> dict | None:
    # [FUNCTION DOC] Converts one saved candidate result record from an earlier generation run into
    # a reusable import descriptor, or None if nothing reusable exists.
    # Inputs: record (saved result.json dict); source (frozen PR source dict); directory (the
    # earlier run directory); reuse_evidence (bool, default False) to also import execution trials.
    # Behavior: requires record id == source id. Locates the task from record.quality.task_path
    # (expected hash = quality.bundle_hash) or from record.constructed (local + task_relative, no
    # expected hash); returns None if neither exists. The resolved task must lie inside directory.
    # Reads task.toml metadata.repo2env and requires recipe 'tasksmith', recipe_version '1', and
    # source url/head/base, source diff SHA-256 and workspace_strategy to match the source. Checks
    # the bundle hash. Returns None if the saved review has a blocking 'reference' issue. Collects
    # probe definitions from 'probe' trials, deduplicated by name (conflicting definitions raise).
    # With reuse_evidence, imports baseline/oracle/rollout trial results (must be non-symlink,
    # inside directory, matching hash and result SHA-256) via import_trial, skipping infrastructure
    # failures (exception or missing reward); duplicate roles raise.
    # Output: dict with task, bundle_hash, generation_run, probes, and optionally trials (role ->
    # observed trial dict). Raises ValueError on any identity mismatch. Side effects: filesystem
    # reads.
    if record.get("id") != source["id"]:
        raise ValueError("Saved candidate belongs to a different PR")
    quality = record.get("quality", {})
    if quality:
        task = Path(quality["task_path"])
        expected = quality["bundle_hash"]
    elif record.get("constructed"):
        built = record["constructed"]
        task = Path(built["local"]) / built["value"]["task_relative"]
        expected = None
    else:
        return None
    task = task.resolve()
    if not task.is_relative_to(directory.resolve()):
        raise ValueError("Saved generation points outside its run directory")
    metadata = tomllib.loads((task / "task.toml").read_text())["metadata"]["repo2env"]
    binding = {
        "recipe": "tasksmith",
        "recipe_version": "1",
        "source_url": source["url"],
        "source_head": source["head"],
        "source_base": source["base"],
        "source_diff_sha256": hashlib.sha256(source["source_diff"].encode()).hexdigest(),
        "workspace_strategy": source["workspace_strategy"],
    }
    if any(metadata.get(key) != value for key, value in binding.items()):
        raise ValueError("Saved generation metadata differs from the frozen PR")
    identity = task_identity(task)
    if expected is not None and identity != expected:
        raise ValueError("Imported task differs from its saved evidence")
    # A draft whose requirements conflict with the frozen reference needs fresh
    # design. Its probes were judged against that invalid draft; retain them in
    # the original run rather than treating them as proven PR counterexamples.
    issues = (quality.get("review") or {}).get("issues", [])
    if any(
        item.get("category") == "reference" and item.get("severity") == "blocking"
        for item in issues
    ):
        return None
    retained = {}
    for trial in quality.get("trials", []):
        if trial.get("role") == "probe" and trial.get("probe"):
            probe = SemanticProbe.model_validate(trial["probe"])
            if probe.name in retained and retained[probe.name] != probe.model_dump():
                raise ValueError("Saved probes disagree on the same control identity")
            retained[probe.name] = probe.model_dump()
    result = {
        "task": str(task),
        "bundle_hash": identity,
        "generation_run": str(directory.resolve()),
        "probes": list(retained.values()),
    }
    if reuse_evidence:
        trials = {}
        for item in quality.get("trials", []):
            if item.get("role") not in {"baseline", "oracle", "rollout"}:
                continue
            trial = TrialRecord.model_validate(item)
            path = Path(trial.result)
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Reusable execution evidence points outside its generation run")
            if trial.bundle_hash != identity or digest(path) != trial.result_sha256:
                raise ValueError("Reusable execution evidence changed or belongs to another task")
            observed = import_trial(path, task, trial.role)
            # Infrastructure failures need fresh execution. Legitimate solver
            # failures remain useful evidence about the unchanged task.
            if observed.exception_type is not None or observed.reward is None:
                continue
            if trial.role in trials:
                raise ValueError("Reusable evidence contains duplicate execution roles")
            trials[trial.role] = observed.model_dump(mode="json")
        result["trials"] = trials
    return result


def load_generation(
    directory: Path, panel: Panel, *, reuse_evidence: bool = False
) -> tuple[list[dict], dict]:
    # [FUNCTION DOC] Loads an earlier Tasksmith run directory for reuse.
    # Inputs: directory (earlier run root); panel (Panel that must be identical); reuse_evidence
    # (bool). Reads directory/panel.json, requires its configuration.panel to equal
    # panel.model_dump() and its source URLs to equal panel.prs (trailing slash stripped, same
    # order). For each source, reads candidates/<id>/result.json if present (must be inside
    # directory) and passes it to generation_task.
    # Output: tuple (sources list from panel.json, dict source_id -> import descriptor for the
    # reusable ones). Raises ValueError on mismatches; propagates JSON/IO errors.
    previous = json.loads((directory / "panel.json").read_text())
    if previous["configuration"]["panel"] != panel.model_dump():
        raise ValueError("Imported generation must have the identical frozen panel")
    sources = previous["sources"]
    if [source["url"] for source in sources] != [url.rstrip("/") for url in panel.prs]:
        raise ValueError("Imported source identities differ from the frozen panel")
    imported = {}
    for source in sources:
        candidate = directory / "candidates" / source["id"] / "result.json"
        if not candidate.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Saved candidate points outside its run directory")
        if not candidate.is_file():
            continue
        value = generation_task(
            json.loads(candidate.read_text()), source, directory, reuse_evidence=reuse_evidence
        )
        if value is not None:
            imported[source["id"]] = value
    return sources, imported
