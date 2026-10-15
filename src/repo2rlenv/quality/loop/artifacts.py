# [MODULE DOC] Read-only ingestion of Harbor task directories and trial evidence, plus transactional
# text repairs and semantic-probe variants. Every new revision is built in a temporary sibling
# directory, checked, then atomically renamed into place; its bundle hash (metadata.repo2env) is
# refreshed and a JSON record (repair.json / probe.json) is saved beside it. Constant
# EXPECTED_PASSES_CONTRACT names tests/contract.json, the owned-task test contract.
"""Read-only task/evidence ingestion and transactional text repairs."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import tempfile
import tomllib
from pathlib import Path

import tomli_w

from repo2rlenv.emitter.bundle import inspect_bundle, relative_asset_path
from repo2rlenv.emitter.evaluation import generated_evaluation
from repo2rlenv.execution.lifecycle import save_record
from repo2rlenv.quality.loop.models import Repair, SemanticProbe, TrialRecord
from repo2rlenv.quality.loop.verifier_policy import (
    TASKSMITH_BEHAVIOR_PATH,
    check_behavioral_verifier,
)

EXPECTED_PASSES_CONTRACT = "tests/contract.json"


def digest(path: Path) -> str:
    # [FUNCTION DOC] Returns the hex SHA-256 of the file at path (streamed via hashlib.file_digest).
    # Raises OSError if unreadable. No side effects.
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def task_identity(task: Path) -> str:
    # [FUNCTION DOC] Returns the task's content bundle hash as computed by
    # emitter.bundle.inspect_bundle. Raises ValueError if the task records a claimed bundle hash
    # that no longer matches its content (integrity failed). Tasks without a claimed hash are
    # accepted.
    identity = inspect_bundle(task)
    if identity["claimed_hash"] is not None and not identity["integrity_passed"]:
        raise ValueError("Task differs from its recorded bundle hash")
    return identity["bundle_hash"]


def parse_task(task: Path):
    # [FUNCTION DOC] Loads the directory as a Harbor Task object (harbor.models.task.task.Task) and
    # returns it. Raises ValueError if the optional harbor extra is not installed.
    try:
        from harbor.models.task.task import Task
    except ImportError as exc:
        raise ValueError("Install repo2rlenv[harbor] to review Harbor task definitions") from exc
    return Task(task)


def snapshot(task: Path, destination: Path) -> Path:
    # [FUNCTION DOC] Copies a task to destination atomically and returns destination. Inputs: task
    # (source dir), destination (target dir). If destination already exists it must have the same
    # identity (else ValueError) and is reused. Otherwise copies into a temp dir under destination's
    # parent, re-verifies identity (ValueError 'Task changed during snapshot'), then renames.
    # Creates parents.
    identity = task_identity(task)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if task_identity(destination) != identity:
            raise ValueError("Existing snapshot differs from the source")
        return destination
    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=destination.parent) as temporary:
        copied = Path(temporary) / task.name
        shutil.copytree(task, copied)
        if task_identity(copied) != identity:
            raise ValueError("Task changed during snapshot")
        copied.rename(destination)
    return destination


def refresh_identity(task: Path) -> str:
    # [FUNCTION DOC] Recomputes and stores the bundle hash in task.toml metadata.repo2env when the
    # task carries one. If the hash changed it also replaces metadata.repo2env.evaluation with a
    # fresh generated_evaluation record bound to the new hash, with reason_codes ['task_changed']
    # (invalidating prior acceptance). Rewrites task.toml (via tomli_w, formatting not preserved).
    # Returns task_identity(task) afterwards. Tasks without bundle_hash metadata are not modified.
    config_path = task / "task.toml"
    config = tomllib.loads(config_path.read_text())
    metadata = config.get("metadata", {}).get("repo2env", {})
    if "bundle_hash" in metadata:
        identity = inspect_bundle(task)["bundle_hash"]
        if identity != metadata["bundle_hash"]:
            metadata["evaluation"] = {
                **generated_evaluation(subject_bundle_hash=identity),
                "reason_codes": ["task_changed"],
                "detail": "Task content changed; validate this revision before acceptance.",
            }
        metadata["bundle_hash"] = identity
        config_path.write_text(tomli_w.dumps(config))
    return task_identity(task)


def edit_path(value: str) -> str:
    # [FUNCTION DOC] HELPER: Validates a repair edit path and returns it unchanged. 'instruction.md'
    # is allowed as-is; anything else must pass emitter.bundle.relative_asset_path (which raises on
    # unsafe or non-asset paths).
    if value == "instruction.md":
        return value
    relative_asset_path(value)
    return value


def apply_repair(task: Path, repair: Repair, destination: Path) -> Path:
    # [FUNCTION DOC] Applies a model-authored Repair to a new copy of task at destination and
    # returns destination. Each Edit with nonempty old must match exactly once in its file (replaced
    # once); an Edit with empty old creates a new file exclusively (fails if it exists) with mode
    # 755/644 per edit.executable. append_expected_passes is appended to tests/contract.json.
    # Raises: FileExistsError if destination exists; ValueError if total new text exceeds 150000
    # chars, if appends and a text edit of tests/contract.json are combined, if an old text is not
    # unique, if instruction.md ends up blank, if a tasksmith task's behavioral verifier fails
    # check_behavioral_verifier, or if the hash is unchanged and no probe_replacements were given;
    # ValueError from task_identity if the source fails integrity. Side effects: temp dir under the
    # destination parent, refresh_identity on the copy, atomic rename, and writes repair.json
    # (parent_hash, bundle_hash, repair dump) next to destination.
    """Apply exact replacements and case-ID appends to a new, unverified copy."""
    task_identity(task)
    if destination.exists():
        raise FileExistsError(destination)
    if (
        sum(len(edit.new) for edit in repair.edits)
        + sum(len(value) for value in repair.append_expected_passes)
        > 150000
    ):
        raise ValueError("Repair exceeds the text-edit limit")
    if repair.append_expected_passes and any(
        edit_path(edit.path) == EXPECTED_PASSES_CONTRACT for edit in repair.edits
    ):
        raise ValueError("Cannot append expected passes and text-edit tests/contract.json together")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".repair-", dir=destination.parent) as temporary:
        copied = Path(temporary) / task.name
        shutil.copytree(task, copied)
        for edit in repair.edits:
            path = copied / edit_path(edit.path)
            if edit.old:
                text = path.read_text()
                if text.count(edit.old) != 1:
                    raise ValueError(f"Repair old text must occur exactly once: {edit.path}")
                path.write_text(text.replace(edit.old, edit.new, 1))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x") as handle:
                    handle.write(edit.new)
                path.chmod(0o755 if edit.executable else 0o644)
        if repair.append_expected_passes:
            _append_expected_passes(copied, repair.append_expected_passes)
        if not (copied / "instruction.md").read_text().strip():
            raise ValueError("Repair erased the task instruction")
        config = tomllib.loads((copied / "task.toml").read_text())
        behavior = copied / TASKSMITH_BEHAVIOR_PATH
        if (
            config.get("metadata", {}).get("repo2env", {}).get("recipe") == "tasksmith"
            and behavior.is_file()
        ):
            check_behavioral_verifier(behavior.read_text())
        new_hash = refresh_identity(copied)
        if new_hash == task_identity(task) and not repair.probe_replacements:
            raise ValueError("Repair did not change the task")
        copied.rename(destination)
    save_record(
        destination.parent / "repair.json",
        {
            "parent_hash": task_identity(task),
            "bundle_hash": new_hash,
            "repair": repair.model_dump(mode="json"),
        },
    )
    return destination


def _append_expected_passes(task: Path, additions: list[str]) -> None:
    # [FUNCTION DOC] HELPER: Appends case IDs to tests/contract.json's expected_passes list and
    # rewrites the file (indent 2, trailing newline). Raises ValueError if the contract file is
    # missing, has duplicate JSON keys, lacks a list of nonempty string IDs, already contains
    # duplicates, or already contains any of the additions.
    def unique_fields(pairs):
        # [FUNCTION DOC] HELPER: JSON object_pairs_hook that builds a dict from key/value pairs and
        # raises ValueError on a duplicate key.
        fields = {}
        for key, value in pairs:
            if key in fields:
                raise ValueError(f"tests/contract.json contains duplicate field: {key}")
            fields[key] = value
        return fields

    path = task / EXPECTED_PASSES_CONTRACT
    if not path.is_file():
        raise ValueError("Appending expected passes requires an existing tests/contract.json file")
    contract = json.loads(path.read_text(), object_pairs_hook=unique_fields)
    existing = contract.get("expected_passes") if isinstance(contract, dict) else None
    if not isinstance(existing, list) or any(
        not isinstance(value, str) or not value.strip() for value in existing
    ):
        raise ValueError("tests/contract.json must contain a list of nonempty expected-pass IDs")
    if len(set(existing)) != len(existing):
        raise ValueError("tests/contract.json contains duplicate expected-pass IDs")
    if set(existing).intersection(additions):
        raise ValueError("Appended expected-pass IDs already exist in tests/contract.json")
    contract["expected_passes"] = [*existing, *additions]
    path.write_text(json.dumps(contract, indent=2) + "\n")


def probe_variant(task: Path, probe: SemanticProbe, destination: Path) -> Path:
    # [FUNCTION DOC] Builds a private semantic-probe variant of a task at destination and returns
    # it. Snapshots the task, moves solution/solve.sh to solution/quality-original-solve.sh and
    # writes a new solve.sh wrapper that runs the original reference, then the probe script via bash
    # -eu -c, then prints the __QUALITY_PROBE_COMPLETED__ receipt. When tests/contract.json declares
    # submitted_files/submitted_roots, it also copies probe_audit.py and a boundary JSON
    # (submitted_files, submitted_roots, immutable_assets, require_valid_python for wrong_solution
    # probes focused on model_behavior/compiled_execution) into solution/ and runs the audit
    # 'before' and 'after' the probe script. Instruction, tests and environment stay unchanged.
    # Raises FileExistsError if destination exists; ValueError if the task has no solution/solve.sh
    # or already uses a reserved probe filename. Side effects: refresh_identity on the variant and
    # probe.json (parent_hash, bundle_hash, probe dump) next to destination.
    """Private reference-then-mutation control; instruction/tests/build stay exact."""
    if destination.exists():
        raise FileExistsError(destination)
    original = task / "solution/solve.sh"
    if not original.is_file():
        raise ValueError("Semantic probes require solution/solve.sh")
    snapshot(task, destination)
    private = destination / "solution/quality-original-solve.sh"
    if private.exists():
        raise ValueError("Task already uses the reserved probe filename")
    (destination / "solution/solve.sh").rename(private)
    before, after = "", ""
    contract_path = task / "tests/contract.json"
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text())
        if contract.get("submitted_files") or contract.get("submitted_roots"):
            # Scope this guard to the explicit collection contract of owned
            # repository tasks. Other Harbor tasks may have different boundaries.
            audit = destination / "solution/quality-probe-audit.py"
            boundary = destination / "solution/quality-probe-contract.json"
            if audit.exists() or boundary.exists():
                raise ValueError("Task already uses a reserved probe audit filename")
            shutil.copyfile(Path(__file__).with_name("probe_audit.py"), audit)
            boundary.write_text(
                json.dumps(
                    {
                        **{
                            key: contract.get(key, {} if key == "immutable_assets" else [])
                            for key in ("submitted_files", "submitted_roots", "immutable_assets")
                        },
                        "require_valid_python": probe.kind == "wrong_solution"
                        and probe.focus in {"model_behavior", "compiled_execution"},
                    }
                )
            )
            command = (
                "python /solution/quality-probe-audit.py "
                "/solution/quality-probe-contract.json /tmp/quality-probe-before.json "
            )
            before, after = command + "before\n", "\n" + command + "after\n"
    wrapper = destination / "solution/solve.sh"
    wrapper.write_text(
        "#!/bin/bash\nset -eu\nbash /solution/quality-original-solve.sh\n"
        + before
        # A successful exit in the supplied script must not skip the trusted
        # change audit or completion receipt. Nonzero exits still stop setup.
        + "bash -eu -c "
        + shlex.quote(probe.script)
        + "\n"
        + after
        + "\nprintf '__QUALITY_PROBE_COMPLETED__\\n'\n"
    )
    wrapper.chmod(0o755)
    identity = refresh_identity(destination)
    save_record(
        destination.parent / "probe.json",
        {
            "parent_hash": task_identity(task),
            "bundle_hash": identity,
            "probe": probe.model_dump(mode="json"),
        },
    )
    return destination


def import_trial(path: Path, task: Path, role: str) -> TrialRecord:
    # [FUNCTION DOC] Imports existing trial evidence for task under a role and returns a
    # TrialRecord. Input path may be a receipt (trial.json, or a directory containing one), a Harbor
    # result.json, or a single-trial job directory. Receipt binding: state must be completed,
    # bundle_hash equal the task identity, trial_id match a slug pattern, exactly one adjacent
    # <trial_id>/*/result.json that is not a symlink and stays inside the receipt dir, and its
    # sha256 must match result_sha256. Otherwise harbor_checksum binding: exactly one result.json
    # containing task_checksum, equal to the parsed task's checksum. Role checks: baseline must use
    # agent 'nop', oracle 'oracle'; rollout must not be nop/oracle/unknown. Reward, exception type,
    # model and agent/exit-code.txt are read into the record. Raises ValueError on any violation
    # (also from task_identity/parse_task).
    """Import a receipt, trial result, or single-trial job; require content binding."""
    identity = task_identity(task)
    receipt = None
    if path.is_dir() and (path / "trial.json").is_file():
        path = path / "trial.json"
    if path.is_file() and path.name == "trial.json":
        receipt = json.loads(path.read_text())
        if receipt.get("state") != "completed" or receipt.get("bundle_hash") != identity:
            raise ValueError("Imported receipt is incomplete or belongs to a different task")
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,60}", receipt.get("trial_id", "")):
            raise ValueError("Imported receipt has an invalid trial ID")
        candidates = list((path.parent / receipt["trial_id"]).glob("*/result.json"))
        if len(candidates) != 1:
            raise ValueError("Receipt must have exactly one adjacent Harbor trial result")
        result = candidates[0]
        if result.is_symlink() or not result.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError("Imported receipt cannot redirect evidence outside its directory")
        if digest(result) != receipt["result_sha256"]:
            raise ValueError("Imported trial result changed after collection")
        binding = "receipt"
    else:
        candidates = (
            [path] if path.is_file() else [path / "result.json", *path.glob("*/result.json")]
        )
        results = []
        for candidate in candidates:
            if candidate.is_file():
                data = json.loads(candidate.read_text())
                if "task_checksum" in data:
                    results.append(candidate)
        if len(results) != 1:
            raise ValueError("Supply exactly one Harbor trial, not a multi-task job")
        result = results[0]
        if json.loads(result.read_text())["task_checksum"] != parse_task(task).checksum:
            raise ValueError("Harbor checksum does not match the supplied task")
        binding = "harbor_checksum"
    data = json.loads(result.read_text())
    exit_path = result.parent / "agent/exit-code.txt"
    agent_config = data.get("config", {}).get("agent", {})
    agent = agent_config.get("name") or agent_config.get("import_path") or "unknown"
    if role in {"baseline", "oracle"} and agent != {"baseline": "nop", "oracle": "oracle"}[role]:
        raise ValueError(f"Wrong agent for imported {role} evidence")
    if role == "rollout" and agent in {"nop", "oracle", "unknown"}:
        raise ValueError("A rollout must come from a solver, not a deterministic control")
    return TrialRecord(
        role=role,
        bundle_hash=identity,
        result=str(result.resolve()),
        result_sha256=digest(result),
        agent=agent,
        model=agent_config.get("model_name"),
        reward=(data.get("verifier_result") or {}).get("rewards", {}).get("reward"),
        exception_type=(data.get("exception_info") or {}).get("exception_type"),
        binding=binding,
        agent_exit_code=int(exit_path.read_text()) if exit_path.exists() else None,
    )
