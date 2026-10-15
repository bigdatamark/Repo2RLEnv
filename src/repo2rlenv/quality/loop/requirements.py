# [MODULE DOC] Explicit quality requirements (currently required probe focus areas) stored in
# task.toml under metadata.repo2env.quality_requirements and bound to the task's bundle identity.
# Adding requirements produces an annotated copy with a new identity rather than editing the
# original.
"""Explicit quality requirements bound to a task's executable bundle identity."""

from __future__ import annotations

import shutil
import tempfile
import tomllib
from collections.abc import Sequence
from pathlib import Path

import tomli_w
from pydantic import Field, model_validator

from repo2rlenv.quality.loop.artifacts import refresh_identity, task_identity
from repo2rlenv.quality.loop.models import ProbeFocus, Record


class QualityRequirements(Record):
    # [CLASS DOC] Model for metadata.repo2env.quality_requirements: probe_focus, a list of up to 4
    # unique ProbeFocus values (normalized to sorted order).
    probe_focus: list[ProbeFocus] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def unique_focus(self):
        # [FUNCTION DOC] Validator: raises ValueError on duplicate focus entries, then sorts
        # probe_focus in place. Returns self.
        if len(set(self.probe_focus)) != len(self.probe_focus):
            raise ValueError("Required probe focus entries must be unique")
        self.probe_focus = sorted(self.probe_focus)
        return self


def task_probe_focus(task: Path) -> set[ProbeFocus]:
    # [FUNCTION DOC] Returns the set of required probe focus areas from the task's task.toml (empty
    # when absent). Raises ValueError if metadata or metadata.repo2env is not a table, or pydantic
    # ValidationError for invalid requirements.
    """Read strict, explicit requirements; absent metadata preserves legacy behavior."""
    configuration = tomllib.loads((task / "task.toml").read_text())
    metadata = configuration.get("metadata", {})
    if not isinstance(metadata, dict) or not isinstance(metadata.get("repo2env", {}), dict):
        raise ValueError("Task quality requirements must belong to metadata.repo2env")
    requirements = metadata.get("repo2env", {}).get("quality_requirements", {})
    return set(QualityRequirements.model_validate(requirements).probe_focus)


def with_probe_requirements(task: Path, destination: Path, required: Sequence[ProbeFocus]) -> Path:
    # [FUNCTION DOC] Returns a task path whose requirements include required. If already satisfied,
    # returns the resolved original. Otherwise copies the task (keeping symlinks) to a temp dir
    # beside destination, writes the merged sorted probe_focus and bundle_hash into task.toml, calls
    # refresh_identity (which resets evaluation), and renames it to destination unless an identical
    # destination already exists. Raises ValueError if destination is a symlink or inside the task,
    # if the task changed during copying/annotation, or if an existing destination has a different
    # identity.
    """Publish an annotated copy, preserving originals and refusing stale evidence.

    A semantic no-op returns the original. A changed requirement set creates a new
    identity and resets evaluation through refresh_identity. An identical existing
    destination is resumable; a conflicting destination is never overwritten.
    """
    task = task.resolve()
    source_hash = task_identity(task)
    requested = set(QualityRequirements(probe_focus=list(required)).probe_focus)
    existing = task_probe_focus(task)
    if requested <= existing:
        return task
    destination = destination.absolute()
    if destination.is_symlink() or destination.resolve().is_relative_to(task):
        raise ValueError("Annotated quality input must be outside the original task")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".quality-input-", dir=destination.parent) as temporary:
        copied = Path(temporary) / task.name
        shutil.copytree(task, copied, symlinks=True)
        if task_identity(copied) != source_hash:
            raise ValueError("Task changed while copying its quality requirements")
        path = copied / "task.toml"
        configuration = tomllib.loads(path.read_text())
        metadata = configuration.setdefault("metadata", {}).setdefault("repo2env", {})
        metadata["quality_requirements"] = {"probe_focus": sorted(existing | requested)}
        metadata["bundle_hash"] = source_hash
        path.write_text(tomli_w.dumps(configuration))
        identity = refresh_identity(copied)
        if destination.exists() and task_identity(destination) != identity:
            raise ValueError("Existing quality input has different task content or requirements")
        if task_identity(task) != source_hash:
            raise ValueError("Original task changed while annotating its quality requirements")
        if not destination.exists():
            copied.rename(destination)
    return destination
