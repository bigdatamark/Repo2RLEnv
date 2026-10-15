# [MODULE DOC] Tests for emitter.bundle TaskBundle/write_bundle/inspect_bundle/bundle_hash:
# integrity, identity sensitivity, path traversal, overwrite/resume, symlinks and privileged modes.
from __future__ import annotations

import copy

import pytest

from repo2rlenv.emitter.bundle import (
    TaskBundle,
    TaskFile,
    bundle_hash,
    inspect_bundle,
    write_bundle,
)


@pytest.fixture
def bundle():
    # [FUNCTION DOC] HELPER (fixture): A TaskBundle with Dockerfile, a binary file and executable
    # solve/test scripts.
    return TaskBundle(
        name="fixture",
        org="tests",
        instruction="Restore the expected behavior.",
        metadata={"recipe": "fixture", "recipe_version": "1", "reward_kinds": ["test_execution"]},
        files={
            "environment/Dockerfile": TaskFile.text("FROM python:3.12-slim\n"),
            "environment/input.bin": TaskFile(b"\x00\xff"),
            "solution/solve.sh": TaskFile.text("#!/bin/sh\nexit 0\n", executable=True),
            "tests/test.sh": TaskFile.text("#!/bin/sh\nexit 1\n", executable=True),
        },
    )


def test_roundtrip_and_verifier_tamper(bundle, tmp_path):
    # [FUNCTION DOC] Written bundle passes integrity, preserves binary bytes, and fails integrity
    # after tests/test.sh is edited.
    path = write_bundle(bundle, tmp_path)
    assert inspect_bundle(path)["integrity_passed"]
    assert (path / "environment/input.bin").read_bytes() == b"\x00\xff"
    (path / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    assert not inspect_bundle(path)["integrity_passed"]


@pytest.mark.parametrize("change", ["image", "mode", "reward", "timeout"])
def test_execution_contract_changes_identity(bundle, change):
    # [FUNCTION DOC] Parametrized changes to image, file mode, reward kinds or verifier timeout
    # change bundle_hash.
    revised = copy.deepcopy(bundle)
    if change == "image":
        revised.files["environment/Dockerfile"] = TaskFile.text("FROM python:3.13-slim\n")
    elif change == "mode":
        revised.files["environment/input.bin"] = TaskFile(b"\x00\xff", executable=True)
    elif change == "reward":
        revised.metadata["reward_kinds"] = ["different"]
    else:
        revised.verifier_timeout_sec += 1
    assert bundle_hash(bundle) != bundle_hash(revised)


@pytest.mark.parametrize(
    "name",
    ["../secret", "/tmp/secret", "tests/../secret", "tests//file", "tests/./file", "tests\\escape"],
)
def test_asset_traversal_rejected_before_emission(bundle, name, tmp_path):
    # [FUNCTION DOC] Invalid asset paths raise ValueError 'asset path' and nothing is written.
    bundle.files[name] = TaskFile.text("invalid")
    with pytest.raises(ValueError, match="asset path"):
        write_bundle(bundle, tmp_path)
    assert not list(tmp_path.iterdir())


def test_never_overwrites_existing_task(bundle, tmp_path):
    # [FUNCTION DOC] A second write raises FileExistsError; the original stays intact.
    path = write_bundle(bundle, tmp_path)
    with pytest.raises(FileExistsError):
        write_bundle(bundle, tmp_path)
    assert inspect_bundle(path)["integrity_passed"]


def test_resume_recovers_only_an_identical_completed_export(bundle, tmp_path):
    # [FUNCTION DOC] resume=True returns the existing path for an identical bundle and raises 'does
    # not match' for a changed one.
    path = write_bundle(bundle, tmp_path)
    assert write_bundle(bundle, tmp_path, resume=True) == path
    bundle.instruction = "A different task must not reuse an existing export."
    with pytest.raises(ValueError, match="does not match"):
        write_bundle(bundle, tmp_path, resume=True)


def test_symlink_and_privileged_mode_rejected(bundle, tmp_path):
    # [FUNCTION DOC] inspect_bundle raises for a symlink and for a setuid mode.
    path = write_bundle(bundle, tmp_path)
    (path / "tests/link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        inspect_bundle(path)
    (path / "tests/link").unlink()
    (path / "tests/test.sh").chmod(0o4755)
    with pytest.raises(ValueError, match="mode"):
        inspect_bundle(path)
