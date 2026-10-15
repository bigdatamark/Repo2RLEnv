# [MODULE DOC] Tests for the terminalworld recipe pieces: EnvironmentBuild validation,
# recording loading with PII screening, the HTML MetadataParser, resumable recording acquisition
# receipts, and filesystem snapshot diffing.
from __future__ import annotations

import hashlib
import json

import pytest

from repo2rlenv.pipelines.recipes.terminalworld.capture import changes, filesystem_state
from repo2rlenv.pipelines.recipes.terminalworld.source import MetadataParser, load_recordings


def test_recording_environment_can_start_without_extra_fixture_files():
    # [FUNCTION DOC] EnvironmentBuild
    # accepts an empty file list, but a setup that switches USER raises ValueError ("cannot
    # replace").
    from repo2rlenv.pipelines.recipes.terminalworld.materialize import EnvironmentBuild

    environment = EnvironmentBuild(
        environment_setup="RUN mkdir -p /workspace/project",
        environment_files=[],
        solution_shell="#!/bin/bash\nset -eu\necho result > /workspace/project/output.txt\n",
        self_review="This workflow creates its deliverables from an empty workspace.",
    )
    assert environment.environment_files == []
    with pytest.raises(ValueError, match="cannot replace"):
        environment.model_validate({**environment.model_dump(), "environment_setup": "USER root"})


def test_recording_screen_excludes_flagged_text_from_author_input(tmp_path):
    # [FUNCTION DOC] A recording containing a
    # public IPv4 address is flagged pii_public_ipv4, its transcript is blanked, and a 64-char hash
    # is
    # still recorded.
    first = tmp_path / "123"
    first.mkdir()
    (first / "info.json").write_text(json.dumps({"title": "An input recording", "id": "123"}))
    (first / "recording.txt").write_text("download from 203.0.113.4 then run a task")
    record = load_recordings(tmp_path)[0]
    assert record["filter_flags"] == ["pii_public_ipv4"]
    assert record["transcript"] == ""
    assert len(record["transcript_sha256"]) == 64


def test_metadata_reads_description_and_ignores_unrelated_secrets():
    # [FUNCTION DOC] The parser keeps the
    # og:title, ignores the csrf meta, and collects nested description text only.
    parser = MetadataParser()
    parser.feed(
        '<meta name="csrf-token" content="ignore-me"><meta property="og:title" content="Task title"><div class="description">Workflow <div>details</div> follow.</div><div>unrelated</div>'
    )
    assert parser.metadata == {"title": "Task title"}
    assert "".join(parser.description) == "Workflow details follow."


def test_acquisition_marks_partial_inputs_and_binds_completion_to_cache(tmp_path, monkeypatch):
    # [FUNCTION DOC] fetch_recordings
    # writes acquisition.json as "running" during downloads, "interrupted" when a download raises,
    # and
    # "completed" with the SHA-256 of retrieval.json on success.
    from repo2rlenv.pipelines.recipes.terminalworld import source

    def interrupted(client, url, limit):
        # [FUNCTION DOC] HELPER:
        # Fake download asserting state "running" then raising RuntimeError.
        assert json.loads((tmp_path / "acquisition.json").read_text())["state"] == "running"
        raise RuntimeError("Disconnected before the input shard finished")

    monkeypatch.setattr(source, "_download", interrupted)
    with pytest.raises(RuntimeError, match="Disconnected"):
        source.fetch_recordings(["123"], tmp_path)
    assert json.loads((tmp_path / "acquisition.json").read_text())["state"] == "interrupted"

    def download(client, url, limit):
        # [FUNCTION DOC] HELPER:
        # Fake download asserting state "running" and serving robots, a transcript or a metadata
        # page.
        assert json.loads((tmp_path / "acquisition.json").read_text())["state"] == "running"
        if url.endswith("robots.txt"):
            return "User-agent: *\nAllow: /\n"
        return (
            "echo fixture\n" * 20
            if url.endswith(".txt")
            else '<meta property="og:title" content="Fixture">'
        )

    monkeypatch.setattr(source, "_download", download)
    monkeypatch.setattr(source.time, "sleep", lambda _: None)
    assert source.fetch_recordings(["123"], tmp_path)["123"]["downloaded"]
    done = json.loads((tmp_path / "acquisition.json").read_text())
    assert done["state"] == "completed"
    assert (
        done["retrieval_sha256"]
        == hashlib.sha256((tmp_path / "retrieval.json").read_bytes()).hexdigest()
    )


def test_snapshot_detects_new_modified_and_deleted_files(tmp_path):
    # [FUNCTION DOC] changes(before, after) between two
    # filesystem_state snapshots reports created, modified and deleted paths.
    edited = tmp_path / "edited.txt"
    edited.write_text("before")
    removed = tmp_path / "removed.txt"
    removed.write_text("remove")
    before = filesystem_state([tmp_path])
    edited.write_text("after!")
    removed.unlink()
    created = tmp_path / "created.txt"
    created.write_text("created")
    delta = changes(before, filesystem_state([tmp_path]))
    assert delta == {
        "created": [str(created)],
        "modified": [str(edited)],
        "deleted": [str(removed)],
    }
