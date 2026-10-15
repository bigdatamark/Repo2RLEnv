# [MODULE DOC] Unit tests for repo2rlenv.quality.loop.probe_audit (submission_files / changed_files): semantic
# probes must change collected submission source; comment-only edits, unrelated files and immutable
# assets do not count; paths must be canonical and non-symlinked. Uses tmp_path only; no network.
from pathlib import Path

import pytest

from repo2rlenv.quality.loop.probe_audit import changed_files, submission_files


def test_probe_requires_a_change_to_collected_source(tmp_path):
    # [FUNCTION DOC] Verifies changed_files raises ValueError('unchanged') when only an uncollected file or a
    # comment-only edit occurs, and reports src/answer.py once its syntax actually changes. Input:
    # tmp_path fixture. Side effects: writes files under tmp_path.
    source = tmp_path / "src"
    source.mkdir()
    module = source / "answer.py"
    module.write_text("raise RuntimeError('must never import this file')\n")
    contract = {"submitted_roots": ["src"], "submitted_files": ["src/answer.py"]}
    before = submission_files(tmp_path, contract)
    (tmp_path / "extra-check.txt").write_text("passed")
    with pytest.raises(ValueError, match="unchanged"):
        changed_files(before, submission_files(tmp_path, contract))
    module.write_text("# A comment is not an alternative implementation\n" + module.read_text())
    with pytest.raises(ValueError, match="unchanged"):
        changed_files(before, submission_files(tmp_path, contract))
    module.write_text("def answer(): return 42\n")
    assert set(changed_files(before, submission_files(tmp_path, contract))) == {"src/answer.py"}


def test_probe_tracks_new_and_removed_modules_but_not_immutable_assets(tmp_path):
    # [FUNCTION DOC] Verifies deletions and new modules under submitted_roots are reported (after=None /
    # before=None) while edits to immutable_assets are ignored (raising 'unchanged'). Input:
    # tmp_path. Side effects: writes/deletes files in tmp_path.
    source = tmp_path / "src"
    source.mkdir()
    module = source / "answer.py"
    module.write_text("value = 1\n")
    asset = source / "vocab.json"
    asset.write_text("{}")
    contract = {
        "submitted_roots": ["src"],
        "submitted_files": ["src/answer.py", "src/vocab.json"],
        "immutable_assets": {"src/vocab.json": "protected"},
    }
    before = submission_files(tmp_path, contract)
    asset.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="unchanged"):
        changed_files(before, submission_files(tmp_path, contract))
    module.unlink()
    (source / "helper.py").write_text("value = 1\n")
    changes = changed_files(before, submission_files(tmp_path, contract))
    assert set(changes) == {"src/answer.py", "src/helper.py"}
    assert changes["src/answer.py"]["after"] is None
    assert changes["src/helper.py"]["before"] is None


@pytest.mark.parametrize(
    "name", ["../answer.py", "/answer.py", "src/../answer.py", "src\\answer.py"]
)
def test_probe_rejects_noncanonical_paths(tmp_path, name):
    # [FUNCTION DOC] Parametrized over non-canonical names (parent traversal, absolute, embedded '..', backslash);
    # verifies submission_files raises ValueError matching 'canonical'. Inputs: tmp_path, name.
    with pytest.raises(ValueError, match="canonical"):
        submission_files(tmp_path, {"submitted_files": [name]})


def test_probe_does_not_follow_submission_links(tmp_path):
    # [FUNCTION DOC] Verifies submission_files refuses a symlinked submitted file with ValueError('symlink'). Side
    # effects: creates a symlink in tmp_path pointing at this test file.
    (tmp_path / "source.py").symlink_to(Path(__file__))
    with pytest.raises(ValueError, match="symlink"):
        submission_files(tmp_path, {"submitted_files": ["source.py"]})


def test_probe_keeps_byte_evidence_when_python_syntax_cannot_be_parsed(tmp_path):
    # [FUNCTION DOC] Verifies unparsable Python yields syntax_sha256=None but byte-level change detection still
    # reports a change after a valid rewrite. Side effects: writes answer.py in tmp_path.
    module = tmp_path / "answer.py"
    module.write_text("not valid Python !")
    contract = {"submitted_files": ["answer.py"]}
    before = submission_files(tmp_path, contract)
    assert before["answer.py"]["syntax_sha256"] is None
    module.write_text("def answer(): return 42\n")
    assert changed_files(before, submission_files(tmp_path, contract))
