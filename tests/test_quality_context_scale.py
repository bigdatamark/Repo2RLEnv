# [MODULE DOC] Scale tests for EvidenceContext (repo2rlenv.quality.loop.context): large file
# inventories stay addressable within a bounded payload (grouped by directory, or moved to a
# searchable inventory document when too large), rejected oversized reads leave the context
# reusable, and a selected small test suite keeps its helper and final assertions within the
# character budget. Uses tmp_path only.
from __future__ import annotations

import json

import pytest

from repo2rlenv.quality.loop.context import EvidenceContext
from repo2rlenv.quality.loop.models import ReadRequest


@pytest.mark.parametrize("count", [800, 2400])
def test_large_inventory_retains_every_addressable_file(tmp_path, count):
    # [FUNCTION DOC] For 800 or 2400 generated source files, asserts the JSON payload stays <= 86000 chars; at 800
    # the inventory_by_directory map reconstructs exactly the full inventory, at 2400 the
    # inventory_document catalogue is searchable for the last module; finally reads line 1 of the
    # last module and checks its content.
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("Repair the package behavior.")
    for index in range(count):
        path = task / "environment/source/package" / f"module_{index:04}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"VALUE = {index}\n")
    context = EvidenceContext(task, [], limit=16000)
    payload = json.loads(context.payload())
    assert len(json.dumps(payload)) <= 86000
    requested = f"environment/source/package/module_{count - 1:04}.py"
    if count == 800:
        entries = {
            "/".join(filter(None, [prefix, name])): size
            for prefix, files in payload["inventory_by_directory"].items()
            for name, size in files.items()
        }
        assert entries == {item["path"]: item["bytes"] for item in context.inventory}
    else:
        catalogue = payload["inventory_document"]
        assert payload["inventory_files"] == len(context.inventory)
        query = f"module_{count - 1:04}.py"
        context.read_more([ReadRequest(path=catalogue, start_line=1, end_line=1, query=query)])
        assert requested in context.documents[catalogue + ":search=" + query]
    context.read_more([ReadRequest(path=requested, start_line=1, end_line=1, query=None)])
    assert context.documents[requested + ":L1-L1"] == f"VALUE = {count - 1}\n"


def test_rejected_large_read_does_not_poison_next_review(tmp_path):
    # [FUNCTION DOC] Asserts an over-budget 100-line read raises 'context budget' without mutating documents, a
    # subsequent 1-line read succeeds, and payload() still works.
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("Repair the package behavior.")
    (task / "large.txt").write_text(("x" * 1000 + "\n") * 100)
    context = EvidenceContext(task, [], limit=16000)
    before = dict(context.documents)
    with pytest.raises(ValueError, match="context budget"):
        context.read_more([ReadRequest(path="large.txt", start_line=1, end_line=100, query=None)])
    assert context.documents == before
    context.read_more([ReadRequest(path="large.txt", start_line=1, end_line=1, query=None)])
    assert context.documents["large.txt:L1-L1"] == "x" * 1000 + "\n"
    context.payload()


@pytest.mark.parametrize("limit", [16000, 64000])
def test_selected_small_suite_keeps_helpers_and_final_assertions_within_budget(tmp_path, limit):
    # [FUNCTION DOC] For limits 16000 and 64000, writes a contract-selected test file with a helper, long filler
    # and a final test; asserts the helper is always present, the whole file is included at 64000,
    # and at 16000 the file is marked partial in context.omitted while total size stays within the
    # limit.
    (tmp_path / "instruction.md").write_text("Implement the requested public behavior.")
    tests = tmp_path / "tests/source/tests"
    tests.mkdir(parents=True)
    content = (
        "def fixture_helper():\n    return 'independent fixture'\n\n"
        + "# Additional fixture explanation.\n" * 450
        + "def test_final_behavior():\n    assert fixture_helper() == 'independent fixture'\n"
    )
    (tests / "behavior.py").write_text(content)
    (tmp_path / "tests/contract.json").write_text(
        json.dumps({"test_paths": ["tests/behavior.py"], "expected_passes": []})
    )
    context = EvidenceContext(tmp_path, [], limit=limit)
    name = "tests/source/tests/behavior.py"
    assert "def fixture_helper" in context.documents[name]
    if limit == 64000:
        assert context.documents[name] == content
    else:
        assert name + ": partial text; request line ranges if needed" in context.omitted
        assert sum(map(len, context.documents.values())) <= limit
    context.payload()
