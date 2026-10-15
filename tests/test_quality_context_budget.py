# [MODULE DOC] Tests for EvidenceContext.read_more budgeting in repo2rlenv.quality.loop.context:
# requested source excerpts (literal searches and explicit line ranges) must fit the remaining
# character budget without shortening previously supplied evidence, collapse overlapping search
# windows, declare omitted windows, raise ValueError with the request and remaining budget when
# nothing fits, and apply multi-read batches atomically. Also checks _search_excerpts' no-match
# message. Uses tmp_path only.
"""Requested source excerpts fit without shortening previously supplied evidence."""

from __future__ import annotations

import pytest

from repo2rlenv.quality.loop.context import EvidenceContext, _search_excerpts
from repo2rlenv.quality.loop.models import ReadRequest


def packed_context(tmp_path, source, *, remaining):
    # [FUNCTION DOC] HELPER: Writes instruction.md and source.txt (with `source`) into tmp_path, builds an
    # EvidenceContext with a 100000-char limit, then pads documents with 'evidence/retained.txt' so
    # exactly `remaining` characters of budget are left. Inputs: tmp_path, source text, keyword-only
    # remaining (int). Output: EvidenceContext.
    (tmp_path / "instruction.md").write_text("Preserve the documented behavior.\n")
    (tmp_path / "source.txt").write_text(source)
    context = EvidenceContext(tmp_path, [], limit=100000)
    used = sum(map(len, context.documents.values()))
    context.documents["evidence/retained.txt"] = "R" * (context.limit - used - remaining)
    return context


def search():
    # [FUNCTION DOC] HELPER: Returns a ReadRequest for a literal search of 'MARKER' in source.txt (line range 1-1).
    return ReadRequest(path="source.txt", query="MARKER", start_line=1, end_line=1)


def test_overlapping_search_windows_do_not_duplicate_source_to_exhaust_budget(tmp_path):
    # [FUNCTION DOC] With 8 nearby MARKER lines and 5092 chars remaining, asserts the search excerpt fits, merges
    # overlapping windows (fewer than 8 '[Lines ' headers) yet contains all 8 targets, notes the
    # 8-window limit, and leaves prior documents and the total within budget.
    lines = [f"# line {index:03}: preserve this exact source text\n" for index in range(100)]
    for index in (7, 13, 24, 25, 29, 32, 36, 46):
        lines[index] = f"def target_{index}():  # MARKER\n"
    context = packed_context(tmp_path, "".join(lines), remaining=5092)
    before = dict(context.documents)
    context.read_more([search()])
    excerpt = context.documents["source.txt:search=MARKER"]
    assert len(excerpt) <= 5092
    assert excerpt.count("[Lines ") < 8
    assert excerpt.count("def target_") == 8
    assert "Search limited to the first 8 matching windows" in excerpt
    assert all(context.documents[key] == value for key, value in before.items())
    assert sum(map(len, context.documents.values())) <= context.limit


def test_budget_limited_search_keeps_complete_windows_and_declares_omissions(tmp_path):
    # [FUNCTION DOC] With 12 MARKERs on one very long line, asserts only one complete window (columns 1-3000 of
    # line 1) is kept, '7 additional search window(s) omitted' is declared, and existing
    # documents/total budget are preserved.
    source = ("MARKER " + "x" * 5000) * 12
    context = packed_context(tmp_path, source, remaining=5092)
    before = dict(context.documents)
    context.read_more([search()])
    excerpt = context.documents["source.txt:search=MARKER"]
    assert excerpt.count("MARKER") == 1
    assert "7 additional search window(s) omitted" in excerpt
    assert "Line 1, columns 1-3000" in excerpt
    assert source[:3000] in excerpt
    assert all(context.documents[key] == value for key, value in before.items())
    assert sum(map(len, context.documents.values())) <= context.limit


def test_repeated_read_never_shortens_an_existing_citable_excerpt(tmp_path):
    # [FUNCTION DOC] After a first search and filling the rest of the budget, asserts repeating the same search
    # leaves all documents unchanged.
    context = packed_context(tmp_path, ("MARKER " + "x" * 5000) * 12, remaining=26000)
    context.read_more([search()])
    context.documents["evidence/later.txt"] = "L" * (
        context.limit - sum(map(len, context.documents.values()))
    )
    before = dict(context.documents)
    context.read_more([search()])
    assert context.documents == before


def test_unfittable_search_reports_path_and_budget_without_mutation(tmp_path):
    # [FUNCTION DOC] With only 100 chars left, asserts read_more raises ValueError mentioning the path, the budget,
    # '100 characters remain' and a hint to use a short line range, without mutating documents.
    context = packed_context(tmp_path, "MARKER " + "x" * 5000, remaining=100)
    before = dict(context.documents)
    with pytest.raises(
        ValueError, match=r"source\.txt: Additional reads exceeded context budget"
    ) as error:
        context.read_more([search()])
    assert "100 characters remain" in str(error.value)
    assert "short explicit line range" in str(error.value)
    assert context.documents == before


def test_requested_full_range_is_not_silently_clipped(tmp_path):
    # [FUNCTION DOC] Asserts an explicit L1-L10 request needing 1010 chars with 200 remaining raises a precise
    # ValueError instead of clipping, with no mutation.
    context = packed_context(tmp_path, ("x" * 100 + "\n") * 10, remaining=200)
    before = dict(context.documents)
    with pytest.raises(ValueError) as error:
        context.read_more([ReadRequest(path="source.txt", query=None, start_line=1, end_line=10)])
    assert "source.txt:L1-L10 requires 1010 characters but 200 remain" in str(error.value)
    assert context.documents == before


def test_failed_multi_read_is_atomic_and_describes_the_unfittable_request(tmp_path):
    # [FUNCTION DOC] Asserts that when the second of two reads cannot fit, ValueError names 'source.txt:L2-L4' and
    # the first read is not applied either.
    context = packed_context(tmp_path, ("x" * 100 + "\n") * 10, remaining=200)
    before = dict(context.documents)
    with pytest.raises(ValueError, match=r"source\.txt:L2-L4"):
        context.read_more(
            [
                ReadRequest(path="source.txt", query=None, start_line=1, end_line=1),
                ReadRequest(path="source.txt", query=None, start_line=2, end_line=4),
            ]
        )
    assert context.documents == before


def test_bounded_search_preserves_no_match_result():
    # [FUNCTION DOC] Asserts _search_excerpts returns '[No literal matches found]' when the query is absent.
    assert (
        _search_excerpts(["unrelated text"], "MARKER", maximum=100) == "[No literal matches found]"
    )
