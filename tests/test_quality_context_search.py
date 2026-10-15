# [MODULE DOC] Tests for literal search excerpts in EvidenceContext.read_more
# (repo2rlenv.quality.loop.context) on pathological inputs: minified single-line traces are
# excerpted by column windows rather than reading the whole line, and a match next to an oversized
# unrelated line is still returned compactly. Uses tmp_path only.
from __future__ import annotations

import json

from repo2rlenv.quality.loop.context import EvidenceContext
from repo2rlenv.quality.loop.models import ReadRequest


def test_search_handles_minified_trace_without_reading_entire_line(tmp_path):
    # [FUNCTION DOC] Writes a one-line JSON trace with 12 'write marker' hits in long strings; asserts the search
    # excerpt is < 25000 chars, contains 8 matches (window cap), notes omitted surrounding text and
    # uses 'Line 1, columns' headers; payload() still works.
    (tmp_path / "instruction.md").write_text("Repair the package behavior.")
    trace = json.dumps({"steps": ["write marker " + "x" * 5000 for _ in range(12)]})
    (tmp_path / "trajectory.json").write_text(trace)
    context = EvidenceContext(tmp_path, [], limit=32000)
    context.read_more(
        [ReadRequest(path="trajectory.json", query="write marker", start_line=1, end_line=1)]
    )
    excerpt = context.documents["trajectory.json:search=write marker"]
    assert len(excerpt) < 25000
    assert excerpt.count("write marker") == 8
    assert "surrounding text omitted" in excerpt
    assert "Line 1, columns" in excerpt
    context.payload()


def test_search_preserves_match_next_to_an_oversized_unrelated_line(tmp_path):
    # [FUNCTION DOC] Asserts a match on line 2 following a 50000-char line is returned with header 'Line 2, columns
    # 1-19' in an excerpt under 1000 chars.
    (tmp_path / "instruction.md").write_text("Repair the package behavior.")
    (tmp_path / "source.txt").write_text("x" * 50000 + "\nTHE RELEVANT MATCH\n")
    context = EvidenceContext(tmp_path, [], limit=16000)
    context.read_more([ReadRequest(path="source.txt", query="RELEVANT", start_line=1, end_line=1)])
    excerpt = context.documents["source.txt:search=RELEVANT"]
    assert "THE RELEVANT MATCH" in excerpt
    assert "Line 2, columns 1-19" in excerpt
    assert len(excerpt) < 1000
