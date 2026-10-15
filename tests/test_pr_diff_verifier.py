# [MODULE DOC] Unit tests for pipelines._pr_diff_verifier, the stdlib-only in-container pr_diff reward, tested as
# ordinary Python: parsing primitives (file_paths, hunk_ranges), the six components (format_valid,
# size_sanity, file_targeting F1, region_overlap with 5-line slack, changes-only similarity,
# llm_judge), llm_judge routing (Anthropic Messages API by default, OpenAI-compatible
# /chat/completions when an endpoint is set) with urllib.request.urlopen mocked, env-driven judge
# config (_judge_config_from_env), combine's weight redistribution when components are None, and the
# retuned _DEFAULT_WEIGHTS. Module constants: _ENDPOINT (fake local server URL), _OPENAI_STYLE_OK
# (canned OpenAI reply, score 0.6), _JUDGE_VARS (env vars cleared per test), _WEIGHTS (sample
# weights).
# Side effects: env changes via monkeypatch only; no real network.
# KNOWN ISSUE: the module docstring says the verifier is base64-baked into pr_diff tasks, but
# pr_diff
# now ships it as a plain tests/verifier.py aux file (see test_pipeline_pr_diff.py), so that wording
# is
# stale.
"""Unit tests for `pipelines._pr_diff_verifier` — the in-container reward.

This is the same module that's base64-baked into pr_diff tasks. Testing
it as ordinary Python (rather than as a string blob) catches regressions
without spinning up Docker.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from repo2rlenv.pipelines._pr_diff_verifier import (
    _judge_config_from_env,
    _normalize_changes_only,
    combine,
    file_paths,
    file_targeting,
    format_valid,
    hunk_ranges,
    llm_judge,
    region_overlap,
    similarity,
    size_sanity,
)

# ---------------------------------------------------------------------------
# Parsing primitives
# ---------------------------------------------------------------------------


def test_file_paths_extracts_b_paths() -> None:
    # [FUNCTION DOC] Checks file_paths returns the set of b/ paths from 'diff --git' headers.
    diff = (
        "diff --git a/src/foo.py b/src/foo.py\n"
        "index 1..2 100644\n"
        "diff --git a/tests/x.py b/tests/x.py\n"
    )
    assert file_paths(diff) == {"src/foo.py", "tests/x.py"}


def test_file_paths_empty() -> None:
    # [FUNCTION DOC] Checks an empty diff yields an empty set.
    assert file_paths("") == set()


def test_hunk_ranges_single_file() -> None:
    # [FUNCTION DOC] Checks hunk_ranges maps a file to inclusive new-side (start, end) ranges from '+start,count'
    # headers.
    diff = "diff --git a/foo.py b/foo.py\n@@ -10,3 +10,5 @@\n@@ -100,1 +102,1 @@\n"
    out = hunk_ranges(diff)
    assert out == {"foo.py": [(10, 14), (102, 102)]}


def test_hunk_ranges_no_count_default_one() -> None:
    # [FUNCTION DOC] Checks a hunk header without a count is treated as a single line.
    """`@@ -10 +10 @@` (no comma-count) means 1 line."""
    diff = "diff --git a/x b/x\n@@ -10 +10 @@\n"
    assert hunk_ranges(diff) == {"x": [(10, 10)]}


# ---------------------------------------------------------------------------
# format_valid
# ---------------------------------------------------------------------------


def test_format_valid_real_diff() -> None:
    # [FUNCTION DOC] Checks a diff with header and +/- lines scores 1.0.
    assert format_valid("diff --git a/x.py b/x.py\n@@ -1 +1 @@\n-old\n+new\n") == 1.0


def test_format_valid_empty_zero() -> None:
    # [FUNCTION DOC] Checks an empty diff scores 0.0.
    assert format_valid("") == 0.0


def test_format_valid_no_diff_header_zero() -> None:
    # [FUNCTION DOC] Checks +/- lines without a 'diff --git' header score 0.0.
    """Just `+/-` lines with no `diff --git` header isn't a real patch."""
    assert format_valid("+something\n-something_else\n") == 0.0


def test_format_valid_header_no_changes_zero() -> None:
    # [FUNCTION DOC] Checks a header with only ---/+++ file markers and no changes scores 0.0.
    """Header but no +/- changes (only file markers) isn't a real patch."""
    assert format_valid("diff --git a/x b/x\n--- a/x\n+++ b/x\n") == 0.0


# ---------------------------------------------------------------------------
# file_targeting (F1)
# ---------------------------------------------------------------------------


def test_file_targeting_exact_match_one() -> None:
    # [FUNCTION DOC] Checks identical file sets score 1.0.
    diff = "diff --git a/foo.py b/foo.py\n"
    assert file_targeting(diff, diff) == 1.0


def test_file_targeting_no_overlap_zero() -> None:
    # [FUNCTION DOC] Checks disjoint file sets score 0.0.
    o = "diff --git a/foo.py b/foo.py\n"
    p = "diff --git a/bar.py b/bar.py\n"
    assert file_targeting(o, p) == 0.0


def test_file_targeting_partial_recall_two_of_three() -> None:
    # [FUNCTION DOC] Checks F1 on TP=2, FN=1, FP=2 equals 4/7.
    """Oracle touches {a, b, c}; predicted touches {a, b, d, e}. F1 should
    give substantial credit (TP=2, FN=1, FP=2 → F1 = 4/(4+1+2) ≈ 0.57)."""
    o = "diff --git a/a b/a\ndiff --git a/b b/b\ndiff --git a/c b/c\n"
    p = "diff --git a/a b/a\ndiff --git a/b b/b\ndiff --git a/d b/d\ndiff --git a/e b/e\n"
    got = file_targeting(o, p)
    assert got == pytest.approx(4 / 7)


def test_file_targeting_is_not_jaccard() -> None:
    # [FUNCTION DOC] Sanity check the same case scores above 0.5 (F1, not Jaccard's 0.4).
    """Sanity check that we use F1, not Jaccard. Same case as above:
    Jaccard would be 2/5 = 0.4; F1 = 4/7 ≈ 0.57. Verify we get F1.
    """
    o = "diff --git a/a b/a\ndiff --git a/b b/b\ndiff --git a/c b/c\n"
    p = "diff --git a/a b/a\ndiff --git a/b b/b\ndiff --git a/d b/d\ndiff --git a/e b/e\n"
    got = file_targeting(o, p)
    assert got > 0.5  # F1 region, not Jaccard


def test_file_targeting_both_empty_one() -> None:
    # [FUNCTION DOC] Checks two empty diffs score 1.0.
    assert file_targeting("", "") == 1.0


def test_file_targeting_predicted_empty_zero() -> None:
    # [FUNCTION DOC] Checks an empty prediction against a non-empty oracle scores 0.0.
    o = "diff --git a/foo.py b/foo.py\n"
    assert file_targeting(o, "") == 0.0


# ---------------------------------------------------------------------------
# size_sanity
# ---------------------------------------------------------------------------


def test_size_sanity_equal_sizes_one() -> None:
    # [FUNCTION DOC] Checks equal-size diffs score 1.0.
    diff = "diff --git a/x b/x\n@@ -1 +1 @@\n-old\n+new\n"
    assert size_sanity(diff, diff) == 1.0


def test_size_sanity_rampage_low() -> None:
    # [FUNCTION DOC] Checks a 20-changed-line prediction vs a 2-line oracle scores 2/20 (min/max ratio).
    """Predicted is 10x oracle → score ~0.1."""
    o = "diff --git a/x b/x\n+a\n-a\n"  # 2 lines
    p_lines = "\n".join(["+a"] * 20)  # 20 lines
    p = f"diff --git a/x b/x\n{p_lines}\n"
    s = size_sanity(o, p)
    assert s == pytest.approx(2 / 20)


def test_size_sanity_predicted_empty_zero() -> None:
    # [FUNCTION DOC] Checks an empty prediction scores 0.0.
    o = "diff --git a/x b/x\n+a\n-a\n"
    assert size_sanity(o, "") == 0.0


def test_size_sanity_both_empty_one() -> None:
    # [FUNCTION DOC] Checks two empty diffs score 1.0.
    assert size_sanity("", "") == 1.0


# ---------------------------------------------------------------------------
# region_overlap
# ---------------------------------------------------------------------------


def test_region_overlap_exact_match_one() -> None:
    # [FUNCTION DOC] Checks identical hunks score 1.0.
    diff = "diff --git a/x b/x\n@@ -10,3 +10,5 @@\n"
    assert region_overlap(diff, diff) == 1.0


def test_region_overlap_wrong_file_zero() -> None:
    # [FUNCTION DOC] Checks same ranges in a different file score 0.0.
    o = "diff --git a/foo b/foo\n@@ -10,3 +10,5 @@\n"
    p = "diff --git a/bar b/bar\n@@ -10,3 +10,5 @@\n"
    assert region_overlap(o, p) == 0.0


def test_region_overlap_off_by_three_lines_within_slack() -> None:
    # [FUNCTION DOC] Checks a hunk a few lines away still matches within the 5-line slack.
    """Slack=5 lines: a hunk 3 lines off should still match."""
    o = "diff --git a/x b/x\n@@ -10,3 +10,3 @@\n"  # lines 10-12
    p = "diff --git a/x b/x\n@@ -15,3 +15,3 @@\n"  # lines 15-17, 3-line gap
    assert region_overlap(o, p) == 1.0


def test_region_overlap_off_by_far_no_match() -> None:
    # [FUNCTION DOC] Checks a hunk ~90 lines away does not match.
    """Slack=5: 50 lines off → no match."""
    o = "diff --git a/x b/x\n@@ -10,1 +10,1 @@\n"
    p = "diff --git a/x b/x\n@@ -100,1 +100,1 @@\n"
    assert region_overlap(o, p) == 0.0


def test_region_overlap_one_of_two_hunks_matched() -> None:
    # [FUNCTION DOC] Checks matching one of two oracle hunks scores 0.5 (fraction of oracle hunks covered).
    o = "diff --git a/x b/x\n@@ -10,1 +10,1 @@\n@@ -50,1 +50,1 @@\n"
    p = "diff --git a/x b/x\n@@ -10,1 +10,1 @@\n"
    assert region_overlap(o, p) == 0.5


# ---------------------------------------------------------------------------
# similarity (changes-only)
# ---------------------------------------------------------------------------


def test_similarity_identical_one() -> None:
    # [FUNCTION DOC] Checks identical diffs have similarity 1.0.
    diff = "diff --git a/x b/x\n@@ -1 +1 @@\n-old\n+new\n"
    assert similarity(diff, diff) == 1.0


def test_similarity_predicted_empty_zero() -> None:
    # [FUNCTION DOC] Checks an empty prediction has similarity 0.0.
    o = "diff --git a/x b/x\n@@ -1 +1 @@\n-old\n+new\n"
    assert similarity(o, "") == 0.0


def test_similarity_ignores_context_lines() -> None:
    # [FUNCTION DOC] Checks diffs with the same +/- changes but different context and hunk positions score 1.0.
    """Context lines (no +/- prefix) must NOT contribute to similarity.

    Two diffs with the SAME +/- changes but different context should be
    scored as identical.
    """
    o = "diff --git a/x b/x\n@@ -1,3 +1,3 @@\n ctx_a\n-old\n+new\n ctx_b\n"
    p = "diff --git a/x b/x\n@@ -100,3 +100,3 @@\n different_ctx_1\n-old\n+new\n different_ctx_2\n"
    assert similarity(o, p) == 1.0


def test_normalize_changes_only_drops_metadata_and_context() -> None:
    # [FUNCTION DOC] Checks _normalize_changes_only keeps only ---/+++ and -/+ change lines, dropping diff/index/@@
    # headers and context.
    raw = (
        "diff --git a/x b/x\n"
        "index 1..2 100644\n"
        "--- a/x\n"
        "+++ b/x\n"
        "@@ -1,3 +1,3 @@\n"
        " context_line\n"
        "-old_line\n"
        "+new_line\n"
        " more_context\n"
    )
    out = _normalize_changes_only(raw)
    assert out == ["--- a/x", "+++ b/x", "-old_line", "+new_line"]


# ---------------------------------------------------------------------------
# llm_judge (network calls fully mocked)
# ---------------------------------------------------------------------------


def test_llm_judge_no_api_key_returns_none() -> None:
    # [FUNCTION DOC] Checks an empty api_key returns (None, 'no_api_key') without calling out.
    score, status = llm_judge(instruction="i", oracle="o", predicted="p", api_key="")
    assert score is None
    assert status == "no_api_key"


def test_llm_judge_empty_predicted_returns_zero() -> None:
    # [FUNCTION DOC] Checks an empty prediction returns (0.0, 'empty_predicted').
    score, status = llm_judge(instruction="i", oracle="o", predicted="", api_key="key")
    assert score == 0.0
    assert status == "empty_predicted"


def test_llm_judge_happy_path() -> None:
    # [FUNCTION DOC] Mocks urlopen to return an Anthropic-shaped reply with JSON score 0.73; checks (0.73, 'ok').
    """Mock the API to return a well-formed score."""
    fake_response = (
        b'{"content": [{"text": "{\\"score\\": 0.73, '
        b'\\"reasoning\\": \\"plausible fix in right region\\"}"}]}'
    )

    with mock.patch("urllib.request.urlopen") as m:
        m.return_value.__enter__.return_value.read.return_value = fake_response
        score, status = llm_judge(
            instruction="fix the bug",
            oracle="diff --git a/x b/x\n",
            predicted="diff --git a/x b/x\n+fix\n",
            api_key="sk-test",
        )
    assert score == pytest.approx(0.73)
    assert status == "ok"


def test_llm_judge_network_error_returns_none() -> None:
    # [FUNCTION DOC] Mocks urlopen to raise URLError; checks (None, 'network').
    import urllib.error

    with mock.patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        score, status = llm_judge(instruction="i", oracle="o", predicted="p", api_key="sk-test")
    assert score is None
    assert status == "network"


def test_llm_judge_parse_error_returns_none() -> None:
    # [FUNCTION DOC] Mocks a reply with prose and no JSON object; checks (None, 'missing_score').
    """Model returns prose without a JSON object — judge should bail."""
    fake_response = b'{"content": [{"text": "Plain prose, no JSON object here at all"}]}'

    with mock.patch("urllib.request.urlopen") as m:
        m.return_value.__enter__.return_value.read.return_value = fake_response
        score, status = llm_judge(instruction="i", oracle="o", predicted="p", api_key="sk-test")
    assert score is None
    assert status == "missing_score"


def test_llm_judge_clamps_out_of_range_score() -> None:
    # [FUNCTION DOC] Mocks a reply with score 1.5; checks it is clamped to 1.0 with status 'ok'.
    """Model returns score=1.5 → clamp to 1.0."""
    fake_response = b'{"content": [{"text": "{\\"score\\": 1.5, \\"reasoning\\": \\"x\\"}"}]}'
    with mock.patch("urllib.request.urlopen") as m:
        m.return_value.__enter__.return_value.read.return_value = fake_response
        score, status = llm_judge(instruction="i", oracle="o", predicted="p", api_key="sk-test")
    assert score == 1.0
    assert status == "ok"


# ---------------------------------------------------------------------------
# llm_judge — routing: Anthropic by default, OpenAI-compatible via endpoint
# ---------------------------------------------------------------------------


_ENDPOINT = "http://127.0.0.1:8000/v1"
_OPENAI_STYLE_OK = (
    b'{"choices": [{"message": {"role": "assistant", '
    b'"content": "{\\"score\\": 0.6, \\"reasoning\\": \\"ok\\"}"}}]}'
)


def _judge_via(endpoint, response: bytes, **kw):
    # [FUNCTION DOC] HELPER: Calls llm_judge with fixed instruction/oracle/predicted, the given endpoint and extra
    # kwargs (api_key popped from kw, default 'EMPTY'; mutates kw), while urlopen is mocked to
    # return `response` bytes. Output: (score, status, the urllib Request passed to urlopen). Raises
    # whatever llm_judge raises; would fail with TypeError/IndexError if urlopen was never called.
    """Run llm_judge with urlopen mocked; return (score, status, Request)."""
    with mock.patch("urllib.request.urlopen") as m:
        m.return_value.__enter__.return_value.read.return_value = response
        score, status = llm_judge(
            instruction="i",
            oracle="diff --git a/x b/x\n",
            predicted="diff --git a/x b/x\n+fix\n",
            api_key=kw.pop("api_key", "EMPTY"),
            endpoint=endpoint,
            **kw,
        )
        req = m.call_args.args[0]
    return score, status, req


def test_llm_judge_default_route_is_anthropic() -> None:
    # [FUNCTION DOC] Checks endpoint=None posts to api.anthropic.com/v1/messages with an x-api-key header, no
    # Authorization, and no temperature in the body.
    fake = b'{"content": [{"text": "{\\"score\\": 0.5}"}]}'
    score, status, req = _judge_via(None, fake, api_key="sk-ant", model="claude-x")
    assert status == "ok" and score == pytest.approx(0.5)
    assert req.full_url == "https://api.anthropic.com/v1/messages"
    assert req.get_header("X-api-key") == "sk-ant"
    assert req.get_header("Authorization") is None
    assert "temperature" not in json.loads(req.data)  # default route untouched


def test_llm_judge_endpoint_uses_openai_compatible_route() -> None:
    # [FUNCTION DOC] Checks an endpoint routes to <endpoint>/chat/completions with 'Bearer EMPTY' auth, no x-api-
    # key, the given model, a user message and temperature 0; score 0.6 parsed.
    score, status, req = _judge_via(_ENDPOINT, _OPENAI_STYLE_OK, model="Qwen/Qwen3.5-4B")
    assert status == "ok" and score == pytest.approx(0.6)
    assert req.full_url == _ENDPOINT + "/chat/completions"
    assert req.get_header("Authorization") == "Bearer EMPTY"
    assert req.get_header("X-api-key") is None
    body = json.loads(req.data)
    assert body["model"] == "Qwen/Qwen3.5-4B"
    assert body["messages"][0]["role"] == "user"
    assert body["temperature"] == 0  # greedy: small local judges are noisy otherwise


def test_llm_judge_endpoint_trailing_slash_and_explicit_key() -> None:
    # [FUNCTION DOC] Checks a trailing slash on the endpoint is normalized and an explicit key is used as Bearer
    # token.
    _, status, req = _judge_via(_ENDPOINT + "/", _OPENAI_STYLE_OK, api_key="srv", model="m")
    assert status == "ok"
    assert req.full_url == _ENDPOINT + "/chat/completions"
    assert req.get_header("Authorization") == "Bearer srv"


def test_llm_judge_endpoint_requires_model() -> None:
    # [FUNCTION DOC] Checks an endpoint with empty model returns (None, 'no_judge_model').
    score, status = llm_judge(
        instruction="i", oracle="o", predicted="p", api_key="EMPTY", model="", endpoint=_ENDPOINT
    )
    assert score is None
    assert status == "no_judge_model"


def test_llm_judge_endpoint_anthropic_shaped_reply_is_parse_error() -> None:
    # [FUNCTION DOC] Checks an Anthropic-shaped reply on the OpenAI route yields (None, 'parse') rather than
    # crashing.
    """A server answering in Anthropic's shape on the OpenAI route is a parse failure, not a crash."""
    anthropic_shaped = b'{"content": [{"text": "{\\"score\\": 0.9}"}]}'
    score, status, _ = _judge_via(_ENDPOINT, anthropic_shaped, model="m")
    assert score is None
    assert status == "parse"


def test_llm_judge_endpoint_null_content_is_parse_error() -> None:
    # [FUNCTION DOC] Checks a null message content (thinking-mode servers) yields (None, 'parse').
    """Thinking-mode servers can return content=null with the text elsewhere."""
    null_content = b'{"choices": [{"message": {"content": null, "reasoning_content": "..."}}]}'
    score, status, _ = _judge_via(_ENDPOINT, null_content, model="m")
    assert score is None
    assert status == "parse"


def test_llm_judge_endpoint_network_error() -> None:
    # [FUNCTION DOC] Checks URLError on the endpoint route yields (None, 'network').
    import urllib.error

    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        score, status = llm_judge(
            instruction="i",
            oracle="o",
            predicted="p",
            api_key="EMPTY",
            model="m",
            endpoint=_ENDPOINT,
        )
    assert score is None
    assert status == "network"


# ---------------------------------------------------------------------------
# _judge_config_from_env — what main() hands to llm_judge
# ---------------------------------------------------------------------------


_JUDGE_VARS = ("ANTHROPIC_API_KEY", "R2E_JUDGE_MODEL", "R2E_JUDGE_ENDPOINT", "R2E_JUDGE_API_KEY")


def _env(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    # [FUNCTION DOC] HELPER: Clears all _JUDGE_VARS from the environment and sets the given name=value pairs via
    # monkeypatch. Output: None. Side effects: env changes, undone by monkeypatch at teardown.
    for name in _JUDGE_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_judge_config_default_route_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Checks with no env vars the config is ('', 'claude-haiku-4-5-20251001', None).
    _env(monkeypatch)
    assert _judge_config_from_env() == ("", "claude-haiku-4-5-20251001", None)


def test_judge_config_default_route_with_key_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Checks ANTHROPIC_API_KEY is whitespace-stripped and R2E_JUDGE_MODEL overrides the model.
    _env(monkeypatch, ANTHROPIC_API_KEY=" sk-ant ", R2E_JUDGE_MODEL="claude-x")
    assert _judge_config_from_env() == ("sk-ant", "claude-x", None)


def test_judge_config_endpoint_withholds_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Checks that with R2E_JUDGE_ENDPOINT set the Anthropic key is not forwarded (api_key 'EMPTY')
    # and model/endpoint come from env.
    _env(
        monkeypatch,
        ANTHROPIC_API_KEY="sk-ant",
        R2E_JUDGE_ENDPOINT=_ENDPOINT,
        R2E_JUDGE_MODEL="Qwen/Qwen3.5-4B",
    )
    api_key, model, endpoint = _judge_config_from_env()
    assert api_key == "EMPTY"
    assert (model, endpoint) == ("Qwen/Qwen3.5-4B", _ENDPOINT)


def test_judge_config_endpoint_explicit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Checks R2E_JUDGE_API_KEY is used as the key on the endpoint route.
    _env(monkeypatch, R2E_JUDGE_ENDPOINT=_ENDPOINT, R2E_JUDGE_MODEL="m", R2E_JUDGE_API_KEY="srv")
    assert _judge_config_from_env() == ("srv", "m", _ENDPOINT)


def test_judge_config_endpoint_without_model_has_no_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC] Checks the endpoint route has no default model (''), key 'EMPTY'.
    _env(monkeypatch, R2E_JUDGE_ENDPOINT=_ENDPOINT)
    api_key, model, endpoint = _judge_config_from_env()
    assert api_key == "EMPTY"
    assert model == ""  # llm_judge turns this into status "no_judge_model"
    assert endpoint == _ENDPOINT


def test_judge_config_blank_endpoint_means_default_route(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Checks a whitespace-only endpoint falls back to the default Anthropic route.
    _env(monkeypatch, R2E_JUDGE_ENDPOINT="  ", ANTHROPIC_API_KEY="sk-ant")
    assert _judge_config_from_env() == ("sk-ant", "claude-haiku-4-5-20251001", None)


# ---------------------------------------------------------------------------
# combine — weight redistribution when judge is None
# ---------------------------------------------------------------------------


_WEIGHTS = {
    "format_valid": 0.05,
    "size_sanity": 0.05,
    "file_targeting": 0.10,
    "region_overlap": 0.20,
    "similarity": 0.20,
    "llm_judge": 0.40,
}


def test_combine_all_components_present() -> None:
    # [FUNCTION DOC] Checks all components at 1.0 combine to 1.0.
    comps: dict[str, float | None] = {
        "format_valid": 1.0,
        "size_sanity": 1.0,
        "file_targeting": 1.0,
        "region_overlap": 1.0,
        "similarity": 1.0,
        "llm_judge": 1.0,
    }
    assert combine(comps, _WEIGHTS) == pytest.approx(1.0)


def test_combine_judge_missing_redistributes_weight() -> None:
    # [FUNCTION DOC] Checks a None judge with all others 1.0 still gives 1.0 (weight renormalized over present
    # components).
    """All deterministic = 1.0, judge = None → score should still be 1.0
    (judge weight redistributes proportionally)."""
    comps: dict[str, float | None] = {
        "format_valid": 1.0,
        "size_sanity": 1.0,
        "file_targeting": 1.0,
        "region_overlap": 1.0,
        "similarity": 1.0,
        "llm_judge": None,
    }
    assert combine(comps, _WEIGHTS) == pytest.approx(1.0)


def test_combine_partial_credit() -> None:
    # [FUNCTION DOC] Checks all present components at 0.5 with judge None give 0.5.
    """0.5 across the board → final 0.5 regardless of weight redistribution."""
    comps: dict[str, float | None] = dict.fromkeys(_WEIGHTS, 0.5)
    comps["llm_judge"] = None
    assert combine(comps, _WEIGHTS) == pytest.approx(0.5)


def test_combine_all_none_returns_zero() -> None:
    # [FUNCTION DOC] Checks all-None components give 0.0.
    comps: dict[str, float | None] = dict.fromkeys(_WEIGHTS, None)
    assert combine(comps, _WEIGHTS) == 0.0


# ---------------------------------------------------------------------------
# Default weights (post pilot-v4 retune)
# ---------------------------------------------------------------------------


def test_default_weights_sum_to_one() -> None:
    # [FUNCTION DOC] Checks _DEFAULT_WEIGHTS sum to 1.0.
    """The retuned defaults must still sum to 1.0 so reward stays in [0,1]."""
    from repo2rlenv.pipelines._pr_diff_verifier import _DEFAULT_WEIGHTS

    assert sum(_DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_default_weights_match_pilot_recommendations() -> None:
    # [FUNCTION DOC] Pins the retune: format_valid weight 0, llm_judge >= 0.4, similarity < region_overlap.
    """Codify the post-pilot retune so weights can't silently regress."""
    from repo2rlenv.pipelines._pr_diff_verifier import _DEFAULT_WEIGHTS

    assert _DEFAULT_WEIGHTS["format_valid"] == 0.0  # zero discriminative signal
    assert _DEFAULT_WEIGHTS["llm_judge"] >= 0.4  # most informative component
    assert (
        _DEFAULT_WEIGHTS["similarity"] < _DEFAULT_WEIGHTS["region_overlap"]
    )  # correlated → don't double-count
