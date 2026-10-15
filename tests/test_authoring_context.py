# [MODULE DOC] Tests for quality.authoring_context.bounded_context: large evidence is serialised to
# JSON within an 80k-char budget by sampling long lists (reporting total/omitted counts) without
# mutating the input, and small evidence round-trips exactly. KNOWN ISSUE: this module lacks the
# 'from __future__ import annotations' line that the project invariants require for every module.
import json

from repo2rlenv.quality.authoring_context import bounded_context


def test_large_parametrized_evidence_is_sampled_without_changing_verifier_contract():
    # [FUNCTION DOC] 3000 long test ids plus a huge log are bounded to <=80000 chars; FAIL_TO_PASS
    # reports total 3000 and 2992 omitted; the original lists are unchanged.
    identities = [f"test_value[{index}-" + "x" * 1000 + "]" for index in range(3000)]
    contract = {"FAIL_TO_PASS": identities, "PASS_TO_PASS": identities.copy()}
    evidence = {"title": "Handle nested values", "contrast": contract, "log": "Failure\n" * 100000}
    text = bounded_context(evidence)
    assert len(text) <= 80000
    result = json.loads(text)
    assert result["contrast"]["FAIL_TO_PASS"]["total_items"] == 3000
    assert result["contrast"]["FAIL_TO_PASS"]["omitted_items"] == 2992
    assert len(contract["FAIL_TO_PASS"]) == len(contract["PASS_TO_PASS"]) == 3000


def test_small_evidence_is_preserved_exactly():
    # [FUNCTION DOC] Small evidence round-trips through JSON unchanged.
    evidence = {"source": "def value(x): return x", "tests": ["test_one", "test_two"]}
    assert json.loads(bounded_context(evidence)) == evidence
