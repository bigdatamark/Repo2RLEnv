# [MODULE DOC] Unit tests for the equivalence_tests pipeline's pure-Python pieces:
# _extract_test_section (parse the '[Test]' section of an LLM reply), uses_both_names (test must
# import/use both <name> and reference_<name>), _rename_function_source,
# _stub_module/_oracle_module (task_module.py shapes), build_equivalence_dockerfile baking the
# stub, the stub->oracle gold patch (including a real `git apply` solvability check), the bootstrap
# contract, option defaults and _build_task's per-host reference URL. Side effects: one test runs
# git in tmp_path (skipped if git is missing); another writes a Harbor task under tmp_path. No
# Docker, network or LLM.
"""equivalence_tests — helpers, builders, and contract conformance.

The function extractor is exercised in test_function_extractor.py.
Here we cover the pipeline's pure-Python pieces:

  - _extract_test_section (LLM response parsing)
  - test_uses_both_names (test must reference both `name` and `reference_name`)
  - _stub_module / _oracle_module (task_module.py shape)
  - _rename_function_source (rename in source text)
  - build_equivalence_dockerfile + stub→oracle gold patch (solvability)
  - Pipeline contract (requires_bootstrap, missing-bootstrap rejection)
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from repo2rlenv.pipelines._eval_script import make_unified_diff
from repo2rlenv.pipelines._function_extractor import FunctionCandidate
from repo2rlenv.pipelines.equivalence_tests import (
    EquivalenceTestsPipeline,
    _extract_test_section,
    _oracle_module,
    _rename_function_source,
    _stub_module,
    build_equivalence_dockerfile,
    uses_both_names,
)
from repo2rlenv.sources import SourceKind
from repo2rlenv.spec.options import EquivalenceTestsOptions


def _candidate() -> FunctionCandidate:
    # [FUNCTION DOC] HELPER: Returns a FunctionCandidate for 'add(x, y)' at src/calc.py lines 1-2
    # with no docstring.
    return FunctionCandidate(
        relative_path="src/calc.py",
        name="add",
        lineno=1,
        end_lineno=2,
        body_loc=1,
        source="def add(x, y):\n    return x + y\n",
        docstring="",
        arg_names=("x", "y"),
    )


def _make_candidate(
    name: str = "add",
    arg_names: tuple[str, ...] = ("x", "y"),
    source: str = 'def add(x, y):\n    """Sum."""\n    return x + y\n',
    docstring: str = "Sum.",
    body_loc: int = 2,
) -> FunctionCandidate:
    # [FUNCTION DOC] HELPER: Returns a FunctionCandidate at src/foo.py lines 1-3 with configurable
    # name (default 'add'), arg_names, source (default a docstring'd add), docstring (default
    # 'Sum.') and body_loc (default 2).
    return FunctionCandidate(
        relative_path="src/foo.py",
        name=name,
        lineno=1,
        end_lineno=3,
        body_loc=body_loc,
        source=source,
        docstring=docstring,
        arg_names=arg_names,
    )


# ---------------------------------------------------------------------------
# _extract_test_section
# ---------------------------------------------------------------------------


def test_extracts_test_section_with_fence():
    # [FUNCTION DOC] Checks a fenced python block under '[Test]' is extracted with fences stripped.
    text = (
        "[Test]\n"
        "```python\n"
        "from task_module import add, reference_add\n"
        "def test_add():\n"
        "    assert add(1, 2) == reference_add(1, 2)\n"
        "```\n"
    )
    parsed = _extract_test_section(text)
    assert parsed is not None
    assert "from task_module import add" in parsed.code
    assert not parsed.code.startswith("```")


def test_extracts_test_section_without_fence():
    # [FUNCTION DOC] Checks unfenced code under '[Test]' is extracted.
    text = (
        "[Test]\n"
        "from task_module import add, reference_add\n"
        "def test_add():\n"
        "    assert add(1, 2) == reference_add(1, 2)\n"
    )
    parsed = _extract_test_section(text)
    assert parsed is not None
    assert "def test_add" in parsed.code


def test_extracts_test_section_case_insensitive():
    # [FUNCTION DOC] Checks a lowercase '[test]' header is recognized.
    text = (
        "[test]\n"
        "from task_module import add, reference_add\n"
        "def test_x():\n"
        "    assert add(1, 1) == reference_add(1, 1)\n"
    )
    parsed = _extract_test_section(text)
    assert parsed is not None


def test_extract_test_section_returns_none_when_missing():
    # [FUNCTION DOC] Checks text without a test section yields None.
    text = "Here is some content but no test section."
    assert _extract_test_section(text) is None


def test_extract_test_section_stops_at_next_section():
    # [FUNCTION DOC] Checks extraction stops at the next bracketed section header ('[Notes]').
    text = (
        "[Test]\nfrom task_module import x, reference_x\ndef test_x(): pass\n[Notes]\nignore me\n"
    )
    parsed = _extract_test_section(text)
    assert parsed is not None
    assert "ignore me" not in parsed.code


# ---------------------------------------------------------------------------
# test_uses_both_names
# ---------------------------------------------------------------------------


def test_uses_both_names_happy_path():
    # [FUNCTION DOC] Checks code importing and calling both add and reference_add is accepted.
    code = (
        "from task_module import add, reference_add\n"
        "def test_add():\n"
        "    assert add(1, 2) == reference_add(1, 2)\n"
    )
    assert uses_both_names(code, "add")


def test_uses_both_names_missing_candidate():
    # [FUNCTION DOC] Checks a test using only reference_add is rejected (trivial test).
    """Test that imports only the reference is trivial — reject it."""
    code = (
        "from task_module import reference_add\n"
        "def test_add():\n"
        "    assert reference_add(1, 2) == 3\n"
    )
    assert not uses_both_names(code, "add")


def test_uses_both_names_missing_reference():
    # [FUNCTION DOC] Checks a test using only add is rejected.
    code = "from task_module import add\ndef test_add():\n    assert add(1, 2) == 3\n"
    assert not uses_both_names(code, "add")


def test_uses_both_names_missing_import():
    # [FUNCTION DOC] Checks a test that calls both names but never imports from task_module is
    # rejected.
    code = "def test_add():\n    assert add(1, 2) == reference_add(1, 2)\n"
    assert not uses_both_names(code, "add")


# ---------------------------------------------------------------------------
# _rename_function_source
# ---------------------------------------------------------------------------


def test_rename_simple_def():
    # [FUNCTION DOC] Checks the def line is renamed to reference_add.
    src = 'def add(x, y):\n    """Sum."""\n    return x + y\n'
    out = _rename_function_source(src, "add", "reference_add")
    assert out.startswith("def reference_add(x, y):")


def test_rename_preserves_decorators():
    # [FUNCTION DOC] Checks decorators are kept when renaming.
    src = "@staticmethod\ndef add(x, y):\n    return x + y\n"
    out = _rename_function_source(src, "add", "reference_add")
    assert "@staticmethod" in out
    assert "def reference_add" in out


def test_rename_only_renames_def_line():
    # [FUNCTION DOC] Documents the known limitation that recursive calls inside the body keep the
    # old name; only the def line is renamed.
    """Internal recursive calls keep the old name — known v0.7 limitation."""
    src = "def fact(n):\n    return 1 if n <= 1 else n * fact(n - 1)\n"
    out = _rename_function_source(src, "fact", "reference_fact")
    assert "def reference_fact" in out
    # The recursive call still says `fact(n - 1)` — that's the documented v0.7 trade-off
    assert "fact(n - 1)" in out


# ---------------------------------------------------------------------------
# _stub_module / _oracle_module
# ---------------------------------------------------------------------------


def test_stub_module_has_stubbed_candidate():
    # [FUNCTION DOC] Checks the stub module contains reference_add, an add def, and a
    # NotImplementedError stub.
    c = _make_candidate()
    out = _stub_module(c)
    assert "def reference_add(x, y):" in out
    assert "def add(x, y):" in out
    assert "raise NotImplementedError" in out


def test_oracle_module_has_two_definitions():
    # [FUNCTION DOC] Checks the oracle module defines both reference_add and a real add with no
    # NotImplementedError.
    c = _make_candidate()
    out = _oracle_module(c)
    assert "def reference_add(x, y):" in out
    assert "def add(x, y):" in out
    assert "raise NotImplementedError" not in out


# ---------------------------------------------------------------------------
# Stub bake + stub→oracle gold patch (issue #54 — solvable by non-oracle agents)
# ---------------------------------------------------------------------------


def test_dockerfile_bakes_stub_module():
    # [FUNCTION DOC] Checks build_equivalence_dockerfile uses the given base image and
    # base64-decodes the stub into /workspace/task_module.py, and that the stub contains
    # reference_add and the stub raise.
    """Every agent must start with the stub (reference_<name> present) — so it's
    baked into the image, not packed into the OracleAgent-only solution/."""
    stub = _stub_module(_candidate())
    df = build_equivalence_dockerfile("local/img:abc", stub)
    assert "FROM local/img:abc" in df
    assert "base64 -d > /workspace/task_module.py" in df
    # reference_<name> + stub must both be in the baked payload
    assert "reference_add" in stub
    assert "raise NotImplementedError" in stub


def test_gold_patch_is_stub_to_oracle_modify_diff():
    # [FUNCTION DOC] Checks the stub->oracle diff modifies task_module.py (no 'new file mode') and
    # removes the NotImplementedError line.
    """Gold patch transforms the baked stub into the oracle (not a new file)."""
    cand = _candidate()
    diff = make_unified_diff(_stub_module(cand), _oracle_module(cand), "task_module.py")
    assert "diff --git a/task_module.py b/task_module.py" in diff
    assert "new file mode" not in diff  # modify, not create
    assert "raise NotImplementedError" in diff  # the stub line is removed


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_oracle_patch_applies_to_baked_stub(tmp_path: Path):
    # [FUNCTION DOC] End-to-end solvability: git-inits tmp_path, writes the stub and the gold diff,
    # runs `git apply`, and checks the result equals the oracle (modulo trailing newlines). Side
    # effects: subprocess git calls and files in tmp_path; skipped when git is not on PATH. Raises
    # CalledProcessError on git failure.
    """End-to-end solvability: applying the gold patch to the baked stub via
    `git apply` (exactly what solve.sh does) yields the oracle module."""
    cand = _candidate()
    stub = _stub_module(cand)
    oracle = _oracle_module(cand)
    diff = make_unified_diff(stub, oracle, "task_module.py")

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "task_module.py").write_text(stub)  # what the image bakes
    (tmp_path / "patch.diff").write_text(diff)
    subprocess.run(["git", "apply", "patch.diff"], cwd=tmp_path, check=True)

    got = (tmp_path / "task_module.py").read_text()
    assert got.rstrip("\n") == oracle.rstrip("\n")
    assert "raise NotImplementedError" not in got  # stub really replaced


# ---------------------------------------------------------------------------
# Pipeline contract
# ---------------------------------------------------------------------------


def test_equivalence_tests_requires_bootstrap_attr():
    # [FUNCTION DOC] Checks EquivalenceTestsPipeline.requires_bootstrap is True.
    assert EquivalenceTestsPipeline.requires_bootstrap is True


def test_equivalence_tests_rejects_missing_bootstrap():
    # [FUNCTION DOC] Checks constructing the pipeline with bootstrap=None raises RuntimeError
    # matching 'requires a BootstrapResult'.
    from repo2rlenv.spec.input import (
        GenerationInput,
        LLMSpec,
        OutputSpec,
        PipelineName,
        PipelineSpec,
        RepoSpec,
    )

    gen_input = GenerationInput(
        repo=RepoSpec(url="huggingface/trl"),
        pipeline=PipelineSpec(name=PipelineName.EQUIVALENCE_TESTS, options={}),
        llm=LLMSpec(provider="anthropic", model="claude-sonnet-4-6"),
        output=OutputSpec(destination="./out", org="x", dataset_name="y"),
    )
    with pytest.raises(RuntimeError, match="requires a BootstrapResult"):
        EquivalenceTestsPipeline(gen_input, EquivalenceTestsOptions(), bootstrap=None)


def test_equivalence_tests_options_defaults():
    # [FUNCTION DOC] Checks EquivalenceTestsOptions defaults: limit 50, min_loc 5, max_loc 60, both
    # stub- fail/oracle-pass requirements True.
    opts = EquivalenceTestsOptions()
    assert opts.limit == 50
    assert opts.min_loc == 5
    assert opts.max_loc == 60
    assert opts.require_test_fails_with_stub is True
    assert opts.require_test_passes_with_oracle is True


# ---------------------------------------------------------------------------
# _build_task: reference URL must follow source_kind, not always github.com
# ---------------------------------------------------------------------------


def _stub_pipeline_for_build_task(source_kind=SourceKind.GITHUB):
    # [FUNCTION DOC] HELPER: Builds an EquivalenceTestsPipeline via __new__ with only what
    # _build_task reads: zero cost, a SimpleNamespace local python bootstrap (not pushed) and a
    # MagicMock input (repo o/r, ref main, access auto, given source_kind, org default, LLM
    # qualified name). Input: source_kind (default GITHUB). Output: stub pipeline.
    """A pipeline instance with just enough scaffolding to call `_build_task`."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    pipe = EquivalenceTestsPipeline.__new__(EquivalenceTestsPipeline)
    pipe._llm_cost_usd = 0.0
    pipe.bootstrap = SimpleNamespace(
        image_tag="local/r2e-bootstrap/o__r:abc",
        image_digest="local/r2e-bootstrap/o__r:abc",
        pushed_to_registry=False,
        language=SimpleNamespace(value="python"),
    )
    pipe.input = MagicMock()
    pipe.input.repo.owner_name = ("o", "r")
    pipe.input.repo.ref = "main"
    pipe.input.repo.access = "auto"
    pipe.input.repo.source_kind = source_kind
    pipe.input.output.org = "default"
    pipe.input.llm.qualified_name = "test-provider/test-model"
    pipe._progress_cb = None
    return pipe


@pytest.mark.parametrize(
    ("source_kind", "expected_reference"),
    [
        (SourceKind.GITHUB, "https://github.com/o/r/blob/main/src/calc.py#L1-L2"),
        (SourceKind.GITLAB, "https://gitlab.com/o/r/-/blob/main/src/calc.py#L1-2"),
        (SourceKind.LOCAL, None),
    ],
)
def test_build_task_reference_host_matches_source(source_kind, expected_reference):
    # [FUNCTION DOC] Parametrized over GITHUB/GITLAB/LOCAL: checks repo2env 'reference' is a GitHub
    # blob URL with '#L1-L2', a GitLab '/-/blob/' URL with '#L1-2', or absent for local.
    """The 'reference' provenance URL must point at the candidate's actual
    host, not always github.com: equivalence_tests sets no
    required_capabilities and runs on any source (sources.py), and a
    GitLab- or local-sourced task previously got a github.com link that
    does not resolve."""
    pipe = _stub_pipeline_for_build_task(source_kind=source_kind)
    cand = _candidate()
    task = pipe._build_task(
        cand, "def test_add():\n    assert add(1, 2) == 3\n", test_filename="test_r2e_deadbeef.py"
    )
    if expected_reference is None:
        # TOML has no null, so a local checkout must drop the key, not null it.
        assert "reference" not in task.repo2env
    else:
        assert task.repo2env["reference"] == expected_reference


def test_build_task_local_source_writes_a_loadable_task_toml(tmp_path):
    # [FUNCTION DOC] Writes a LOCAL-source task with write_harbor_task into tmp_path and checks
    # task.toml parses without a 'reference' key. Side effects: files under tmp_path.
    """A None reference would crash tomli_w.dumps (TOML has no null) inside
    write_harbor_task. Drive the real emitter, not just the dict, so a
    regression here fails loudly instead of only at push time."""
    import tomllib

    from repo2rlenv.emitter.harbor import write_harbor_task

    pipe = _stub_pipeline_for_build_task(source_kind=SourceKind.LOCAL)
    cand = _candidate()
    task = pipe._build_task(
        cand, "def test_add():\n    assert add(1, 2) == 3\n", test_filename="test_r2e_deadbeef.py"
    )
    task_path = write_harbor_task(task, tmp_path)
    loaded = tomllib.loads((task_path / "task.toml").read_text())
    assert "reference" not in loaded["metadata"]["repo2env"]
