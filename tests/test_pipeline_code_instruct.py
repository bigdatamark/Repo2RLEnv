# [MODULE DOC] Unit tests for the code_instruct pipeline's pure-Python pieces: make_solution_diff
# (gold patch contains only task_module.py), build_code_instruct_dockerfile shape, the shared
# all_tests_passed log check, the CodeInstructPipeline contract (requires_bootstrap, missing-
# bootstrap error, option defaults) and _build_task's provenance 'reference' URL per source kind.
# Sampler/parser helpers are covered elsewhere (test_oss_instruct_helpers.py). Side effects: one
# test writes a Harbor task directory under pytest's tmp_path; no network, Docker or LLM.
"""code_instruct — diff builder, dockerfile shape, pipeline contract.

Sampler / parser / decontam are tested in test_oss_instruct_helpers.py.
Here we cover the pipeline-level pure-Python pieces and the contract.
"""

from __future__ import annotations

import pytest

from repo2rlenv.pipelines._eval_script import all_tests_passed as _all_tests_passed
from repo2rlenv.pipelines._oss_instruct import ParsedTask, Seed
from repo2rlenv.pipelines.code_instruct import (
    CodeInstructPipeline,
    build_code_instruct_dockerfile,
    make_solution_diff,
)
from repo2rlenv.sources import SourceKind
from repo2rlenv.spec.options import CodeInstructOptions

# ---------------------------------------------------------------------------
# make_solution_diff — gold patch carries ONLY task_module.py (issue #54)
# ---------------------------------------------------------------------------


def test_solution_diff_has_single_header():
    # [FUNCTION DOC] Checks make_solution_diff emits exactly one 'diff --git' header, for
    # task_module.py.
    diff = make_solution_diff(task_module_code="def add(x, y):\n    return x + y\n")
    assert diff.count("diff --git ") == 1
    assert "diff --git a/task_module.py b/task_module.py" in diff


def test_solution_diff_excludes_test_file():
    # [FUNCTION DOC] Regression for issue #54: the gold patch must not include the grading test
    # file; checks there is exactly one new-file section and no 'test_r2e' path.
    """Regression for #54: the grading test must NOT be packed into the gold
    patch — it ships under tests/ so non-oracle agents can reach it."""
    diff = make_solution_diff(task_module_code="x = 1\n")
    assert "test_r2e" not in diff
    assert diff.count("new file mode") == 1
    assert diff.count("--- /dev/null") == 1


def test_solution_diff_hunk_line_counts():
    # [FUNCTION DOC] Checks the new-file hunk header counts lines correctly ('@@ -0,0 +1,3 @@' for
    # 3 lines).
    diff = make_solution_diff(task_module_code="line1\nline2\nline3\n")
    assert "@@ -0,0 +1,3 @@" in diff


def test_solution_diff_handles_missing_trailing_newline():
    # [FUNCTION DOC] Checks the '\ No newline at end of file' marker is emitted when the code lacks
    # a trailing newline.
    """If a file doesn't end with \\n, we emit the `\\ No newline at end of file` line."""
    diff = make_solution_diff(task_module_code="x = 1")  # no trailing newline
    assert "\\ No newline at end of file" in diff


# ---------------------------------------------------------------------------
# build_code_instruct_dockerfile
# ---------------------------------------------------------------------------


def test_dockerfile_minimal_shape():
    # [FUNCTION DOC] Checks build_code_instruct_dockerfile uses the given image in FROM, applies no
    # patch at build time ('git apply' absent) and installs git defensively via apt-get.
    df = build_code_instruct_dockerfile("local/img:abc")
    assert df.startswith("# Auto-generated") or "FROM local/img:abc" in df
    assert "FROM local/img:abc" in df
    # No patching at build time (unlike pr_runtime)
    assert "git apply" not in df
    # Defensive git install (so `git config` works inside container)
    assert "apt-get install" in df


# ---------------------------------------------------------------------------
# _all_tests_passed
# ---------------------------------------------------------------------------


def test_all_tests_passed_detects_passed_summary():
    # [FUNCTION DOC] Checks all_tests_passed returns True for a pytest summary with only passes.
    log = "==== 3 passed in 0.12s ===="
    assert _all_tests_passed(log)


def test_all_tests_passed_rejects_failed():
    # [FUNCTION DOC] Checks all_tests_passed returns False when the summary reports any failure.
    log = "==== 1 failed, 2 passed in 0.12s ===="
    assert not _all_tests_passed(log)


def test_all_tests_passed_rejects_no_collected():
    # [FUNCTION DOC] Checks all_tests_passed returns False when zero tests were collected.
    log = "ERROR: collected 0 items"
    assert not _all_tests_passed(log)


def test_all_tests_passed_rejects_collection_error():
    # [FUNCTION DOC] Checks all_tests_passed returns False on an import/collection error log.
    log = "ImportError: No module named 'task_module'\nERRORS\ncollected 0 items / 1 error\n"
    assert not _all_tests_passed(log)


# ---------------------------------------------------------------------------
# Pipeline contract
# ---------------------------------------------------------------------------


def test_code_instruct_requires_bootstrap_attr():
    # [FUNCTION DOC] Checks CodeInstructPipeline.requires_bootstrap is True.
    assert CodeInstructPipeline.requires_bootstrap is True


def test_code_instruct_rejects_missing_bootstrap():
    # [FUNCTION DOC] Builds a GenerationInput for code_instruct and checks constructing the
    # pipeline with bootstrap=None raises RuntimeError mentioning 'requires a BootstrapResult'.
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
        pipeline=PipelineSpec(name=PipelineName.CODE_INSTRUCT, options={}),
        llm=LLMSpec(provider="anthropic", model="claude-sonnet-4-6"),
        output=OutputSpec(destination="./out", org="x", dataset_name="y"),
    )
    with pytest.raises(RuntimeError, match="requires a BootstrapResult"):
        CodeInstructPipeline(gen_input, CodeInstructOptions(), bootstrap=None)


def test_code_instruct_options_defaults():
    # [FUNCTION DOC] Checks CodeInstructOptions defaults: limit 50, seed_min_loc 30, seed_max_loc
    # 200, and both oracle fail/pass requirements True.
    opts = CodeInstructOptions()
    assert opts.limit == 50
    assert opts.seed_min_loc == 30
    assert opts.seed_max_loc == 200
    assert opts.require_test_fails_without_oracle is True
    assert opts.require_test_passes_with_oracle is True


# ---------------------------------------------------------------------------
# _build_task: reference URL must follow source_kind, not always github.com
# ---------------------------------------------------------------------------


def _stub_pipeline_for_build_task(source_kind=SourceKind.GITHUB):
    # [FUNCTION DOC] HELPER: Builds a CodeInstructPipeline via __new__ (skipping __init__) with
    # only the attributes _build_task reads: zero LLM cost, a SimpleNamespace bootstrap (local
    # image tag/digest, not pushed, python language), and a MagicMock input with repo owner/name
    # ('o','r'), ref 'main', access 'auto', the given source_kind, org 'default' and an LLM
    # qualified name. Input: source_kind (SourceKind, default GITHUB). Output: the stub pipeline.
    # Side effects: none.
    """A pipeline instance with just enough scaffolding to call `_build_task`."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    pipe = CodeInstructPipeline.__new__(CodeInstructPipeline)
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


def _seed_and_parsed():
    # [FUNCTION DOC] HELPER: Returns a (Seed, ParsedTask) pair: a seed at src/calc.py lines 10-20
    # with an add function and a parsed task with problem text, a test and a solution.
    seed = Seed(
        relative_path="src/calc.py",
        start_line=10,
        end_line=20,
        text="def add(x, y):\n    return x + y\n",
    )
    parsed = ParsedTask(
        problem="Implement add(x, y).",
        test_code="def test_add():\n    assert add(1, 2) == 3\n",
        solution_code="def add(x, y):\n    return x + y\n",
    )
    return seed, parsed


@pytest.mark.parametrize(
    ("source_kind", "expected_reference"),
    [
        (SourceKind.GITHUB, "https://github.com/o/r/blob/main/src/calc.py#L10-L20"),
        (SourceKind.GITLAB, "https://gitlab.com/o/r/-/blob/main/src/calc.py#L10-20"),
        (SourceKind.LOCAL, None),
    ],
)
def test_build_task_reference_host_matches_source(source_kind, expected_reference):
    # [FUNCTION DOC] Parametrized over GITHUB/GITLAB/LOCAL: checks _build_task's repo2env
    # 'reference' is a GitHub blob URL, a GitLab '/-/blob/' URL with 'L10-20' range, or absent
    # entirely for local sources (TOML cannot hold null).
    """The 'reference' provenance URL must point at the seed's actual host,
    not always github.com: code_instruct sets no required_capabilities and
    runs on any source (sources.py), and a GitLab- or local-sourced task
    previously got a github.com link that does not resolve."""
    pipe = _stub_pipeline_for_build_task(source_kind=source_kind)
    seed, parsed = _seed_and_parsed()
    task = pipe._build_task(seed, parsed, test_filename="test_r2e_deadbeef.py")
    if expected_reference is None:
        # TOML has no null, so a local checkout must drop the key, not null it.
        assert "reference" not in task.repo2env
    else:
        assert task.repo2env["reference"] == expected_reference


def test_build_task_local_source_writes_a_loadable_task_toml(tmp_path):
    # [FUNCTION DOC] Drives the real write_harbor_task emitter with a LOCAL-source task into
    # tmp_path and checks the resulting task.toml parses and has no 'reference' in
    # metadata.repo2env. Side effects: writes files under tmp_path.
    """A None reference would crash tomli_w.dumps (TOML has no null) inside
    write_harbor_task. Drive the real emitter, not just the dict, so a
    regression here fails loudly instead of only at push time."""
    import tomllib

    from repo2rlenv.emitter.harbor import write_harbor_task

    pipe = _stub_pipeline_for_build_task(source_kind=SourceKind.LOCAL)
    seed, parsed = _seed_and_parsed()
    task = pipe._build_task(seed, parsed, test_filename="test_r2e_deadbeef.py")
    task_path = write_harbor_task(task, tmp_path)
    loaded = tomllib.loads((task_path / "task.toml").read_text())
    assert "reference" not in loaded["metadata"]["repo2env"]
