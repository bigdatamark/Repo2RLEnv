# [MODULE DOC] Typed LLM author output for terminal tasks and its materialization into a standalone
# Harbor task bundle. Pydantic models validate the author's environment files, setup Dockerfile
# fragment, pytest tests, weights and reference solution; dockerfile_for_setup builds the owned
# Dockerfile; emit_draft writes the full bundle (environment, solution, tests incl. the copied
# grade.py and test_results.py parser, contract.json) via emitter.bundle.write_bundle with no-
# network agent and verifier policies.
"""Typed author output and standalone Harbor materialization."""

from __future__ import annotations

import ast
import json
import math
import re
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from repo2rlenv.emitter.bundle import TaskBundle, TaskFile, relative_asset_path, write_bundle
from repo2rlenv.pipelines.recipes.catalog import RecipeInfo


class EnvironmentFile(BaseModel):
    # [CLASS DOC] Pydantic model (extra fields forbidden) for one author-supplied file placed under
    # the task's environment/ directory. Fields: path (str, relative path under environment/),
    # content (str text), executable (bool, whether to set the executable bit).
    model_config = ConfigDict(extra="forbid")
    path: str
    content: str
    executable: bool


class TestWeight(BaseModel):
    # [CLASS DOC] Pydantic model (extra forbidden) giving the reward weight of one pytest test.
    # Fields: name (str, top-level test function name); weight (float, 0 < weight <= 1).
    model_config = ConfigDict(extra="forbid")
    name: str
    weight: float = Field(gt=0, le=1)


class EnvironmentDefinition(BaseModel):
    # [CLASS DOC] Pydantic model (extra forbidden) for the author-defined environment:
    # environment_setup (str, up to 16000 chars of Dockerfile lines appended after the owned base)
    # and environment_files (up to 100 EnvironmentFile items). Validated so it cannot override the
    # owned Dockerfile or base image.
    model_config = ConfigDict(extra="forbid")
    environment_setup: str = Field(max_length=16000)
    environment_files: list[EnvironmentFile] = Field(max_length=100)

    @model_validator(mode="after")
    def valid_environment(self):
        # [FUNCTION DOC] Post-validation check. For each environment file, validates
        # "environment/<path>" with relative_asset_path (which raises on unsafe paths), rejects a
        # file named exactly "Dockerfile", and rejects duplicate paths. Rejects any setup line whose
        # stripped, upper-cased form starts with "FROM ", "USER ", "ENTRYPOINT " or "CMD ". Output:
        # self. Raises: ValueError on any violation.
        paths = []
        for item in self.environment_files:
            relative_asset_path("environment/" + item.path)
            if item.path == "Dockerfile":
                raise ValueError("Environment files cannot replace the owned Dockerfile")
            paths.append(item.path)
        if len(set(paths)) != len(paths):
            raise ValueError("Environment files contain duplicate paths")
        if any(
            line.lstrip().upper().startswith(("FROM ", "USER ", "ENTRYPOINT ", "CMD "))
            for line in self.environment_setup.splitlines()
        ):
            raise ValueError("Environment setup cannot replace the base image, user or entry point")
        return self


class TerminalDraft(EnvironmentDefinition):
    # [CLASS DOC] Complete author draft for a terminal task, extending EnvironmentDefinition with:
    # instruction (40- 16000 chars), tests_python (40-50000 chars of pytest source), solution_shell
    # (20-50000 chars bash reference solution), weights (5-10 TestWeight entries), self_review
    # (20-8000 chars author notes).
    instruction: str = Field(min_length=40, max_length=16000)
    tests_python: str = Field(min_length=40, max_length=50000)
    solution_shell: str = Field(min_length=20, max_length=50000)
    weights: list[TestWeight] = Field(min_length=5, max_length=10)
    self_review: str = Field(min_length=20, max_length=8000)

    @model_validator(mode="after")
    def complete_contract(self):
        # [FUNCTION DOC] Post-validation contract check. Parses tests_python with ast (a SyntaxError
        # is not a ValueError, so it propagates unwrapped rather than as a pydantic ValidationError)
        # and collects top-level sync function names starting with "test_". Requires weight names to
        # be unique and exactly equal to that set, weights to sum to 1.0 (abs tol 0.001), and
        # solution_shell to begin with "#!/bin/bash" followed by a newline. Output: self. Raises:
        # ValueError (wrapped by pydantic) on violation; SyntaxError on unparsable tests.
        tree = ast.parse(self.tests_python)
        tests = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        }
        weights = {item.name: item.weight for item in self.weights}
        if len(weights) != len(self.weights) or tests != weights.keys():
            raise ValueError("Weights must name each top-level pytest test exactly once")
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=0.001):
            raise ValueError("Test weights must sum to one")
        if not self.solution_shell.startswith("#!/bin/bash\n"):
            raise ValueError("Reference must start with a bash shebang")
        return self


def dockerfile_for_setup(setup: str, agent_user: str | None = None) -> str:
    # [FUNCTION DOC] Build the owned task Dockerfile text. Inputs: setup (str) - author Dockerfile
    # lines appended (right-stripped) after the base; agent_user (str or None, default None) -
    # optional unprivileged user the agent runs as. Base: python:3.12-slim with apt packages bash
    # tmux curl git jq sqlite3, pinned pytest 9.0.3 and uv 0.10.9, PYTHONDONTWRITEBYTECODE and
    # PYTEST_DISABLE_PLUGIN_AUTOLOAD env, WORKDIR /workspace and COPY of the build context; WORKDIR
    # /workspace is re-set after setup. With agent_user: creates the user if missing, chowns
    # /workspace and its home, registers each .git directory found under those trees as a system git
    # safe.directory (so the root verifier can use them), and sets HOME. Output: Dockerfile string.
    # Raises: ValueError if agent_user is not a simple lowercase Unix name (letter then up to 31 of
    # [a-z0-9_]).
    if agent_user is not None and re.fullmatch(r"[a-z][a-z0-9_]{0,31}", agent_user) is None:
        raise ValueError("Agent user must be a simple Unix username")
    dockerfile = (
        "FROM python:3.12-slim\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends bash tmux curl git jq sqlite3 "
        "&& rm -rf /var/lib/apt/lists/*\n"
        "RUN python -m pip install --no-cache-dir pytest==9.0.3 uv==0.10.9\n"
        "ENV PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1\n"
        "WORKDIR /workspace\nCOPY . /workspace\n" + setup.rstrip() + "\nWORKDIR /workspace\n"
    )
    if agent_user:
        # The verifier runs as root after ownership passes to the learner. Trust
        # only repositories created by the authored image, not arbitrary paths
        # the learner might introduce later or a global safe.directory wildcard.
        dockerfile += (
            f"RUN (id -u {agent_user} >/dev/null 2>&1 || useradd -m -s /bin/bash {agent_user}) "
            f"&& mkdir -p /home/{agent_user} && chown -R {agent_user}:{agent_user} /workspace /home/{agent_user}\n"
            f"RUN find /workspace /home/{agent_user} -type d -name .git -prune "
            "-exec sh -c 'git config --system --add safe.directory \"${1%/.git}\"' sh {} \\;\n"
            f"ENV HOME=/home/{agent_user}\n"
        )
    return dockerfile


def emit_draft(
    draft: TerminalDraft,
    destination: Path,
    *,
    name: str,
    org: str,
    recipe: RecipeInfo,
    lineage: dict,
    timeout_sec: int,
    resume: bool = False,
    agent_user: str | None = None,
) -> Path:
    # [FUNCTION DOC] Materialize a validated TerminalDraft into a Harbor task directory. Inputs:
    # draft; destination (Path); keyword-only name, org (task identity), recipe (RecipeInfo for
    # provenance metadata), lineage (dict merged into metadata), timeout_sec (pytest timeout,
    # verifier timeout is this + 30), resume (bool, default False, passed to write_bundle),
    # agent_user (optional). Files written: environment/ author files, environment/Dockerfile,
    # solution/solve.sh (executable), tests/test_outputs.py, tests/weights.json, tests/grade.py and
    # tests/test_results.py (copied from package resources), executable tests/test.sh running
    # grade.py with isolated python, tests/contract.json (test_names, timeout_sec). Metadata
    # includes recipe id/version, pipeline, upstream commit, reward_kinds ["test_execution"],
    # quality_status "exported". Agent and verifier both use network_mode "no-network"; verifier
    # runs as root. Output: Path returned by write_bundle. Raises: ValueError from
    # dockerfile_for_setup; errors from write_bundle (e.g. existing destination without resume).
    # Note: weights.json is written but grade.py awards binary reward (all tests must pass).
    dockerfile = dockerfile_for_setup(draft.environment_setup, agent_user)
    assets = {
        "environment/" + item.path: TaskFile.text(item.content, executable=item.executable)
        for item in draft.environment_files
    }
    assets.update(
        {
            "environment/Dockerfile": TaskFile.text(dockerfile),
            "solution/solve.sh": TaskFile.text(draft.solution_shell, executable=True),
            "tests/test_outputs.py": TaskFile.text(draft.tests_python),
            "tests/weights.json": TaskFile.text(
                json.dumps({item.name: item.weight for item in draft.weights})
            ),
            "tests/grade.py": TaskFile(files(__package__).joinpath("grade.py").read_bytes()),
            "tests/test_results.py": TaskFile(
                files("repo2rlenv.quality").joinpath("test_results.py").read_bytes()
            ),
            "tests/test.sh": TaskFile.text(
                "#!/bin/bash\nset -eu\nexec /usr/local/bin/python -I /tests/grade.py\n",
                executable=True,
            ),
            "tests/contract.json": TaskFile.text(
                json.dumps(
                    {
                        "test_names": [item.name for item in draft.weights],
                        "timeout_sec": timeout_sec,
                    }
                )
            ),
        }
    )
    return write_bundle(
        TaskBundle(
            name=name,
            org=org,
            instruction=draft.instruction,
            files=assets,
            metadata={
                "recipe": recipe.id,
                "recipe_version": recipe.recipe_version,
                "pipeline": recipe.pipeline,
                "upstream_revision": recipe.upstream["commit"],
                "reward_kinds": ["test_execution"],
                "quality_status": "exported",
                **lineage,
            },
            agent={"network_mode": "no-network", **({"user": agent_user} if agent_user else {})},
            verifier={"network_mode": "no-network", "user": "root"},
            verifier_timeout_sec=timeout_sec + 30,
        ),
        destination,
        resume=resume,
    )
