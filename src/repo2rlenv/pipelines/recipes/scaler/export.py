# [MODULE DOC] Build a Harbor bundle for one concrete SCALER instance: the learner writes a boxed
# answer to /workspace/answer.txt; a separate offline verifier image with math-verify runs grade.py
# against the private reference answer and writes a reward in {-1, +1}.
"""Concrete reasoning instances with a private reference answer and native reward scale."""

from __future__ import annotations

import json
from importlib.resources import files

from repo2rlenv.emitter.bundle import TaskBundle, TaskFile, write_bundle
from repo2rlenv.pipelines.recipes.catalog import get_recipe


def export_task(candidate, destination, org, *, resume=False):
    # [FUNCTION DOC] Write the task bundle for a generated candidate and return its path.
    # Inputs: candidate (dict from the worker: id, instruction, reference_answer, answer_contract,
    # family, difficulty, actual_seed and hash fields), destination (Path), org (str), resume (bool,
    # default False, forwarded to write_bundle).
    # Files: environment Dockerfile (python:3.12-slim, tmux, learner uid 1000, empty answer.txt),
    # tests Dockerfile (pins math-verify 0.8.0, copies grade.py, reference.json, answer-
    # contract.json, test.sh, upstream license/notice), test.sh, solution answer.txt and solve.sh.
    # Metadata records recipe/version "2", pipeline and upstream commit from the catalog, reward
    # kind answer_equivalence with range [-1, 1], domain and identity hashes. No network for agent
    # or verifier (1 CPU, 2048 MB), artifact /workspace/answer.txt, 60s verifier timeout.
    # Output: Path. Raises KeyError on missing candidate fields. Side effects: writes files.
    recipe = get_recipe("scaler")
    bundle = TaskBundle(
        name="scaler-" + candidate["id"],
        org=org,
        instruction=candidate["instruction"],
        files={
            "environment/Dockerfile": TaskFile.text(
                "FROM python:3.12-slim\nRUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*\nWORKDIR /workspace\nRUN useradd -m -u 1000 learner && touch answer.txt && chown -R learner:learner /workspace\n"
            ),
            "tests/Dockerfile": TaskFile.text(
                "FROM python:3.12-slim\nRUN python -m pip install --no-cache-dir math-verify==0.8.0\nWORKDIR /workspace\nRUN touch answer.txt\nCOPY grade.py reference.json answer-contract.json test.sh UPSTREAM_LICENSE UPSTREAM_NOTICE /tests/\nRUN chmod 755 /tests/test.sh\n"
            ),
            "tests/test.sh": TaskFile.text(
                "#!/bin/sh\nset -eu\nexec /usr/local/bin/python -I /tests/grade.py\n",
                executable=True,
            ),
            "tests/grade.py": TaskFile(files(__package__).joinpath("grade.py").read_bytes()),
            "tests/reference.json": TaskFile.text(json.dumps(candidate["reference_answer"])),
            "tests/answer-contract.json": TaskFile.text(
                json.dumps(candidate.get("answer_contract", {}), sort_keys=True)
            ),
            "tests/UPSTREAM_LICENSE": TaskFile(
                files(__package__).joinpath("UPSTREAM_LICENSE").read_bytes()
            ),
            "tests/UPSTREAM_NOTICE": TaskFile(
                files(__package__).joinpath("UPSTREAM_NOTICE").read_bytes()
            ),
            "solution/solve.sh": TaskFile.text(
                "#!/bin/sh\nset -eu\ncp /solution/answer.txt /workspace/answer.txt\n",
                executable=True,
            ),
            "solution/answer.txt": TaskFile.text(candidate["reference_answer"] + "\n"),
        },
        metadata={
            "recipe": "scaler",
            "recipe_version": "2",
            "answer_contract": candidate.get("answer_contract", {}),
            "pipeline": recipe.pipeline,
            "upstream_revision": recipe.upstream["commit"],
            "reward_kinds": ["answer_equivalence"],
            "reward_min": -1,
            "reward_max": 1,
            "quality_status": "exported",
            "domain": "algorithmic_reasoning",
            "generation_scope": "released_family_expansion",
            **{
                key: candidate[key]
                for key in (
                    "family",
                    "difficulty",
                    "actual_seed",
                    "source_sha256",
                    "instance_sha256",
                    "reference_code_sha256",
                )
            },
        },
        agent={"user": "learner", "network_mode": "no-network"},
        verifier={
            "user": "root",
            "network_mode": "no-network",
            "environment_mode": "separate",
            "environment": {"network_mode": "no-network", "cpus": 1, "memory_mb": 2048},
        },
        artifacts=[{"source": "/workspace/answer.txt"}],
        verifier_timeout_sec=60,
    )
    return write_bundle(bundle, destination, resume=resume)
