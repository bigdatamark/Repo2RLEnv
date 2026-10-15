# [MODULE DOC] Remote SWE-gen worker: build the healthy PR head, reverse-apply the PR's source diff
# to obtain the defective files, and record test contrast plus test excerpts.
"""Build healthy PR heads, reverse source changes, and measure test contrast."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from repo2rlenv.execution.lifecycle import save_record
from repo2rlenv.execution.python_repository import bootstrap_snapshot, test_image
from repo2rlenv.quality.python_evidence import test_excerpts
from repo2rlenv.quality.test_results import execution_contrast
from repo2rlenv.spec.input import RepoSpec
from repo2rlenv.spec.recipe_options import PRRecipeOptions
from repo2rlenv.ui import console


def generate(source: dict, options: PRRecipeOptions, destination: Path) -> dict:
    # [FUNCTION DOC] Build one SWE-gen candidate.
    # Inputs: source (from fetch_source), options (PRRecipeOptions), destination (Path; must not
    # exist). Bootstraps RepoSpec(url=repo, ref=head, public), requires the healthy suite to pass,
    # copies source files to defective-source, writes source.diff and runs git apply --reverse there
    # (stderr saved to reverse-patch.stderr), tests with the defective files substituted, and
    # computes execution_contrast and F2P excerpts. Output: source fields plus contrast,
    # image_digest and test_evidence (saved to candidate.json). Raises ValueError if healthy fails
    # or the reverse patch does not apply. Side effects: Docker, git subprocess, writes.
    destination.mkdir(parents=True, exist_ok=False)
    repo = RepoSpec(url=source["repo"], ref=source["head"], access="public")
    boot, base = bootstrap_snapshot(repo, options, destination)
    healthy = test_image(boot.image_digest, options, destination / "healthy")
    if healthy.returncode or not healthy.passed:
        raise ValueError("Healthy PR head does not pass the configured test suite")
    defective = destination / "defective-source"
    defective.mkdir()
    for relative in source["source_files"]:
        path = defective / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / relative, path)
    patch = destination / "source.diff"
    patch.write_text(source["source_diff"])
    result = subprocess.run(
        ["git", "apply", "--reverse", str(patch.resolve())],
        cwd=defective,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    (destination / "reverse-patch.stderr").write_text(result.stderr)
    if result.returncode:
        raise ValueError("PR source changes cannot be reversed on the pinned healthy head")
    broken = test_image(
        boot.image_digest,
        options,
        destination / "defective",
        replacements={relative: defective / relative for relative in source["source_files"]},
    )
    contrast = execution_contrast(healthy, broken)
    excerpts = test_excerpts(base, contrast["FAIL_TO_PASS"])
    record = {
        **source,
        "contrast": contrast,
        "image_digest": boot.image_digest,
        "test_evidence": excerpts,
    }
    save_record(destination / "candidate.json", record)
    return record


def main() -> None:
    # [FUNCTION DOC] CLI entry: parse config and output, run generate with config source and
    # validated PRRecipeOptions, print id and contrast as JSON.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.config.read_text())
    result = generate(data["source"], PRRecipeOptions.model_validate(data["options"]), args.output)
    console.json({"id": result["id"], "contrast": result["contrast"]})


if __name__ == "__main__":
    main()
