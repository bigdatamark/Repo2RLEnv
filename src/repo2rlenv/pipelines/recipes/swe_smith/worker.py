# [MODULE DOC] Remote-worker entry point for SWE-smith procedural bug generation. It bootstraps the
# target repository using Repo2RLEnv's existing bootstrap/cache, verifies the healthy test suite
# passes, enumerates single-site source mutations per file (owned mutation enumeration), runs the
# tests on each mutated image, and keeps candidates whose JUnit results contrast with the healthy
# run (strict test-identity comparison). Records, rejected candidates and progress events are
# written under the output directory. Only runs when REPO2RLENV_REMOTE_WORKER=1. Acknowledges SWE-
# smith (MIT).
"""Remote-only procedural generation, reusing Repo2RLEnv's bootstrap/cache.

Acknowledgment: SWE-smith (MIT), pinned in provenance.md. Owned single-site
mutation enumeration and strict JUnit identity comparison replace the upstream
profile registry and validator, while retaining mutation/test/issue generation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

from repo2rlenv.campaigns.events import EventJournal, ProgressEvent
from repo2rlenv.execution.lifecycle import save_record
from repo2rlenv.execution.python_repository import (
    bootstrap_snapshot,
    repository_source_files,
    test_image,
)
from repo2rlenv.pipelines.recipes.swe_smith.mutations import generate_mutations
from repo2rlenv.pipelines.recipes.swe_smith.options import SWESmithOptions
from repo2rlenv.quality.test_results import execution_contrast
from repo2rlenv.spec.input import RepoSpec
from repo2rlenv.ui import console


def generate(repo: RepoSpec, options: SWESmithOptions, destination: Path) -> dict:
    # [FUNCTION DOC] Generate execution-validated mutation candidates for one repository inside a
    # remote worker. Inputs: repo (RepoSpec) - repository to bootstrap; options (SWESmithOptions) -
    # uses seed, max_candidates, max_per_entity, target, exclude_candidate_ids plus bootstrap/test
    # options consumed by helpers; destination (Path) - must not already exist. Behavior: (1)
    # refuses unless env var REPO2RLENV_REMOTE_WORKER == "1"; (2) creates destination and an
    # EventJournal at destination/events.jsonl; (3) bootstrap_snapshot builds/reuses the image and
    # returns (boot, base checkout); (4) runs tests on the healthy image into destination/healthy
    # and requires exit code 0 with at least one passing test; (5) lists source files, shuffles them
    # deterministically with Random(seed); (6) for each file, generate_mutations(original text,
    # relative path, seed, limit= max_candidates); skips excluded ids and entities already at
    # max_per_entity for (file, entity); stops once attempted reaches max_candidates or accepted
    # records reach target; (7) per candidate writes candidates/<id>/{mutated.py, original.py,
    # mutation.diff}, tests the image with the mutated file replacing the original, computes
    # execution_contrast(healthy, defective); RuntimeError/ValueError/ TimeoutExpired reject the
    # candidate (recorded with exception type and message); accepted records (id, source_file,
    # entity, operator, line, repo url, ref, image_digest, contrast) are saved to candidate.json and
    # generation.json is rewritten after each acceptance. Output: dict with "candidates" (accepted
    # records), "rejected", "attempted"; also saved to destination/generation.json. Raises:
    # RuntimeError outside a remote worker; ValueError if healthy tests do not pass; FileExistsError
    # if destination exists or a candidate directory collides. Side effects: filesystem writes,
    # Docker/test subprocesses via helpers, progress events.
    if os.environ.get("REPO2RLENV_REMOTE_WORKER") != "1":
        raise RuntimeError("This entry point runs inside a remote generation worker only")
    destination.mkdir(parents=True, exist_ok=False)
    journal = EventJournal(destination / "events.jsonl")

    def event(stage, state, message, **metrics):
        # [FUNCTION DOC] HELPER: Emit one ProgressEvent for recipe "swe_smith" to the journal.
        # Inputs: stage, state, message (str) and arbitrary keyword metrics. Output: None. Side
        # effect: appends to events.jsonl via the enclosing EventJournal.
        journal.emit(
            ProgressEvent(
                recipe="swe_smith", stage=stage, state=state, message=message, metrics=metrics
            )
        )

    event(
        "bootstrap", "started", "Build the explicit repository profile using the existing bootstrap"
    )
    boot, base = bootstrap_snapshot(repo, options, destination)
    healthy = test_image(boot.image_digest, options, destination / "healthy")
    if healthy.returncode != 0 or not healthy.passed:
        raise ValueError("Healthy repository tests must pass before mutation")
    event(
        "bootstrap",
        "completed",
        "Fresh healthy test suite passed",
        healthy_tests=len(healthy.passed),
    )

    paths = repository_source_files(base, options)
    random.Random(options.seed).shuffle(paths)
    records, rejected, entity_counts = [], [], {}
    attempted = 0
    for path in paths:
        relative = path.relative_to(base).as_posix()
        original = path.read_text()
        candidates = generate_mutations(
            original, relative, seed=options.seed, limit=options.max_candidates
        )
        for candidate in candidates:
            if candidate.id in options.exclude_candidate_ids:
                continue
            key = (relative, candidate.entity)
            if entity_counts.get(key, 0) >= options.max_per_entity:
                continue
            if attempted >= options.max_candidates or len(records) >= options.target:
                break
            attempted += 1
            output = destination / "candidates" / candidate.id
            output.mkdir(parents=True)
            (output / "mutated.py").write_text(candidate.source)
            (output / "original.py").write_text(original)
            (output / "mutation.diff").write_text(candidate.patch)
            event(
                "contrast",
                "started",
                candidate.entity,
                attempted=attempted,
                execution_passed=len(records),
            )
            try:
                defective = test_image(
                    boot.image_digest,
                    options,
                    output / "defective",
                    replacement=(output / "mutated.py", relative),
                )
                contrast = execution_contrast(healthy, defective)
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                rejected.append(
                    {"id": candidate.id, "reason": type(exc).__name__, "detail": str(exc)}
                )
                event("contrast", "failed", str(exc))
                continue
            record = {
                "id": candidate.id,
                "source_file": relative,
                "entity": candidate.entity,
                "operator": candidate.operator,
                "line": candidate.line,
                "repo": repo.url,
                "ref": boot.ref,
                "image_digest": boot.image_digest,
                "contrast": contrast,
            }
            save_record(output / "candidate.json", record)
            records.append(record)
            entity_counts[key] = entity_counts.get(key, 0) + 1
            event(
                "contrast",
                "completed",
                candidate.entity,
                attempted=attempted,
                execution_passed=len(records),
            )
            save_record(
                destination / "generation.json",
                {"candidates": records, "rejected": rejected, "attempted": attempted},
            )
        if attempted >= options.max_candidates or len(records) >= options.target:
            break
    result = {"candidates": records, "rejected": rejected, "attempted": attempted}
    save_record(destination / "generation.json", result)
    return result


def main() -> None:
    # [FUNCTION DOC] CLI entry: parse positional args config (JSON path) and output (Path), load the
    # config's "repo" and "options" objects into RepoSpec and SWESmithOptions, call generate(), then
    # print a JSON summary (attempted, execution_passed = number of accepted candidates) through the
    # console. Output: None. Raises: argparse SystemExit on bad args, KeyError/ValidationError on
    # malformed config, and anything generate() raises.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.config.read_text())
    result = generate(
        RepoSpec.model_validate(data["repo"]),
        SWESmithOptions.model_validate(data["options"]),
        args.output,
    )
    console.json({"attempted": result["attempted"], "execution_passed": len(result["candidates"])})


if __name__ == "__main__":
    main()
