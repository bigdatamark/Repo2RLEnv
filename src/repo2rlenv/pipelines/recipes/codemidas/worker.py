# [MODULE DOC] CodeMidas remote-worker entry point. Runs inside a credential-free remote worker
# in two modes driven by a JSON config: "prepare" (default) bootstraps the repository snapshot,
# checks that a sanitized public build context builds, and discovers candidate public
# function/class anchors to author features against; "evaluate" runs a generated verifier against
# the healthy reference and a defective (implementation-removed) source and records the contrast.
# Invoked as a script: worker CONFIG OUTPUT.
"""Credential-free discovery and reference execution inside a remote worker."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path

from repo2rlenv.execution.lifecycle import save_record
from repo2rlenv.execution.python_repository import (
    _run,
    bootstrap_snapshot,
    repository_source_files,
    test_image,
)
from repo2rlenv.pipelines.recipes.codemidas.models import Feature, Verifier
from repo2rlenv.pipelines.recipes.codemidas.source import (
    implementation_pair,
    symbols,
    validate_assertions,
)
from repo2rlenv.quality.test_results import execution_contrast
from repo2rlenv.spec.input import RepoSpec
from repo2rlenv.spec.recipe_options import CodeMidasOptions


def prepare(repo, options, destination, stack_source=None):
    # [FUNCTION DOC] Discovery phase: bootstraps the repo and selects candidate anchors.
    # Inputs: repo (RepoSpec); options (CodeMidasOptions); destination (Path, must not exist);
    # stack_source (optional Stack manifest dict; when given, provenance is recorded and, for inline
    # materialization, the row is written to destination/stack-source, committed into a fresh git repo
    # via subprocess git, and used as the bootstrap repo).
    # Steps: bootstrap_snapshot gives (boot, base); private_paths are test/docs/example/benchmark/
    # .github/conftest/test_*/ *_test.py/ *.egg-info paths in base; builds the sanitized public context
    # in destination/public-readiness with docker build (600s), logging to public-readiness.log, and
    # removes it on success. For each repository source file outside private paths (and outside
    # underscore-prefixed path parts unless listed in options.source_paths) parses symbols; public
    # symbols spanning at least 12 lines with at least options.min_implementation_statements
    # non-def, non-docstring statements are shuffled (seeded by seed+path), and up to
    # options.max_per_module per module are kept, skipping ids in options.exclude_candidate_ids. Each
    # candidate id is the first 20 hex chars of sha256 over repo/ref/path/anchor. Candidates are
    # shuffled with options.seed and truncated to options.max_candidates.
    # Output: dict {candidates, rejected (syntax-error files), attempted}; also saved to
    # generation.json. Raises: ValueError if the sanitized repository fails to build; FileExistsError
    # if destination exists; subprocess errors from git/docker. Side effects: filesystem, Docker, git.
    destination.mkdir(parents=True, exist_ok=False)
    bootstrap_repo = repo
    source_record = {"source_kind": "github"}
    if stack_source:
        from repo2rlenv.pipelines.recipes.codemidas.stack import materialize, provenance

        source_record = provenance(stack_source, options.stack_materialization)
        if options.stack_materialization == "inline":
            checkout = destination / "stack-source"
            materialize(stack_source, checkout)
            for command in (
                ["init", "-q"],
                ["add", "."],
                [
                    "-c",
                    "user.name=CodeMidas",
                    "-c",
                    "user.email=codemidas@localhost",
                    "commit",
                    "-qm",
                    "Materialize pinned Stack repository row",
                ],
            ):
                _run(["git", "-C", str(checkout), *command])
            bootstrap_repo = RepoSpec(url=str(checkout.resolve()), access="public")
        save_record(destination / "source-provenance.json", source_record)
    boot, base = bootstrap_snapshot(bootstrap_repo, options, destination)
    private_paths = sorted(
        {
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.name
            in {"tests", "test", "docs", "examples", "benchmarks", ".github", "conftest.py"}
            or path.name.startswith("test_")
            or path.name.endswith(("_test.py", ".egg-info"))
        }
    )
    from repo2rlenv.pipelines.recipes.codemidas.sanitize import public_build_context, public_profile

    public = destination / "public-readiness"
    public_build_context(base, public_profile(base, options, private_paths), public)
    ready = _run(["docker", "build", "--quiet", str(public)], timeout=600, check=False)
    (destination / "public-readiness.log").write_text(ready.stdout + ready.stderr)
    if ready.returncode:
        raise ValueError(
            "Sanitized repository cannot build; inspect public-readiness.log before authoring"
        )
    shutil.rmtree(public)
    candidates, rejected = [], []
    for path in repository_source_files(base, options):
        relative = path.relative_to(base).as_posix()
        if any(relative == item or relative.startswith(item + "/") for item in private_paths):
            continue
        # Public names inside private helper modules are not necessarily APIs.
        if relative not in options.source_paths and any(
            part.startswith("_") and part != "__init__.py" for part in Path(relative).parts
        ):
            continue
        source = path.read_text()
        try:
            available = symbols(source)
        except SyntaxError:
            rejected.append({"reason": "unsupported_python_syntax", "path": relative})
            continue
        anchors = [
            (name, node)
            for name, node in available.items()
            if not name.split(".")[-1].startswith("_")
            and node.end_lineno - node.lineno >= 12
            and sum(
                isinstance(child, ast.stmt)
                and not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not (
                    isinstance(child, ast.Expr)
                    and isinstance(child.value, ast.Constant)
                    and isinstance(child.value.value, str)
                )
                for child in ast.walk(node)
            )
            >= options.min_implementation_statements
        ]
        random.Random(f"{options.seed}:{relative}").shuffle(anchors)
        selected = 0
        for name, node in anchors:
            if selected >= options.max_per_module:
                break
            identity = {
                "repo": repo.url,
                "ref": repo.ref if stack_source else boot.ref,
                "path": relative,
                "anchor": name,
            }
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
            if key in options.exclude_candidate_ids:
                continue
            selected += 1
            candidates.append(
                {
                    **identity,
                    "id": key,
                    "image_digest": boot.image_digest,
                    "source": ast.get_source_segment(source, node)[:20000],
                    "private_paths": private_paths,
                    "source_kind": source_record["source_kind"],
                    "source_provenance": source_record,
                }
            )
    random.Random(options.seed).shuffle(candidates)
    candidates = candidates[: options.max_candidates]
    result = {
        "candidates": candidates,
        "rejected": rejected,
        "attempted": len(candidates) + len(rejected),
    }
    save_record(destination / "generation.json", result)
    return result


def evaluate(config, options, destination):
    # [FUNCTION DOC] Evaluation phase: runs a generated verifier against reference and defective code.
    # Inputs: config (dict with feature, verifier, generation dir, image_digest); options
    # (CodeMidasOptions); destination (Path, must not exist).
    # Behavior: validates assertions, derives the defective implementation via implementation_pair,
    # writes the verifier test as test_codemidas_generated.py, runs test_image on the reference (must
    # pass), then with defective files replaced (baseline must have no setup "error" statuses), and
    # stores execution_contrast(healthy, broken). ValueError messages and test timeouts (with phase,
    # error_type "test_timeout" and the last 18000 chars of output) are captured into the result rather
    # than raised. Appends the last 18000 chars of each stage's stdout.txt.
    # Output: result dict, also saved to evaluation.json. Raises: ValueError (uncaught) if the
    # verifier filename already exists in the source base. Side effects: filesystem, Docker runs.
    destination.mkdir(parents=True, exist_ok=False)
    feature, verifier = (
        Feature.model_validate(config["feature"]),
        Verifier.model_validate(config["verifier"]),
    )
    validate_assertions(feature, verifier)
    base = Path(config["generation"]) / "base"
    defective, _ = implementation_pair(base, feature, options.source_paths)
    private = destination / "test_codemidas_generated.py"
    if (base / private.name).exists():
        raise ValueError("Generated verifier path already exists in the source")
    private.write_text(verifier.test_code)
    replacements = {private.name: private}
    result = {"contrast": None}
    phase = "reference"
    try:
        healthy = test_image(
            config["image_digest"], options, destination / "reference", replacements=replacements
        )
        if healthy.returncode:
            raise ValueError("Generated tests fail on the unchanged reference")
        for relative, content in defective.items():
            path = destination / "defective-source" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            replacements[relative] = path
        phase = "baseline"
        broken = test_image(
            config["image_digest"], options, destination / "baseline", replacements=replacements
        )
        if any(status == "error" for status in broken.statuses.values()):
            raise ValueError("Baseline has setup errors rather than behavioral test failures")
        result["contrast"] = execution_contrast(healthy, broken)
    except ValueError as exc:
        result["error"] = str(exc)
    except subprocess.TimeoutExpired as exc:
        # A bounded verifier hang is repair feedback, not a successful baseline
        # failure. test_image has already removed its container in its finally.
        result["error"] = f"{phase} test execution exceeded {options.test_timeout_sec} seconds"
        result["error_type"] = "test_timeout"
        result["phase"] = phase
        output = exc.stdout or b""
        result["timeout_output"] = (
            output.decode(errors="replace") if isinstance(output, bytes) else output
        )[-18000:]
    for stage in ("reference", "baseline"):
        stdout = destination / stage / "stdout.txt"
        result[stage + "_log"] = stdout.read_text()[-18000:] if stdout.exists() else ""
    save_record(destination / "evaluation.json", result)
    return result


def main():
    # [FUNCTION DOC] CLI entry: parses CONFIG and OUTPUT paths, loads the JSON config, validates
    # CodeMidasOptions, and dispatches to evaluate (mode == "evaluate") or prepare (otherwise, with
    # RepoSpec from config["repo"] and optional stack_source). Output: None. Side effects: as callees.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    options = CodeMidasOptions.model_validate(config["options"])
    if config.get("mode") == "evaluate":
        evaluate(config, options, args.output)
    else:
        prepare(
            RepoSpec.model_validate(config["repo"]),
            options,
            args.output,
            config.get("stack_source"),
        )


if __name__ == "__main__":
    main()
