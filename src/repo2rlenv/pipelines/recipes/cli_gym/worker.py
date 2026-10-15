# [MODULE DOC] Remote worker for CLI-Gym, executed on the provider worker via run_generator as
# "python -m ... config.json output_dir". "prepare" mode bootstraps the repo snapshot, proves the
# healthy suite passes, records protected file hashes, installs an isolated verifier venv and
# samples candidates. "evaluate" mode applies an inversion in a fresh container, checks protected
# files, commits the damaged image and runs the suite with and without the recovery script to
# produce contrast evidence. All containers run with no network and resource limits and are
# force-removed afterwards.
"""Remote healthy baseline, environment inversion, restoration and test evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import uuid
from pathlib import Path

from repo2rlenv.execution.job import container_labels
from repo2rlenv.execution.lifecycle import save_record
from repo2rlenv.execution.python_repository import bootstrap_snapshot, test_image
from repo2rlenv.pipelines.recipes.cli_gym.models import Inversion
from repo2rlenv.quality.test_results import parse_junit
from repo2rlenv.spec.input import RepoSpec
from repo2rlenv.spec.recipe_options import EnvironmentRepairOptions

_CONTROL = "/opt/repo2rlenv-verifier/bin/python"


def run(argv, *, timeout=60, check=True):
    # [FUNCTION DOC] HELPER: Runs a subprocess with captured UTF-8 text output. Inputs: argv (list),
    # keyword timeout (seconds, default 60), check (default True; raise CalledProcessError on
    # non-zero). Output: CompletedProcess. Raises TimeoutExpired / CalledProcessError.
    return subprocess.run(
        argv, capture_output=True, text=True, encoding="utf-8", timeout=timeout, check=check
    )


def create(image: str) -> str:
    # [FUNCTION DOC] HELPER: Creates and starts a sandbox container from image with repo2rlenv
    # labels, a random name "cli-inversion-<hex>", no network, 1 CPU, 2g memory, 256 pids, all
    # capabilities dropped, no-new-privileges, workdir /workspace, running "sleep infinity". Input:
    # image (str). Output: container name. Raises CalledProcessError on docker failure.
    name = "cli-inversion-" + uuid.uuid4().hex
    run(
        [
            "docker",
            "create",
            *container_labels(),
            "--name",
            name,
            "--network",
            "none",
            "--cpus",
            "1",
            "--memory",
            "2g",
            "--pids-limit",
            "256",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--workdir",
            "/workspace",
            image,
            "sleep",
            "infinity",
        ]
    )
    run(["docker", "start", name])
    return name


def prepare(repo: RepoSpec, options: EnvironmentRepairOptions, destination: Path) -> dict:
    # [FUNCTION DOC] Prepare mode. Inputs: repo (RepoSpec), options (EnvironmentRepairOptions),
    # destination (Path, must not exist). Bootstraps a snapshot (bootstrap_snapshot) and runs the
    # healthy test image; raises ValueError if tests fail or nothing passed. Hashes every regular
    # non-symlink file under options.source_paths and test_paths (skipping __pycache__). In a
    # temporary container it creates the /opt/repo2rlenv-verifier venv, captures "pip list" JSON and
    # commits the container to a new image (container always removed). Then builds
    # options.max_candidates candidates with a seeded RNG: each has repo url, ref, index, seed, a
    # 20-char sha256 id of that identity, a direction cycled from options.directions, up to 50
    # sampled passing tests and the committed image id. Writes environment.json (protected hashes,
    # required_tests = all passing tests, packages, first 800 file paths) and generation.json.
    # Output: dict {candidates, rejected: [], attempted}.
    destination.mkdir(parents=True, exist_ok=False)
    boot, base = bootstrap_snapshot(repo, options, destination)
    healthy = test_image(boot.image_digest, options, destination / "healthy")
    if healthy.returncode or not healthy.passed:
        raise ValueError("CLI-Gym needs a nonempty, passing gold test suite")
    protected = {}
    for relative in [*options.source_paths, *options.test_paths]:
        root = base / relative
        for path in [root] if root.is_file() else sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                protected[path.relative_to(base).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    container = create(boot.image_digest)
    try:
        run(
            [
                "docker",
                "exec",
                container,
                "python",
                "-m",
                "venv",
                "--copies",
                "/opt/repo2rlenv-verifier",
            ]
        )
        packages = run(
            ["docker", "exec", container, "python", "-m", "pip", "list", "--format=json"]
        ).stdout
        image = run(["docker", "commit", container]).stdout.strip()
    finally:
        run(["docker", "rm", "-f", container], check=False)
    rng = random.Random(options.seed)
    tests = sorted(healthy.passed)
    candidates = []
    for index in range(options.max_candidates):
        identity = {"repo": repo.url, "ref": boot.ref, "index": index, "seed": options.seed}
        candidates.append(
            {
                **identity,
                "id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
                    :20
                ],
                "direction": options.directions[index % len(options.directions)],
                "sampled_tests": rng.sample(tests, min(50, len(tests))),
                "image_digest": image,
            }
        )
    save_record(
        destination / "environment.json",
        {
            "protected": protected,
            "required_tests": tests,
            "packages": json.loads(packages),
            "file_paths": sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())[
                :800
            ],
        },
    )
    result = {"candidates": candidates, "rejected": [], "attempted": len(candidates)}
    save_record(destination / "generation.json", result)
    return result


def suite(image: str, options, output: Path, recovery: Path | None = None) -> dict:
    # [FUNCTION DOC] Runs the repository test suite in a fresh container of image. Inputs: image,
    # options (test_timeout_sec), output (Path, must not exist), recovery (optional Path to a
    # recovery script). If recovery is given it is copied in and run with bash; on non-zero exit
    # returns {returncode, stage: "recovery", passed: [], log tail} (writes recovery.log). Then runs
    # pytest on tests with a junit XML, writes tests.log, copies results.xml out and parses passed
    # ids (parse errors ignored). Writes result.json and returns {returncode, passed (sorted ids),
    # log (last 18000 chars)}. Container is always removed. Timeouts raise TimeoutExpired.
    output.mkdir(parents=True, exist_ok=False)
    container = create(image)
    try:
        if recovery is not None:
            run(["docker", "cp", str(recovery), f"{container}:/tmp/recovery.sh"])
            result = run(
                ["docker", "exec", container, "bash", "/tmp/recovery.sh"],
                timeout=options.test_timeout_sec,
                check=False,
            )
            (output / "recovery.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                return {
                    "returncode": result.returncode,
                    "stage": "recovery",
                    "passed": [],
                    "log": (result.stdout + result.stderr)[-18000:],
                }
        result = run(
            [
                "docker",
                "exec",
                container,
                "python",
                "-m",
                "pytest",
                "tests",
                "-q",
                "--tb=short",
                "--junitxml=/tmp/results.xml",
            ],
            timeout=options.test_timeout_sec,
            check=False,
        )
        log = result.stdout + result.stderr
        (output / "tests.log").write_text(log)
        copied = run(
            ["docker", "cp", f"{container}:/tmp/results.xml", str(output / "results.xml")],
            check=False,
        )
        passed = []
        if copied.returncode == 0:
            try:
                parsed = parse_junit(
                    (output / "results.xml").read_text(), returncode=result.returncode
                )
                passed = sorted(parsed.passed)
            except ValueError:
                pass
        evidence = {"returncode": result.returncode, "passed": passed, "log": log[-18000:]}
        save_record(output / "result.json", evidence)
        return evidence
    finally:
        run(["docker", "rm", "-f", container], check=False)


def evaluate(config, options, destination: Path) -> dict:
    # [FUNCTION DOC] Evaluate mode. Inputs: config (dict with inversion, generation dir path,
    # candidate.image_digest), options, destination (Path, must not exist). Writes destruction.sh
    # and recovery.sh, starts a container from the candidate image and runs the destruction script.
    # If it fails: result contrast False, stage "inversion". Otherwise runs a hashing script with
    # the isolated verifier python to list protected files that are missing or changed; if any (or
    # the check fails) contrast False, stage "protected_files". Else removes the script, commits the
    # damaged image, runs suite() for baseline (no recovery) and reference (with recovery). contrast
    # is True when required tests are nonempty, not all pass at baseline, reference exits 0 and all
    # required pass in reference; also records baseline, reference, image_digest and
    # missing_baseline_tests. Saves evaluation.json and returns the result dict. The original
    # container is always removed (committed images are not).
    destination.mkdir(parents=True, exist_ok=False)
    inversion = Inversion.model_validate(config["inversion"])
    generation = Path(config["generation"])
    environment = json.loads((generation / "environment.json").read_text())
    destruction = destination / "destruction.sh"
    destruction.write_text(inversion.destruction_shell)
    recovery = destination / "recovery.sh"
    recovery.write_text(inversion.recovery_shell)
    container = create(config["candidate"]["image_digest"])
    try:
        run(["docker", "cp", str(destruction), f"{container}:/tmp/inversion.sh"])
        changed = run(
            ["docker", "exec", container, "bash", "/tmp/inversion.sh"],
            timeout=options.test_timeout_sec,
            check=False,
        )
        if changed.returncode:
            result = {
                "contrast": False,
                "stage": "inversion",
                "log": (changed.stdout + changed.stderr)[-18000:],
            }
        else:
            script = "import hashlib,json,pathlib,sys; expected=json.loads(sys.argv[1]); bad=[p for p,h in expected.items() if not (pathlib.Path('/workspace')/p).is_file() or hashlib.sha256((pathlib.Path('/workspace')/p).read_bytes()).hexdigest()!=h]; sys.stdout.write(json.dumps(bad))"
            checked = run(
                [
                    "docker",
                    "exec",
                    container,
                    _CONTROL,
                    "-I",
                    "-c",
                    script,
                    json.dumps(environment["protected"]),
                ],
                check=False,
            )
            if checked.returncode or json.loads(checked.stdout):
                result = {
                    "contrast": False,
                    "stage": "protected_files",
                    "log": "Inversion modified source/tests or broke the isolated control interpreter",
                }
            else:
                run(["docker", "exec", container, "rm", "-f", "/tmp/inversion.sh"])
                image = run(["docker", "commit", container]).stdout.strip()
                baseline = suite(image, options, destination / "baseline")
                reference = suite(image, options, destination / "reference", recovery)
                required = set(environment["required_tests"])
                result = {
                    "contrast": bool(required)
                    and not required <= set(baseline["passed"])
                    and reference["returncode"] == 0
                    and required <= set(reference["passed"]),
                    "baseline": baseline,
                    "reference": reference,
                    "image_digest": image,
                    "missing_baseline_tests": sorted(required - set(baseline["passed"])),
                }
    finally:
        run(["docker", "rm", "-f", container], check=False)
    save_record(destination / "evaluation.json", result)
    return result


def main() -> None:
    # [FUNCTION DOC] CLI entrypoint: parses positional config (JSON path) and output (Path),
    # validates options as EnvironmentRepairOptions, and runs evaluate when config["mode"] ==
    # "evaluate", otherwise prepare with the config's RepoSpec. Output: None.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    options = EnvironmentRepairOptions.model_validate(config["options"])
    if config.get("mode") == "evaluate":
        evaluate(config, options, args.output)
    else:
        prepare(RepoSpec.model_validate(config["repo"]), options, args.output)


if __name__ == "__main__":
    main()
