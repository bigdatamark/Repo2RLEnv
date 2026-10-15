# [MODULE DOC] Remote-worker module (run via execution.generation.run_generator) that builds a
# TerminalWorld candidate environment with Docker and replays the reference inside a locked-down
# container using capture.py, producing snapshot.json with the observed filesystem effects.
"""Remote Docker build and reference replay for recording-driven test synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import uuid
from importlib.resources import files
from pathlib import Path

from repo2rlenv.execution.job import container_labels
from repo2rlenv.execution.lifecycle import save_record
from repo2rlenv.pipelines.recipes.terminal.draft import EnvironmentDefinition, dockerfile_for_setup


def capture(config: dict, destination: Path) -> dict:
    # [FUNCTION DOC] Build the image and capture the reference's effects. Inputs: config (dict with
    # "environment" (EnvironmentDefinition data), "solution_shell" (str) and "timeout_sec" (int));
    # destination (Path, must not exist). Steps: validate the environment, write files into
    # destination/environment with mode 755/644 and an owned Dockerfile (no agent user); docker
    # build tagged "repo2rlenv-recording-" + 24 hex chars of SHA-256(config) with a 600 s timeout,
    # logging to build.log. On build failure returns and saves {returncode, stage "build", last
    # 18000 chars of log}. Otherwise writes reference.sh, creates a container (repo2rlenv labels, no
    # network, 1 CPU, 2 GiB, 256 pids, all capabilities dropped, no-new-privileges, workdir
    # /workspace) whose command is "python /tmp/capture.py /tmp/reference.sh <timeout>", copies
    # reference.sh and capture.py into /tmp, then starts it attached (timeout = timeout_sec + 60)
    # logging to replay.log. A nonzero container exit yields {returncode, stage "capture", log
    # tail}; otherwise copies /tmp/recording-snapshot.json out and loads it. Adds image_digest
    # (image ID) and saves snapshot.json. Always force-removes the container (the image is kept).
    # Output: result dict. Raises: FileExistsError, ValidationError, CalledProcessError or
    # TimeoutExpired from Docker steps (build timeout included).
    destination.mkdir(parents=True, exist_ok=False)
    environment = EnvironmentDefinition.model_validate(config["environment"])
    context = destination / "environment"
    context.mkdir()
    for asset in environment.environment_files:
        path = context / asset.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(asset.content)
        path.chmod(0o755 if asset.executable else 0o644)
    (context / "Dockerfile").write_text(dockerfile_for_setup(environment.environment_setup))
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:24]
    tag = "repo2rlenv-recording-" + identity
    with (destination / "build.log").open("w") as log:
        built = subprocess.run(
            ["docker", "build", "-t", tag, str(context)],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
    if built.returncode:
        result = {
            "returncode": built.returncode,
            "stage": "build",
            "log": (destination / "build.log").read_text()[-18000:],
        }
        save_record(destination / "snapshot.json", result)
        return result
    reference = destination / "reference.sh"
    reference.write_text(config["solution_shell"])
    name = "recording-replay-" + uuid.uuid4().hex
    try:
        subprocess.run(
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
                tag,
                "python",
                "/tmp/capture.py",
                "/tmp/reference.sh",
                str(config["timeout_sec"]),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        for local, remote in [
            (reference, "reference.sh"),
            (Path(str(files(__package__).joinpath("capture.py"))), "capture.py"),
        ]:
            subprocess.run(
                ["docker", "cp", str(local), f"{name}:/tmp/{remote}"],
                check=True,
                capture_output=True,
                timeout=30,
            )
        with (destination / "replay.log").open("w") as log:
            replay = subprocess.run(
                ["docker", "start", "-a", name],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=config["timeout_sec"] + 60,
            )
        if replay.returncode:
            result = {
                "returncode": replay.returncode,
                "stage": "capture",
                "log": (destination / "replay.log").read_text()[-18000:],
            }
        else:
            subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{name}:/tmp/recording-snapshot.json",
                    str(destination / "snapshot.json"),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            result = json.loads((destination / "snapshot.json").read_text())
        image = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", tag],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        result["image_digest"] = image.stdout.strip()
        save_record(destination / "snapshot.json", result)
        return result
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


def main() -> None:
    # [FUNCTION DOC] CLI: positional config (JSON path) and output (Path); loads the config and
    # calls capture(). Output: None.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    capture(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
