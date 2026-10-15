# [MODULE DOC] Standalone FrontierSmith verifier copied into each task as tests/grade.py (stdlib
# only, Linux, run as root). Loads the private generator/scorer/optional feasibility modules,
# sanity-checks them, runs the submitted /workspace/solution.py per generated case as an
# unprivileged, resource-limited process, and writes the mean reward to /logs/verifier.
# Infrastructure faults raise (no reward); bad submissions score zero per case.
"""Standalone Linux verifier: private scoring, unprivileged bounded submissions.

Copied into Harbor tasks. Requires only Python's standard library. Infrastructure
errors exit without a reward; malformed, missing or failing submissions score zero.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


def load(path, name):
    # [FUNCTION DOC] HELPER: Imports a Python file as a module under the given name and returns it.
    # Raises: whatever the module raises on import. Side effects: executes the module code.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def limits():
    # [FUNCTION DOC] HELPER: preexec hook for the submission process; sets rlimits CPU 3s, address
    # space 256 MiB, file size 1 MiB, 32 processes and 64 open files. Output: None.
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def finite_reward(value):
    # [FUNCTION DOC] HELPER: Validates a reward: must be exactly int or float (bool rejected), finite,
    # and within [0, 1]. Output: float. Raises: ValueError otherwise.
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Scorer must return a finite numeric reward in [0,1]")
    return float(value)


def grade_output(scorer, validator, instance, output, status):
    # [FUNCTION DOC] Scores one case. Inputs: scorer and optional validator modules, instance, output,
    # status. Output: (score, feasible) where score is the scorer result if status is "completed"
    # else 0.0, and feasible is the validator's bool or None without a validator.
    # Raises: ValueError if the validator returns a non-bool, an infeasible output gets a nonzero
    # score, or a second scoring call differs (non-determinism).
    score = finite_reward(scorer.score(instance, output)) if status == "completed" else 0.0
    feasible = validator.is_feasible(instance, output) if validator else None
    if validator and type(feasible) is not bool:
        raise ValueError("Feasibility validator must return bool")
    if feasible is False and score != 0:
        raise ValueError("Scorer rewards an infeasible output")
    if status == "completed" and score != finite_reward(scorer.score(instance, output)):
        raise ValueError("Scorer is not deterministic")
    return score, feasible


def generated_cases(generator, seed):
    # [FUNCTION DOC] Generates test cases with generator.generate(seed) and validates them: list of
    # 8-16 dicts, JSON-serializable without NaN, at most 2 MiB, and identical on a second call.
    # Output: list of cases. Raises: ValueError (or TypeError from json) on violations.
    cases = generator.generate(seed)
    if (
        not isinstance(cases, list)
        or not 8 <= len(cases) <= 16
        or any(type(case) is not dict for case in cases)
    ):
        raise ValueError("Generator must return 8-16 JSON objects")
    serialized = json.dumps(cases, sort_keys=True, allow_nan=False)
    if len(serialized.encode()) > 2 * 1024 * 1024:
        raise ValueError("Generated test data exceeds 2 MiB")
    if serialized != json.dumps(generator.generate(seed), sort_keys=True, allow_nan=False):
        raise ValueError("Generator is not deterministic")
    return cases


def run_solution(path, instance, directory):
    # [FUNCTION DOC] Runs the submission on one instance as uid/gid 65534 in its own session with a
    # minimal env, rlimits and a 5s wall timeout, feeding the instance JSON on stdin; kills the
    # process group afterwards. Inputs: path (script), instance (dict), directory (cwd).
    # Output: (parsed dict, "completed") or (None, status) with status timeout, program_error,
    # output_limit (over 1 MiB), invalid_json (NaN/Infinity become None) or invalid_shape.
    # Side effects: subprocess, temp files.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            [sys.executable, "-I", str(path)],
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            cwd=directory,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"},
            user=65534,
            group=65534,
            extra_groups=[],
            start_new_session=True,
            preexec_fn=limits,
        )
        try:
            process.communicate(json.dumps(instance).encode(), timeout=5)
        except subprocess.TimeoutExpired:
            return None, "timeout"
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        if process.returncode != 0:
            return None, "program_error"
        stdout.seek(0)
        output = stdout.read(1024 * 1024 + 1)
        if len(output) > 1024 * 1024:
            return None, "output_limit"
        try:
            parsed = json.loads(output, parse_constant=lambda _: None)
        except (ValueError, UnicodeDecodeError, RecursionError):
            return None, "invalid_json"
        if not isinstance(parsed, dict):
            return None, "invalid_shape"
        return parsed, "completed"


def main():
    # [FUNCTION DOC] Verifier entry point. Requires euid 0; locks down tests/, /solution and
    # /logs/verifier permissions and removes stale reward files; loads contract and private modules;
    # generates cases for seed and checks generation for seed+1 and seed+2; checks the scorer gives 0
    # (and validator False) for malformed outputs; reads /workspace/solution.py safely (no symlinks,
    # regular file, at most 64 KiB) into a temp dir as read-only; grades each case; writes
    # scores.json, reward.txt and reward.json with the mean reward and prints a JSON summary.
    # Raises: RuntimeError if not root; ValueError for broken infrastructure. Side effects: filesystem,
    # subprocesses, stdout.
    if os.geteuid() != 0:
        raise RuntimeError("Verifier requires root to isolate submitted code")
    tests = Path(__file__).resolve().parent
    tests.chmod(0o700)
    if Path("/solution").exists():
        Path("/solution").chmod(0o700)
    log = Path("/logs/verifier")
    log.mkdir(parents=True, exist_ok=True)
    os.chown(log, 0, 0)
    log.chmod(0o700)
    for filename in ("reward.txt", "reward.json", "scores.json"):
        (log / filename).unlink(missing_ok=True)
    contract = json.loads((tests / "contract.json").read_text())
    generator = load(tests / "generator.py", "private_generator")
    scorer = load(tests / "scorer.py", "private_scorer")
    validator = (
        load(tests / "feasibility.py", "private_feasibility")
        if contract.get("explicit_feasibility")
        else None
    )
    cases = generated_cases(generator, contract["seed"])
    smoke_seeds = [contract["seed"] + offset for offset in (0, 1, 2)]
    for seed in smoke_seeds[1:]:
        generated_cases(generator, seed)
    for case in cases:
        for invalid in (None, True, [], {}, {"__invalid__": True}):
            if validator and validator.is_feasible(case, invalid) is not False:
                raise ValueError("Feasibility validator accepts malformed output")
            if finite_reward(scorer.score(case, invalid)) != 0:
                raise ValueError("Scorer accepts malformed output")

    solution = Path("/workspace/solution.py")
    code = None
    try:
        fd = os.open(solution, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (FileNotFoundError, OSError):
        fd = None
    if fd is not None:
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if stat.S_ISREG(info.st_mode) and info.st_size <= 65536:
                code = stream.read(65537)
                if len(code) > 65536:
                    code = None
    rows = []
    with tempfile.TemporaryDirectory(prefix="frontiersmith-") as temporary:
        directory = Path(temporary)
        directory.chmod(0o755)
        submitted = directory / "solution.py"
        if code is not None:
            submitted.write_bytes(code)
            submitted.chmod(0o444)
        for index, case in enumerate(cases):
            output, status = (
                run_solution(submitted, case, directory)
                if code is not None
                else (None, "missing_solution")
            )
            score, feasible = grade_output(scorer, validator, case, output, status)
            rows.append({"case": index, "reward": score, "status": status, "feasible": feasible})
    reward = sum(row["reward"] for row in rows) / len(rows)
    (log / "scores.json").write_text(
        json.dumps(
            {"reward": reward, "cases": rows, "generator_seeds_checked": smoke_seeds}, indent=2
        )
    )
    (log / "reward.txt").write_text(str(reward))
    (log / "reward.json").write_text(json.dumps({"reward": reward}))
    print(json.dumps({"reward": reward, "case_count": len(rows)}))


if __name__ == "__main__":
    main()
