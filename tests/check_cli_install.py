# [MODULE DOC] Standalone smoke check for an installed repo2rlenv CLI, run outside the checkout (not
# a pytest module). Invokes the installed console script in a temp dir, exercises --version/--help,
# each subcommand's help, pipelines list/describe for implemented recipes, JSON error output, and
# validates a written Unicode Harbor task with --deep. Run as a script it prints the JSON result via
# console.json. No cloud calls.
"""Smoke-test an installed CLI from outside the checkout, without cloud calls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from repo2rlenv.emitter.harbor import HarborTask, write_harbor_task
from repo2rlenv.ui import console


def check() -> dict:
    # [FUNCTION DOC] Locates the repo2rlenv executable in the interpreter's scripts dir
    # (repo2rlenv.exe on Windows), creates a temp directory (name contains a space) and runs:
    # --version (must contain the installed package version), --help, '<cmd> --help' for
    # generate/validate/push/pull/tasksmith/quality/pipelines, 'pipelines list --json' and
    # 'pipelines describe' for each implemented recipe (ids must match), 'tasksmith|quality show
    # missing.json --json' expecting exit 2 with exactly {error, message}. Then writes a pr_diff
    # HarborTask containing non-ASCII text, checks instruction.md round-trips, and runs 'validate
    # --deep' on it. Returns {status: 'passed', version, recipes: count}. Raises AssertionError on
    # any failure and subprocess.TimeoutExpired after 60s per call. Side effects: subprocesses, temp
    # files.
    executable = Path(sysconfig.get_path("scripts")) / (
        "repo2rlenv.exe" if sys.platform == "win32" else "repo2rlenv"
    )
    with TemporaryDirectory(prefix="repo2rlenv CLI ") as directory:
        root = Path(directory)

        def cli(*args: str, expected: int = 0) -> str:
            # [FUNCTION DOC] HELPER: Runs the installed executable with args in the temp dir with
            # PYTHON_DOTENV_DISABLED=1 and PYTHONUTF8=1 added to the environment, UTF-8 captured
            # output, 60s timeout. Asserts the return code equals expected (default 0, message is
            # stderr+stdout) and returns stdout.
            result = subprocess.run(
                [str(executable), *args],
                cwd=root,
                env={**os.environ, "PYTHON_DOTENV_DISABLED": "1", "PYTHONUTF8": "1"},
                capture_output=True,
                encoding="utf-8",
                timeout=60,
                check=False,
            )
            assert result.returncode == expected, result.stderr + result.stdout
            return result.stdout

        assert version("repo2rlenv") in cli("--version")
        assert "generate" in cli("--help")
        for command in (
            "generate",
            "validate",
            "push",
            "pull",
            "tasksmith",
            "quality",
            "pipelines",
        ):
            assert "usage:" in cli(command, "--help")
        listing = json.loads(cli("pipelines", "list", "--json"))
        recipes = [item for item in listing["recipes"] if item["implemented"]]
        for recipe in recipes:
            detail = json.loads(
                cli("pipelines", "describe", recipe["pipeline"], "--recipe", recipe["id"], "--json")
            )
            assert detail["id"] == recipe["id"]
        for command in ("tasksmith", "quality"):
            error = json.loads(cli(command, "show", "missing.json", "--json", expected=2))
            assert set(error) == {"error", "message"}
        task = HarborTask(
            name="unicode-task",
            org="smoke",
            description="Handle café input",
            instruction="# Task\n\nPreserve café and 日本語 in the output.\n",
            oracle_diff="--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-1\n+2\n",
            repo2env={"pipeline": "pr_diff", "pipeline_version": "0.1.0", "repo": "smoke/demo"},
        )
        output = write_harbor_task(task, root / "task output")
        assert (output / "instruction.md").read_text(encoding="utf-8") == task.instruction
        cli("validate", str(root / "task output"), "--deep")
    return {"status": "passed", "version": version("repo2rlenv"), "recipes": len(recipes)}


if __name__ == "__main__":
    console.json(check())
