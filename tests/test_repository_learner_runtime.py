# [MODULE DOC] Tests that exported repository tasks' learner Dockerfile startup logic (login-profile
# installer and the BASH_ENV non-interactive hook) puts the task venv first on PATH, preserves the
# user's own startup files, and is absent from the verifier image. Executes the emitted shell
# snippets against temporary fake HOME/venv/bin directories with /bin/sh and /bin/bash; never runs
# task code or Docker.
"""Exercise emitted learner startup using temporary shell fixtures, never task code."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from repo2rlenv.pipelines.recipes.repository.export import export_repository_task, repository_build
from repo2rlenv.spec.recipe_options import PythonRepositoryProfile


def exported_task(tmp_path: Path, *, venv: bool = True):
    # [FUNCTION DOC] HELPER: Builds a tiny source tree (src/core.py) under tmp_path/source and
    # exports a repository task via export_repository_task with a defective 'value = 0' and
    # reference 'value = 1'. Inputs: tmp_path; venv (bool, default True) mapped to
    # PythonRepositoryProfile.use_system_site_packages. Output: (task directory Path, profile
    # options). Side effects: writes under tmp_path.
    base = tmp_path / "source"
    (base / "src").mkdir(parents=True)
    (base / "src/core.py").write_text("value = 1\n")
    options = PythonRepositoryProfile(
        source_paths=["src"], test_paths=["tests"], use_system_site_packages=venv
    )
    task = export_repository_task(
        base=base,
        defective={"src/core.py": b"value = 0\n"},
        reference={"src/core.py": b"value = 1\n"},
        options=options,
        instruction="Restore the public value.",
        destination=tmp_path / "tasks",
        name="learner-runtime-fixture",
        org="test",
        contrast={"FAIL_TO_PASS": ["test_value"], "PASS_TO_PASS": []},
        metadata={"recipe": "tasksmith", "recipe_version": "1"},
    )
    return task, options


def shell_fixture(tmp_path: Path):
    # [FUNCTION DOC] HELPER: Creates a 'learner home' dir, a fake 'task runtime' venv and a 'base
    # bin' dir, each bin holding python/python3 shell scripts that print 'base' or 'task-venv'.
    # Output: (learner_home, venv path, env dict with HOME, minimal PATH and BASE_BIN). Side
    # effects: writes files, chmod 0755.
    learner_home = tmp_path / "learner home"
    learner_home.mkdir()
    venv = tmp_path / "task runtime"
    base_bin = tmp_path / "base bin"
    for directory, value in ((base_bin, "base"), (venv / "bin", "task-venv")):
        directory.mkdir(parents=True)
        for name in ("python", "python3"):
            executable = directory / name
            executable.write_text(f"#!/bin/sh\nprintf '%s\\n' {value}\n")
            executable.chmod(0o755)
    # A child-only HOME keeps the test away from developer/system login files.
    env = {"HOME": str(learner_home), "PATH": "/usr/bin:/bin", "BASE_BIN": str(base_bin)}
    return learner_home, venv, env


def startup_result(learner_home: Path, env: dict[str, str]):
    # [FUNCTION DOC] HELPER: Picks the first existing login file (.bash_profile, .bash_login,
    # .profile) in learner_home, runs bash --noprofile --norc that resets PATH to BASE_BIN, sources
    # that file, then prints STARTUP_KIND, BASHRC_RAN, outputs of python and python3, and
    # VIRTUAL_ENV. Inputs: learner_home, env. Output: list of stdout lines. Raises
    # CalledProcessError/TimeoutExpired (10s) on failure.
    selected = next(
        (
            name
            for name in (".bash_profile", ".bash_login", ".profile")
            if (learner_home / name).exists()
        ),
        None,
    )
    # Model a system-profile PATH reset, then execute the actual emitted user
    # startup. Do not source this machine's /etc/profile or real home files.
    command = 'export PATH="$BASE_BIN:/usr/bin:/bin"; '
    if selected:
        command += f'. "$HOME/{selected}"; '
    command += (
        'printf "%s\\n" "${STARTUP_KIND-none}" "${BASHRC_RAN-no}"; '
        'python; python3; printf "%s\\n" "${VIRTUAL_ENV-none}"'
    )
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", command],
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    return result.stdout.splitlines()


@pytest.mark.parametrize("selected", [None, ".profile", ".bash_login", ".bash_profile"])
def test_emitted_login_preserves_startup_and_restores_runtime_last(tmp_path, selected):
    # [FUNCTION DOC] Parametrized over which user login file exists (none or one of three). Writes
    # login files that source .bashrc (which clobbers PATH) and return early, checks base python is
    # selected first, then extracts the Dockerfile 'RUN sh -c' installer, checks its argv layout,
    # runs its script against the temp home/venv, and asserts afterwards the original startup still
    # runs, the venv python wins, VIRTUAL_ENV is set, and original files are preserved
    # (.bash_profile moved to .repo2rlenv-original-bash-profile).
    task, _ = exported_task(tmp_path)
    learner_home, venv, env = shell_fixture(tmp_path)
    startup_names = [".bash_profile", ".bash_login", ".profile"]
    originals = {}
    if selected:
        (learner_home / ".bashrc").write_text('export BASHRC_RAN=yes\nexport PATH="$BASE_BIN"\n')
        for name in startup_names[startup_names.index(selected) :]:
            text = (
                f"export STARTUP_KIND={shlex.quote(name)}\n"
                '. "$HOME/.bashrc"\n'
                "return 0\n"
                "export STARTUP_KIND=incorrectly-executed-after-return\n"
            )
            (learner_home / name).write_text(text)
            originals[name] = text
    before = startup_result(learner_home, env)
    assert before[2:4] == ["base", "base"]
    command = next(
        line.removeprefix("RUN ")
        for line in (task / "environment/Dockerfile").read_text().splitlines()
        if line.startswith("RUN sh -c ")
    )
    argv = shlex.split(command)
    assert argv[:2] == ["sh", "-c"]
    assert argv[3:6] == ["--", "/home/learner", "/opt/tasksmith-venv"]
    # Execute the exact generated installer on temporary home/runtime paths.
    # The Docker-only chown remains outside this command and is never run here.
    subprocess.run(
        ["/bin/sh", "-c", argv[2], "--", str(learner_home), str(venv)],
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    after = startup_result(learner_home, env)
    assert after == [
        selected or "none",
        "yes" if selected else "no",
        "task-venv",
        "task-venv",
        str(venv),
    ]
    for name, contents in originals.items():
        preserved = ".repo2rlenv-original-bash-profile" if name == ".bash_profile" else name
        assert (learner_home / preserved).read_text() == contents


@pytest.mark.parametrize("reset_before_child", [False, True])
def test_emitted_noninteractive_hook_restores_runtime_without_user_startup(
    tmp_path, reset_before_child
):
    # [FUNCTION DOC] Parametrized on whether PATH is reset before a child bash. Makes every user
    # startup file fail loudly, confirms baseline uses base python, extracts the Dockerfile 'RUN
    # printf' line writing /etc/repo2rlenv-venv.sh, re-runs that printf with the temp venv into a
    # local file, asserts the Dockerfile sets ENV BASH_ENV, and with BASH_ENV pointing to the file
    # the venv python and VIRTUAL_ENV are selected without user startup running.
    task, _ = exported_task(tmp_path)
    learner_home, venv, env = shell_fixture(tmp_path)
    for name in (".bash_profile", ".bash_login", ".profile", ".bashrc"):
        (learner_home / name).write_text("echo unexpected-user-startup; exit 91\n")
    command = 'python; python3; printf "%s\\n" "${VIRTUAL_ENV-none}"'
    if reset_before_child:
        command = 'export PATH="$BASE_BIN:/usr/bin:/bin"; exec /bin/bash -c ' + shlex.quote(command)
    env["PATH"] = env["BASE_BIN"] + ":/usr/bin:/bin"

    def execute():
        # [FUNCTION DOC] HELPER: Runs the closure's command in bash --noprofile --norc with the
        # closure env (10s timeout, check=True) and returns stdout lines.
        return subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-c", command],
            env=env,
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout.splitlines()

    assert execute() == ["base", "base", "none"]
    dockerfile = (task / "environment/Dockerfile").read_text()
    emitted = next(
        line.removeprefix("RUN ")
        for line in dockerfile.splitlines()
        if line.startswith("RUN printf ")
    )
    argv = shlex.split(emitted)
    assert argv[0] == "printf"
    assert argv[2:4] == ["/opt/tasksmith-venv", "/opt/tasksmith-venv"]
    assert argv[4:] == [">", "/etc/repo2rlenv-venv.sh"]
    startup = tmp_path / "noninteractive runtime.sh"
    # Execute the emitted printf, changing only its runtime arguments and output
    # location. No developer startup files, target imports or images are touched.
    with startup.open("w") as stream:
        subprocess.run(
            [*argv[:2], str(venv), str(venv)],
            env=env,
            stdout=stream,
            stderr=subprocess.PIPE,
            check=True,
            timeout=10,
        )
    assert "ENV BASH_ENV=/etc/repo2rlenv-venv.sh\n" in dockerfile
    env["BASH_ENV"] = str(startup)
    assert execute() == ["task-venv", "task-venv", str(venv)]


@pytest.mark.parametrize("venv", [False, True])
def test_runtime_fix_is_only_a_learner_layer_and_public_hint(tmp_path, venv):
    # [FUNCTION DOC] Parametrized on venv. Asserts learner and verifier Dockerfiles share the
    # repository_build prefix, the profile/BASH_ENV layer and instruction hint exist only in the
    # learner image when venv is used, tests/test.sh execs the right interpreter on /tests/grade.py,
    # and environment/solution sources hold defective/reference contents respectively.
    task, options = exported_task(tmp_path, venv=venv)
    learner = (task / "environment/Dockerfile").read_text()
    verifier = (task / "tests/Dockerfile").read_text()
    prefix = repository_build(options)
    assert learner.startswith(prefix) and verifier.startswith(prefix)
    assert (".bash_profile" in learner) is venv
    assert ("ENV BASH_ENV=/etc/repo2rlenv-venv.sh" in learner) is venv
    assert ".bash_profile" not in verifier
    assert "BASH_ENV" not in verifier and "repo2rlenv-venv.sh" not in verifier
    assert ("Use `/opt/tasksmith-venv/bin/python`" in (task / "instruction.md").read_text()) is venv
    expected = "/opt/tasksmith-venv/bin/python" if venv else "/usr/local/bin/python"
    assert (
        task / "tests/test.sh"
    ).read_text() == f"#!/bin/sh\nset -eu\nexec {expected} -I /tests/grade.py\n"
    assert (task / "environment/source/src/core.py").read_text() == "value = 0\n"
    assert (task / "solution/reference/src/core.py").read_text() == "value = 1\n"
