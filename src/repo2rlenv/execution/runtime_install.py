# [MODULE DOC] Standalone script run on the remote POSIX worker (via python -c) that installs the
# uploaded repo2rlenv wheel into a content-addressed venv, serialized across controllers with a file
# lock.
"""Serialize runtime installation on the remote Linux worker, across controllers."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


def install(directory: Path, uploaded: Path):
    # [FUNCTION DOC] Takes an exclusive flock on directory/.install.lock. If directory/installed
    # exists returns. Otherwise checks the wheel's SHA-256 equals directory.name (ValueError), moves
    # it into directory, creates directory/venv with uv from /opt/repo2rlenv/bin/python if missing,
    # installs the wheel with extras mutation,harbor (reinstalling repo2rlenv), and writes the
    # installed marker. Always deletes the uploaded file and removes its (assumed empty) upload
    # directory. Raises RuntimeError on Windows, CalledProcessError on uv failures.
    if sys.platform == "win32":
        raise RuntimeError("Runtime installation requires a remote POSIX worker")
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    try:
        with (directory / ".install.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (directory / "installed").is_file():
                return
            if hashlib.sha256(uploaded.read_bytes()).hexdigest() != directory.name:
                raise ValueError("Uploaded runtime differs from its content address")
            wheel = directory / uploaded.name
            os.replace(uploaded, wheel)
            python = directory / "venv/bin/python"
            if not python.exists():
                subprocess.run(
                    [
                        "uv",
                        "venv",
                        "--python",
                        "/opt/repo2rlenv/bin/python",
                        str(directory / "venv"),
                    ],
                    check=True,
                )
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--reinstall-package",
                    "repo2rlenv",
                    str(wheel) + "[mutation,harbor]",
                ],
                check=True,
            )
            (directory / "installed").write_text(directory.name + "\n")
    finally:
        uploaded.unlink(missing_ok=True)
        uploaded.parent.rmdir()


if __name__ == "__main__":
    install(Path(sys.argv[1]), Path(sys.argv[2]))
