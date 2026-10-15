# [MODULE DOC] Bounded transfer of the owned repo2rlenv runtime wheel to a remote worker and safe
# unpacking of downloaded evidence archives on the controller (never executing or following links).
"""Bounded transfer of owned runtime code and remote generation evidence."""

from __future__ import annotations

import hashlib
import tarfile
import tempfile
import uuid
import zipfile
from importlib.resources import files
from pathlib import Path, PurePosixPath

from repo2rlenv.execution.base import RemoteWorker


def check_runtime_wheel(wheel: Path) -> str:
    # [FUNCTION DOC] Verifies a built wheel matches the installed repo2rlenv package byte-for-byte:
    # every file under the package root (excluding __pycache__ and .pyc) must exist in the wheel zip
    # at "repo2rlenv/<rel>" with identical content. Input: wheel (Path). Output: SHA-256 hex digest
    # of the wheel file. Raises ValueError if a file is missing or differs ("run uv build"). Reads
    # files only.
    """Refuse a stale wheel: every installed owned source/asset must match it."""
    root = Path(str(files("repo2rlenv")))
    with zipfile.ZipFile(wheel) as archive:
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            name = "repo2rlenv/" + path.relative_to(root).as_posix()
            try:
                content = archive.read(name)
            except KeyError as exc:
                raise ValueError(f"Worker wheel is missing {name}; run uv build") from exc
            if content != path.read_bytes():
                raise ValueError(f"Worker wheel is stale at {name}; run uv build")
    return hashlib.sha256(wheel.read_bytes()).hexdigest()


def install_runtime(worker: RemoteWorker, wheel: Path, log_dir: Path) -> str:
    # [FUNCTION DOC] Installs the owned runtime wheel on a worker under the content-addressed
    # directory /work/runtime/<digest>. Validates the wheel (check_runtime_wheel), creates the
    # directory, and returns early if an "installed" marker exists. Otherwise uploads the wheel into
    # a unique upload-<uuid> subdirectory and runs the runtime_install.py source with
    # /opt/repo2rlenv/bin/python -c (args: runtime dir, uploaded wheel path, 600s timeout). Writes
    # install.stdout/install.stderr into log_dir (created). Inputs: worker (RemoteWorker), wheel
    # (Path), log_dir (Path). Output: digest str. Raises RemoteCommandError on failed remote
    # commands; ValueError from wheel check.
    digest = check_runtime_wheel(wheel)
    remote_dir = "/work/runtime/" + digest
    worker.exec(["mkdir", "-p", remote_dir], timeout=30).checked("Runtime directory")
    ready = worker.exec(["test", "-f", remote_dir + "/installed"], timeout=30)
    if ready.returncode == 0:
        return digest
    upload_dir = remote_dir + "/upload-" + uuid.uuid4().hex
    worker.exec(["mkdir", "-p", upload_dir], timeout=30).checked("Runtime upload directory")
    remote = upload_dir + "/" + wheel.name
    worker.upload(wheel, remote)
    installer = files("repo2rlenv.execution").joinpath("runtime_install.py").read_text()
    result = worker.exec(
        [
            "/opt/repo2rlenv/bin/python",
            "-c",
            installer,
            remote_dir,
            remote,
        ],
        timeout=600,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "install.stdout").write_text(result.stdout)
    (log_dir / "install.stderr").write_text(result.stderr)
    result.checked("Install the owned runtime")
    return digest


def runtime_python(digest: str) -> str:
    # [FUNCTION DOC] Returns the remote interpreter path /work/runtime/<digest>/venv/bin/python.
    # Input: digest (must be 64 lowercase hex chars). Raises ValueError otherwise. No side effects.
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Runtime identity must be a SHA-256 digest")
    return "/work/runtime/" + digest + "/venv/bin/python"


def unpack_evidence(
    archive: Path, destination: Path, *, root_name: str, max_bytes: int = 512 * 1024 * 1024
) -> Path:
    # [FUNCTION DOC] Safely extracts a downloaded .tar.gz evidence archive to destination/root_name
    # and returns that path. Rejects (ValueError) archives with more than 50,000 members or total
    # member size above max_bytes (default 512 MiB), absolute paths, "..", empty names, entries not
    # under root_name, any non-file/non-directory member (links, devices) and duplicate paths.
    # Raises FileExistsError if the target already exists (or is a symlink). Extracts with the
    # "data" filter into a temporary directory inside destination, then renames into place (atomic
    # on the same filesystem). Creates destination.
    """Never execute downloaded code or follow archive links on the controller."""
    target = destination / root_name
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        if len(members) > 50_000 or sum(member.size for member in members) > max_bytes:
            raise ValueError("Remote evidence exceeds the archive limits")
        names = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != root_name
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("Remote archive has unsafe paths, links or special files")
            if str(path) in names:
                raise ValueError("Remote archive has duplicate paths")
            names.add(str(path))
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".evidence-", dir=destination) as temporary:
            stream.extractall(temporary, members=members, filter="data")
            (Path(temporary) / root_name).rename(target)
    return target
