# [MODULE DOC] Modal VM-runtime implementation of the RemoteWorker contract. The sandbox runs
# dockerd as its entrypoint; Docker and task processes execute remotely under the app repo2rlenv-
# owned-generation.
"""Modal VM generation worker. Docker and task processes execute remotely."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from repo2rlenv.execution.base import CommandResult, WorkerSpec

APP_NAME = "repo2rlenv-owned-generation"


class ModalWorker:
    # [CLASS DOC] RemoteWorker backed by a modal.Sandbox; id is the sandbox object_id.
    def __init__(self, sandbox):
        # [FUNCTION DOC] Stores the sandbox and sets id from sandbox.object_id.
        self.sandbox = sandbox
        self.id = sandbox.object_id

    @classmethod
    def create(cls, spec: WorkerSpec) -> ModalWorker:
        # [FUNCTION DOC] Classmethod: builds the image from spec.snapshot_id when set, otherwise
        # from spec.image (default ubuntu:24.04, Python 3.12 added) with Docker packages, git, curl,
        # uv==0.10.9, /work, /evidence and a venv at /opt/repo2rlenv on PATH. Creates a sandbox
        # running dockerd (overlay2) with the spec's name, CPU, memory and timeout and the
        # vm_runtime experimental option, in the app looked up or created by APP_NAME. Uses Modal
        # credentials from the environment. Returns a ModalWorker.
        import modal

        image = modal.Image.from_registry(spec.image or "ubuntu:24.04", add_python="3.12")
        image = (
            modal.Image.from_id(spec.snapshot_id)
            if spec.snapshot_id
            else (
                image.env({"DEBIAN_FRONTEND": "noninteractive"})
                .apt_install(
                    "docker.io",
                    "docker-buildx",
                    "docker-compose-v2",
                    "git",
                    "curl",
                    "ca-certificates",
                    "python3-venv",
                )
                .pip_install("uv==0.10.9")
                .run_commands("mkdir -p /work /evidence", "python -m venv /opt/repo2rlenv")
                .env({"PATH": "/opt/repo2rlenv/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"})
            )
        )
        sandbox = modal.Sandbox.create(
            "/usr/bin/dockerd",
            "--storage-driver=overlay2",
            app=modal.App.lookup(APP_NAME, create_if_missing=True),
            name=spec.name,
            image=image,
            cpu=(spec.cpus, spec.cpus),
            memory=spec.memory_mb,
            timeout=spec.timeout_sec,
            experimental_options={"vm_runtime": True},
        )
        return cls(sandbox)

    @classmethod
    def connect(cls, worker_id: str) -> ModalWorker:
        # [FUNCTION DOC] Classmethod: wraps the existing sandbox modal.Sandbox.from_id(worker_id).
        import modal

        return cls(modal.Sandbox.from_id(worker_id))

    def exec(
        self, argv: list[str], *, timeout: int = 600, env: dict[str, str] | None = None
    ) -> CommandResult:
        # [FUNCTION DOC] Runs argv in the sandbox with timeout; env (if given) is passed as a Modal
        # secret. Drains stdout and stderr concurrently in two threads to avoid pipe deadlock,
        # waits, and returns CommandResult(returncode, stdout, stderr). Raises ValueError for empty
        # argv or non-positive timeout.
        import modal

        if not argv or timeout <= 0:
            raise ValueError("Remote execution needs an argv and positive timeout")
        process = self.sandbox.exec(
            *argv,
            timeout=timeout,
            secrets=[modal.Secret.from_dict(env)] if env else [],
        )
        # Drain both streams concurrently so a full stderr pipe cannot deadlock a
        # controller that is still reading stdout.
        with ThreadPoolExecutor(max_workers=2) as pool:
            stdout = pool.submit(process.stdout.read)
            stderr = pool.submit(process.stderr.read)
            process.wait()
            return CommandResult(process.returncode, stdout.result(), stderr.result())

    def upload(self, local: Path, remote: str) -> None:
        # [FUNCTION DOC] Copies a local regular (non-symlink) file into the sandbox. Raises
        # ValueError otherwise.
        if not local.is_file() or local.is_symlink():
            raise ValueError("Upload requires a regular local file")
        self.sandbox.filesystem.copy_from_local(str(local), remote)

    def download(self, remote: str, local: Path) -> None:
        # [FUNCTION DOC] Copies a remote sandbox file to local, creating the parent directory.
        local.parent.mkdir(parents=True, exist_ok=True)
        self.sandbox.filesystem.copy_to_local(remote, str(local))

    def terminate(self) -> None:
        # [FUNCTION DOC] Terminates the sandbox and waits for completion.
        self.sandbox.terminate(wait=True)
