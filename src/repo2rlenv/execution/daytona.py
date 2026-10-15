# [MODULE DOC] Daytona implementation of the RemoteWorker contract. Creates an ephemeral Ubuntu
# 24.04 sandbox with Docker, Python venv at /opt/repo2rlenv and uv; Docker daemon readiness is
# probed separately by the caller (lifecycle.prepare_docker).
"""Daytona generation worker with explicit Docker readiness probing by the caller."""

from __future__ import annotations

import shlex
import tempfile
from pathlib import Path

from repo2rlenv.execution.base import CommandResult, WorkerSpec
from repo2rlenv.execution.read_retry import retry_read


class DaytonaWorker:
    # [CLASS DOC] RemoteWorker backed by a Daytona sandbox. Holds the Daytona client, the sandbox
    # object and id (= sandbox.id).
    def __init__(self, client, sandbox):
        # [FUNCTION DOC] Stores client and sandbox and sets id from sandbox.id.
        self.client = client
        self.sandbox = sandbox
        self.id = sandbox.id

    @classmethod
    def create(cls, spec: WorkerSpec) -> DaytonaWorker:
        # [FUNCTION DOC] Classmethod: creates a sandbox from spec. Uses spec.image or a built image
        # (ubuntu:24.04 + docker.io/buildx/compose, python3-venv, git, curl, /work and /evidence
        # dirs, venv at /opt/repo2rlenv with uv==0.10.9, PATH set). Runs as root with resources
        # cpu=spec.cpus, memory in GiB rounded up, disk=spec.disk_gb, label repo2rlenv-
        # worker=<name>, auto-stop after timeout_sec/60 minutes (min 1), ephemeral, 180s create
        # timeout. Reads Daytona credentials from the environment (daytona SDK). Returns a
        # DaytonaWorker.
        from daytona import CreateSandboxFromImageParams, Daytona, Image, Resources

        client = Daytona()
        image = spec.image or Image.base("ubuntu:24.04").run_commands(
            "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "docker.io docker-buildx docker-compose-v2 python3 python3-venv git curl ca-certificates",
            "mkdir -p /work /evidence",
            "python3 -m venv /opt/repo2rlenv && /opt/repo2rlenv/bin/python -m pip install uv==0.10.9",
        ).env({"PATH": "/opt/repo2rlenv/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"})
        sandbox = client.create(
            CreateSandboxFromImageParams(
                image=image,
                name=spec.name,
                os_user="root",
                resources=Resources(
                    cpu=spec.cpus, memory=(spec.memory_mb + 1023) // 1024, disk=spec.disk_gb
                ),
                labels={"repo2rlenv-worker": spec.name},
                auto_stop_interval=max(1, spec.timeout_sec // 60),
                auto_delete_interval=0,
                ephemeral=True,
            ),
            timeout=180,
        )
        return cls(client, sandbox)

    @classmethod
    def connect(cls, worker_id: str) -> DaytonaWorker:
        # [FUNCTION DOC] Classmethod: fetches an existing sandbox by worker_id (with retry_read on
        # transient errors) and wraps it.
        from daytona import Daytona

        client = Daytona()
        return cls(client, retry_read(lambda: client.get(worker_id)))

    def exec(
        self, argv: list[str], *, timeout: int = 600, env: dict[str, str] | None = None
    ) -> CommandResult:
        # [FUNCTION DOC] Runs argv as a shell-joined command via sandbox.process.exec with env and
        # timeout. Returns CommandResult(exit_code, result or ""); stderr is always empty because
        # Daytona returns a combined output. Raises ValueError for empty argv or non-positive
        # timeout.
        if not argv or timeout <= 0:
            raise ValueError("Remote execution needs an argv and positive timeout")
        result = self.sandbox.process.exec(shlex.join(argv), env=env, timeout=timeout)
        return CommandResult(result.exit_code, result.result or "")

    def upload(self, local: Path, remote: str) -> None:
        # [FUNCTION DOC] Uploads a local regular (non-symlink) file to remote. Raises ValueError
        # otherwise.
        if not local.is_file() or local.is_symlink():
            raise ValueError("Upload requires a regular local file")
        self.sandbox.fs.upload_file(str(local), remote)

    def download(self, remote: str, local: Path) -> None:
        # [FUNCTION DOC] Downloads remote into a temporary file beside local (retrying transient
        # errors), then atomically replaces local. Creates local's parent directory.
        local.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local.parent) as temporary:
            partial = Path(temporary) / "download"
            retry_read(lambda: self.sandbox.fs.download_file(remote, str(partial)))
            partial.replace(local)

    def terminate(self) -> None:
        # [FUNCTION DOC] Deletes the sandbox via the client (180s timeout).
        self.client.delete(self.sandbox, timeout=180)
