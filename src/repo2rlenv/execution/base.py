# [MODULE DOC] Minimal provider contract for remote workers: a WorkerSpec model, CommandResult, a
# RemoteCommandError that hides command output, the RemoteWorker Protocol and factory functions that
# lazily import the Modal or Daytona implementation. There is no local fallback.
"""Small provider contract for remote commands and files, with no local fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkerSpec(BaseModel):
    # [CLASS DOC] Pydantic model (extra fields forbidden) describing a remote worker to create:
    # provider ("modal" or "daytona", default modal), cpus 1-16 (default 2), memory_mb 1024-65536
    # (default 4096), disk_gb 5-100 (default 10), timeout_sec 60-14400 (default 3600), optional
    # image, optional Modal snapshot_id matching "im-<alnum>", and a required name (lowercase slug,
    # starts with a letter, max 61 chars) that doubles as the provider recovery handle.
    model_config = ConfigDict(extra="forbid")
    provider: Literal["modal", "daytona"] = "modal"
    cpus: int = Field(default=2, ge=1, le=16)
    memory_mb: int = Field(default=4096, ge=1024, le=65536)
    disk_gb: int = Field(default=10, ge=5, le=100)
    timeout_sec: int = Field(default=3600, ge=60, le=14400)
    image: str | None = None
    snapshot_id: str | None = Field(default=None, pattern=r"^im-[A-Za-z0-9]+$")
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,60}$")

    @model_validator(mode="after")
    def snapshot_provider(self):
        # [FUNCTION DOC] After-validator: raises ValueError when snapshot_id is set together with a
        # non-Modal provider or an explicit image. Returns self otherwise.
        if self.snapshot_id and (self.provider != "modal" or self.image):
            raise ValueError(
                "A Modal worker snapshot cannot be combined with another image/provider"
            )
        return self


@dataclass(frozen=True)
class CommandResult:
    # [CLASS DOC] Frozen dataclass for a remote command outcome: returncode (int), stdout (str) and
    # stderr (str, default empty).
    returncode: int
    stdout: str
    stderr: str = ""

    def checked(self, context: str) -> CommandResult:
        # [FUNCTION DOC] Returns self when returncode is 0; otherwise raises
        # RemoteCommandError(context, self).
        # Input: context (str label for the error message).
        if self.returncode:
            raise RemoteCommandError(context, self)
        return self


class RemoteCommandError(RuntimeError):
    # [CLASS DOC] RuntimeError for a failed remote command. Message contains only the context and
    # exit code (never stdout/stderr, which may hold secrets); the full CommandResult is kept on
    # .result.
    def __init__(self, context: str, result: CommandResult):
        # [FUNCTION DOC] Builds the message "<context> failed with exit code <n>" and stores result.
        # Inputs: context (str), result (CommandResult).
        # Logs belong in the evidence store; arbitrary command output may contain
        # credentials or private task context and must not enter exception text.
        super().__init__(f"{context} failed with exit code {result.returncode}")
        self.result = result


class RemoteWorker(Protocol):
    # [CLASS DOC] Structural Protocol every provider worker implements: an id attribute plus exec,
    # upload, download and terminate.
    id: str

    def exec(
        self, argv: list[str], *, timeout: int = 600, env: dict[str, str] | None = None
    ) -> CommandResult:
        # [FUNCTION DOC] Protocol: run argv remotely with a timeout in seconds (default 600) and
        # optional extra environment variables; return a CommandResult.
        ...

    def upload(self, local: Path, remote: str) -> None:
        # [FUNCTION DOC] Protocol: copy a local regular file to the remote path.
        ...

    def download(self, remote: str, local: Path) -> None:
        # [FUNCTION DOC] Protocol: copy a remote file to a local path.
        ...

    def terminate(self) -> None:
        # [FUNCTION DOC] Protocol: destroy the remote worker.
        ...


def connect_worker(provider: str, worker_id: str) -> RemoteWorker:
    # [FUNCTION DOC] Reconnects to an existing worker by id. provider "modal" ->
    # ModalWorker.connect, "daytona" -> DaytonaWorker.connect (lazy imports). Raises ValueError for
    # any other provider.
    if provider == "modal":
        from repo2rlenv.execution.modal import ModalWorker

        return ModalWorker.connect(worker_id)
    if provider == "daytona":
        from repo2rlenv.execution.daytona import DaytonaWorker

        return DaytonaWorker.connect(worker_id)
    raise ValueError(f"Unsupported remote provider {provider!r}")


def create_worker(spec: WorkerSpec) -> RemoteWorker:
    # [FUNCTION DOC] Creates a new worker from a WorkerSpec: ModalWorker.create for provider
    # "modal", otherwise DaytonaWorker.create. Side effects: provisions billable remote compute.
    if spec.provider == "modal":
        from repo2rlenv.execution.modal import ModalWorker

        return ModalWorker.create(spec)
    from repo2rlenv.execution.daytona import DaytonaWorker

    return DaytonaWorker.create(spec)
