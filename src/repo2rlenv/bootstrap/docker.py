# [MODULE DOC] Thin subprocess wrappers over the `docker` CLI (no docker-py) used by bootstrap: a
# non-raising command runner returning ExecResult, daemon availability check, image pulls
# (optionally streaming progress lines to a callback via a reader thread), and DockerSandbox, a
# long-lived container that receives a copy of the repo (docker cp, not a bind mount, so docker
# commit captures it), executes bash -lc commands with persistent state, and can be
# committed/pushed/removed.
"""Thin wrappers around the `docker` CLI for the bootstrap agent.

We use the CLI directly rather than `docker-py` to keep deps light and to
match how Harbor's own executors invoke docker. All commands run via
subprocess; output is captured and returned to the caller.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExecResult:
    # [CLASS DOC] Slotted dataclass for a finished command: exit_code (124 means timeout), stdout,
    # stderr and duration_sec.
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float

    @property
    def ok(self) -> bool:
        # [FUNCTION DOC] Property: True when exit_code == 0.
        return self.exit_code == 0

    def truncated(self, max_chars: int = 4000) -> str:
        # [FUNCTION DOC] Returns stdout plus, if stderr is non-blank, a "--- stderr ---" separator
        # and stderr. If longer than max_chars (default 4000) keeps only the last max_chars
        # characters, prefixed by a note with the number of elided chars. Output: str for LLM
        # prompts.
        """Combined stdout+stderr, truncated for putting in an LLM prompt."""
        combined = self.stdout
        if self.stderr.strip():
            combined += "\n--- stderr ---\n" + self.stderr
        if len(combined) > max_chars:
            tail = combined[-max_chars:]
            return f"... [{len(combined) - max_chars} chars elided] ...\n{tail}"
        return combined


class DockerError(RuntimeError):
    # [CLASS DOC] RuntimeError raised for Docker failures that callers must handle (daemon down,
    # pull, run, cp, commit failures, or use of a cleaned-up sandbox).
    pass


def _run(args: list[str], *, timeout: int = 600, input_text: str | None = None) -> ExecResult:
    # [FUNCTION DOC] HELPER: Runs args with subprocess.run (captured text output, UTF-8, optional
    # stdin text, timeout default 600s). Never raises on non-zero exit; on timeout returns exit_code
    # 124 with any partial stdout (decoded if bytes) and a "[timeout after Ns]" stderr. Output:
    # ExecResult. May raise OSError if the executable is missing.
    """Run a subprocess and return ExecResult. Never raises on non-zero exit."""
    start = time.monotonic()
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            input=input_text,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ExecResult(
            exit_code=124,
            stdout=(exc.stdout or b"").decode(errors="replace")
            if isinstance(exc.stdout, (bytes, bytearray))
            else (exc.stdout or ""),
            stderr=f"[timeout after {timeout}s]",
            duration_sec=time.monotonic() - start,
        )
    return ExecResult(
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        duration_sec=time.monotonic() - start,
    )


def is_docker_available() -> bool:
    # [FUNCTION DOC] Returns True if `docker version --format {{.Server.Version}}` exits 0 within 15
    # seconds (i.e. the daemon is reachable). Side effects: subprocess.
    """Return True if `docker version` succeeds.

    Uses a generous timeout: `docker version` round-trips to the daemon and on
    macOS/Docker-Desktop a cold call can take 3-4s, which under concurrent load
    blows past a tight 5s budget and yields a spurious "daemon not running".
    """
    return _run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=15).ok


def pull_image(image: str, *, platform: str | None = None, timeout: int = 600) -> ExecResult:
    # [FUNCTION DOC] Runs `docker pull [--platform P] image` with the given timeout (default 600s).
    # Output: ExecResult; never raises on failure.
    args = ["docker", "pull"]
    if platform:
        args.extend(["--platform", platform])
    args.append(image)
    return _run(args, timeout=timeout)


# Matches the last "<layer>: <Status> [...]  12.3MB/45.7MB" line in a chunk
_DOCKER_PROGRESS_RE = re.compile(
    r"(?P<status>Pulling|Downloading|Extracting|Verifying|Waiting|Pull complete|Already exists)"
    r"[^\r\n]*",
)


def pull_image_streaming(
    image: str,
    *,
    platform: str | None = None,
    timeout: int = 600,
    on_progress: Callable[[str], None] | None = None,
) -> ExecResult:
    # [FUNCTION DOC] Pulls an image while reporting progress. Without on_progress it delegates to
    # pull_image. Otherwise starts docker pull with stdout+stderr merged, a daemon thread reads
    # 256-char chunks, splits on CR/LF and passes each new non-empty line (up to 120 chars) to
    # on_progress (callback errors swallowed). On timeout the process is killed and exit 124
    # returned. Output: ExecResult whose combined output is placed in stdout on success or stderr on
    # failure.
    """Pull an image, streaming the most recent progress line to a callback.

    Docker uses carriage returns to overwrite the same line in a TTY; we read
    chunks and split on '\\r' and '\\n' to grab the most-recent meaningful line.
    Falls back to plain pull_image() if no callback is supplied.
    """
    if on_progress is None:
        return pull_image(image, platform=platform, timeout=timeout)

    args = ["docker", "pull"]
    if platform:
        args.extend(["--platform", platform])
    args.append(image)

    start = time.monotonic()
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    out_buf: list[str] = []
    last_line = ""

    def _pump() -> None:
        # [FUNCTION DOC] HELPER: Reader-thread body: drains proc.stdout in 256-char chunks into
        # out_buf and forwards every non-empty line that matches the docker progress regex or
        # differs from the previous line to on_progress, truncated to 120 chars; mutates the
        # enclosing last_line.
        nonlocal last_line
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            out_buf.append(chunk)
            # Split on both \r (overwrites) and \n (new lines)
            parts = re.split(r"[\r\n]+", chunk)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                m = _DOCKER_PROGRESS_RE.search(part)
                if m or part != last_line:
                    last_line = part
                    try:
                        on_progress(part[:120])
                    except Exception:
                        pass

    pump_thread = threading.Thread(target=_pump, daemon=True)
    pump_thread.start()
    try:
        exit_code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        pump_thread.join(timeout=2)
        return ExecResult(
            exit_code=124,
            stdout="".join(out_buf),
            stderr=f"[timeout after {timeout}s]",
            duration_sec=time.monotonic() - start,
        )
    pump_thread.join(timeout=2)
    full = "".join(out_buf)
    return ExecResult(
        exit_code=exit_code,
        stdout=full if exit_code == 0 else "",
        stderr=full if exit_code != 0 else "",
        duration_sec=time.monotonic() - start,
    )


class DockerSandbox:
    # [CLASS DOC] Handle to a long-lived bootstrap container. Create with DockerSandbox.start(), run
    # commands with exec(), inspect with read_file()/list_dir(), snapshot with commit(), publish
    # with push() and remove with cleanup(); usable as a context manager that always cleans up.
    # Attributes: container_id, repo_mount (host path the repo was copied from), platform, _alive
    # flag.
    """A long-lived Docker container the bootstrap agent runs commands in.

    Lifecycle:
        sb = DockerSandbox.start(base_image, repo_dir, platform="linux/amd64")
        sb.exec("pip install -e .")
        sb.exec("pytest --collect-only")
        digest = sb.commit("r2e/bootstrap/django:abc123")
        sb.cleanup()

    Use as a context manager to guarantee cleanup:
        with DockerSandbox.start(...) as sb: ...
    """

    def __init__(self, container_id: str, repo_mount: str, platform: str):
        # [FUNCTION DOC] Stores the container id, host repo path and platform and marks the sandbox
        # alive. Does not touch Docker; use start() to create a container.
        self.container_id = container_id
        self.repo_mount = repo_mount
        self.platform = platform
        self._alive = True

    @classmethod
    def start(
        cls,
        base_image: str,
        repo_dir: Path,
        *,
        platform: str = "linux/amd64",
        workdir: str = "/workspace",
        env: dict[str, str] | None = None,
        timeout: int = 600,
        on_phase=None,
    ) -> DockerSandbox:
        # [FUNCTION DOC] Classmethod that creates a ready sandbox. Steps: raise DockerError if the
        # daemon is unavailable; generate name r2e-bootstrap-<8 hex>; emit on_phase("pull_start");
        # skip the pull when the image already exists locally, else pull (streaming progress as
        # "pull_progress" events) and raise DockerError on failure; emit
        # "pull_done"/"sandbox_start"; `docker run -d --platform P --name N -w workdir [-e K=V...]
        # image sleep infinity`; mkdir workdir and docker cp the contents of repo_dir into it
        # (removing the container and raising DockerError on failure); emit "sandbox_done". Inputs:
        # base_image, repo_dir, platform (default linux/amd64), workdir (default /workspace), env
        # vars, pull timeout (default 600s), on_phase(name, details) callback whose exceptions are
        # swallowed. Output: DockerSandbox. Side effects: network pulls, new container.
        if not is_docker_available():
            raise DockerError("Docker daemon is not running (or `docker` is not on PATH).")
        name = f"r2e-bootstrap-{uuid.uuid4().hex[:8]}"
        # Pull first so the start-up isn't conflated with image-not-found errors.
        if on_phase is not None:
            try:
                on_phase("pull_start", {"detail": base_image})
            except Exception:
                pass

        # Wire progress lines through on_phase so the UI doesn't look frozen
        # during long base-image pulls (golang:1.23 etc. can take 30–60s).
        def _pull_progress(line: str) -> None:
            # [FUNCTION DOC] HELPER: Forwards a pull progress line to on_phase as a "pull_progress"
            # event with {"detail": line}; no-op without on_phase; swallows callback errors.
            if on_phase is None:
                return
            try:
                on_phase("pull_progress", {"detail": line})
            except Exception:
                pass

        # If the image is already in the local Docker cache (e.g. a tag we
        # just committed in a previous bootstrap, or a local/r2e-bootstrap/...
        # tag that was never pushed), skip the pull — `docker pull` would
        # fail trying to contact a registry that doesn't have it.
        already_local = _run(
            ["docker", "image", "inspect", base_image, "--format", "{{.Id}}"],
            timeout=10,
        ).ok
        if not already_local:
            pull = pull_image_streaming(
                base_image,
                platform=platform,
                timeout=timeout,
                on_progress=_pull_progress if on_phase is not None else None,
            )
            if not pull.ok:
                raise DockerError(f"failed to pull {base_image!r}: {pull.stderr.strip()[:400]}")
        if on_phase is not None:
            try:
                on_phase("pull_done", {"detail": base_image})
                on_phase("sandbox_start", {"detail": name})
            except Exception:
                pass
        # IMPORTANT: we COPY the repo into the container instead of bind-mounting.
        # Bind mounts aren't captured by `docker commit`, so a bind-mounted /workspace
        # would leave the committed image with an empty repo dir. We want the repo
        # baked into the image's filesystem.
        args = [
            "docker",
            "run",
            "-d",
            "--platform",
            platform,
            "--name",
            name,
            "-w",
            workdir,
        ]
        for k, v in (env or {}).items():
            args.extend(["-e", f"{k}={v}"])
        args.extend([base_image, "sleep", "infinity"])
        result = _run(args, timeout=60)
        if not result.ok:
            raise DockerError(f"docker run failed: {result.stderr.strip()[:400]}")
        cid = result.stdout.strip()
        if not cid:
            raise DockerError("docker run did not return a container id")
        # Create workdir + copy repo contents into it
        mk = _run(["docker", "exec", cid, "mkdir", "-p", workdir], timeout=15)
        if not mk.ok:
            _run(["docker", "rm", "-f", cid], timeout=10)
            raise DockerError(f"mkdir {workdir} failed: {mk.stderr.strip()[:400]}")
        # `docker cp <src>/. <cid>:<dest>` copies contents (not the dir itself)
        cp = _run(
            ["docker", "cp", f"{repo_dir.resolve()}/.", f"{cid}:{workdir}"],
            timeout=180,
        )
        if not cp.ok:
            _run(["docker", "rm", "-f", cid], timeout=10)
            raise DockerError(f"docker cp into container failed: {cp.stderr.strip()[:400]}")
        if on_phase is not None:
            try:
                on_phase("sandbox_done", {"detail": name})
            except Exception:
                pass
        return cls(container_id=cid, repo_mount=str(repo_dir.resolve()), platform=platform)

    def exec(self, command: str, *, timeout: int = 300) -> ExecResult:
        # [FUNCTION DOC] Runs command via `docker exec <id> bash -lc <command>` (timeout default
        # 300s). State in the container persists between calls. Output: ExecResult (non-zero exit
        # does not raise). Raises DockerError if the sandbox was cleaned up.
        """Run a shell command inside the container. State persists across calls."""
        if not self._alive:
            raise DockerError("sandbox has been cleaned up; cannot exec")
        # Wrap in bash -lc so users can write multi-line / piped commands.
        args = ["docker", "exec", self.container_id, "bash", "-lc", command]
        return _run(args, timeout=timeout)

    def read_file(self, path: str, *, max_bytes: int = 50_000) -> str:
        # [FUNCTION DOC] Reads up to max_bytes (default 50000) of a container path with head -c
        # (path shell-quoted). Output: file text. Raises DockerError if the command fails (e.g.
        # missing file) or the sandbox is gone.
        """Read a file path inside the container, capped at max_bytes."""
        r = self.exec(f"head -c {max_bytes} {shlex.quote(path)}", timeout=30)
        if not r.ok:
            raise DockerError(f"read_file({path!r}) failed: {r.stderr.strip()[:200]}")
        return r.stdout

    def list_dir(self, path: str = ".") -> list[str]:
        # [FUNCTION DOC] Lists up to 200 entries of a container directory (ls -1A, path quoted,
        # default "."). Output: list of non-blank names; empty list on command failure. Note:
        # because the ls output is piped to head, the pipeline exit status is head's, so a missing
        # directory usually yields an empty list rather than a failure.
        r = self.exec(f"ls -1A {shlex.quote(path)} | head -200", timeout=10)
        if not r.ok:
            return []
        return [line for line in r.stdout.splitlines() if line.strip()]

    def commit(self, tag: str, *, message: str = "repo2rlenv bootstrap", timeout: int = 600) -> str:
        # [FUNCTION DOC] Commits the container to image tag with a commit message (default
        # "repo2rlenv bootstrap", timeout default 600s). Then inspects the tag: returns its first
        # RepoDigest if any, else the local image Id, else the tag itself if inspection fails or
        # JSON is bad. Raises DockerError if the sandbox is gone or docker commit fails.
        """Commit the container to an image. Returns the image's content digest.

        Default 600s timeout because large full-stack-app images (Django + OCR +
        ML deps; multi-GB final layer) can take several minutes to write out.
        """
        if not self._alive:
            raise DockerError("sandbox has been cleaned up; cannot commit")
        r = _run(["docker", "commit", "-m", message, self.container_id, tag], timeout=timeout)
        if not r.ok:
            raise DockerError(f"docker commit failed: {r.stderr.strip()[:400]}")
        # Resolve the image's RepoDigests (only present after a push) OR the local Id
        inspect = _run(["docker", "image", "inspect", tag, "--format", "{{json .}}"], timeout=10)
        if inspect.ok:
            try:
                data = json.loads(inspect.stdout)
                if isinstance(data, list):
                    data = data[0] if data else {}
                digests = data.get("RepoDigests") or []
                if digests:
                    return digests[0]
                return data.get("Id", tag)
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
        return tag

    def push(self, tag: str, *, timeout: int = 900) -> bool:
        # [FUNCTION DOC] Runs `docker push tag` (timeout default 900s). Output: True on success;
        # False on failure, after logging a warning with the stderr excerpt. Never raises.
        r = _run(["docker", "push", tag], timeout=timeout)
        if not r.ok:
            logger.warning("docker push failed for %s: %s", tag, r.stderr.strip()[:400])
            return False
        return True

    def cleanup(self) -> None:
        # [FUNCTION DOC] Force-removes the container (docker rm -f, 30s timeout) and marks the
        # sandbox dead; idempotent. Errors from docker are ignored.
        if not self._alive:
            return
        # `rm -f` stops + removes in one go
        _run(["docker", "rm", "-f", self.container_id], timeout=30)
        self._alive = False

    def __enter__(self) -> DockerSandbox:
        # [FUNCTION DOC] Context manager entry; returns self.
        return self

    def __exit__(self, *exc) -> None:
        # [FUNCTION DOC] Context manager exit; calls cleanup() and does not suppress exceptions.
        self.cleanup()
