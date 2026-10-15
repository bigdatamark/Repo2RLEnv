# [MODULE DOC] Installs the pinned Node.js agent runtime libraries that ship in
# tasksmith/author/runtimes (package.json + lockfile) into a content-addressed directory under the
# user's home cache, so they live outside the source and wheel trees.
"""Install pinned essential agent libraries outside source and wheel trees."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path


def sources() -> Path:
    # [FUNCTION DOC] HELPER: Returns the Path of the bundled runtime manifest directory
    # (author/runtimes next to this module). No side effects.
    return Path(__file__).parent / "author/runtimes"


def directory() -> Path:
    # [FUNCTION DOC] HELPER: Computes the install directory for the runtimes: SHA-256 over the
    # concatenated bytes of all regular files in sources() (sorted), truncated to 20 hex chars,
    # under ~/.cache/repo2rlenv/tasksmith/runtimes/<key>. Output: Path. Side effects: reads the
    # manifest files.
    key = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(sources().iterdir()) if path.is_file())
    ).hexdigest()[:20]
    return Path.home() / ".cache/repo2rlenv/tasksmith/runtimes" / key


def install() -> Path:
    # [FUNCTION DOC] Creates directory(), copies every regular file from sources() into it, then
    # runs `npm ci --prefix <target> --no-audit --no-fund` (600 s timeout). Output: the target Path.
    # Raises subprocess.CalledProcessError / TimeoutExpired (or FileNotFoundError if npm is
    # missing).
    # Side effects: filesystem writes, subprocess, network via npm. Reruns reinstall every time.
    target = directory()
    target.mkdir(parents=True, exist_ok=True)
    for source in sources().iterdir():
        if source.is_file():
            shutil.copyfile(source, target / source.name)
    subprocess.run(
        ["npm", "ci", "--prefix", str(target), "--no-audit", "--no-fund"], check=True, timeout=600
    )
    return target
