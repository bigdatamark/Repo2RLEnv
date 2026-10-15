# [MODULE DOC] Script copied into a disposable replay container (as /tmp/capture.py) by
# worker.capture. It snapshots file metadata under /workspace, /app and /home/user, runs the
# reference bash script with a timeout, snapshots again, and writes /tmp/recording-snapshot.json
# describing the observed filesystem changes plus the reference's combined output. Uses only the
# standard library.
"""Runs inside a disposable container to capture a reference's filesystem changes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def filesystem_state(roots: list[Path], limit: int = 4000) -> dict:
    # [FUNCTION DOC] Collect metadata for regular files under the given roots. Inputs: roots (list
    # of Path; missing or non-directory roots are skipped); limit (int, default 4000, max number of
    # files recorded). Walks without following symlinks, pruning .git, __pycache__, .venv and
    # node_modules, in sorted order; skips symlinks and non-regular files; ignores files that raise
    # OSError. Output: dict mapping absolute path string to {size, mode (permission bits), mtime_ns,
    # prefix_sha256 (SHA-256 of the first 64 KiB)}; returns early once limit entries are recorded.
    # Side effects: file reads.
    result = {}
    for root in roots:
        if not root.is_dir():
            continue
        for directory, children, names in os.walk(root, followlinks=False):
            children[:] = sorted(
                name
                for name in children
                if name not in (".git", "__pycache__", ".venv", "node_modules")
            )
            for name in sorted(names):
                path = Path(directory) / name
                if path.is_symlink() or not path.is_file():
                    continue
                if len(result) >= limit:
                    return result
                try:
                    stat = path.stat()
                    with path.open("rb") as handle:
                        prefix = handle.read(65536)
                    result[str(path)] = {
                        "size": stat.st_size,
                        "mode": stat.st_mode & 0o777,
                        "mtime_ns": stat.st_mtime_ns,
                        "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                    }
                except OSError:
                    continue
    return result


def changes(before: dict, after: dict) -> dict:
    # [FUNCTION DOC] Diff two filesystem_state dicts. Inputs: before, after. Output: dict with lists
    # "created" (paths only in after), "modified" (paths in both whose metadata dicts differ, so an
    # mtime-only change counts), and "deleted" (paths only in before). Pure function.
    return {
        "created": [path for path in after if path not in before],
        "modified": [path for path in after if path in before and after[path] != before[path]],
        "deleted": [path for path in before if path not in after],
    }


def main() -> None:
    # [FUNCTION DOC] Capture entry point. Arguments: argv[1] = path of the reference bash script,
    # argv[2] = timeout in seconds. Takes a before-snapshot, runs "bash <script>" with stdout and
    # stderr merged into /tmp/reference-output.txt; on timeout records timed_out true and returncode
    # 124. Takes an after-snapshot and diffs it. For up to 40 created/modified paths, stores the
    # first 2048 bytes decoded as UTF-8 (or "<binary file>" if a NUL byte appears, "<unreadable or
    # binary>" on OSError or decode error). Writes /tmp/recording-snapshot.json with returncode,
    # timed_out, solution_stdout (first 16000 chars of output), changes, changed_contents,
    # initial_files, final_files and the scan/prefix limits. Output: None. Raises:
    # IndexError/ValueError on missing or bad argv. Side effects: subprocess and files in /tmp; the
    # script's own exit status is 0 even if the reference fails.
    roots = [Path(path) for path in ("/workspace", "/app", "/home/user")]
    before = filesystem_state(roots)
    output = Path("/tmp/reference-output.txt")
    timed_out = False
    with output.open("w") as handle:
        try:
            result = subprocess.run(
                ["bash", sys.argv[1]],
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=int(sys.argv[2]),
            )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            timed_out, returncode = True, 124
    after = filesystem_state(roots)
    delta = changes(before, after)
    contents = {}
    for path in (delta["created"] + delta["modified"])[:40]:
        try:
            with Path(path).open("rb") as handle:
                prefix = handle.read(2048)
            contents[path] = prefix.decode("utf-8") if b"\0" not in prefix else "<binary file>"
        except (OSError, UnicodeDecodeError):
            contents[path] = "<unreadable or binary>"
    with output.open() as handle:
        stdout = handle.read(16000)
    Path("/tmp/recording-snapshot.json").write_text(
        json.dumps(
            {
                "returncode": returncode,
                "timed_out": timed_out,
                "solution_stdout": stdout,
                "changes": delta,
                "changed_contents": contents,
                "initial_files": before,
                "final_files": after,
                "file_scan_limit": 4000,
                "content_prefix_limit": 2048,
            }
        )
    )


if __name__ == "__main__":
    main()
