# [MODULE DOC] Fallback for reversing a PR source patch whose target files use CRLF line endings
# while the fetched diff was newline-normalized to LF. It rebuilds exact CRLF hunks, proves the
# result with git blob IDs from the frozen diff, and only then swaps the reconstructed tree into
# place.
"""Recover newline-normalized PR patches without changing pinned source bytes."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


def _blob(content: bytes) -> str:
    # [FUNCTION DOC] HELPER: Returns the git blob SHA-1 hex id for the given bytes (sha1 over 'blob
    # <len>\0' + content).
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def _crlf_text(content: bytes) -> bool:
    # [FUNCTION DOC] HELPER: Returns True when content is valid UTF-8, contains at least one CRLF,
    # and after removing all CRLF pairs contains no stray CR, LF or NUL bytes (i.e. purely CRLF
    # text). Else False.
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    remaining = content.replace(b"\r\n", b"")
    return b"\r\n" in content and not any(value in remaining for value in (b"\r", b"\n", b"\0"))


def reverse_crlf_patch(root: Path, patch: Path) -> bool:
    # [FUNCTION DOC] Attempts to reverse-apply patch to root after an ordinary `git apply --reverse`
    # failed.
    # Inputs: root (directory to modify, replaced in place on success); patch (Path to the LF diff).
    # Behavior: splits the diff into per-file blocks with identical a/b paths; rejects absolute or
    # '..' paths (ValueError). For each existing non-symlink file that is pure CRLF text, requires
    # an 'index pre..post' line whose post id prefixes the file's current blob id (else ValueError),
    # then converts LF-terminated context/added/removed hunk lines to CRLF, except a line followed
    # by the 'No newline at end of file' marker. Returns False if no block changed. Otherwise copies
    # root into a temporary directory beside patch, runs `git -c core.autocrlf=false apply
    # --reverse` there (120 s), raises ValueError if it fails or any file's resulting blob does not
    # match the preimage id (an all-zero preimage means the file must be absent). On success writes
    # source-crlf.diff and source-crlf-fallback.json (original/applied patch SHA-256, operation,
    # files) next to patch, then swaps the staged tree into root (renaming root to a backup and
    # restoring it if the swap fails).
    # Output: True on success. Side effects: subprocess git, filesystem writes, directory renames.
    """Try exact CRLF hunk bytes after ordinary reverse application failed.

    Git blob IDs in the frozen diff must identify both the original CRLF head
    and the reconstructed preimage. Only a completely checked temporary copy
    replaces ``root``; the caller's pinned checkout and original patch stay intact.
    Mixed-ending/binary files and missing blob IDs are never normalized.
    """
    original = patch.read_bytes()
    blocks = re.split(rb"(?=^diff --git )", original, flags=re.MULTILINE)
    bindings = []
    for index, block in enumerate(blocks):
        header = re.match(rb"diff --git a/(.+) b/\1\n", block)
        if header is None:
            continue
        name = header[1].decode("utf-8")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("CRLF patch paths must be repository-relative")
        path = root / name
        if not path.is_file() or path.is_symlink():
            continue
        content = path.read_bytes()
        if not _crlf_text(content):
            continue
        ids = re.search(
            rb"^index ([0-9a-f]{7,40})\.\.([0-9a-f]{7,40})(?: [0-7]{6})?\n", block, re.MULTILINE
        )
        if ids is None or not _blob(content).startswith(ids[2].decode()):
            raise ValueError(f"CRLF fallback requires the frozen postimage blob: {name}")
        lines = block.splitlines(keepends=True)
        in_hunk = False
        changed = False
        for number, line in enumerate(lines):
            if line.startswith(b"@@ "):
                in_hunk = True
            elif in_hunk and line[:1] in (b" ", b"+", b"-"):
                # Git removes the patch's final LF for this marker. Adding CR
                # would change an unterminated source line into a different blob.
                no_newline = number + 1 < len(lines) and lines[number + 1].startswith(
                    b"\\ No newline at end of file"
                )
                if line.endswith(b"\n") and not line.endswith(b"\r\n") and not no_newline:
                    lines[number] = line[:-1] + b"\r\n"
                    changed = True
        if changed:
            blocks[index] = b"".join(lines)
            bindings.append(
                {"path": name, "preimage": ids[1].decode(), "postimage": _blob(content)}
            )
    if not bindings:
        return False
    adapted = b"".join(blocks)
    with tempfile.TemporaryDirectory(prefix=".crlf-reverse-", dir=patch.parent) as directory:
        directory = Path(directory)
        staged = directory / "source"
        shutil.copytree(root, staged)
        adapted_path = directory / "source.diff"
        adapted_path.write_bytes(adapted)
        result = subprocess.run(
            # core.autocrlf=false: `staged` isn't a git repository, so `git apply`
            # would otherwise fall back to the caller's global config and risk
            # re-normalizing the exact CRLF bytes this fallback just reconstructed.
            ["git", "-c", "core.autocrlf=false", "apply", "--reverse", str(adapted_path.resolve())],
            cwd=staged,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            check=False,
        )
        if result.returncode:
            raise ValueError("CRLF patch still does not apply exactly: " + result.stderr[-6000:])
        for binding in bindings:
            path = staged / binding["path"]
            expected = binding["preimage"]
            matches = (
                not path.exists()
                if set(expected) == {"0"}
                else (path.is_file() and _blob(path.read_bytes()).startswith(expected))
            )
            if not matches:
                raise ValueError(
                    f"CRLF fallback differs from the frozen preimage blob: {binding['path']}"
                )
        (patch.parent / "source-crlf.diff").write_bytes(adapted)
        (patch.parent / "source-crlf-fallback.json").write_text(
            json.dumps(
                {
                    "original_patch_sha256": hashlib.sha256(original).hexdigest(),
                    "applied_patch_sha256": hashlib.sha256(adapted).hexdigest(),
                    "operation": "reverse",
                    "files": bindings,
                },
                indent=2,
            )
            + "\n"
        )
        backup = directory / "original"
        root.rename(backup)
        try:
            staged.rename(root)
        except BaseException:
            backup.rename(root)
            raise
    return True
