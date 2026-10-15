# [MODULE DOC] Cross-platform exclusive process lock on an open file handle, used for controller
# receipts and shared worker checkouts. POSIX uses fcntl.flock; Windows uses msvcrt.locking on byte
# 0.
"""Process locks for controller receipts and shared worker checkouts."""

from __future__ import annotations

import errno
import sys
import time
from typing import IO


def lock_file(handle: IO, *, blocking: bool = False) -> None:
    # [FUNCTION DOC] Exclusively lock an open file handle until it is closed; never deletes the
    # path.
    # Inputs: handle (open file object); blocking (default False). POSIX: flock LOCK_EX, adding
    # LOCK_NB when not blocking (raises BlockingIOError/OSError if held). Windows: seeks to 0 and
    # tries a non-blocking 1-byte lock in a loop; on EACCES/EAGAIN/EDEADLK either raises
    # BlockingIOError (non-blocking) or sleeps 0.1 s and retries forever (blocking); other OSErrors
    # propagate. Returns None. Side effect: moves the file position to 0 on Windows.
    """Exclusively lock an open file until it closes, without deleting its path.

    POSIX retains flock semantics, including interoperability with older
    controllers. Windows locks byte zero; the region may extend beyond EOF,
    so an empty lock file needs no write. All callers must keep the same path
    and close their handle on success, failure or cancellation.
    """
    if sys.platform != "win32":
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        return

    import msvcrt

    handle.seek(0)
    while True:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise
            if not blocking:
                raise BlockingIOError(errno.EAGAIN, "Another process owns this lock") from error
            # LK_LOCK gives up after ten retries; match flock's indefinite wait.
            time.sleep(0.1)
