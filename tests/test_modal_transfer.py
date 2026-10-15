# [MODULE DOC] Tests for execution.modal_transfer.upload_directory: a directory is archived,
# uploaded, sha256-verified remotely, retried with identical bytes on corruption or transport
# error, extracted only after verification, always cleaned up, and recorded in a JSON receipt under
# <accounting dir>/transfers. Uses an in-memory fake environment.
"""Corrupted uploads must never reach artifact extraction or replay a trial."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from repo2rlenv.execution.modal_transfer import upload_directory


class Transfer:
    # [CLASS DOC] HELPER: Fake Modal environment exposing the private SDK hooks upload_directory
    # uses. Holds accounting.directory, an iterator of per-upload faults, counters (uploads,
    # extractions, cleanups) and the uploaded payloads.
    _sandbox = object()
    _default_shell = "bash"

    def __init__(self, directory, faults):
        # [FUNCTION DOC] HELPER: Stores the receipts directory and fault list (True = truncate the
        # upload, an exception = raise, None = clean) and zeroes counters.
        self.accounting = SimpleNamespace(directory=directory)
        self.faults = iter(faults)
        self.uploads = self.extractions = self.cleanups = 0
        self.payloads = []

    async def _sdk_upload_file(self, path, target):
        # [FUNCTION DOC] HELPER: Records the uploaded bytes; applies the next fault by raising it
        # or storing a truncated copy as remote_data, else stores the full bytes.
        self.uploads += 1
        data = path.read_bytes()
        self.payloads.append(data)
        fault = next(self.faults, None)
        if isinstance(fault, BaseException):
            raise fault
        self.remote_data = data if not fault else data[:-8]

    async def _sdk_exec(self, command, **kwargs):
        # [FUNCTION DOC] HELPER: Simulates shell commands: 'sha256sum' returns the digest of
        # remote_data, 'rm -f' counts a cleanup, anything else counts an extraction and asserts the
        # remote data matches the last payload. Returns a result with stdout/stderr/return_code 0.
        if command.startswith("sha256sum"):
            output = hashlib.sha256(self.remote_data).hexdigest() + "  archive.tar.gz\n"
        elif command.startswith("rm -f"):
            self.cleanups += 1
            output = ""
        else:
            self.extractions += 1
            assert self.remote_data == self.payloads[-1]
            output = ""
        return SimpleNamespace(stdout=output, stderr="", return_code=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [True, ConnectionError("transfer response lost")])
async def test_transfer_retries_identical_bytes_before_extraction(tmp_path, fault):
    # [FUNCTION DOC] Parametrized over truncation or ConnectionError on the first attempt: asserts
    # 2 uploads of identical bytes, 1 extraction, 1 cleanup and a 'completed' receipt whose first
    # attempt has an error and second is verified.
    source = tmp_path / "source"
    source.mkdir()
    (source / "answer.py").write_text("answer = 42\n")
    environment = Transfer(tmp_path / "receipts", [fault, None])
    await upload_directory(environment, source, "/workspace/submitted")
    assert environment.uploads == 2 and environment.extractions == 1 and environment.cleanups == 1
    assert environment.payloads[0] == environment.payloads[1]
    receipt = json.loads(next((tmp_path / "receipts/transfers").glob("*.json")).read_text())
    assert receipt["state"] == "completed"
    assert "error" in receipt["attempts"][0] and receipt["attempts"][1]["verified"]


@pytest.mark.asyncio
async def test_repeated_corruption_stops_without_partial_extraction(tmp_path):
    # [FUNCTION DOC] Asserts three corrupted uploads raise RuntimeError mentioning checksum, with
    # no extraction, one cleanup and a 'failed' receipt.
    source = tmp_path / "source"
    source.mkdir()
    (source / "answer.py").write_text("answer = 42\n")
    environment = Transfer(tmp_path / "receipts", [True, True, True])
    with pytest.raises(RuntimeError, match="checksum"):
        await upload_directory(environment, source, "/workspace/submitted")
    assert environment.uploads == 3 and environment.extractions == 0 and environment.cleanups == 1
    receipt = json.loads(next((tmp_path / "receipts/transfers").glob("*.json")).read_text())
    assert receipt["state"] == "failed"


@pytest.mark.asyncio
async def test_transfer_cancellation_is_not_retried(tmp_path):
    # [FUNCTION DOC] Asserts CancelledError on upload propagates after a single attempt, with no
    # extraction and one cleanup.
    source = tmp_path / "source"
    source.mkdir()
    environment = Transfer(tmp_path / "receipts", [asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await upload_directory(environment, source, "/workspace/submitted")
    assert environment.uploads == 1 and environment.extractions == 0 and environment.cleanups == 1
