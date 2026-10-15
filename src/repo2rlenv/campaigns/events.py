# [MODULE DOC] Typed progress events and an append-only JSON Lines journal used by owned recipes for
# both the terminal view and durable machine-readable logs. Writes are serialized with a thread
# lock; non-progress events are fsynced.
"""Typed progress events for the terminal view and durable machine-readable logs."""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ProgressEvent(BaseModel):
    # [CLASS DOC] Pydantic model (extra fields forbidden) for one progress record. Fields: time
    # (ISO-8601 UTC timestamp, defaults to now), recipe, stage, state
    # (started|progress|completed|failed|cancelled), task_id (optional), message (default ''),
    # metrics (dict of str to int/float/str).
    model_config = ConfigDict(extra="forbid")
    time: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    recipe: str
    stage: str
    state: Literal["started", "progress", "completed", "failed", "cancelled"]
    task_id: str | None = None
    message: str = ""
    metrics: dict[str, int | float | str] = Field(default_factory=dict)


class EventJournal:
    # [CLASS DOC] Append-only JSONL journal of ProgressEvent records at a file path. Attributes:
    # path; _lock (threading.Lock serializing appends within the process).
    def __init__(self, path: Path):
        # [FUNCTION DOC] Stores path, creates its parent directories and the lock. Side effect:
        # mkdir.
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: ProgressEvent) -> None:
        # [FUNCTION DOC] Appends one event as a JSON line. Opens the file in append+read binary mode
        # under the lock; if the file is non-empty and its last byte is not a newline (a torn
        # previous write) raises ValueError instead of appending. Writes and flushes; for any state
        # other than 'progress' also fsyncs for durability. Returns None. Side effects: file write.
        data = event.model_dump_json() + "\n"
        with self._lock, self.path.open("a+b") as handle:
            if handle.tell():
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    raise ValueError(
                        "Journal has an incomplete tail; reconcile it before appending"
                    )
            handle.write(data.encode("utf-8"))
            handle.flush()
            if event.state != "progress":
                os.fsync(handle.fileno())

    def read(self) -> list[dict[str, Any]]:
        # [FUNCTION DOC] Returns all complete records as validated dicts (model_dump of
        # ProgressEvent), or [] if the file is missing. Any trailing bytes after the last newline (a
        # partial record from a killed writer) are ignored but left on disk. Raises json/pydantic
        # errors if a complete line is corrupt.
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        # A process killed during append can leave a final partial record. Keep
        # that tail on disk for recovery; corruption of a complete line is an error.
        records = raw.split(b"\n")[:-1]
        return [ProgressEvent.model_validate(json.loads(line)).model_dump() for line in records]
