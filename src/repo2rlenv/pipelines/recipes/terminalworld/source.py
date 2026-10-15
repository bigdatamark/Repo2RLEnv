# [MODULE DOC] Acquire public asciinema text recordings and load the native recording-directory
# layout (<dir>/<id>/{recording.txt, info.json}). Fetching honors robots.txt, caps response sizes
# (_LIMIT = 160000 bytes for transcripts), uses a fixed research _USER_AGENT, sleeps 1 s per
# recording, and keeps per-ID outcomes in retrieval.json plus an acquisition.json state receipt for
# restartability. Loading screens transcripts with privacy.screen. Runnable as a CLI (main).
"""Acquire public text recordings and load the native recording-directory layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import Path
from urllib.robotparser import RobotFileParser

import httpx

from repo2rlenv.execution.lifecycle import now, save_record
from repo2rlenv.pipelines.recipes.terminalworld.privacy import screen
from repo2rlenv.ui import console

_USER_AGENT = "Repo2RLEnv-RecordingResearch/1.0"
_LIMIT = 160000


class MetadataParser(HTMLParser):
    # [CLASS DOC] HTMLParser subclass that extracts og:title / og:description meta content into
    # metadata (keys "title"/"description") and collects text inside the first div whose class list
    # contains "description" (tracking nested div depth) into the description list.
    def __init__(self):
        # [FUNCTION DOC] Initialize the parser with empty metadata dict, description list and
        # nesting depth 0.
        super().__init__()
        self.metadata = {}
        self.description = []
        self._description_depth = 0

    def handle_starttag(self, tag, attrs):
        # [FUNCTION DOC] Record og:title/og:description meta content; for div tags increment the
        # depth when already inside the description div, or enter it (depth 1) when the class list
        # contains "description". Output: None.
        attributes = dict(attrs)
        if tag == "meta" and attributes.get("property") in ("og:title", "og:description"):
            self.metadata[attributes["property"].removeprefix("og:")] = attributes.get(
                "content", ""
            )
        if tag == "div":
            if self._description_depth:
                self._description_depth += 1
            elif "description" in attributes.get("class", "").split():
                self._description_depth = 1

    def handle_endtag(self, tag):
        # [FUNCTION DOC] Decrement the description depth on a closing div while inside the
        # description. Output: None.
        if tag == "div" and self._description_depth:
            self._description_depth -= 1

    def handle_data(self, data):
        # [FUNCTION DOC] Append text data to description while inside the description div. Output:
        # None.
        if self._description_depth:
            self.description.append(data)


def _download(client, url: str, limit: int) -> str:
    # [FUNCTION DOC] HELPER: Stream a GET request and return the body as text decoded as UTF-8 with
    # replacement. Inputs: client (httpx.Client), url (str), limit (int max bytes). Raises:
    # httpx.HTTPStatusError on non-2xx, other httpx.HTTPError on transport errors, ValueError when
    # the body exceeds limit. Also used by discovery.py.
    with client.stream("GET", url) as response:
        response.raise_for_status()
        chunks = []
        length = 0
        for chunk in response.iter_bytes():
            length += len(chunk)
            if length > limit:
                raise ValueError("Recording response exceeds the configured text limit")
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


@contextmanager
def _acquisition(destination: Path, ids: list[str]):
    # [FUNCTION DOC] HELPER: Context manager recording acquisition state in
    # destination/acquisition.json. Writes a "running" record (started_at, SHA-256 of the requested
    # ID list, count of unique IDs) on entry; on any exception (including KeyboardInterrupt)
    # rewrites it as "interrupted" with the exception type and re-raises; on success writes
    # "completed" with finished_at and the SHA-256 of retrieval.json. Inputs: destination (Path),
    # ids (list of str). Yields None. Raises: OSError if retrieval.json is missing at completion.
    """Distinguish a complete input shard from a cache still being filled."""
    path = destination / "acquisition.json"
    record = {
        "state": "running",
        "started_at": now(),
        "requested_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "requested_count": len(set(ids)),
    }
    save_record(path, record)
    try:
        yield
    except BaseException as exc:
        save_record(path, {**record, "state": "interrupted", "exception_type": type(exc).__name__})
        raise
    else:
        save_record(
            path,
            {
                **record,
                "state": "completed",
                "finished_at": now(),
                "retrieval_sha256": hashlib.sha256(
                    (destination / "retrieval.json").read_bytes()
                ).hexdigest(),
            },
        )


def fetch_recordings(ids: list[str], destination: Path) -> dict:
    # [FUNCTION DOC] Fetch page metadata and the .txt transcript for each recording ID (never
    # solutions). Inputs: ids (1-2000 numeric strings; ValueError otherwise), destination (Path).
    # Loads existing retrieval.json results; within _acquisition and an httpx client, downloads
    # robots.txt, then for each unique ID: cached results are skipped (ValueError if a recorded
    # download's recording.txt is missing); otherwise checks robots for the page and .txt, parses
    # the page (1 MiB cap) with MetadataParser, downloads the text (_LIMIT cap), rejects text
    # shorter than 100 non-blank chars or starting with an HTML doctype, writes <id>/recording.txt
    # and <id>/info.json (og metadata, id, url, description, transcript SHA-256). HTTP/ValueError
    # failures are recorded as error_type (plus http_status). Saves retrieval.json after each ID and
    # sleeps 1 s. Output: dict ID -> result {url, downloaded, transcript_sha256 or error info}. Side
    # effects: network, filesystem, sleeping.
    """Fetch metadata and .txt only; retain per-input outcomes for restart."""
    if (
        not ids
        or len(ids) > 2000
        or any(re.fullmatch(r"[0-9]{1,12}", value) is None for value in ids)
    ):
        raise ValueError("Supply one to 2000 numeric public recording IDs")
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / "retrieval.json"
    results = json.loads(receipt.read_text()) if receipt.exists() else {}
    with (
        _acquisition(destination, ids),
        httpx.Client(
            timeout=30, follow_redirects=True, headers={"User-Agent": _USER_AGENT}
        ) as client,
    ):
        rules = _download(client, "https://asciinema.org/robots.txt", 64000)
        (destination / "robots.txt").write_text(rules)
        robots = RobotFileParser()
        robots.parse(rules.splitlines())
        for rec_id in dict.fromkeys(ids):
            folder = destination / rec_id
            if rec_id in results:
                if results[rec_id].get("downloaded") and not (folder / "recording.txt").is_file():
                    raise ValueError("Retrieved recording disappeared; restore its recorded input")
                continue
            url = f"https://asciinema.org/a/{rec_id}"
            result = {"url": url, "downloaded": False}
            try:
                if not all(robots.can_fetch(_USER_AGENT, path) for path in (url, url + ".txt")):
                    raise ValueError("Recording page or text disallowed by robots.txt")
                metadata = MetadataParser()
                metadata.feed(_download(client, url, 1024 * 1024))
                text = _download(client, url + ".txt", _LIMIT)
                if len(text.strip()) < 100 or text.lstrip().lower().startswith("<!doctype html"):
                    raise ValueError("No usable plain-text recording")
                folder.mkdir(exist_ok=True)
                (folder / "recording.txt").write_text(text)
                save_record(
                    folder / "info.json",
                    {
                        **metadata.metadata,
                        "id": rec_id,
                        "url": url,
                        "description": " ".join(metadata.description).strip()
                        or metadata.metadata.get("description", ""),
                        "transcript_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    },
                )
                result.update(
                    downloaded=True, transcript_sha256=hashlib.sha256(text.encode()).hexdigest()
                )
            except (httpx.HTTPError, ValueError) as exc:
                result["error_type"] = type(exc).__name__
                if isinstance(exc, httpx.HTTPStatusError):
                    result["http_status"] = exc.response.status_code
            results[rec_id] = result
            save_record(receipt, results)
            time.sleep(1)
    return results


def load_recordings(path: Path) -> list[dict]:
    # [FUNCTION DOC] Load recording seeds from a directory. Input: path - either a single recording
    # folder (contains recording.txt) or a parent of recording folders (sorted subdirectories).
    # Folders lacking either recording.txt or info.json are skipped. Output: list of seed dicts with
    # source "asciinema_public_recording", id, title (fallback folder name), url (fallback asciinema
    # URL from folder name), description, transcript (emptied when screening flags it),
    # transcript_sha256 (of the original text) and filter_flags. Raises: ValueError if a transcript
    # exceeds _LIMIT bytes or info.json exceeds 64000 bytes, or if no valid pairs are found; JSON/OS
    # errors on bad files.
    folders = (
        [path]
        if (path / "recording.txt").is_file()
        else sorted(p for p in path.iterdir() if p.is_dir())
    )
    records = []
    for folder in folders:
        transcript = folder / "recording.txt"
        info = folder / "info.json"
        if not transcript.is_file() or not info.is_file():
            continue
        if transcript.stat().st_size > _LIMIT or info.stat().st_size > 64000:
            raise ValueError("Recording exceeds the supported bounded input profile")
        text = transcript.read_text()
        metadata = json.loads(info.read_text())
        flags = screen(text)
        records.append(
            {
                "source": "asciinema_public_recording",
                "id": str(metadata.get("id", folder.name)),
                "title": metadata.get("title") or folder.name,
                "url": metadata.get("url") or f"https://asciinema.org/a/{folder.name}",
                "description": metadata.get("description", ""),
                "transcript": "" if flags else text,
                "transcript_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "filter_flags": flags,
            }
        )
    if not records:
        raise ValueError("No native info.json/recording.txt input pairs found")
    return records


def main() -> None:
    # [FUNCTION DOC] CLI: --ids-json (Path to a JSON list of IDs) and --out (Path), both required.
    # Calls fetch_recordings and prints a console key/value panel with attempted and downloaded
    # counts. Output: None.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = fetch_recordings(json.loads(args.ids_json.read_text()), args.out)
    console.kv(
        {
            "attempted": len(result),
            "downloaded": sum(bool(r["downloaded"]) for r in result.values()),
        },
        title="Recording acquisition",
    )


if __name__ == "__main__":
    main()
