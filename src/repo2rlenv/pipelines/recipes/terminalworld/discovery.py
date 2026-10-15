# [MODULE DOC] Bounded, restartable discovery of public asciinema recording IDs from explore feeds
# (FEEDS: public, recent, featured, popular) and explicit public profiles, following TerminalWorld's
# explore step. Respects robots.txt, sleeps 1 s between new page fetches, caps a run at 200 pages,
# collects only numeric IDs, and caches each page's outcome as a JSON receipt so a restart reuses
# completed reads. Runnable as a CLI (main).
"""Bounded public recording discovery, following TerminalWorld's explore feeds.

Only numeric source IDs are collected. Each page's identity and outcome is
retained, so a restart reuses completed reads rather than changing the input pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.robotparser import RobotFileParser

import httpx

from repo2rlenv.execution.lifecycle import now, save_record
from repo2rlenv.pipelines.recipes.terminalworld.source import _USER_AGENT, _download
from repo2rlenv.ui import console

FEEDS = {
    "public": "/explore/public",
    "recent": "/explore/recordings/recent",
    "featured": "/explore/recordings/featured",
    "popular": "/explore/recordings/popular",
}
_ORIGIN = "https://asciinema.org"


class RecordingLinks(HTMLParser):
    # [CLASS DOC] HTMLParser subclass that collects unique recording IDs (in first-seen order) from
    # anchor tags whose href is "/a/<1-12 digits>" with an optional trailing slash, optionally
    # prefixed by https://asciinema.org. Attribute: ids (list of str).
    def __init__(self):
        # [FUNCTION DOC] Initialize the parser and an empty ids list. No inputs besides self.
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        # [FUNCTION DOC] HTMLParser callback: for "a" tags, match the href against the recording-
        # link pattern and append the numeric ID if not already present. Inputs: tag (str), attrs
        # (list of name/value pairs). Output: None; mutates self.ids.
        if tag != "a":
            return
        href = dict(attrs).get("href") or ""
        match = re.fullmatch(r"(?:https://asciinema\.org)?/a/([0-9]{1,12})/?", href)
        if match and match[1] not in self.ids:
            self.ids.append(match[1])


def discover_recordings(
    destination: Path,
    *,
    feeds: list[str],
    pages_per_feed: int = 5,
    profiles: list[str] | None = None,
) -> list[str]:
    # [FUNCTION DOC] Crawl feed and profile index pages and return de-duplicated recording IDs.
    # Inputs: destination (Path cache dir); keyword-only feeds (list of FEEDS keys), pages_per_feed
    # (int, default 5, 1-50), profiles (optional list of "/~name" paths; de-duplicated preserving
    # order). Validation: at least one feed or profile, all feeds known, profile paths match
    # /~[A-Za-z0-9_.-]{1,80}, sources x pages <= 200; ValueError otherwise. Profiles get source
    # names "profile-" + first 16 hex chars of their SHA-256. Behavior: downloads and saves
    # robots.txt (64000-byte cap); for each source and page 1..N uses pages/<source>-<NNN>.json if
    # present (ValueError if its url differs), otherwise checks robots, downloads the page (1 MiB
    # cap), parses IDs, records ids and html_sha256, or records error_type (and http_status for
    # status errors) on httpx.HTTPError/ValueError; saves the receipt and sleeps 1 s. Validates
    # cached IDs are numeric strings (ValueError otherwise), appends new IDs, and stops a source
    # when a page has no IDs or repeats an earlier page's ID list. Output: list of ID strings, also
    # saved to ids.json. Side effects: network, filesystem, sleeping. Note: a failure while
    # downloading robots.txt propagates.
    """Cache bounded, robots-permitted index pages; never fetch task solutions."""
    profiles = list(dict.fromkeys(profiles or []))
    if (not feeds and not profiles) or any(feed not in FEEDS for feed in feeds):
        raise ValueError("Choose one or more supported public recording feeds")
    if any(re.fullmatch(r"/~[a-zA-Z0-9_.-]{1,80}", profile) is None for profile in profiles):
        raise ValueError("Profiles must be explicit public asciinema /~name paths")
    if not 1 <= pages_per_feed <= 50:
        raise ValueError("Discovery requires one to 50 pages per feed")
    sources = [(feed, FEEDS[feed]) for feed in dict.fromkeys(feeds)]
    sources.extend(
        ("profile-" + hashlib.sha256(profile.encode()).hexdigest()[:16], profile)
        for profile in profiles
    )
    if len(sources) * pages_per_feed > 200:
        raise ValueError("A discovery run may request at most 200 pages")
    destination.mkdir(parents=True, exist_ok=True)
    selected: list[str] = []
    with httpx.Client(
        timeout=30, follow_redirects=True, headers={"User-Agent": _USER_AGENT}
    ) as client:
        rules = _download(client, _ORIGIN + "/robots.txt", 64000)
        (destination / "robots.txt").write_text(rules)
        robots = RobotFileParser()
        robots.parse(rules.splitlines())
        for feed, path in sources:
            prior_pages: set[str] = set()
            for page in range(1, pages_per_feed + 1):
                url = _ORIGIN + path + f"?page={page}"
                receipt = destination / "pages" / f"{feed}-{page:03d}.json"
                if receipt.exists():
                    record = json.loads(receipt.read_text())
                    if record.get("url") != url:
                        raise ValueError("Cached discovery page belongs to a different source")
                else:
                    record = {"url": url, "checked_at": now(), "ids": []}
                    try:
                        if not robots.can_fetch(_USER_AGENT, url):
                            raise ValueError("Recording index disallowed by robots.txt")
                        html = _download(client, url, 1024 * 1024)
                        parser = RecordingLinks()
                        parser.feed(html)
                        record.update(
                            ids=parser.ids,
                            html_sha256=hashlib.sha256(html.encode()).hexdigest(),
                        )
                    except (httpx.HTTPError, ValueError) as exc:
                        record["error_type"] = type(exc).__name__
                        if isinstance(exc, httpx.HTTPStatusError):
                            record["http_status"] = exc.response.status_code
                    save_record(receipt, record)
                    time.sleep(1)
                if any(
                    not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,12}", value)
                    for value in record["ids"]
                ):
                    raise ValueError("Cached recording IDs are invalid")
                selected.extend(value for value in record["ids"] if value not in selected)
                signature = hashlib.sha256(json.dumps(record["ids"]).encode()).hexdigest()
                # Empty, failed or repeated pages end this feed. Other feeds can
                # still supply IDs without retrying a failed page in this run.
                if not record["ids"] or signature in prior_pages:
                    break
                prior_pages.add(signature)
    save_record(destination / "ids.json", selected)
    return selected


def main() -> None:
    # [FUNCTION DOC] CLI: --out (required Path), --feeds (choices from FEEDS), --profiles-json (Path
    # to a JSON list of profile paths), --pages-per-feed (default 5). Rejects a non-list or non-
    # string profiles file via parser.error. Feeds default to all FEEDS unless profiles are given
    # (then none). Calls discover_recordings and prints a console key/value panel with the ID count
    # and ids.json path. Output: None.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--feeds", nargs="+", choices=sorted(FEEDS))
    parser.add_argument("--profiles-json", type=Path, help="Explicit public /~name paths")
    parser.add_argument("--pages-per-feed", type=int, default=5)
    args = parser.parse_args()
    profiles = json.loads(args.profiles_json.read_text()) if args.profiles_json else None
    if profiles is not None and (
        not isinstance(profiles, list) or any(not isinstance(item, str) for item in profiles)
    ):
        parser.error("--profiles-json must contain a list of public profile paths")
    feeds = args.feeds if args.feeds is not None else ([] if profiles else list(FEEDS))
    ids = discover_recordings(
        args.out, feeds=feeds, pages_per_feed=args.pages_per_feed, profiles=profiles
    )
    console.kv({"recording_ids": len(ids), "path": args.out / "ids.json"}, title="Discovery")


if __name__ == "__main__":
    main()
