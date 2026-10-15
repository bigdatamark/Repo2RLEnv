# [MODULE DOC] Shared console singleton (`console`, an R2EConsole) plus logging integration:
# install_logging routes the root logger through a stderr RichHandler and mutes noisy libraries;
# ensure_utf8_output makes stdout/stderr able to encode the UI glyphs.
"""The shared Repo2RLEnv console + logger integration.

`console` is a module-level singleton everyone imports. Methods cover the
common stamped-line patterns (success/info/warn/error), section headers,
and key-value panels.

Logging is routed through `rich.logging.RichHandler` so plain log lines from
pipelines and runners get colored consistently with the rest of the UI.
When a Live is active, noisy library loggers (litellm/httpx) are temporarily
silenced so they don't tear the live display.
"""

from __future__ import annotations

import codecs
import json
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from repo2rlenv.ui.primitives import kv_panel
from repo2rlenv.ui.theme import GLYPH, STYLE


def _should_use_rich() -> bool:
    # [FUNCTION DOC] HELPER: Returns False if NO_COLOR or CI env vars are set, stdout is not a TTY,
    # or TERM is 'dumb'; True otherwise. Reads environment only.
    """Use Rich UI only on a real terminal. Respect NO_COLOR / CI / dumb."""
    if os.environ.get("NO_COLOR") or os.environ.get("CI"):
        return False
    if not sys.stdout.isatty():
        return False
    return os.environ.get("TERM") != "dumb"


class R2EConsole:
    # [CLASS DOC] Wrapper around one rich Console offering stamped-line helpers, sections, key/value
    # and generic panels, a raw print passthrough and a JSON record writer. live_active tracks
    # whether a Live display is running (set by live_view).
    """Single-instance wrapper around rich.console.Console.

    Adds stamped-line helpers (success/info/warn/error) and tracks whether a
    Live is currently active so noisy log streams can be suppressed.
    """

    def __init__(self, *, force_terminal: bool | None = None):
        # [FUNCTION DOC] Creates the underlying Console (force_terminal: optional bool, default
        # auto-detect) and sets live_active False.
        self.console = Console(force_terminal=force_terminal)
        self.live_active = False

    # ----- stamped lines ------------------------------------------------------

    def success(self, msg: str) -> None:
        # [FUNCTION DOC] Prints msg prefixed by the green success glyph. Output: None; writes to
        # stdout.
        self.console.print(f"[{STYLE.SUCCESS}]{GLYPH.SUCCESS}[/] {msg}")

    def info(self, msg: str) -> None:
        # [FUNCTION DOC] Prints msg prefixed by the cyan info glyph. Output: None; writes to stdout.
        self.console.print(f"[{STYLE.INFO}]{GLYPH.INFO}[/] {msg}")

    def warn(self, msg: str) -> None:
        # [FUNCTION DOC] Prints msg prefixed by the yellow warning glyph. Output: None; writes to
        # stdout.
        self.console.print(f"[{STYLE.WARN}]{GLYPH.WARN}[/] {msg}")

    def error(self, msg: str) -> None:
        # [FUNCTION DOC] Prints msg prefixed by the red error glyph (to the console's stdout, not
        # stderr).
        self.console.print(f"[{STYLE.ERROR}]{GLYPH.ERROR}[/] {msg}")

    def dim(self, msg: str) -> None:
        # [FUNCTION DOC] Prints msg in the dim style. Output: None.
        self.console.print(f"[{STYLE.DIM}]{msg}[/]")

    # ----- structured output --------------------------------------------------

    @contextmanager
    def section(self, title: str) -> Iterator[None]:
        # [FUNCTION DOC] Context manager: prints a header-styled rule with title, yields, then
        # always prints a dim closing rule.
        """Context manager that prints a rule before/after a block of output."""
        self.console.rule(f"[{STYLE.HEADER}]{title}[/]", style=STYLE.HEADER)
        try:
            yield
        finally:
            self.console.rule(style=STYLE.DIM)

    def kv(self, pairs: dict[str, Any], *, title: str | None = None) -> None:
        # [FUNCTION DOC] Prints kv_panel(pairs, title=title). Output: None.
        """Render a key/value table inside a panel."""
        self.console.print(kv_panel(pairs, title=title))

    def panel(self, body: Any, *, title: str | None = None, style: str = STYLE.PANEL_INFO) -> None:
        # [FUNCTION DOC] Prints body inside a non-expanding Panel with optional title and border
        # style (default STYLE.PANEL_INFO). Output: None.
        self.console.print(Panel(body, title=title, border_style=style, expand=False))

    # ----- escape hatch -------------------------------------------------------

    def print(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Escape hatch: forwards all args/kwargs to Console.print.
        self.console.print(*args, **kwargs)

    def json(self, value: Any) -> None:
        # [FUNCTION DOC] Writes value as one sorted-key JSON line (non-serializable values via str)
        # directly to the console's file, bypassing Rich markup/wrapping, then flushes.
        """Write one machine-readable record, without terminal wrapping or ANSI."""
        self.console.file.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        self.console.file.flush()


def _make_default_console() -> R2EConsole:
    # [FUNCTION DOC] HELPER: Returns a new default R2EConsole; used to build the module singleton.
    return R2EConsole()


# Module-level singleton. Import this everywhere.
console: R2EConsole = _make_default_console()


# ---------------------------------------------------------------------------
# Logging integration
# ---------------------------------------------------------------------------

_LOGGING_INSTALLED = False


def install_logging(*, level: int = logging.INFO, propagate_noisy: bool = False) -> None:
    # [FUNCTION DOC] Idempotently configures logging (guarded by a module-global flag).
    # Inputs: level (default logging.INFO); propagate_noisy (default False). Replaces the root
    # logger's handlers with a single stderr RichHandler (rich tracebacks, markup, level shown, no
    # time/path) and sets the root level; unless propagate_noisy, sets
    # litellm/LiteLLM/httpx/httpcore/anthropic/openai loggers to WARNING. Output: None. Mutates
    # global logging state.
    """Route logging through Rich so log lines blend with console output.

    Idempotent. Call once at CLI entry. `propagate_noisy=False` mutes
    litellm/httpx/anthropic chatter at the root level so it doesn't dominate
    output; the noisy libraries still emit at WARNING+.
    """
    global _LOGGING_INSTALLED
    if _LOGGING_INSTALLED:
        return
    _LOGGING_INSTALLED = True

    handler = RichHandler(
        console=Console(stderr=True),
        rich_tracebacks=True,
        markup=True,
        show_path=False,
        show_time=False,
        show_level=True,
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    if not propagate_noisy:
        for noisy in ("litellm", "LiteLLM", "httpx", "httpcore", "anthropic", "openai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def ensure_utf8_output() -> None:
    # [FUNCTION DOC] Reconfigures sys.stdout and sys.stderr to UTF-8 (errors='replace') when their
    # encoding is not already UTF-8 and they support reconfigure; does nothing if PYTHONIOENCODING
    # is set. Output: None.
    """Let stdout/stderr encode the UI's glyphs (✓ ✗ ⚠ ─).

    Call once at CLI entry. When stdout is a pipe or a redirected file, Python
    on Windows encodes it with the ANSI code page (e.g. cp1252), which has none
    of those glyphs, so the first stamped line raised UnicodeEncodeError and the
    command exited 1. Such streams are switched to UTF-8, as `PYTHONUTF8=1`
    would do. An explicit `PYTHONIOENCODING` is left alone.
    """
    if os.environ.get("PYTHONIOENCODING"):
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None and not _is_utf8(getattr(stream, "encoding", None)):
            reconfigure(encoding="utf-8", errors="replace")


def _is_utf8(encoding: str | None) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if the codec name normalizes to 'utf-8'; False for
    # None/unknown encodings.
    try:
        return codecs.lookup(encoding or "").name == "utf-8"
    except LookupError:
        return False


def should_use_rich() -> bool:  # re-export for views
    # [FUNCTION DOC] Public re-export of _should_use_rich for the views. Output: bool.
    return _should_use_rich()
