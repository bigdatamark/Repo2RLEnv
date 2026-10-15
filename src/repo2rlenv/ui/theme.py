# [MODULE DOC] Single source of truth for UI colors (STYLE), glyphs (GLYPH) and the agent action ->
# style map (ACTION_STYLE_MAP keyed by action name: BASH, READ_FILE, LIST_DIR, SAVE_SETUP, GIVE_UP,
# INVALID).
"""Single source of truth for the Repo2RLEnv UI look-and-feel.

Change a color or glyph here; every CLI command picks it up.
"""

from __future__ import annotations

from typing import Final


class STYLE:
    # [CLASS DOC] Namespace of Rich style strings: stamped-line colors, header/highlight/muted,
    # panel border colors and per-agent-action colors. Constants only.
    SUCCESS: Final[str] = "bold green"
    WARN: Final[str] = "bold yellow"
    ERROR: Final[str] = "bold red"
    INFO: Final[str] = "cyan"
    DIM: Final[str] = "dim"
    HEADER: Final[str] = "bold cyan"
    HIGHLIGHT: Final[str] = "bright_yellow"
    MUTED: Final[str] = "grey50"

    PANEL_SUCCESS: Final[str] = "green"
    PANEL_WARN: Final[str] = "yellow"
    PANEL_ERROR: Final[str] = "red"
    PANEL_INFO: Final[str] = "cyan"
    PANEL_DIM: Final[str] = "grey39"

    ACTION_BASH: Final[str] = "bright_green"
    ACTION_READ: Final[str] = "cyan"
    ACTION_LIST: Final[str] = "cyan"
    ACTION_SAVE: Final[str] = "bold green"
    ACTION_GIVE_UP: Final[str] = "bold red"
    ACTION_INVALID: Final[str] = "yellow"


class GLYPH:
    # [CLASS DOC] Namespace of glyph strings: status marks (success, warn, error, info, pending,
    # arrow, bullet) and emoji used for bootstrap phases. Constants only.
    SUCCESS: Final[str] = "✓"
    WARN: Final[str] = "⚠"
    ERROR: Final[str] = "✗"
    INFO: Final[str] = "ⓘ"
    PENDING: Final[str] = "·"
    ARROW: Final[str] = "→"
    BULLET: Final[str] = "•"

    # Phase glyphs (bootstrap phases panel)
    PHASE_CLONE: Final[str] = "📥"
    PHASE_PULL: Final[str] = "🐳"
    PHASE_SANDBOX: Final[str] = "📦"
    PHASE_AGENT: Final[str] = "🤖"
    PHASE_COMMIT: Final[str] = "💾"
    PHASE_PUSH: Final[str] = "📤"
    PHASE_CACHE_HIT: Final[str] = "♻️ "


ACTION_STYLE_MAP: Final[dict[str, str]] = {
    "BASH": STYLE.ACTION_BASH,
    "READ_FILE": STYLE.ACTION_READ,
    "LIST_DIR": STYLE.ACTION_LIST,
    "SAVE_SETUP": STYLE.ACTION_SAVE,
    "GIVE_UP": STYLE.ACTION_GIVE_UP,
    "INVALID": STYLE.ACTION_INVALID,
}
