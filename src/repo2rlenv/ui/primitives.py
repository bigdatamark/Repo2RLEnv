# [MODULE DOC] Pure builder functions returning Rich renderables (panels and tables) styled with the
# theme. No output side effects.
"""Reusable panel + table builders. Pure functions returning Rich renderables."""

from __future__ import annotations

from typing import Any

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from repo2rlenv.ui.theme import GLYPH, STYLE


def kv_panel(
    pairs: dict[str, Any], *, title: str | None = None, border_style: str = STYLE.PANEL_INFO
) -> Panel:
    # [FUNCTION DOC] Renders a dict as aligned key/value lines (keys dim, left-padded to the longest
    # key + 2, values str()) inside a non-expanding Panel. Inputs: pairs; title (optional);
    # border_style (default STYLE.PANEL_INFO). Output: Panel.
    """Render a dict as a two-column key/value table inside a Panel."""
    body = Text()
    width = max((len(k) for k in pairs), default=0) + 2
    for i, (k, v) in enumerate(pairs.items()):
        body.append(f"{k:<{width}}", style=STYLE.DIM)
        body.append(str(v))
        if i < len(pairs) - 1:
            body.append("\n")
    return Panel(body, title=title, border_style=border_style, expand=False)


def success_panel(body: Any, *, title: str = "Success") -> Panel:
    # [FUNCTION DOC] Wraps body in a non-expanding green Panel titled with the success glyph and
    # title (default 'Success'). Output: Panel.
    return Panel(
        body,
        title=f"[{STYLE.SUCCESS}]{GLYPH.SUCCESS} {title}[/]",
        border_style=STYLE.PANEL_SUCCESS,
        expand=False,
    )


def error_panel(body: Any, *, title: str = "Failed") -> Panel:
    # [FUNCTION DOC] Wraps body in a non-expanding red Panel titled with the error glyph and title
    # (default 'Failed'). Output: Panel.
    return Panel(
        body,
        title=f"[{STYLE.ERROR}]{GLYPH.ERROR} {title}[/]",
        border_style=STYLE.PANEL_ERROR,
        expand=False,
    )


def warn_panel(body: Any, *, title: str = "Warning") -> Panel:
    # [FUNCTION DOC] Wraps body in a non-expanding yellow Panel titled with the warning glyph and
    # title (default 'Warning'). Output: Panel.
    return Panel(
        body,
        title=f"[{STYLE.WARN}]{GLYPH.WARN} {title}[/]",
        border_style=STYLE.PANEL_WARN,
        expand=False,
    )


def header_panel(
    line1: Text, line2: Text | None = None, *, border_style: str = STYLE.PANEL_INFO
) -> Panel:
    # [FUNCTION DOC] Builds a full-width Panel from one Text line, or a Group of two when line2 is
    # given.
    # Inputs: line1; line2 (optional); border_style (default STYLE.PANEL_INFO). Output: Panel.
    body: RenderableType = Group(line1, line2) if line2 is not None else line1
    return Panel(body, border_style=border_style, expand=True)


def styled_table(*, headers: list[str], rows: list[list[Any]], title: str | None = None) -> Table:
    # [FUNCTION DOC] Builds an expanding borderless Table with bold magenta headers. Inputs: headers
    # (column names); rows (lists of values, each str()'d); title (optional). Output: Table.
    """Shared table style: bold magenta header, no edge, no padding."""
    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta",
        show_edge=False,
        pad_edge=False,
        box=None,
        expand=True,
    )
    for h in headers:
        table.add_column(h)
    for row in rows:
        table.add_row(*[str(c) for c in row])
    return table
