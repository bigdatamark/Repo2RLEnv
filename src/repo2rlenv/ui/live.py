# [MODULE DOC] Context managers for Rich Live displays that mark the console as live and temporarily
# raise noisy library loggers (litellm, httpx, httpcore, anthropic, openai, repo2rlenv.bootstrap) to
# WARNING so the display doesn't tear.
"""Live display helper that quiets noisy loggers while active."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import RenderableType
from rich.live import Live

from repo2rlenv.ui.console import console

_NOISY_LOGGERS = (
    "litellm",
    "LiteLLM",
    "httpx",
    "httpcore",
    "anthropic",
    "openai",
    "repo2rlenv.bootstrap",
)


@contextmanager
def quiet_libraries() -> Iterator[None]:
    # [FUNCTION DOC] Context manager: saves each noisy logger's level, sets it to WARNING for the
    # block and restores the saved levels afterwards (even on exception). Yields None. Mutates
    # global logging state.
    """Raise the level of noisy libs to WARNING for the duration of the block."""
    saved: list[tuple[logging.Logger, int]] = []
    for name in _NOISY_LOGGERS:
        lg = logging.getLogger(name)
        saved.append((lg, lg.level))
        lg.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for lg, level in saved:
            lg.setLevel(level)


@contextmanager
def live_view(
    renderable: RenderableType,
    *,
    refresh_per_second: int = 8,
    transient: bool = False,
    quiet_noisy: bool = True,
) -> Iterator[Live]:
    # [FUNCTION DOC] Context manager wrapping rich.live.Live on the shared console.
    # Inputs: renderable; refresh_per_second (default 8); transient (default False); quiet_noisy
    # (default True, wraps the block in quiet_libraries). Sets console.live_active True for the
    # block and False afterwards. Yields the Live object (vertical_overflow 'visible').
    """Standard Live wrapper: tracks the active flag on the console + quiets noise."""
    console.live_active = True
    try:
        if quiet_noisy:
            with quiet_libraries():
                with Live(
                    renderable,
                    console=console.console,
                    refresh_per_second=refresh_per_second,
                    transient=transient,
                    vertical_overflow="visible",
                ) as live:
                    yield live
        else:
            with Live(
                renderable,
                console=console.console,
                refresh_per_second=refresh_per_second,
                transient=transient,
                vertical_overflow="visible",
            ) as live:
                yield live
    finally:
        console.live_active = False
