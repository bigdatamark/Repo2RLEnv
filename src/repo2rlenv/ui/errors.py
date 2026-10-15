# [MODULE DOC] Uniform error reporting for CLI adapters: one JSON record or one plain line, with
# optional traceback on stderr.
"""One error record for CLI adapters; verbose diagnostics stay on stderr."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

from repo2rlenv.ui.console import console


def report_error(exc: Exception, *, json_output: bool, verbose: bool = False) -> int:
    # [FUNCTION DOC] Reports an exception and returns exit code 2.
    # Inputs: exc; json_output (bool) -> write {error: class name, message} via console.json to
    # stdout, else print '<Class>: <message>' to stderr; verbose (bool, default False) -> also print
    # the current traceback to stderr (meaningful only inside an except block). Output: int 2.
    if json_output:
        console.json({"error": type(exc).__name__, "message": str(exc)})
    else:
        Console(stderr=True).print(Text(f"{type(exc).__name__}: {exc}"))
    if verbose:
        Console(stderr=True).print_exception(show_locals=False)
    return 2
