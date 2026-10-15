# [MODULE DOC] Rich progress view for owned-recipe runs. RecipeView holds the latest
# stage/state/task/message and a metrics dict; recipe_events() is a context manager yielding a
# 'record' callback that journals each ProgressEvent and renders it as JSON, as a plain console
# line, or into a Rich Live display.
"""One recipe-aware view, rendered from the same events used by plain/JSON logs."""

from __future__ import annotations

from contextlib import contextmanager

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from repo2rlenv.campaigns.events import EventJournal, ProgressEvent
from repo2rlenv.ui import console
from repo2rlenv.ui.console import should_use_rich
from repo2rlenv.ui.live import live_view


class RecipeView:
    # [CLASS DOC] Mutable snapshot of a recipe run's progress, rendered as three Rich panels
    # (header, current stage, progress/usage metrics). Fed by update() with ProgressEvent objects.
    def __init__(self, recipe: str, source: str, target: int, provider: str):
        # [FUNCTION DOC] Stores recipe name, source, candidate target count and provider;
        # initialises stage 'preflight', state 'started', empty task/message and metrics {attempted,
        # exported, accepted} = 0. Side effects: none.
        self.recipe, self.source, self.target, self.provider = recipe, source, target, provider
        self.stage = "preflight"
        self.state = "started"
        self.task = ""
        self.message = ""
        self.metrics: dict[str, int | float | str] = {"attempted": 0, "exported": 0, "accepted": 0}

    def update(self, event: ProgressEvent) -> None:
        # [FUNCTION DOC] Copies stage, state, task_id (None -> '') and message from the event and
        # merges event.metrics into self.metrics (overwriting keys). Returns None. Mutates self
        # only.
        self.stage, self.state = event.stage, event.state
        self.task, self.message = event.task_id or "", event.message
        self.metrics.update(event.metrics)

    def render(self):
        # [FUNCTION DOC] Builds and returns a Rich Group of three Panels: header (recipe · provider,
        # source, candidate target), status (stage · state, task, message) and a grid table of
        # metrics with underscores in keys replaced by spaces. Pure; no output.
        header = Text(
            f"{self.recipe} · {self.provider}\n{self.source}\nCandidate target: {self.target}"
        )
        status = Text(f"{self.stage} · {self.state}\n{self.task}\n{self.message}")
        table = Table.grid(padding=(0, 2))
        for key, value in self.metrics.items():
            table.add_row(Text(key.replace("_", " ")), Text(str(value)))
        return Group(
            Panel(header, title="Repo2RLEnv"),
            Panel(status, title="Current stage"),
            Panel(table, title="Progress and usage"),
        )


@contextmanager
def recipe_events(
    view: RecipeView, journal: EventJournal, *, plain: bool = False, json_output: bool = False
):
    # [FUNCTION DOC] Context manager. Inputs: view (RecipeView), journal (EventJournal), keyword-
    # only plain (default False) and json_output (default False). Yields a callable taking a
    # ProgressEvent. When plain, json_output, or should_use_rich() is false, yields record directly;
    # otherwise opens a live_view of view.render() and yields a wrapper that passes the Live into
    # record. Side effects: Live terminal display for the duration of the context.
    def record(event: ProgressEvent, live=None):
        # [FUNCTION DOC] HELPER: Emits the event to the journal, updates the view, then outputs it:
        # as console.json(event.model_dump()) when json_output, by refreshing the Live display when
        # live is given, else as a plain console line '[recipe/stage] state: message'. Returns None.
        journal.emit(event)
        view.update(event)
        if json_output:
            console.json(event.model_dump())
        elif live is not None:
            live.update(view.render())
        else:
            console.print(Text(f"[{event.recipe}/{event.stage}] {event.state}: {event.message}"))

    if plain or json_output or not should_use_rich():
        yield record
    else:
        with live_view(view.render()) as live:
            yield lambda event: record(event, live)
