# [MODULE DOC] CLI surface for pipeline/recipe discovery (the "pipelines list|describe" subcommand)
# and the generic runner for owned recipes (run_recipe). Output goes through the repo2rlenv.ui
# console; no credentials are needed for discovery.
"""Recipe discovery uses the existing CLI console and needs no runtime credentials."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from rich.table import Table
from rich.text import Text

from repo2rlenv.pipelines.recipes.catalog import IMPLEMENTATIONS, describe, get_recipe, recipes
from repo2rlenv.ui import console
from repo2rlenv.ui.errors import report_error


def cmd_pipelines(args: argparse.Namespace) -> int:
    # [FUNCTION DOC] Handler for "pipelines list" and "pipelines describe". Input: args
    # (argparse.Namespace with pipeline_action, json, and for describe: pipeline, recipe; optional
    # verbose). Describe: looks up the recipe, errors (via report_error, returning its exit code) if
    # unknown or if it belongs to a different pipeline; prints describe() as JSON or a key/value
    # panel (pipeline, recipe, status, input kinds, scope, target, upstream repo/paper, source pin
    # commit/version, RFC). List: builds native rows from PIPELINES (skipping classes with
    # native_supported False; status experimental/stable from cls.experimental) and owned rows from
    # the catalog; prints JSON {native, recipes} or a Rich table. Output: int exit code (0 on
    # success). Side effects: console output.
    from repo2rlenv.pipelines import PIPELINES

    if args.pipeline_action == "describe":
        try:
            recipe = get_recipe(args.recipe)
            if recipe.pipeline != args.pipeline:
                raise ValueError(f"Recipe {recipe.id} belongs to pipeline {recipe.pipeline}")
        except ValueError as exc:
            return report_error(exc, json_output=args.json, verbose=getattr(args, "verbose", False))
        data = describe(recipe)
        if args.json:
            console.json(data)
        else:
            console.kv(
                {
                    "pipeline": recipe.pipeline,
                    "recipe": recipe.id,
                    "status": data["status"],
                    "input": ", ".join(recipe.source_kinds),
                    "scope": recipe.summary,
                    "target": recipe.target,
                    "upstream": recipe.upstream.get("repository", recipe.upstream.get("paper")),
                    "source pin": recipe.upstream.get(
                        "commit", recipe.upstream.get("version", "not published")
                    ),
                    "RFC": recipe.rfc,
                },
                title=recipe.title,
            )
        return 0
    native = [
        {
            "pipeline": name,
            "recipe": "native",
            "status": "experimental" if cls.experimental else "stable",
        }
        for name, cls in PIPELINES.items()
        if getattr(cls, "native_supported", True)
    ]
    owned = [describe(recipe) for recipe in recipes()]
    if args.json:
        console.json({"native": native, "recipes": owned})
        return 0
    table = Table(title="Generation pipelines")
    for heading in ("Pipeline", "Recipe", "Status", "Source"):
        table.add_column(heading)
    for entry in native:
        table.add_row(entry["pipeline"], "native", entry["status"], "repository")
    for entry in owned:
        table.add_row(
            *(
                Text(value)
                for value in (
                    entry["pipeline"],
                    entry["id"],
                    entry["status"],
                    ", ".join(entry["source_kinds"]),
                )
            )
        )
    console.print(table)
    return 0


def run_recipe(input, *, plain: bool, json_output: bool) -> int:
    # [FUNCTION DOC] Runs an owned recipe for a GenerationInput with a live progress view. Keyword
    # inputs: plain (bool, plain-text event output), json_output (bool). Validates that the recipe's
    # pipeline and accepted source kinds match the input and that it is implemented (ValueError
    # otherwise); parses the recipe options, imports and instantiates the implementation class, and
    # refuses hf:// destinations (ValueError; publish separately with push). Opens an EventJournal
    # at campaign_dir/runs/run_id/events.jsonl and a RecipeView, wires the emit callback into the
    # pipeline and runs it into the local destination. On any exception emits a run event
    # ("cancelled" for KeyboardInterrupt, else "failed") and returns 130 for KeyboardInterrupt or
    # re-raises otherwise. On success emits a completed event with attempted/exported/skipped
    # metrics. Output: 0 if at least one task was exported else 1. Note the pipeline is constructed
    # before the hf:// destination check.
    from repo2rlenv.campaigns.events import EventJournal, ProgressEvent
    from repo2rlenv.spec.options import parse_options
    from repo2rlenv.ui.views.recipe import RecipeView, recipe_events

    recipe = get_recipe(input.pipeline.recipe)
    if recipe.pipeline != input.pipeline.name or input.source.kind not in recipe.source_kinds:
        raise ValueError("Recipe family or source kind does not match the input")
    if recipe.id not in IMPLEMENTATIONS:
        raise ValueError(f"{recipe.id} is planned and has no executable implementation yet")
    module, name = IMPLEMENTATIONS[recipe.id].split(":")
    options = parse_options(recipe.pipeline, input.pipeline.options, recipe=recipe.id)
    pipeline = getattr(importlib.import_module(module), name)(input, options)
    if input.output.destination.startswith("hf://"):
        raise ValueError(
            "Generate owned tasks to a local artifact directory, then publish with repo2rlenv push"
        )
    execution = input.execution
    journal = EventJournal(execution.campaign_dir / "runs" / execution.run_id / "events.jsonl")
    view = RecipeView(recipe.id, input.source_label, pipeline.options.target, "remote")
    with recipe_events(view, journal, plain=plain, json_output=json_output) as emit:
        pipeline.set_event_callback(emit)
        try:
            result = pipeline.run(Path(input.output.destination))
        except BaseException as exc:
            emit(
                ProgressEvent(
                    recipe=recipe.id,
                    stage="run",
                    state="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                    message=f"{type(exc).__name__}: inspect the run receipts before resuming",
                )
            )
            if isinstance(exc, KeyboardInterrupt):
                return 130
            raise
        emit(
            ProgressEvent(
                recipe=recipe.id,
                stage="run",
                state="completed",
                message="Export complete; independent quality validation is still required",
                metrics={
                    "attempted": result.candidates,
                    "exported": result.emitted,
                    "accepted": 0,
                    "skipped": result.skipped,
                },
            )
        )
    return 0 if result.emitted else 1


def add_discovery_parser(subparsers) -> None:
    # [FUNCTION DOC] Registers the "pipelines" subcommand on the given argparse subparsers with
    # actions "list" (--json) and "describe" (positional pipeline, required --recipe, --json), both
    # dispatching to cmd_pipelines via set_defaults(func=...). Output: None.
    parser = subparsers.add_parser(
        "pipelines", help="Discover pipelines, recipes and their requirements"
    )
    actions = parser.add_subparsers(dest="pipeline_action", required=True)
    listing = actions.add_parser("list", help="List native pipelines and owned recipe status")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=cmd_pipelines)
    details = actions.add_parser("describe", help="Show a recipe's input, scope and provenance")
    details.add_argument("pipeline")
    details.add_argument("--recipe", required=True)
    details.add_argument("--json", action="store_true")
    details.set_defaults(func=cmd_pipelines)
