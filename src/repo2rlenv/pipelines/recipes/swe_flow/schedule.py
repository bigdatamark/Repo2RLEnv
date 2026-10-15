# [MODULE DOC] Pure AST helpers for SWE-Flow: catalog top-level functions, group traced tests by
# their core-function sets into an ordered development schedule, and skeletonize source for one
# step. Node keys are "path:first_line:name" where first_line includes decorators.
"""Group observed core dependencies and introduce only newly developed functions."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path


def function_catalog(base: Path, paths: list[str]) -> dict:
    # [FUNCTION DOC] Catalog top-level synchronous functions under source paths.
    # Inputs: base (root), paths (files or directories; directories scanned recursively for .py).
    # Output: dict key -> {path, name, line, source}. Raises SyntaxError on unparsable files.
    """This first profile covers top-level synchronous Python functions."""
    found = {}
    for relative in paths:
        source = base / relative
        for path in [source] if source.is_file() else sorted(source.rglob("*.py")):
            text = path.read_text()
            tree = ast.parse(text)
            name = path.relative_to(base).as_posix()
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                start = min([node.lineno] + [item.lineno for item in node.decorator_list])
                key = f"{name}:{start}:{node.name}"
                found[key] = {
                    "path": name,
                    "name": node.name,
                    "line": node.lineno,
                    "source": ast.get_source_segment(text, node),
                }
    return found


def development_schedule(traces: list[dict], catalog: dict) -> list[dict]:
    # [FUNCTION DOC] Order development steps from call traces.
    # Inputs: traces (records with test_id, core_nodes, target_nodes, edges, truncated), catalog.
    # Skips truncated traces; groups tests by frozenset(core_nodes within catalog). Groups are
    # visited by (size, sorted keys); for each, new = core minus already developed, targets = union
    # of target nodes intersected with new; a step is emitted only when both are non-empty, with
    # core_nodes, nodes_to_develop, target_nodes, dependent_nodes (new minus targets), test_ids,
    # previous_tests and deduplicated edges. All visited groups mark their nodes developed and add
    # tests to previous_tests.
    # Output: list of step dicts (step index = position).
    groups = defaultdict(list)
    for trace in traces:
        if trace.get("truncated"):
            continue
        core = frozenset(set(trace["core_nodes"]) & catalog.keys())
        if core:
            groups[core].append(trace)
    developed, previous_tests, result = set(), [], []
    for core, records in sorted(groups.items(), key=lambda item: (len(item[0]), sorted(item[0]))):
        new = core - developed
        tests = [record["test_id"] for record in records]
        targets = set().union(*(set(record["target_nodes"]) for record in records)) & new
        if new and targets:
            result.append(
                {
                    "step": len(result),
                    "core_nodes": sorted(core),
                    "nodes_to_develop": sorted(new),
                    "target_nodes": sorted(targets),
                    "dependent_nodes": sorted(new - targets),
                    "test_ids": tests,
                    "previous_tests": list(previous_tests),
                    "edges": sorted(set(tuple(edge) for row in records for edge in row["edges"])),
                }
            )
        developed.update(new)
        previous_tests.extend(tests)
    return result


def skeletonize(source: str, *, path: str, schedule: dict, docstrings: dict | None = None) -> str:
    # [FUNCTION DOC] Produce defective source for a schedule step.
    # Inputs: source, keyword path (relative), schedule (step dict), docstrings (key -> text,
    # optional). Removes top-level functions listed in dependent_nodes, replaces target-node bodies
    # with a docstring (provided or a default) followed by an ellipsis expression, keeps everything
    # else.
    # Output: ast.unparse text plus newline (comments and formatting are lost).
    """Stub target entry points and remove newly introduced dependencies."""
    tree = ast.parse(source)
    docstrings = docstrings or {}
    kept = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            start = min([node.lineno] + [item.lineno for item in node.decorator_list])
            key = f"{path}:{start}:{node.name}"
            if key in schedule["dependent_nodes"]:
                continue
            if key in schedule["target_nodes"]:
                node.body = [
                    ast.Expr(
                        ast.Constant(docstrings.get(key, "Implement the documented behavior."))
                    ),
                    ast.Expr(ast.Constant(Ellipsis)),
                ]
        kept.append(node)
    tree.body = kept
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
