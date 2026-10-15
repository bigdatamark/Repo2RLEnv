# [MODULE DOC] Python change analysis for history recipes: path-root membership, per-entity AST
# fingerprints that ignore docstrings, and native SWE-smith/R2E-style bounds on how a commit edits
# source entities and whether its tests correspond to them.
"""Python entity and test correspondence filters used by historical recipes."""

from __future__ import annotations

import ast
from pathlib import PurePosixPath


def within(path: str, roots: list[str]) -> bool:
    # [FUNCTION DOC] HELPER: True if POSIX path equals any root or lies beneath one. Inputs: path
    # (str), roots (list of str). Output: bool.
    value = PurePosixPath(path)
    return any(
        value == PurePosixPath(root) or PurePosixPath(root) in value.parents for root in roots
    )


def entities(source: str) -> tuple[dict[str, str], list[str]]:
    # [FUNCTION DOC] Fingerprints Python source. Input: source text. Output: (definitions, statements)
    # where definitions maps dotted qualified names of every (nested) class/function to an AST dump
    # without positions (docstrings stripped; emptied bodies become pass) and statements lists dumps of
    # module-level non-definition, non-docstring statements. Raises: SyntaxError for invalid source.
    tree = ast.parse(source)
    definitions = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = (
                node.body[1:] if isinstance(node, ast.Module) else node.body[1:] or [ast.Pass()]
            )

    class Visitor(ast.NodeVisitor):
        # [CLASS DOC] HELPER: Local NodeVisitor that records each class/function definition under its
        # dotted prefix into the enclosing definitions dict.
        prefix = ""

        def definition(self, node):
            # [FUNCTION DOC] HELPER: Visits a definition: extends the prefix with the node name, stores its
            # dump, recurses into children, then restores the prefix. Side effects: mutates definitions.
            previous = self.prefix
            self.prefix += node.name + "."
            definitions[self.prefix[:-1]] = ast.dump(node, include_attributes=False)
            self.generic_visit(node)
            self.prefix = previous

        visit_FunctionDef = definition
        visit_AsyncFunctionDef = definition
        visit_ClassDef = definition

    Visitor().visit(tree)
    statements = [
        ast.dump(node, include_attributes=False)
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    return definitions, statements


def check_entities(
    source_pairs: dict[str, tuple[str, str]], test_pairs: dict[str, tuple[str, str]], options
) -> dict:
    # [FUNCTION DOC] Classifies a commit's Python changes and enforces bounds.
    # Inputs: source_pairs and test_pairs (path -> (old text, new text)); options (require_bug_edit,
    # max_added_entities, max_edited_entities, max_statement_entities, require_test_match).
    # Computes added/edited/deleted source entities and the count of changed module-level statements
    # (symmetric difference); test entities whose fingerprint is new or changed. A test match holds if
    # any changed test name contains "issue" or names/references (id or attr) an edited entity.
    # Output: dict with sorted added, edited, deleted, statement_changes, test_match, test_entities.
    # Raises: ValueError if no behavior change, no test change, bug-edit bounds are violated (when
    # required) or no test match (when required).
    added, edited, deleted, statements = [], [], [], 0
    test_changes = []
    for path, (old, new) in source_pairs.items():
        before, before_statements = entities(old)
        after, after_statements = entities(new)
        added.extend(f"{path}:{name}" for name in after.keys() - before.keys())
        deleted.extend(f"{path}:{name}" for name in before.keys() - after.keys())
        edited.extend(
            f"{path}:{name}" for name in before.keys() & after.keys() if before[name] != after[name]
        )
        statements += len(set(before_statements) ^ set(after_statements))
    for path, (old, new) in test_pairs.items():
        before, _ = entities(old)
        after, _ = entities(new)
        for name in after:
            if before.get(name) != after[name]:
                test_changes.append({"file": path, "name": name, "code": after[name]})
    if not (added or edited or statements):
        raise ValueError("No non-docstring Python behavior change")
    if not test_changes:
        raise ValueError("No added or modified test entity")
    if options.require_bug_edit and (
        deleted
        or not edited
        or len(added) > options.max_added_entities
        or len(edited) > options.max_edited_entities
        or statements > options.max_statement_entities
    ):
        raise ValueError("Outside the native bug-edit entity bounds")
    matched = any("issue" in item["name"].lower() for item in test_changes)
    for entity in edited:
        name = entity.split(":", 1)[1].split(".")[-1]
        matched |= any(
            name in item["name"]
            or f"id='{name}'" in item["code"]
            or f"attr='{name}'" in item["code"]
            for item in test_changes
        )
    if options.require_test_match and not matched:
        raise ValueError("Edited test does not name or call a changed source entity")
    return {
        "added": sorted(added),
        "edited": sorted(edited),
        "deleted": sorted(deleted),
        "statement_changes": statements,
        "test_match": matched,
        "test_entities": [{"file": row["file"], "name": row["name"]} for row in test_changes],
    }
