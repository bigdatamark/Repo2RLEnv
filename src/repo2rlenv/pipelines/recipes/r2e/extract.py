# [MODULE DOC] Static AST helpers for R2E: a module-level dependency slice for one function (used as
# LLM context) and a stub that replaces a function body with NotImplementedError (the defective
# task state). Nothing from the repository is imported or executed.
"""Bounded module-level dependency slices without executing repository imports."""

from __future__ import annotations

import ast


def dependency_slice(source: str, function_name: str) -> str:
    # [FUNCTION DOC] Returns the minimal set of top-level statements a function depends on.
    # Inputs: source (module text), function_name. Maps each top-level name (defs, classes, imports
    # by alias or first component, assigned names) to the last statement binding it, then follows
    # loaded names transitively from function_name. Output: unparsed source of the selected
    # statements in original order (empty if the function is not found). Raises: ValueError if the
    # slice exceeds 24000 chars; SyntaxError for invalid source.
    tree = ast.parse(source)
    bindings = {}
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.asname or alias.name.split(".")[0] for alias in node.names]
        else:
            names = [
                item.id
                for item in ast.walk(node)
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
            ]
        for name in names:
            bindings[name] = index
    pending, selected = [function_name], set()
    while pending:
        name = pending.pop()
        index = bindings.get(name)
        if index is None or index in selected:
            continue
        selected.add(index)
        pending.extend(
            item.id
            for item in ast.walk(tree.body[index])
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
        )
    result = ast.unparse(
        ast.Module(body=[tree.body[index] for index in sorted(selected)], type_ignores=[])
    )
    if len(result) > 24000:
        raise ValueError("Dependency slice exceeds the initial R2E context profile")
    return result


def stub(
    source: str, function_name: str, docstring: str = "Implement the specified behavior."
) -> str:
    # [FUNCTION DOC] Replaces the body of every top-level (non-async) function named function_name
    # with the given docstring and raise NotImplementedError. Inputs: source, function_name,
    # docstring (default "Implement the specified behavior."). Output: unparsed module text with a
    # trailing newline (comments and formatting are lost). Raises: ValueError if not found.
    tree = ast.parse(source)
    found = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            node.body = [
                ast.Expr(ast.Constant(docstring)),
                ast.Raise(ast.Call(ast.Name("NotImplementedError", ast.Load()), [], [])),
            ]
            found = True
    if not found:
        raise ValueError("Target function is absent from the module")
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
