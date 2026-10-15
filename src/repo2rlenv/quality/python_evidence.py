# [MODULE DOC] Extracts excerpts of specific repository test functions as reviewer evidence by
# parsing test files statically (ast), never importing the target code.
"""Select repository test evidence statically, without importing target code."""

from __future__ import annotations

import ast
from pathlib import Path


def _methods(node: ast.AST, parents: tuple[str, ...] = ()):
    # [FUNCTION DOC] HELPER: Generator yielding (parent class-name tuple, function node) for every
    # function/async function directly in node or nested inside classes (recursively through
    # classes, not into function bodies). Input: AST node, parents prefix. Output: iterator of
    # tuples.
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            yield from _methods(child, (*parents, child.name))
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield parents, child


def test_excerpts(
    root: Path, identities: list[str], *, additional_sources: dict[str, str] | None = None
) -> dict[str, str]:
    # [FUNCTION DOC] Returns source excerpts for the requested test identities. Inputs: root (repo
    # dir, searched recursively for test*.py), identities (dotted module path then '.' or '::', e.g.
    # 'pkg.test_x::TestC::test_y[param]'), additional_sources (optional relative-path -> source text
    # overrides/additions). For each file with matching identities it keeps the module's top-level
    # imports plus every function whose '<module>[.<Class>...]::<name>' equals an identity with its
    # parametrization suffix stripped, including decorators. Output: dict relative path -> excerpts
    # joined by blank lines. Raises ValueError if nothing matched; SyntaxError on unparsable files.
    # Side effects: reads files.
    excerpts = {}
    additions = additional_sources or {}
    paths = set(root.rglob("test*.py")) | {root / name for name in additions}
    for path in sorted(paths):
        module = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
        relevant = [
            identity
            for identity in identities
            if identity.startswith(module + ".") or identity.startswith(module + "::")
        ]
        if not relevant:
            continue
        relative = path.relative_to(root).as_posix()
        source = additions[relative] if relative in additions else path.read_text()
        tree = ast.parse(source)
        nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]

        for parents, node in _methods(tree):
            prefix = ".".join((module, *parents)) + "::" + node.name
            if any(identity.split("[", 1)[0] == prefix for identity in relevant):
                nodes.append(node)
        excerpts[path.relative_to(root).as_posix()] = "\n\n".join(
            "\n".join(
                source.splitlines()[
                    min(
                        [node.lineno]
                        + [item.lineno for item in getattr(node, "decorator_list", [])]
                    )
                    - 1 : node.end_lineno
                ]
            )
            for node in nodes
        )
    if not excerpts:
        raise ValueError("Could not resolve failing test evidence")
    return excerpts
