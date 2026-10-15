# [MODULE DOC] HELPER: AST function extractor used by the `equivalence_tests` pipeline. Parses
# Python source files of a cloned repo and yields FunctionCandidate records for module-level,
# synchronous functions that pass R2E-style filters: name not private/test/main-like, at least one
# named argument, body LOC within [min_loc, max_loc], an explicit `return <expr>`, no side-effect-
# suggesting substrings (_BODY_SIDE_EFFECT_PATTERNS: I/O, subprocess, env, logging, print,
# click/flask context, time, randomness, ...), and self-containment (only references its own
# locals, builtins, or the _ALLOWED_EXTERNAL_NAMES stdlib allowlist). Side effects: filesystem
# reads only.
# KNOWN ISSUE: the module docstring lists "must have docstring" as a filter, but
# extract_from_module does not require a docstring.
"""AST function extractor for `equivalence_tests`.

Walks a parsed module and yields `FunctionCandidate` records for every
module-level function that survives R2E-style filters. The pipeline picks
candidates and asks the LLM to write equivalence tests against each.

v0.7 scope: module-level functions only. Class methods (with `self` / `cls`)
and class hierarchies are deferred — they need either dependency slicing
or class-context inlining to be self-contained.

Acknowledgment
--------------
Filter set mirrors the spirit of R2E's `src/r2e/repo_builder/fut_extractor/`
(LOC bounds, must have docstring, must have explicit return, exclude
test/main/demo names). Implementation is original Python stdlib.
"""

from __future__ import annotations

import ast
import fnmatch
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class FunctionCandidate:
    # [CLASS DOC] Frozen slotted dataclass: one module-level function chosen as an equivalence-
    # test seed. Fields: relative_path (POSIX path in repo), name, lineno / end_lineno (1-indexed,
    # inclusive, of the def line; decorators not counted in lineno), body_loc (lines from first
    # body statement to end), source (full text including decorators), docstring
    # (ast.get_docstring or ""), arg_names (tuple of positional-only, regular and keyword-only
    # argument names in order; excludes *args/**kwargs).
    """One module-level function picked as an equivalence-test seed."""

    relative_path: str  # POSIX, e.g. "src/foo/bar.py"
    name: str  # function name
    lineno: int  # 1-indexed start
    end_lineno: int  # 1-indexed end (inclusive)
    body_loc: int  # # lines in the function body (signature excluded)
    source: str  # the full `def ...: ...` text from the file
    docstring: str  # extracted via ast.get_docstring (empty if none)
    arg_names: tuple[str, ...]  # positional + keyword arg names, in order


# Names that are almost never useful task candidates.
_SKIP_NAME_PREFIXES = ("_",)
_SKIP_EXACT_NAMES = {"main", "setup", "run", "init", "cli", "wrapper"}

# Patterns inside the body that suggest the function isn't behaviorally pure
# enough for equivalence testing in a standalone module.
_BODY_SIDE_EFFECT_PATTERNS = (
    "open(",
    "subprocess.",
    "os.system",
    "os.environ",
    "os.remove",
    "os.rename",
    "os.path.exists",
    "os.path.isfile",
    "os.path.isdir",
    "os.getcwd",
    "os.chdir",
    "shutil.",
    "requests.",
    "http.",
    "urllib.",
    "socket.",
    "logging.",
    "print(",
    "sys.exit",
    "sys.stdout",
    "sys.stderr",
    "sys.stdin",
    "sys.argv",
    "input(",
    # v0.8.7 (equivalence_tests self-improvement) — click / flask / web-app
    # -specific context patterns. These generate `oracle_does_not_satisfy_test`
    # failures because tests can't easily replicate the required context.
    "click.echo",
    "click.get_current_context",
    "click.Context",
    "click.launch",
    "click.pause",
    "click.confirm",
    "click.prompt",
    "flask.current_app",
    "flask.request",
    "flask.session",
    "flask.g",
    "flask.render_template",
    "flask.url_for",
    "flask.redirect",
    "flask.abort",
    "current_app.",
    "request.",
    "session.",
    "get_current_context",
    "warnings.warn",
    "tempfile.",
    "threading.",
    "asyncio.",
    "time.sleep",
    "time.time",
    "datetime.now",
    "datetime.utcnow",
    "datetime.today",
    "random.random",
    "random.randint",
    "random.choice",
    "random.shuffle",
    "uuid.",
    "secrets.",
)


def _is_excluded(relative_path: str, exclude_globs: list[str]) -> bool:
    # [FUNCTION DOC] HELPER: True if relative_path matches any fnmatch pattern in exclude_globs.
    # Output: bool.
    return any(fnmatch.fnmatch(relative_path, pat) for pat in exclude_globs)


def _function_body_loc(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    # [FUNCTION DOC] HELPER: Number of lines spanned by the function body: node.end_lineno -
    # body[0].lineno + 1, clamped at 0. Returns 0 for an empty body or missing line numbers.
    # Input: (Async)FunctionDef node. Note the first body statement may be the docstring, so
    # docstring lines are counted.
    """# lines in the function body (signature/decorators excluded).

    Uses end_lineno - body[0].lineno + 1 so we don't count the `def` line.
    Falls back to 0 if line numbers aren't populated (synthetic AST).
    """
    if not node.body:
        return 0
    first = node.body[0]
    start = getattr(first, "lineno", 0)
    end = getattr(node, "end_lineno", 0)
    if not (start and end):
        return 0
    return max(0, end - start + 1)


def _function_source(source_lines: list[str], node: ast.FunctionDef) -> str:
    # [FUNCTION DOC] HELPER: Returns the function's source text sliced from source_lines
    # (0-indexed list of lines), starting at the earliest decorator line (or the def line) through
    # end_lineno, joined by newlines.
    # Inputs: source_lines, node. Output: str.
    """Slice the full `def <name>(...): ...` text out of the source.

    Includes decorators. Uses ast's lineno/end_lineno (1-indexed).
    """
    decorator_lines = [getattr(d, "lineno", node.lineno) for d in node.decorator_list]
    start = min([node.lineno, *decorator_lines]) - 1
    end = getattr(node, "end_lineno", node.lineno)
    return "\n".join(source_lines[start:end])


def _arg_names(node: ast.FunctionDef) -> tuple[str, ...]:
    # [FUNCTION DOC] HELPER: Returns the tuple of positional-only, regular and keyword-only
    # parameter names of node in order; *args and **kwargs are excluded.
    """Positional + keyword arg names (no *args / **kwargs)."""
    args = node.args
    out: list[str] = []
    for a in args.posonlyargs + args.args + args.kwonlyargs:
        out.append(a.arg)
    return tuple(out)


def _has_explicit_return(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    # [FUNCTION DOC] HELPER: True iff any Return node with a value exists inside the function
    # (bare `return` ignored).
    # KNOWN ISSUE: the inline comment says nested functions/classes are not descended into, but
    # ast.walk does descend, so a `return x` inside a nested def makes the outer function pass
    # this filter.
    """True iff the function body contains a `return <expr>` (not bare `return`)."""
    for sub in ast.walk(node):
        if sub is node:
            continue
        # Don't descend into nested function/class definitions
        if isinstance(sub, ast.Return) and sub.value is not None:
            return True
    return False


def _body_has_side_effect(source: str) -> bool:
    # [FUNCTION DOC] HELPER: True if any pattern in _BODY_SIDE_EFFECT_PATTERNS is a plain
    # substring of the function source (so it also matches inside strings/comments, e.g.
    # "request." in "self.request.").
    """Heuristic: function body contains a substring suggesting side effects."""
    return any(pat in source for pat in _BODY_SIDE_EFFECT_PATTERNS)


# Whitelist of "safe" top-level names the function may reference beyond its own
# args + Python builtins. Everything else means the function depends on a
# module-level import that `task_module.py` won't have — which crashes
# import time in the sandbox, showing up as `oracle_does_not_satisfy_test`.
# Stays tight on purpose: adding a symbol here is a decision to accept a
# common stdlib usage; the extractor already forbids most side-effecting
# patterns via `_BODY_SIDE_EFFECT_PATTERNS`.
_ALLOWED_EXTERNAL_NAMES: frozenset[str] = frozenset(
    {
        # Stdlib modules commonly used by pure functions
        "collections",
        "functools",
        "itertools",
        "operator",
        "re",
        "math",
        "string",
        "textwrap",
        "shlex",
        "json",
        "base64",
        "hashlib",
        "unicodedata",
        "typing",
        "types",
        "enum",
        "dataclasses",
        # Common typing generics — commonly used without a `typing.` prefix
        "Any",
        "Optional",
        "Union",
        "Iterable",
        "Iterator",
        "Sequence",
        "Mapping",
        "MutableMapping",
        "MutableSequence",
        "Callable",
        "TypeVar",
        "Generic",
        "Literal",
        "Final",
        "ClassVar",
        # Standalone imports commonly used as decorators / helpers
        "cache",
        "lru_cache",
        "wraps",
        "partial",
        "reduce",
        "chain",
        "product",
        "combinations",
        "permutations",
        "defaultdict",
        "OrderedDict",
        "Counter",
        "namedtuple",
        "deque",
        # Type-hint helpers
        "dataclass",
        "field",
        "Enum",
        "IntEnum",
        "StrEnum",
        "auto",
        # Type aliases like `t = typing` are common in mature repos, but a
        # short alias resolves to a locally-imported module, so we keep the
        # allowlist tight and rely on the annotation stripper below to make
        # the def line self-contained.
    }
)


def _collect_scope_names(node: ast.AST) -> set[str]:
    # [FUNCTION DOC] HELPER: Collects names locally bound in one function scope. Input: an AST
    # node (if FunctionDef, AsyncFunctionDef or Lambda, its parameters including *args/**kwargs
    # are added). Then, for each body statement, walks without entering nested
    # function/class/lambda bodies and adds: Assign target Names (any depth),
    # AnnAssign/AugAssign/NamedExpr Name targets, for/async-for targets, with/async-with `as`
    # targets, plain-import bound names (first dotted segment or asname), from-import
    # names/asnames, nested def/class names, and except-handler names. Output: set of str. No side
    # effects.
    # KNOWN ISSUE: comprehension targets and match-statement captures are not collected, so
    # functions using comprehensions with free-looking loop variables are rejected by the self-
    # containment gate.
    """Collect names that would be locally bound in this function scope.

    Includes function args, local assignments (`x = ...`, walrus, augmented,
    annotated), for/with/except targets, nested def/class names, and any
    inline imports. Does NOT recurse into nested function/class bodies —
    those introduce their own scopes.
    """
    names: set[str] = set()
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        args = node.args
        for a in args.posonlyargs + args.args + args.kwonlyargs:
            names.add(a.arg)
        if args.vararg is not None:
            names.add(args.vararg.arg)
        if args.kwarg is not None:
            names.add(args.kwarg.arg)

    def walk_body(items):
        # [FUNCTION DOC] HELPER: Nested closure that iterates the given statements, walks each
        # with _walk_no_nested_scope, and adds every binding form listed in the parent's doc to
        # the enclosing `names` set (mutates it).
        for stmt in items:
            for sub in _walk_no_nested_scope(stmt):
                if isinstance(sub, ast.Assign):
                    for tgt in sub.targets:
                        for n in ast.walk(tgt):
                            if isinstance(n, ast.Name):
                                names.add(n.id)
                elif (
                    (isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name))
                    or (isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Name))
                    or (isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name))
                ):
                    names.add(sub.target.id)
                elif isinstance(sub, ast.For | ast.AsyncFor):
                    for n in ast.walk(sub.target):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
                elif isinstance(sub, ast.With | ast.AsyncWith):
                    for item in sub.items:
                        if item.optional_vars is not None:
                            for n in ast.walk(item.optional_vars):
                                if isinstance(n, ast.Name):
                                    names.add(n.id)
                elif isinstance(sub, ast.Import):
                    for alias in sub.names:
                        names.add((alias.asname or alias.name).split(".")[0])
                elif isinstance(sub, ast.ImportFrom):
                    for alias in sub.names:
                        names.add(alias.asname or alias.name)
                elif isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) or (
                    isinstance(sub, ast.ExceptHandler) and sub.name
                ):
                    names.add(sub.name)

    if hasattr(node, "body"):
        walk_body(node.body)
    return names


def _walk_no_nested_scope(root: ast.AST) -> Iterator[ast.AST]:
    # [FUNCTION DOC] HELPER: Breadth-first generator over root and its descendants (root itself is
    # always yielded) that does not enter child FunctionDef/AsyncFunctionDef/ClassDef/Lambda nodes
    # (they are not yielded either). Output: iterator of AST nodes.
    """Like ast.walk but does not descend into nested function/class bodies."""
    todo: list[ast.AST] = [root]
    while todo:
        item = todo.pop(0)
        yield item
        for child in ast.iter_child_nodes(item):
            if isinstance(
                child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
            ):
                continue
            todo.append(child)


def _references_only_safe_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    # [FUNCTION DOC] HELPER: Self-containment gate. True iff every Load-context Name in the
    # function (walked without nested scopes, so including decorators, defaults and annotations)
    # is a local of the function (via _collect_scope_names), a Python builtin, or in
    # _ALLOWED_EXTERNAL_NAMES. Direct child nested functions of node are checked too, against the
    # outer scope plus their own locals. Returns False on the first unresolved name. Imports
    # `builtins` lazily. No side effects.
    # KNOWN ISSUE: skipping ast.arguments nodes does not skip their children (the walker still
    # yields annotation/default Names), so annotations ARE checked despite the comment saying they
    # are stripped and ignored. Also nested functions are only found as direct AST children of the
    # def (not inside if/for blocks), and nested classes/lambdas are never checked.
    """True iff the function body only references its own args / builtins / a
    small whitelist of stdlib modules.

    This is the equivalence_tests self-containment gate: pre-v0.8.7 the
    extractor happily picked functions like `click.decorators.argument`,
    which annotates its own params with `Argument` and `FC` — types that
    don't exist in the standalone `task_module.py` we bake. Result:
    task_module.py fails at import, verifier crashes with
    `oracle_does_not_satisfy_test`. Filtering here means we never even
    ask the LLM for a test on such a function.

    Scope-aware: nested function/class definitions introduce their own
    scopes, so their arg names / locals don't leak into the outer check.
    """
    import builtins

    builtin_names = set(dir(builtins))
    allowed = builtin_names | _ALLOWED_EXTERNAL_NAMES

    outer_scope = _collect_scope_names(node)

    # Names referenced in the outer function's BODY (not annotations — those
    # get stripped by `strip_annotations` at bake time so annotation-only
    # external refs don't crash import). Also excludes nested scopes.
    for sub in _walk_no_nested_scope(node):
        # Skip annotation subtrees entirely — they don't survive to runtime.
        if isinstance(sub, ast.arguments):
            continue
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            if sub.id in outer_scope or sub.id in allowed:
                continue
            return False

    # Recurse into nested function bodies — they need to resolve too, but
    # against a scope that includes the outer's names + their own locals.
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            child_scope = outer_scope | _collect_scope_names(child)
            for sub in _walk_no_nested_scope(child):
                if isinstance(sub, ast.arguments):
                    continue
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    if sub.id in child_scope or sub.id in allowed:
                        continue
                    return False
    return True


def _is_skipped_name(name: str) -> bool:
    # [FUNCTION DOC] HELPER: True if name is in _SKIP_EXACT_NAMES
    # (main/setup/run/init/cli/wrapper), starts with "test_", or starts with "_" (private/dunder).
    # Output: bool.
    if name in _SKIP_EXACT_NAMES:
        return True
    if name.startswith("test_"):
        return True
    return any(name.startswith(prefix) for prefix in _SKIP_NAME_PREFIXES)


def extract_from_module(
    path: Path,
    source: str,
    *,
    relative_path: str,
    min_loc: int,
    max_loc: int,
) -> list[FunctionCandidate]:
    # [FUNCTION DOC] Parses source and returns FunctionCandidates for every top-level synchronous
    # function passing all filters, in source order. Inputs: path (unused), source (file text),
    # keyword-only relative_path (POSIX, stored on candidates), min_loc/max_loc (inclusive body-
    # LOC bounds). Filters in order: skip async, _is_skipped_name, no named args, body LOC out of
    # range, no explicit return, side-effect substring in source (IndexError slicing -> skip), not
    # self-contained. Output: list (empty on SyntaxError). No side effects.
    """Parse `source` and return every module-level function that survives filters.

    Returns an empty list on SyntaxError.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    source_lines = source.splitlines()
    out: list[FunctionCandidate] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        # Async functions can't be tested with plain pytest assertions; skip
        if isinstance(node, ast.AsyncFunctionDef):
            continue
        # Filter: name
        if _is_skipped_name(node.name):
            continue
        # Filter: argument count (no zero-arg, no *args/**kwargs only)
        names = _arg_names(node)
        if not names:
            continue
        # Filter: body size
        body_loc = _function_body_loc(node)
        if body_loc < min_loc or body_loc > max_loc:
            continue
        # Filter: explicit return
        if not _has_explicit_return(node):
            continue
        # Filter: side-effect heuristics on the source
        try:
            src = _function_source(source_lines, node)
        except IndexError:
            continue
        if _body_has_side_effect(src):
            continue
        # Filter: self-containment. The candidate must reference only its
        # own args / builtins / a small stdlib allowlist. Otherwise
        # task_module.py fails at import (unresolved names in signatures
        # like `Argument`, `FC`, `t.Any`).
        if not _references_only_safe_names(node):
            continue

        out.append(
            FunctionCandidate(
                relative_path=relative_path,
                name=node.name,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", node.lineno),
                body_loc=body_loc,
                source=src,
                docstring=ast.get_docstring(node) or "",
                arg_names=names,
            )
        )
    return out


def walk_repo(
    clone_dir: Path,
    *,
    file_glob: str,
    exclude_glob: list[str],
    min_loc: int,
    max_loc: int,
) -> Iterator[FunctionCandidate]:
    # [FUNCTION DOC] Generator: for every regular file in clone_dir matching file_glob (Path.glob)
    # whose POSIX relative path does not match an exclude_glob pattern, reads it as UTF-8 (replace
    # errors; OSError -> skip) and yields from extract_from_module with the given min_loc/max_loc.
    # Inputs: clone_dir; keyword-only file_glob, exclude_glob, min_loc, max_loc. Side effects:
    # filesystem reads.
    """Walk `clone_dir`, yield FunctionCandidate from every matching .py file."""
    for path in clone_dir.glob(file_glob):
        if not path.is_file():
            continue
        # POSIX separators on every OS: this path feeds task IDs, reference URLs
        # and metadata, which must not depend on the generating host.
        rel = path.relative_to(clone_dir).as_posix()
        if _is_excluded(rel, exclude_glob):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        yield from extract_from_module(
            path, text, relative_path=rel, min_loc=min_loc, max_loc=max_loc
        )
