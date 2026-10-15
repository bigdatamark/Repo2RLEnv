# [MODULE DOC] HELPER: Shared machinery for the `code_instruct` pipeline (Magicoder OSS-Instruct-
# style recipe).
# Provides: repo package detection and top-level symbol listing (for repo anchoring and the
# symbol-collision guard); random seed-snippet sampling from repo source files; a substring
# decontamination check against well-known benchmark phrases; the LLM system/user prompt templates
# and a parser for the three-section response ([Problem Description] / [Test] / [Solution]);
# and the post-LLM quality gates (repo anchoring, symbol collision, test strength, equivalence-
# test
# strength) plus dedup fingerprints. Stdlib only; module-level logger is defined but unused here.
# Side effects: only filesystem reads of the cloned repo.
"""Helpers for the `code_instruct` pipeline: sampling, parsing, decontam.

Magicoder OSS-Instruct is a recipe (prompt template + sampling + parsing),
not a reusable library. This module is the recipe.

Acknowledgment
--------------
Algorithms inspired by Magicoder
(`references/magicoder/src/magicoder/generate_data.py:79-102` and
`magicoder/decontamination/find_substrings.py`). No code copied; we
reimplement against Python stdlib only.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import logging
import random
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Repo package detection — the LLM's oracle must import from THIS package
# ---------------------------------------------------------------------------


def detect_repo_package(clone_dir: Path, repo_name: str) -> str | None:
    # [FUNCTION DOC] Finds the top-level Python package name of a cloned repo.
    # Inputs: clone_dir (repo root Path); repo_name (repo short name, e.g. "attrs").
    # Steps: (1) if pyproject.toml exists, parse it (TOML/OS errors treated as empty) and take
    # project.name, else tool.poetry.name; normalize dashes to underscores and lowercase; accept
    # it only
    # if src/<name>/__init__.py or <name>/__init__.py exists. (2) Otherwise try candidates
    # [repo_name with dashes->underscores lowercased, repo_name lowercased], with "attr" prepended
    # when
    # repo_name is "attrs", checking src/<c>/__init__.py then <c>/__init__.py.
    # Output: the package name string, or None if nothing matched. Raises: nothing. Side effects:
    # reads files.
    # KNOWN ISSUE: docstring says "attr" is always a candidate; code only adds it when repo_name
    # is "attrs".
    # Also the lowercased-but-dash-preserving candidate can never be a valid identifier if it has
    # dashes.
    """Find the top-level Python package name of the target repo.

    Order of attempts:
      1. `pyproject.toml` → `[project.name]` (normalized to underscores)
         or `[tool.poetry.name]`.
      2. `src/<candidate>/__init__.py` for candidate in {repo_name,
         repo_name.replace('-', '_'), 'attr'} (attrs ships as `attr`).
      3. `<candidate>/__init__.py` at repo root, same candidates.

    Returns the package name (a valid Python identifier), or None if
    detection fails — callers should skip repo-anchoring enforcement
    in that case rather than block generation.
    """
    # Try pyproject
    py = clone_dir / "pyproject.toml"
    if py.is_file():
        try:
            data = tomllib.loads(py.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError):
            data = {}
        name = (data.get("project", {}).get("name")) or (
            data.get("tool", {}).get("poetry", {}).get("name")
        )
        if name:
            candidate = name.replace("-", "_").lower()
            # Confirm the folder actually exists
            for base in (clone_dir / "src" / candidate, clone_dir / candidate):
                if (base / "__init__.py").is_file():
                    return candidate
    # Try well-known conventions
    candidates: list[str] = [
        repo_name.replace("-", "_").lower(),
        repo_name.lower(),
    ]
    # attrs is packaged as `attr`
    if repo_name.lower() in ("attrs",):
        candidates.insert(0, "attr")
    for candidate in candidates:
        for base in (clone_dir / "src" / candidate, clone_dir / candidate):
            if (base / "__init__.py").is_file():
                return candidate
    return None


def list_repo_top_level_symbols(clone_dir: Path, pkg_name: str) -> set[str]:
    # [FUNCTION DOC] Returns the set of names of top-level classes and (async) functions defined
    # in any .py file under
    # the repo package directory (src/<pkg_name> preferred, else <pkg_name>), walked recursively.
    # Inputs: clone_dir; pkg_name. Output: set of str (empty if no package dir found). Files that
    # fail to
    # read or parse are skipped silently. Used by check_symbol_collision. Side effects: filesystem
    # reads.
    """Return the set of top-level class/function names defined anywhere in
    the repo's Python package. Used by the symbol-collision guard so that
    the LLM's `task_module.py` doesn't reuse a name that would let an agent
    solve the task by re-exporting the repo's own implementation.

    Cheap: one AST walk per .py file under the package directory.
    Errors are swallowed — corrupt files are treated as contributing no names.
    """
    for base in (clone_dir / "src" / pkg_name, clone_dir / pkg_name):
        if base.is_dir():
            root = base
            break
    else:
        return set()
    names: set[str] = set()
    for py in root.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                names.add(node.name)
    return names


# ---------------------------------------------------------------------------
# Seed sampling
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Seed:
    # [CLASS DOC] Frozen slotted dataclass: one sampled snippet from the target repo. Fields:
    # relative_path (POSIX
    # path relative to the clone), start_line (1-indexed), end_line (1-indexed, inclusive), text
    # (the
    # snippet lines joined by newlines, without line numbers).
    """One sampled snippet from the target repo."""

    relative_path: str  # POSIX-style, e.g. "src/foo/bar.py"
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed, inclusive
    text: str  # the snippet itself (no line numbers)


def is_excluded(relative_path: str, exclude_globs: list[str]) -> bool:
    # [FUNCTION DOC] HELPER: True if relative_path matches any fnmatch glob in exclude_globs.
    # Inputs: relative_path
    # (POSIX str), exclude_globs (list of patterns). Output: bool. No side effects.
    return any(fnmatch.fnmatch(relative_path, pat) for pat in exclude_globs)


def list_source_files(clone_dir: Path, *, file_glob: str, exclude_glob: list[str]) -> list[Path]:
    # [FUNCTION DOC] Lists files in clone_dir matching the glob file_glob (Path.glob semantics, so
    # "**" recurses),
    # keeping only regular files whose POSIX relative path is not matched by any exclude_glob
    # pattern.
    # Inputs: clone_dir; keyword-only file_glob (str), exclude_glob (list of str). Output: list of
    # Paths
    # (absolute under clone_dir, in glob order). Side effects: filesystem traversal.
    """Walk the repo tree matching `file_glob` minus `exclude_glob`."""
    out: list[Path] = []
    for p in clone_dir.glob(file_glob):
        if not p.is_file():
            continue
        rel = p.relative_to(clone_dir).as_posix()
        if is_excluded(rel, exclude_glob):
            continue
        out.append(p)
    return out


def sample_seed(
    files: list[Path],
    clone_dir: Path,
    *,
    rng: random.Random,
    min_loc: int,
    max_loc: int,
    max_attempts: int = 20,
) -> Seed | None:
    # [FUNCTION DOC] Samples a random snippet window from the given files using the supplied RNG.
    # Inputs: files (candidate Paths), clone_dir (to compute relative paths), keyword-only rng
    # (random.Random), min_loc/max_loc (window size bounds in lines), max_attempts (default 20).
    # Each attempt: pick a random file (read errors -> retry), skip if it has fewer than min_loc
    # lines,
    # choose a window size uniformly in [min_loc, min(max_loc, line count)] and a uniform start
    # offset,
    # and accept the chunk if _looks_substantive. Output: a Seed (1-indexed start, inclusive end),
    # or None
    # if files is empty or all attempts fail. Side effects: reads files; advances rng.
    # KNOWN ISSUE: docstring says windows dominated by docstrings are skipped, but
    # _looks_substantive only
    # counts blank, comment and import lines as boring.
    """Pick a random file, pick a random window of `[min_loc..max_loc]` lines.

    Skips files that are too short. Skips windows that are dominated by
    blank lines, imports, docstrings, or comments (those snippets give
    the LLM nothing to work with). Returns None after `max_attempts`.
    """
    if not files:
        return None
    for _ in range(max_attempts):
        f = rng.choice(files)
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        if len(lines) < min_loc:
            continue
        window = rng.randint(min_loc, min(max_loc, len(lines)))
        start = rng.randint(0, max(0, len(lines) - window))
        end = start + window
        chunk_lines = lines[start:end]
        chunk = "\n".join(chunk_lines)
        if _looks_substantive(chunk_lines):
            return Seed(
                # POSIX on every OS: feeds the task ID, reference URL and metadata
                relative_path=f.relative_to(clone_dir).as_posix(),
                start_line=start + 1,
                end_line=end,
                text=chunk,
            )
    return None


_COMMENT_OR_IMPORT_RE = re.compile(r"^\s*(?:#|from\s|import\s)")


def _looks_substantive(chunk_lines: list[str]) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if fewer than 80% of the given lines are "boring"
    # (blank, a comment, or a line
    # starting with `from ` / `import `). Empty input returns False. Input: list of line strings.
    """Reject snippets that are 80%+ comments / imports / blanks."""
    if not chunk_lines:
        return False
    boring = 0
    for ln in chunk_lines:
        s = ln.strip()
        if not s or _COMMENT_OR_IMPORT_RE.match(s):
            boring += 1
    return (boring / len(chunk_lines)) < 0.8


# ---------------------------------------------------------------------------
# Decontamination — substring match against well-known eval benchmarks
# ---------------------------------------------------------------------------


# A minimal seed set. The real Magicoder corpus is huge (~10K+ substrings);
# we cover the most common contamination vectors. Extend per-need.
DEFAULT_BENCHMARK_PHRASES: tuple[str, ...] = (
    # HumanEval-style canonical phrasings (both definition and bare-name forms)
    "from typing import list",
    "def has_close_elements",
    "has_close_elements",
    "def separate_paren_groups",
    "separate_paren_groups",
    "def truncate_number",
    "truncate_number",
    # MBPP common phrasings
    "write a python function to",
    "write a function to find the",
    # APPS competitive phrasings
    "the first line of input contains an integer",
    "given a non-empty array of integers",
    # GSM8K (math word problems)
    "natalia sold clips to",
    "if a car travels",
    # DS-1000
    "import pandas as pd",
    "import numpy as np",
)


def has_benchmark_overlap(text: str, phrases: tuple[str, ...] = DEFAULT_BENCHMARK_PHRASES) -> bool:
    # [FUNCTION DOC] Decontamination check: True if any phrase from phrases (default
    # DEFAULT_BENCHMARK_PHRASES, a small
    # set of HumanEval/MBPP/APPS/GSM8K/DS-1000 phrasings) occurs as a case-insensitive substring
    # of text.
    # Inputs: text; phrases (tuple of str). Output: bool. No side effects.
    """True if any known benchmark phrase appears as a substring (case-insensitive)."""
    lower = text.lower()
    return any(p.lower() in lower for p in phrases)


# ---------------------------------------------------------------------------
# Prompting + parsing
# ---------------------------------------------------------------------------


PROMPT_SYSTEM = """You are a senior Python engineer designing a programming exercise that is genuinely anchored in a specific open-source library. You will be given a code snippet from the library `{pkg_name}` and asked to produce a task that requires the solver to USE `{pkg_name}`'s public API to solve it.

The goal is a task that CANNOT be solved without importing from `{pkg_name}` — a solver working in an isolated file without the library available would be stuck.

Produce three sections — exactly in this order, exactly with these section headers:

[Problem Description]
A clear problem statement that names `{pkg_name}` explicitly. Describe what to implement, what `{pkg_name}` primitive(s) to build on (subclass, extend, wrap, or compose), and the expected behavior. Include 1-2 short input/output examples. Under 200 words.

[Test]
A pytest test file. It MUST `from task_module import ...` (literal module name). Write 3-6 assertions covering normal cases AND edge cases (including at least one `pytest.raises` on an error condition). Use plain `def test_...(): assert ...` — no fixtures. The test may also import from `{pkg_name}` to construct inputs or verify behavior against the library's own types.

[Solution]
The Python source for `task_module.py`. It MUST contain at least one `from {pkg_name} import ...` (or `import {pkg_name}`) that is actually used by the code — subclass a `{pkg_name}` class, call a `{pkg_name}` function, wrap a `{pkg_name}` type, etc. Provide a complete, runnable Python file. Beyond `{pkg_name}` and the Python stdlib, do NOT import any other third-party libraries.

Do NOT name any top-level class or function the same as an existing `{pkg_name}` public symbol — the task-module symbol must be new (e.g. prefix or suffix it so a `grep` of the library source would not find it verbatim).

Output only those three sections in that order. No preamble, no closing notes."""


PROMPT_USER_TEMPLATE = """Target library: `{pkg_name}`.

Inspiration snippet (from `{path}`, lines {start}-{end}):

```python
{snippet}
```

Design a task that requires the solver to import and use `{pkg_name}`'s public API. The solution's `task_module.py` MUST contain a used `from {pkg_name} import ...` (or `import {pkg_name}`). Do not reuse any existing `{pkg_name}` symbol name for your task-module's top-level classes/functions."""


@dataclass(slots=True)
class ParsedTask:
    # [CLASS DOC] Mutable slotted dataclass holding the parsed LLM response: problem (problem
    # statement text),
    # test_code (pytest file source, code fence stripped), solution_code (task_module.py source,
    # fence
    # stripped).
    problem: str
    test_code: str
    solution_code: str


def parse_task_response(text: str) -> ParsedTask | None:
    # [FUNCTION DOC] Parses the LLM response into a ParsedTask by extracting the "[Problem
    # Description]", "[Test]" and
    # "[Solution]" sections (markers matched case-insensitively on their own line). Input: raw
    # text.
    # Output: ParsedTask with problem stripped and test/solution code-fence-stripped, or None if
    # any
    # section is missing or empty. No side effects.
    """Extract the three sections from the LLM's response.

    Tolerant: section markers are matched case-insensitively. Returns
    None if any section is missing or empty.
    """
    problem = _extract_section(text, "Problem Description")
    test = _extract_section(text, "Test")
    solution = _extract_section(text, "Solution")
    if not (problem and test and solution):
        return None
    return ParsedTask(
        problem=problem.strip(),
        test_code=_strip_code_fence(test),
        solution_code=_strip_code_fence(solution),
    )


_SECTION_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _section_re(name: str) -> re.Pattern[str]:
    # [FUNCTION DOC] HELPER: Returns (and memoizes in module-level _SECTION_RE_CACHE) a compiled
    # multiline,
    # case-insensitive regex matching a line consisting only of "[<name>]" with optional
    # whitespace.
    # Input: section name. Output: compiled pattern. Side effects: mutates the global cache.
    if name not in _SECTION_RE_CACHE:
        # Match `[Section Name]` markers, with optional leading/trailing whitespace
        pattern = rf"(?im)^\s*\[\s*{re.escape(name)}\s*\]\s*$"
        _SECTION_RE_CACHE[name] = re.compile(pattern)
    return _SECTION_RE_CACHE[name]


def _extract_section(text: str, name: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the stripped text between the "[<name>]" marker line and the
    # next line that looks
    # like any "[Letters and spaces]" marker (or end of text). Empty string if the marker is
    # absent.
    # Inputs: text, name. Output: str.
    """Return the text between `[<name>]` and the next `[...]` marker (or EOF)."""
    pattern = _section_re(name)
    m = pattern.search(text)
    if not m:
        return ""
    start = m.end()
    # Find the next section marker after this one
    next_marker = re.search(r"(?im)^\s*\[\s*[A-Za-z][A-Za-z ]+\s*\]\s*$", text[start:])
    end = start + next_marker.start() if next_marker else len(text)
    return text[start:end].strip()


_CODE_FENCE_RE = re.compile(r"^```(?:python|py)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    # [FUNCTION DOC] HELPER: Strips surrounding whitespace and, if the whole text is a single ```
    # / ```python / ```py
    # fenced block, returns the inner content stripped; otherwise returns the stripped text
    # unchanged.
    """Strip a single surrounding ``` code fence if present."""
    text = text.strip()
    m = _CODE_FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Test syntactic validation
# ---------------------------------------------------------------------------


def references_task_module(test_code: str) -> bool:
    # [FUNCTION DOC] True iff test_code has a line starting with `from task_module import` or
    # `import task_module`
    # (multiline regex). Guards against self-sufficient tests that pass without the oracle.
    # Input: test source string. Output: bool. Named without a test_ prefix so pytest won't
    # collect it.
    """True iff the test code imports from `task_module` (our convention).

    Without this check, the LLM could write a self-sufficient test that
    passes whether or not we apply the oracle — making the task trivial.

    Named without a `test_` prefix so pytest doesn't try to collect this
    helper as a test function.
    """
    pattern = re.compile(
        r"^\s*(?:from\s+task_module\s+import|import\s+task_module)\b", re.MULTILINE
    )
    return bool(pattern.search(test_code))


# ---------------------------------------------------------------------------
# Quality gates — the four post-LLM checks
# ---------------------------------------------------------------------------


def check_repo_anchoring(solution_code: str, pkg_name: str) -> tuple[bool, str]:
    # [FUNCTION DOC] Quality gate: the oracle task_module.py must import the target package and
    # use it.
    # Inputs: solution_code (source), pkg_name. Parses with ast (SyntaxError -> (False,
    # "solution_syntax_error:<msg>")). Collects every Import/ImportFrom (anywhere in the tree)
    # whose
    # module is pkg_name or a submodule, recording the local bound names (alias asname, else the
    # name for
    # from-imports or the top-level segment for plain imports) and the import's first line number.
    # No such
    # import -> (False, "no_repo_import"). Then, with those import lines removed, accepts if any
    # bound name
    # is "*" or appears as a whole word in the remaining text -> (True, ""); otherwise (False,
    # "repo_import_unused"). Output: (accepted bool, reason str). No side effects.
    # KNOWN ISSUE: only the import statement's first line (lineno) is excluded, so a multi-line
    # parenthesized from-import leaks its imported names into the body text and counts as "used".
    """The oracle `task_module.py` must import the target repo and USE it.

    Passes iff:
      1. AST parses.
      2. At least one `Import`/`ImportFrom` node names `pkg_name` (or a
         submodule `pkg_name.foo`).
      3. Any of the imported names appear in the file body (not just the
         import line). "Used" here is a coarse substring/name check on the
         rest of the source — cheap and catches the common failure mode of
         a bare `import X` that never references X.

    Returns (accepted, reason). Reason is empty on accept.
    """
    try:
        tree = ast.parse(solution_code)
    except SyntaxError as exc:
        return False, f"solution_syntax_error:{exc.msg}"
    imported: list[str] = []  # local names introduced by the imports
    has_repo_import = False
    import_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and (node.module == pkg_name or node.module.startswith(f"{pkg_name}.")):
                has_repo_import = True
                import_lines.add(node.lineno)
                for alias in node.names:
                    imported.append(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == pkg_name or alias.name.startswith(f"{pkg_name}."):
                    has_repo_import = True
                    import_lines.add(node.lineno)
                    imported.append(alias.asname or alias.name.split(".", 1)[0])
    if not has_repo_import:
        return False, "no_repo_import"
    # Check "used" — any imported name appears outside its import line
    lines = solution_code.splitlines()
    body_text = "\n".join(ln for i, ln in enumerate(lines, 1) if i not in import_lines)
    for name in imported:
        # `*` from a star-import — impossible to verify usage; accept as used
        if name == "*":
            return True, ""
        if re.search(rf"\b{re.escape(name)}\b", body_text):
            return True, ""
    return False, "repo_import_unused"


def check_symbol_collision(solution_code: str, repo_symbols: set[str]) -> tuple[bool, str]:
    # [FUNCTION DOC] Quality gate: rejects the oracle if any module top-level class/function name
    # is in repo_symbols
    # (would let an agent grep and re-export the real implementation). Inputs: solution_code,
    # repo_symbols (set of str). Output: (False, "symbol_collides_with_repo:<name>") on the first
    # collision, else (True, ""). A SyntaxError returns (True, "") since anchoring already reports
    # it.
    """Reject if any top-level class/function name in the oracle collides
    with a real symbol in the target repo — that lets an agent `grep`
    the on-disk source and re-export the real implementation to pass the
    binary test.
    """
    try:
        tree = ast.parse(solution_code)
    except SyntaxError:
        # already caught by anchoring check; don't double-report
        return True, ""
    for node in tree.body:
        if (
            isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name in repo_symbols
        ):
            return False, f"symbol_collides_with_repo:{node.name}"
    return True, ""


_ASSERT_TRIVIAL_RE = re.compile(
    r"^\s*assert\s+(True|1|\".+\"|'.+')\s*(?:,.*)?\s*$",
)


def check_test_strength(test_code: str, instruction: str) -> tuple[bool, str]:
    # [FUNCTION DOC] Quality gate for code_instruct test files. Inputs: test_code, instruction
    # (problem text).
    # Parses (SyntaxError -> (False, "test_syntax_error:<msg>")); collects top-level sync `test_*`
    # functions (none -> "no_test_functions"). Walks those functions counting assert statements;
    # an
    # assert is trivial if its test is a constant bool/int/str or a Name == same Name comparison
    # (any
    # compare operator). Any trivial assert -> "trivial_assert_present"; fewer than 3 non-trivial
    # ->
    # "too_few_asserts". If the instruction matches raise/error/invalid/must not/reject/forbidden
    # (case-insensitive word), the test must contain "pytest.raises" or "raises(" substring, else
    # "missing_pytest_raises". Output: (accepted, reason), reason empty on accept.
    # KNOWN ISSUE: docstring rule 2 (at least 3 test functions, or one with >=3 asserts) is not
    # enforced; only total non-trivial asserts >= 3 is. The "x == x" check ignores the operator
    # kind.
    # Module-level _ASSERT_TRIVIAL_RE is defined but never used. Async test functions are ignored.
    """Reject test files that are too weak to be worth grading.

    Rules:
      1. Parses cleanly.
      2. At least 3 top-level `test_*` `def`s OR one `test_*` def with
         ≥3 non-trivial `assert` statements. (Multiple tests preferred.)
      3. Total non-trivial `assert` statements across the file ≥ 3.
      4. If the instruction mentions an error condition (raise/error/
         invalid/must not/reject), the test file must contain at least
         one `pytest.raises(...)` block.
      5. No `assert True` / `assert 1` / `assert "literal"`.
    """
    try:
        tree = ast.parse(test_code)
    except SyntaxError as exc:
        return False, f"test_syntax_error:{exc.msg}"
    test_fns: list[ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            test_fns.append(node)
    if not test_fns:
        return False, "no_test_functions"

    total_asserts = 0
    trivial_asserts = 0
    for fn in test_fns:
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Assert):
                total_asserts += 1
                # `assert True`, `assert False`, `assert <literal>`
                test = sub.test
                if isinstance(test, ast.Constant):
                    if test.value in (True, False) or isinstance(test.value, (int, str)):
                        trivial_asserts += 1
                # `assert x == x`
                elif (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Name)
                    and test.left.id == test.comparators[0].id
                ):
                    trivial_asserts += 1

    non_trivial = total_asserts - trivial_asserts
    if trivial_asserts > 0:
        return False, "trivial_assert_present"
    if non_trivial < 3:
        return False, "too_few_asserts"

    # Error-branch coverage
    err_re = re.compile(r"\b(?:raise|error|invalid|must not|reject|forbidden)\b", re.IGNORECASE)
    if (
        err_re.search(instruction)
        and "pytest.raises" not in test_code
        and "raises(" not in test_code
    ):
        return False, "missing_pytest_raises"

    return True, ""


def check_equivalence_test_strength(
    test_code: str, name: str, *, min_test_cases: int = 5
) -> tuple[bool, str]:
    # [FUNCTION DOC] Quality gate for equivalence_tests test files. Inputs: test_code; name
    # (function under test);
    # keyword-only min_test_cases (default 5). Parses (SyntaxError -> "test_syntax_error:<msg>");
    # requires
    # at least min_test_cases top-level `test_*` functions (else
    # "too_few_test_functions:<n><<min>").
    # Every such function must contain a direct call to `name` and to `reference_<name>` (plain
    # Name
    # calls), else (False, "test_fn_missing_both_names:<fn>") immediately. Constant bool/int/str
    # asserts
    # count as trivial -> "trivial_assert_present". Output: (accepted, reason); reason empty on
    # accept.
    """Reject weak equivalence-test files before sending them to the sandbox.

    Different shape than `check_test_strength`: for equivalence tests the
    LLM writes many `def test_*` bodies where each asserts
    `name(x) == reference_name(x)` on distinct inputs. Weakness modes we
    catch here (Stage-A / Stage-B in the sandbox catch the rest):

      1. Fewer than `min_test_cases` distinct `def test_*` functions.
      2. Any `def test_*` body that references only `reference_<name>` and
         never calls `<name>` (would pass while `<name>` is stubbed).
      3. `assert True` / `assert 1` / other trivial constant asserts.

    The Stage-A gate downstream catches the "test never actually exercises
    both names" case at the module level, so we don't reject candidates
    just because the equivalence assertion goes through intermediate
    variables (`a = name(x); b = ref(x); assert a == b` — very common
    LLM output pattern that a strict same-line gate would reject).

    Returns (accepted, reason). Reason is empty on accept.
    """
    try:
        tree = ast.parse(test_code)
    except SyntaxError as exc:
        return False, f"test_syntax_error:{exc.msg}"
    test_fns = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    ]
    if len(test_fns) < min_test_cases:
        return False, f"too_few_test_functions:{len(test_fns)}<{min_test_cases}"

    ref_name = f"reference_{name}"
    trivial_asserts = 0
    for fn in test_fns:
        body_calls_name = False
        body_calls_ref = False
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Assert):
                test = sub.test
                if isinstance(test, ast.Constant) and (
                    test.value in (True, False) or isinstance(test.value, (int, str))
                ):
                    trivial_asserts += 1
            elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                if sub.func.id == name:
                    body_calls_name = True
                elif sub.func.id == ref_name:
                    body_calls_ref = True
        if not (body_calls_name and body_calls_ref):
            return False, f"test_fn_missing_both_names:{fn.name}"

    if trivial_asserts > 0:
        return False, "trivial_assert_present"
    return True, ""


def task_fingerprints(problem: str, solution_code: str = "") -> set[str]:
    # [FUNCTION DOC] Returns the set of dedup fingerprints for a candidate task. Always includes
    # "p:" + first 16 hex chars
    # of SHA-256 of the first 80 chars of the whitespace-collapsed lowercased problem. If
    # solution_code is
    # given and parses, also adds "s:" + 16-hex SHA-256 of the comma-joined sorted public (no
    # leading
    # underscore) top-level class/function names, when any exist. Output: set of str. No side
    # effects.
    """Return the set of dedup signals for a candidate task.

    Two independent signals — a match on EITHER counts as a duplicate:
      1. Problem-statement head (first 80 normalized chars) — catches
         reworded-but-same-task drift.
      2. Sorted top-level public symbol names in the oracle — catches the
         opposite failure mode where the LLM keeps the same class (e.g.
         `RangedFloatType`) but reframes the problem statement.

    Private/dunder names are ignored so that internal helper renames
    don't split an otherwise-duplicate task into two fingerprints.
    """
    fps: set[str] = set()
    norm = re.sub(r"\s+", " ", problem.strip().lower())
    fps.add("p:" + hashlib.sha256(norm[:80].encode("utf-8")).hexdigest()[:16])
    if solution_code:
        try:
            tree = ast.parse(solution_code)
        except SyntaxError:
            return fps
        symbols = sorted(
            n.name
            for n in tree.body
            if isinstance(n, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and not n.name.startswith("_")
        )
        if symbols:
            fps.add("s:" + hashlib.sha256(",".join(symbols).encode("utf-8")).hexdigest()[:16])
    return fps


def task_fingerprint(problem: str, solution_code: str = "") -> str:
    # [FUNCTION DOC] Legacy single fingerprint: 16-hex SHA-256 of the first 80 normalized chars of
    # the problem (same as
    # the "p:" signal without prefix). solution_code is accepted but ignored. Output: str.
    """Legacy single-fingerprint (problem-only) — kept for compat.

    New callers should use `task_fingerprints` which returns a set of
    independent signals covering both problem-rewording and shared-class
    duplication modes.
    """
    norm = re.sub(r"\s+", " ", problem.strip().lower())
    return hashlib.sha256(norm[:80].encode("utf-8")).hexdigest()[:16]
