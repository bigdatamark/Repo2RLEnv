# [MODULE DOC] Staging of selected historical test modules for r2e_gym: copies configured test
# roots from the new revision under a non-package r2e_tests directory so package fixtures and
# relative imports keep working while only the selected modules run.
"""Stage selected historical tests without breaking their package fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

from repo2rlenv.pipelines.recipes.history.selection import within


def stage_tests(head: Path, old: Path, selected: list[str], roots: list[str]) -> list[str]:
    # [FUNCTION DOC] Copies test roots from head into old/r2e_tests and returns the runnable paths.
    # Inputs: head (new-revision tree), old (old-revision tree, modified), selected (test module paths),
    # roots (configured test roots). Roots are copied shallowest first, skipping ones already covered.
    # Output: list of "r2e_tests/<path>" for each selected module. Raises: ValueError if selection is
    # empty or outside roots, r2e_tests already exists, a root is missing, or a selected module is not
    # present after copying. Side effects: creates directories and copies files.
    """Keep helper files private while executing only the selected test modules.

    Preserve names and relative imports beneath an outer directory that is not
    a Python package. Pytest can then import the original test package instead
    of interpreting its modules as renamed, unrelated top-level files.
    """
    if not selected or any(not within(path, roots) for path in selected):
        raise ValueError("Selected tests must belong to the configured test roots")
    suite = old / "r2e_tests"
    if suite.exists():
        raise ValueError("Input repository already owns reserved r2e_tests directory")
    suite.mkdir()
    for root in sorted(set(roots), key=lambda value: (len(Path(value).parts), value)):
        source, target = head / root, suite / root
        if target.exists():
            continue  # A containing configured root already copied this path.
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        elif source.is_file():
            shutil.copyfile(source, target)
        else:
            raise ValueError(f"Historical test root is missing: {root}")
    if any(not (suite / path).is_file() for path in selected):
        raise ValueError("Selected historical test module is missing")
    return ["r2e_tests/" + path for path in selected]
