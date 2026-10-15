# [MODULE DOC] Read-only preflight for Tasksmith readiness selectors. Before any dependency image
# build or GPU allocation, it confirms that every configured test selector/path prefix names an
# existing, non-symlinked file or directory inside the frozen source snapshot. No imports or pytest
# collection.
# KNOWN ISSUE: this module lacks the `from __future__ import annotations` line required by project
# invariant 7.
"""Read-only readiness path checks before dependency builds or GPU allocation."""

from pathlib import Path

from repo2rlenv.spec.recipe_options import PythonRepositoryProfile


def validate_readiness_paths(snapshot: Path, options: PythonRepositoryProfile) -> None:
    # [FUNCTION DOC] Checks the file prefixes of a Python repository profile's selectors against a
    # materialized snapshot.
    # Inputs: snapshot (Path to the extracted source tree); options (PythonRepositoryProfile). Uses
    # options.test_selectors when non-empty, else options.test_paths. For each entry, the prefix
    # before the first '::' is joined to the resolved snapshot; it fails if any component inside the
    # snapshot is a symlink or if the path is neither a file nor a directory.
    # Output: None. Raises ValueError if snapshot is not a directory, or one ValueError listing
    # every failing entry (field, index, selector, prefix, problem) plus guidance. Side effects:
    # filesystem stat calls only.
    # KNOWN ISSUE: the joined path is not resolved, so a prefix containing '..' can point outside
    # the snapshot; such components are skipped by the symlink check and an existing outside path
    # passes.
    """Check selector file prefixes without importing or collecting repository tests."""
    if not snapshot.is_dir():
        raise ValueError("Readiness preflight requires a materialized source snapshot")
    snapshot = snapshot.resolve()
    field = "test_selectors" if options.test_selectors else "test_paths"
    failures = []
    for index, selector in enumerate(getattr(options, field)):
        prefix = selector.split("::", 1)[0]
        path = snapshot / prefix
        if any(
            part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(snapshot)
        ):
            problem = "symlink paths are unsupported"
        elif not (path.is_file() or path.is_dir()):
            problem = "path is absent from the frozen source snapshot"
        else:
            continue
        failures.append(
            f"profile.options.{field}[{index}]={selector!r} (prefix {prefix!r}): {problem}"
        )
    if failures:
        raise ValueError(
            "Readiness preflight failed before dependency build or GPU allocation:\n"
            + "\n".join(failures)
            + "\nSelect existing upstream files/directories for readiness; future private tests "
            "belong in Design.additional_tests. Test node names are checked later by pytest."
        )
