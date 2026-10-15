# [MODULE DOC] Builds the public (learner-visible) build profile and context for CodeMidas tasks:
# upstream/private tests are removed, but empty __init__.py placeholders are recreated at install
# time so packages still import and build.
"""Remove upstream tests while retaining empty package scaffolding needed to build."""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path

from repo2rlenv.pipelines.recipes.repository.export import private_asset, repository_build


def public_profile(base, options, private_paths):
    # [FUNCTION DOC] Returns a copy of options whose private_test_paths is the sorted union with
    # private_paths. For every __init__.py under base that this profile marks private, collects it
    # as a placeholder; if any, prefixes task_install_command (or install_command when unset) with a
    # shell command that mkdirs their parent dirs and touches the files. Inputs: base (Path),
    # options (repository options model), private_paths (list[str]). Output: new options model.
    profile = options.model_copy(
        update={"private_test_paths": sorted(set(options.private_test_paths + private_paths))}
    )
    placeholders = []
    for path in sorted(base.rglob("__init__.py")):
        relative = path.relative_to(base)
        if private_asset(relative, profile):
            placeholders.append(relative.as_posix())
    if placeholders:
        parents = sorted({str(Path(path).parent) for path in placeholders})
        scaffold = "mkdir -p -- " + shlex.join(parents) + " && touch -- " + shlex.join(placeholders)
        profile = profile.model_copy(
            update={
                "task_install_command": scaffold
                + " && "
                + (options.task_install_command or options.install_command)
            }
        )
    return profile


def public_build_context(base, profile, destination):
    # [FUNCTION DOC] Writes a Docker build context to destination (must not exist): copies every
    # non-private regular file of base into destination/source/ and writes a Dockerfile from
    # repository_build(profile). Raises ValueError if base contains any symlink. Inputs: base,
    # profile, destination. Output: None.
    destination.mkdir(parents=True, exist_ok=False)
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ValueError("Public snapshots cannot contain symlinks")
        relative = path.relative_to(base)
        if path.is_file() and not private_asset(relative, profile):
            target = destination / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    (destination / "Dockerfile").write_text(repository_build(profile))
