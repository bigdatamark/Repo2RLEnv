# [MODULE DOC] Freeze a setuptools-git-versioning project's version for history-free task bundles:
# since exported snapshots have no .git, the dynamic git-derived version would break, so the version
# observed in recorded egg-info PKG-INFO metadata is written statically into pyproject.toml.
"""Freeze observed setuptools-git-versioning metadata for history-free bundles."""

from __future__ import annotations

import re
import tomllib
from email.parser import Parser
from pathlib import Path

import tomli_w


def freeze_version(base: Path, *, test_paths: list[str]) -> tuple[dict[str, bytes], dict[str, str]]:
    # [FUNCTION DOC] Produce file overrides that pin the package version.
    # Inputs: base (snapshot root containing pyproject.toml), test_paths (keyword; private test
    # roots).
    # Behavior: requires [tool.setuptools-git-versioning] enabled = true and "version" in
    # project.dynamic. Finds *.egg-info/PKG-INFO files whose normalized Name matches project.name;
    # they must agree on exactly one non-empty Version. Strips any "+local" suffix and validates the
    # remainder against a simple version charset. Sets project.version, removes "version" from
    # dynamic, disables the plugin, and drops packages under the test roots from
    # tool.setuptools.packages when it is a list. Rewrites each matching PKG-INFO Version header.
    # Output: (overrides: relative path -> new bytes for pyproject.toml and each PKG-INFO, evidence:
    # {package, version, basis}). Raises: ValueError on any precondition failure.
    # Side effects: reads files only (does not write).
    """Use recorded package metadata, without invoking target code or guessing a version."""
    project_file = base / "pyproject.toml"
    configuration = tomllib.loads(project_file.read_text())
    project = configuration.get("project", {})
    plugin = configuration.get("tool", {}).get("setuptools-git-versioning", {})
    if plugin.get("enabled") is not True or "version" not in project.get("dynamic", []):
        raise ValueError("Version freezing requires an enabled setuptools-git-versioning project")

    def normalize(name):
        # [FUNCTION DOC] HELPER: PEP 503-style name normalization: collapse runs of -, _ and . to
        # "-" and lowercase. Input: name (str). Output: str.
        return re.sub(r"[-_.]+", "-", name).lower()

    matches = []
    for path in sorted(base.rglob("*.egg-info/PKG-INFO")):
        metadata = Parser().parsestr(path.read_text())
        if normalize(metadata.get("Name", "")) == normalize(project["name"]):
            matches.append((path, metadata))
    versions = {metadata.get("Version", "") for _, metadata in matches}
    if len(versions) != 1 or not next(iter(versions), ""):
        raise ValueError("Version freezing requires unambiguous recorded package metadata")
    observed = versions.pop()
    # The local version may contain the source Git SHA; it is not a runtime requirement.
    version = observed.split("+", 1)[0]
    if not re.fullmatch(r"[A-Za-z0-9.!_-]+", version):
        raise ValueError("Recorded version is not a valid single-line package version")
    project["version"] = version
    project["dynamic"] = [field for field in project["dynamic"] if field != "version"]
    plugin["enabled"] = False
    setuptools = configuration.get("tool", {}).get("setuptools", {})
    packages = setuptools.get("packages")
    if isinstance(packages, list):
        private = [path.replace("/", ".") for path in test_paths]
        setuptools["packages"] = [
            package
            for package in packages
            if not any(package == root or package.startswith(root + ".") for root in private)
        ]
    overrides = {"pyproject.toml": tomli_w.dumps(configuration).encode()}
    for path, metadata in matches:
        metadata.replace_header("Version", version)
        overrides[path.relative_to(base).as_posix()] = metadata.as_string().encode()
    return overrides, {"package": project["name"], "version": version, "basis": "recorded PKG-INFO"}
