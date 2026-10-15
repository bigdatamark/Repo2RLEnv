# [MODULE DOC] Packaging test: every owned recipe with an implementation must have its upstream
# attribution and license declared in pyproject.toml license-files and THIRD_PARTY_NOTICES.md.
# Reads repo files only.
"""Attribution must be discoverable and declared in distribution metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path

from repo2rlenv.pipelines.recipes.catalog import IMPLEMENTATIONS, recipes

ROOT = Path(__file__).resolve().parents[1]


def test_every_retained_recipe_license_is_declared():
    # [FUNCTION DOC] Parses pyproject.toml, asserts license is 'Apache-2.0 AND MIT', expands
    # license-files globs, and for each catalog recipe present in IMPLEMENTATIONS asserts
    # provenance.md exists, the upstream commit (or version) appears in THIRD_PARTY_NOTICES.md, and
    # UPSTREAM_LICENSE is declared with an MIT/Apache-2.0 license when material is retained, or
    # absent with license NOASSERTION otherwise. Also requires the scaler UPSTREAM_NOTICE to be
    # declared. Inputs/outputs: none. Side effects: reads files.
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license"] == "Apache-2.0 AND MIT"
    declared = {path for pattern in project["license-files"] for path in ROOT.glob(pattern)}
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    assert ROOT / "THIRD_PARTY_NOTICES.md" in declared
    for recipe in recipes():
        if recipe.id not in IMPLEMENTATIONS:
            continue
        folder = ROOT / "src/repo2rlenv/pipelines/recipes" / recipe.id
        assert (folder / "provenance.md").is_file()
        reference = recipe.upstream.get("commit") or recipe.upstream["version"]
        assert reference in notices
        license_file = folder / "UPSTREAM_LICENSE"
        if recipe.upstream.get("retained_material", True):
            assert license_file in declared
            assert recipe.upstream["license"] in {"MIT", "Apache-2.0"}
        else:
            assert not license_file.exists()
            assert recipe.upstream["license"] == "NOASSERTION"
    assert ROOT / "src/repo2rlenv/pipelines/recipes/scaler/UPSTREAM_NOTICE" in declared
