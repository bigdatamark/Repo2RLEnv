# [MODULE DOC] Re-export shim for the SWE-smith recipe options model. SWESmithOptions is defined in
# spec/recipe_options.py (the shared per-pipeline option registry) and re-exported here so recipe
# code can import it from its own package. No behavior; no side effects beyond the import.
"""SWE-smith options, shared with the standard pipeline option registry."""

from __future__ import annotations

from repo2rlenv.spec.recipe_options import SWESmithOptions

__all__ = ["SWESmithOptions"]
