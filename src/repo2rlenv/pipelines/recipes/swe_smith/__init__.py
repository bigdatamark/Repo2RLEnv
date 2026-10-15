# [PACKAGE DOC] Package for the owned SWE-smith recipe (repo_mutate / swe_smith): deterministic
# single-site procedural mutations validated against healthy tests, issue text generation, and
# export with a separate Harbor verifier. No code at package level.
"""SWE-smith-inspired procedural defects, validated against healthy repository tests.

Acknowledgment: SWE-bench/SWE-smith (MIT), revision
9b74ac08118a85c39c356802f7961893af73e07f, swesmith/bug_gen/procedural/python/.
This owned implementation enumerates single-site changes deterministically;
upstream samples probabilistic edits over entities. See provenance.md.
"""

from __future__ import annotations
