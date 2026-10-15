# [MODULE DOC] MkDocs hook module (registered in mkdocs.yml) that regenerates the prompt reference
# pages before MkDocs discovers documentation files, so the ignored docs/pipelines/prompts/ pages
# exist at build time. Imports generate() from the sibling generate_prompt_reference script (relies
# on docs/_tools being on sys.path, as MkDocs hook loading arranges).
# KNOWN ISSUE: lacks the project-wide `from __future__ import annotations` line (invariant 7),
# though it is a docs tool rather than library code.
"""Generate exact prompt references before MkDocs discovers documentation files."""

from pathlib import Path

from generate_prompt_reference import generate


def on_pre_build(config, **kwargs) -> None:
    # [FUNCTION DOC] MkDocs pre-build hook. Inputs: config (MkDocs config mapping; uses
    # config['docs_dir']), extra kwargs ignored. Calls generate() to write prompt reference pages
    # into <docs_dir>/pipelines/prompts. Output: None. Side effects: writes Markdown files;
    # propagates any exception from generation.
    generate(Path(config["docs_dir"]) / "pipelines/prompts")
