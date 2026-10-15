# [PACKAGE DOC] Package for the owned TerminalWorld recipe (pipeline terminal_reconstruct):
# reconstructs terminal tasks from public asciinema text recordings. Modules: discovery (bounded
# feed crawling for recording IDs), source (fetching recordings and loading the native
# info.json/recording.txt layout), privacy (transcript screening), recipe (LLM scoring, solution
# extraction and instruction design), materialize (environment build, remote replay, snapshot-driven
# test authoring), worker and capture (remote Docker replay that records filesystem effects),
# pipeline (Pipeline entry point). Exports nothing itself.
"""Owned recording-driven TerminalWorld reconstruction."""

from __future__ import annotations
