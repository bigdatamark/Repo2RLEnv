# [PACKAGE DOC] Campaigns package: durable campaign accounting and resumable execution records for
# owned recipes.
# Submodules: budget (SQLite spend ledger with reservations), llm (metered LLM calls with receipts),
# structured (schema-validated LLM calls with bounded correction), events (append-only JSONL
# progress journal), scheduling (task-slot allocation across parallel batches), release (immutable
# Harbor release staging + HF Hub publication) and the CLI adapters cli / release_cli.
# The package __init__ itself exports nothing and has no side effects.
"""Durable campaign accounting and resumable execution records."""

from __future__ import annotations
