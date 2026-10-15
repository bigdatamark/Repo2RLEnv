# [MODULE DOC] Plain data types for the bootstrap package: the LanguageHint enum used to choose base
# images and prompt presets, and BootstrapResult, the record returned by ensure_bootstrap and
# persisted as bootstrap.json in the cache. The input-side BootstrapSpec lives in spec/input.py.
"""Data types for the bootstrap module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class LanguageHint(StrEnum):
    # [CLASS DOC] String enum of supported primary-language families: python, node, go, rust, java,
    # c_cpp and unknown. Values are serialized as plain strings (e.g. in the cache JSON).
    PYTHON = "python"
    NODE = "node"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    C_CPP = "c_cpp"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class BootstrapResult:
    # [CLASS DOC] Slotted dataclass describing a finished bootstrap image. Fields: image_digest
    # (pullable repo@sha256 after a push, otherwise the local image Id or tag; the key
    # reproducibility field), image_tag, language, repo ("owner/name"), ref (resolved commit SHA),
    # rebuild_cmds and test_cmds (from the agent's SAVE_SETUP), smoke_passed (soft smoke gate in the
    # live container), iterations, build_time_sec, llm_provider (qualified model name),
    # llm_cost_estimate_usd, dockerfile_reconstruction (Dockerfile text replayed from agent BASH
    # commands or the user Dockerfile), transcript_path, pushed_to_registry, verify_passed
    # (test_cmds succeeded in a fresh container from the committed image), verify_detail (short tail
    # of verify output) and extra (free dict, e.g. source=user_dockerfile).
    """What `ensure_bootstrap()` returns.

    The image_digest is the load-bearing field — once set, the bootstrap is
    fully reproducible: any sandbox can pull `image_digest` and run tasks
    against it without re-running the agent.
    """

    image_digest: str  # ghcr.io/.../bootstrap@sha256:... or local-only digest
    image_tag: str  # human-readable tag we committed to
    language: LanguageHint
    repo: str  # "owner/name"
    ref: str  # commit SHA the bootstrap built from
    rebuild_cmds: list[str]  # commands to re-apply build after a patch
    test_cmds: list[str]  # commands to run the test suite
    smoke_passed: bool  # did the smoke test pass at bootstrap time (in live container)
    iterations: int  # how many agent turns it took
    build_time_sec: float
    llm_provider: str  # "anthropic/claude-sonnet-4-6"
    llm_cost_estimate_usd: float = 0.0  # rough running total
    dockerfile_reconstruction: str = ""  # generated from the agent's commands, for reproducibility
    transcript_path: Path | None = None  # full ReAct transcript for debugging
    pushed_to_registry: bool = False
    verify_passed: bool = False  # did test_cmds work in a FRESH container from the committed image
    verify_detail: str = ""  # short note (last 200 chars of verify output, or skip reason)
    extra: dict = field(default_factory=dict)
