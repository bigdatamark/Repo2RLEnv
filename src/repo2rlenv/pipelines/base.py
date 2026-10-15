# [MODULE DOC] Pipeline contract module. Defines the Pipeline Protocol every synthesis pipeline
# implements (class attributes name / requires_bootstrap / supported_languages / experimental, an
# __init__ taking GenerationInput + validated options + optional bootstrap result, and
# run(out_dir)), the uniform PipelineResult counters returned by run(), LanguageMismatchError, and
# check_language_compatibility() used by the CLI before running a pipeline. Side effects: only a
# logger warning in check_language_compatibility.
"""Pipeline contract — every synthesis pipeline implements this Protocol.

A pipeline:
  1. Takes a `GenerationInput` (the standard input shape) plus its own Options
     model (the validated kwargs).
  2. Emits Harbor-compatible task directories at `out_dir`.
  3. Returns a `PipelineResult` with candidate / emitted / skipped counters
     so the CLI can report yield + QA pass rates uniformly.

Adding a new pipeline = subclass-by-protocol (just match the shape) + register
in `PIPELINES` and `OPTIONS_REGISTRY`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel

from repo2rlenv.bootstrap.spec import LanguageHint
from repo2rlenv.spec.input import GenerationInput, PipelineName

logger = logging.getLogger(__name__)


class LanguageMismatchError(RuntimeError):
    # [CLASS DOC] RuntimeError subclass raised by check_language_compatibility when a pipeline
    # restricted to certain languages is pointed at a repo detected as another language and force
    # is not set. No extra fields.
    """Pipeline can't run against the detected repo language.

    Raised by `check_language_compatibility()` when a Python-only pipeline
    is pointed at a Go / Rust / Node repo without `--force-language`.
    """


@dataclass(slots=True)
class PipelineResult:
    # [CLASS DOC] Mutable slotted dataclass returned by every Pipeline.run. Fields: candidates
    # (int, discovered before filtering), emitted (int, tasks written), skipped (int, sum of skip
    # counts), out_dir (Path where tasks landed), skip_reasons (dict reason -> count). No
    # validation that skipped equals the sum of skip_reasons; callers maintain consistency.
    """Uniform result shape across every pipeline.

    Attributes:
        candidates: Total candidates discovered before filtering.
        emitted: Tasks actually written to `out_dir`.
        skipped: Sum of skip-reason counts.
        out_dir: Where tasks landed.
        skip_reasons: Per-reason counts, e.g. {"draft": 3, "too_many_files": 2}.
    """

    candidates: int
    emitted: int
    skipped: int
    out_dir: Path
    skip_reasons: dict[str, int]


@runtime_checkable
class Pipeline(Protocol):
    # [CLASS DOC] runtime_checkable typing Protocol describing a synthesis pipeline (structural,
    # no inheritance needed). Class attributes: name (PipelineName, required), requires_bootstrap
    # (default False; if True the CLI runs ensure_bootstrap first and passes the BootstrapResult),
    # supported_languages (frozenset of LanguageHint or None = any), experimental (default False;
    # CLI warns before running). Note isinstance checks against a runtime_checkable Protocol only
    # verify member presence.
    """The contract every synthesis pipeline implements.

    Implementations are duck-typed (Protocol, not ABC) so a class doesn't have
    to inherit from anything — just expose:

      - `name: ClassVar[PipelineName]` — the registered identifier
      - `requires_bootstrap: ClassVar[bool]` — True if the pipeline needs a
        working Docker image from the bootstrap phase before it can run
      - `__init__(input, options, bootstrap=None) -> None`
      - `run(out_dir: Path) -> PipelineResult`

    The `Options` arg is whatever Pydantic model is registered for `name` in
    `OPTIONS_REGISTRY`. The dispatcher in `cli.cmd_generate` validates and
    instantiates both before calling `run()`.

    `bootstrap` carries the BootstrapResult (image_digest + test_cmds + ...)
    when `requires_bootstrap=True`. cmd_generate triggers `ensure_bootstrap()`
    automatically; lite pipelines that set `requires_bootstrap=False` get
    `bootstrap=None`.
    """

    name: ClassVar[PipelineName]
    requires_bootstrap: ClassVar[bool] = False
    # Languages this pipeline can handle. `None` means any language is OK.
    # Set to a frozenset of LanguageHint values to restrict (e.g. Python-only
    # pipelines that parse AST or emit pytest verifiers).
    supported_languages: ClassVar[frozenset[LanguageHint] | None] = None
    # Experimental pipelines run normally but emit a warning first: their
    # interface/output may change without notice and quality is not yet on par
    # with the stable set. `cmd_generate` surfaces the warning before running.
    experimental: ClassVar[bool] = False

    def __init__(
        self,
        input: GenerationInput,
        options: BaseModel,
        bootstrap: Any = None,
    ) -> None:
        # [FUNCTION DOC] Protocol signature only (body is an ellipsis). Implementations take input
        # (GenerationInput), options (the Pydantic model registered for this pipeline in
        # OPTIONS_REGISTRY), and bootstrap (BootstrapResult when requires_bootstrap, else None).
        # Output: None.
        ...

    def run(self, out_dir: Path) -> PipelineResult:
        # [FUNCTION DOC] Protocol signature only (body is an ellipsis). Implementations write
        # Harbor task directories under out_dir and return a PipelineResult with
        # candidate/emitted/skipped counters.
        ...


def check_language_compatibility(
    pipeline_cls: type,
    detected: LanguageHint,
    *,
    force: bool = False,
) -> None:
    # [FUNCTION DOC] Verifies a pipeline class can run on the detected repo language. Inputs:
    # pipeline_cls (class, read via getattr for supported_languages and name), detected
    # (LanguageHint), keyword-only force (default False). Returns None if supported_languages is
    # None or contains detected. On mismatch with force: logs a warning (pipeline name, sorted
    # supported language values, detected) and returns. On mismatch without force: raises
    # LanguageMismatchError with a multi-line message suggesting language-agnostic pipelines or
    # --force-language. Side effects: logging only.
    """Verify the pipeline can run against the detected repo language.

    Behavior:
      - `pipeline_cls.supported_languages is None` → any language is fine; returns.
      - detected ∈ supported_languages → returns.
      - mismatch + `force=True` → logs a warning; returns.
      - mismatch + `force=False` → raises `LanguageMismatchError`.
    """
    supported = getattr(pipeline_cls, "supported_languages", None)
    if supported is None:
        return
    if detected in supported:
        return

    pipeline_name = getattr(pipeline_cls, "name", "<unknown>")
    supported_names = ", ".join(sorted(s.value for s in supported))
    detected_name = detected.value if isinstance(detected, LanguageHint) else str(detected)

    if force:
        logger.warning(
            "pipeline %s requires %s; detected %s. Proceeding because --force-language is set.",
            pipeline_name,
            supported_names,
            detected_name,
        )
        return

    raise LanguageMismatchError(
        f"Pipeline {pipeline_name!r} requires {supported_names}; "
        f"this repo is detected as {detected_name!r}.\n"
        f"\n"
        f"This pipeline can't produce valid tasks on {detected_name} code. Options:\n"
        f"  - Pick a language-agnostic pipeline "
        f"(pr_diff / pr_runtime / commit_runtime / cve_patches)\n"
        f"  - Re-run with --force-language to skip this check "
        f"(the pipeline will likely emit 0 tasks)"
    )
