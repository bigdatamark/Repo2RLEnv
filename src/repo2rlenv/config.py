# [MODULE DOC] Loads a GenerationInput from a YAML / TOML / JSON config file and/or CLI-derived
# override dicts. Convention: --config <path> provides the base; CLI flags override file fields via
# a recursive merge, then the result is validated by the GenerationInput Pydantic model.
"""Load GenerationInput from YAML, TOML, or build it from CLI flags.

Convention: --config <path> loads a file; CLI flags override file fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from repo2rlenv.spec.input import GenerationInput


def load_config_file(path: Path) -> dict[str, Any]:
    # [FUNCTION DOC] Reads a UTF-8 config file and returns it as a dict, choosing the parser by
    # suffix: .yaml/.yml (yaml.safe_load; empty file -> {}), .toml (tomllib, falling back to tomli),
    # .json (json). Raises ValueError for any other suffix; propagates read/parse errors. Side
    # effect: file read.
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text) or {}
    if path.suffix == ".toml":
        try:
            import tomllib  # py3.11+
        except ImportError:  # pragma: no cover
            import tomli as tomllib  # type: ignore[import-not-found]
        return tomllib.loads(text)
    if path.suffix == ".json":
        import json

        return json.loads(text)
    raise ValueError(f"unsupported config format: {path.suffix}")


def merge(base: dict, override: dict) -> dict:
    # [FUNCTION DOC] Returns a new dict that is base recursively overlaid with override: None values
    # in override are ignored (keep base); when both sides hold dicts for a key they are merged
    # recursively; otherwise the override value wins. Inputs are not mutated (shallow copy of base
    # at each level).
    """Recursively merge `override` into `base` (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if v is None:
            continue
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def load_generation_input(
    config_path: Path | None,
    overrides: dict[str, Any] | None = None,
) -> GenerationInput:
    # [FUNCTION DOC] Builds a validated GenerationInput. Inputs: config_path (optional Path; loaded
    # with load_config_file), overrides (optional dict merged on top with merge). Returns
    # GenerationInput.model_validate of the result. Raises pydantic ValidationError on invalid data
    # and loader errors.
    raw: dict[str, Any] = {}
    if config_path is not None:
        raw = load_config_file(config_path)
    if overrides:
        raw = merge(raw, overrides)
    return GenerationInput.model_validate(raw)
