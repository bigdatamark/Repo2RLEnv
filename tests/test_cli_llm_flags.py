# [MODULE DOC] Tests for --llm / --llm-fallback / --llm-endpoint / --llm-key-env handling on
# 'generate'
# and 'bootstrap': cli._llm_overrides dict construction and validation, merging with config
# files via load_generation_input, argparse wiring, and cli._drop_config_llm_extras.
"""`--llm-endpoint` / `--llm-key-env` / `--llm-fallback` on `generate` and `bootstrap`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from repo2rlenv import cli as cli_mod
from repo2rlenv.cli import _drop_config_llm_extras, _llm_overrides
from repo2rlenv.config import load_generation_input

ENDPOINT = "http://127.0.0.1:8000/v1"


def _args(**kwargs) -> argparse.Namespace:
    # [FUNCTION DOC] HELPER: Namespace with llm, llm_fallback, llm_endpoint, llm_key_env set
    # to None, overridden by kwargs. Output: argparse.Namespace.
    base = {"llm": None, "llm_fallback": None, "llm_endpoint": None, "llm_key_env": None}
    base.update(kwargs)
    return argparse.Namespace(**base)


def _write_config(tmp_path: Path, llm: dict | None) -> Path:
    # [FUNCTION DOC] HELPER: Writes tmp_path/gen.yaml for repo pallets/click and pipeline
    # pr_diff, adding the
    # given llm block if not None. Inputs: tmp_path, llm (dict or None). Output: Path to file.
    cfg = {
        "repo": {"url": "pallets/click"},
        "pipeline": {"name": "pr_diff"},
        "output": {"destination": str(tmp_path), "org": "o", "dataset_name": "d"},
    }
    if llm is not None:
        cfg["llm"] = llm
    path = tmp_path / "gen.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


class TestLLMOverrides:
    # [CLASS DOC] Tests for cli._llm_overrides.
    def test_nothing_set_leaves_config_untouched(self):
        # [FUNCTION DOC] No flags -> empty override dict.
        assert _llm_overrides(_args()) == {}

    def test_llm_only(self):
        # [FUNCTION DOC] --llm splits into provider and model.
        assert _llm_overrides(_args(llm="anthropic/claude-sonnet-4-6")) == {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
        }

    def test_endpoint_and_key_env_ride_on_primary(self):
        # [FUNCTION DOC] Endpoint and key env attach to the primary; model is split on the
        # first slash only.
        out = _llm_overrides(
            _args(llm="hosted_vllm/Qwen/Qwen3.5-4B", llm_endpoint=ENDPOINT, llm_key_env="K")
        )
        assert out == {
            "provider": "hosted_vllm",
            "model": "Qwen/Qwen3.5-4B",  # split on the first slash only
            "endpoint": ENDPOINT,
            "api_key_env": "K",
        }

    def test_fallback_does_not_inherit_endpoint(self):
        # [FUNCTION DOC] The fallback dict carries only provider/model, not the primary's
        # endpoint.
        out = _llm_overrides(
            _args(llm="openai/local", llm_fallback="anthropic/claude", llm_endpoint=ENDPOINT)
        )
        assert out["endpoint"] == ENDPOINT
        assert out["fallback"] == {"provider": "anthropic", "model": "claude"}

    def test_fallback_composes_with_config_like_endpoint_does(self):
        # [FUNCTION DOC] --llm-fallback with --config but no --llm yields only a fallback
        # override.
        out = _llm_overrides(_args(llm_fallback="anthropic/claude", config="x.yaml"))
        assert out == {"fallback": {"provider": "anthropic", "model": "claude"}}

    @pytest.mark.parametrize("flag", ["llm_fallback", "llm_endpoint", "llm_key_env"])
    def test_secondary_flags_need_llm_or_config(self, flag):
        # [FUNCTION DOC] Secondary flags without --llm or --config raise SystemExit('need
        # --llm'). Inputs: flag.
        with pytest.raises(SystemExit, match="need --llm"):
            _llm_overrides(_args(**{flag: "anthropic/x" if flag == "llm_fallback" else "v"}))

    @pytest.mark.parametrize("flag", ["llm", "llm_fallback"])
    def test_provider_model_shape_enforced(self, flag):
        # [FUNCTION DOC] A value without a slash for --llm/--llm-fallback raises
        # SystemExit('expects provider/model'). Inputs: flag.
        kwargs = {"llm": "anthropic/x"} if flag == "llm_fallback" else {}
        kwargs[flag] = "no-slash"
        with pytest.raises(SystemExit, match="expects provider/model"):
            _llm_overrides(_args(**kwargs))


def test_endpoint_overrides_config_llm_block(tmp_path: Path):
    # [FUNCTION DOC] --llm-endpoint merges onto the config's llm block keeping provider and
    # model. Inputs: tmp_path.
    cfg = _write_config(tmp_path, {"provider": "hosted_vllm", "model": "Qwen/Qwen3.5-4B"})
    overrides = {"llm": _llm_overrides(_args(llm_endpoint=ENDPOINT, config=str(cfg)))}
    gen = load_generation_input(cfg, overrides)
    assert gen.llm is not None
    assert (gen.llm.provider, gen.llm.model, gen.llm.endpoint) == (
        "hosted_vllm",
        "Qwen/Qwen3.5-4B",
        ENDPOINT,
    )


class TestParser:
    # [CLASS DOC] Tests the real argparse tree with command bodies replaced by captures.
    """Real argparse tree, command body swapped for a capture."""

    def test_generate_accepts_flags(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        # [FUNCTION DOC] 'generate' parses --llm-endpoint and --llm-key-env into args.
        # Inputs: monkeypatch, tmp_path.
        seen: dict = {}
        monkeypatch.setattr(cli_mod, "cmd_generate", lambda a: seen.update(vars(a)) or 0)
        rc = cli_mod.main(
            [
                "generate",
                "--repo", "pallets/click",
                "--pipeline", "pr_diff",
                "--llm", "openai/Qwen/Qwen3.5-4B",
                "--llm-endpoint", ENDPOINT,
                "--llm-key-env", "MY_KEY",
                "--out", str(tmp_path),
            ]
        )  # fmt: skip
        assert rc == 0
        assert seen["llm_endpoint"] == ENDPOINT
        assert seen["llm_key_env"] == "MY_KEY"

    def test_bootstrap_accepts_flags(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] 'bootstrap' parses --llm-endpoint; llm_key_env defaults None and
        # there is no llm_fallback attribute. Inputs: monkeypatch.
        seen: dict = {}
        monkeypatch.setattr(cli_mod, "cmd_bootstrap", lambda a: seen.update(vars(a)) or 0)
        rc = cli_mod.main(
            [
                "bootstrap",
                "--repo", "pallets/click",
                "--llm", "hosted_vllm/Qwen/Qwen3.5-4B",
                "--llm-endpoint", ENDPOINT,
            ]
        )  # fmt: skip
        assert rc == 0
        assert seen["llm_endpoint"] == ENDPOINT
        assert seen["llm_key_env"] is None
        assert "llm_fallback" not in seen


class TestDropConfigLLMExtras:
    # [CLASS DOC] Tests for cli._drop_config_llm_extras.
    @staticmethod
    def _load(cfg: Path, **flags) -> tuple:
        # [FUNCTION DOC] HELPER: Builds args with config=cfg plus flags and loads the
        # GenerationInput with LLM overrides. Output: (GenerationInput, Namespace).
        args = _args(config=str(cfg), **flags)
        gen = load_generation_input(cfg, {"llm": _llm_overrides(args)})
        return gen, args

    def test_new_model_drops_config_endpoint_and_key_env(self, tmp_path: Path):
        # [FUNCTION DOC] Switching to a different model via --llm drops the config's
        # endpoint and api_key_env. Inputs: tmp_path.
        cfg = _write_config(
            tmp_path,
            {"provider": "hosted_vllm", "model": "Qwen", "endpoint": ENDPOINT, "api_key_env": "K"},
        )
        gen, args = self._load(cfg, llm="anthropic/claude-sonnet-4-6")
        assert gen.llm.endpoint == ENDPOINT  # what a plain merge leaves behind
        gen = _drop_config_llm_extras(gen, args, cfg)
        assert (gen.llm.provider, gen.llm.model) == ("anthropic", "claude-sonnet-4-6")
        assert gen.llm.endpoint is None
        assert gen.llm.api_key_env is None

    def test_same_model_keeps_config_endpoint(self, tmp_path: Path):
        # [FUNCTION DOC] Re-specifying the same model keeps the config endpoint. Inputs:
        # tmp_path.
        cfg = _write_config(
            tmp_path, {"provider": "hosted_vllm", "model": "Qwen", "endpoint": ENDPOINT}
        )
        gen, args = self._load(cfg, llm="hosted_vllm/Qwen")
        gen = _drop_config_llm_extras(gen, args, cfg)
        assert gen.llm.endpoint == ENDPOINT

    def test_cli_endpoint_survives_model_change(self, tmp_path: Path):
        # [FUNCTION DOC] An explicit --llm-endpoint is kept even when the model changes.
        # Inputs: tmp_path.
        cfg = _write_config(
            tmp_path, {"provider": "hosted_vllm", "model": "Qwen", "endpoint": "http://old/v1"}
        )
        gen, args = self._load(cfg, llm="openai/other", llm_endpoint=ENDPOINT)
        gen = _drop_config_llm_extras(gen, args, cfg)
        assert gen.llm.endpoint == ENDPOINT


def test_missing_llm_block_is_a_clean_error(tmp_path: Path):
    # [FUNCTION DOC] No llm in config and no --llm returns exit 2 instead of a traceback.
    # Inputs: tmp_path.
    """No `llm:` in config, no --llm → field errors, not a traceback."""
    cfg = _write_config(tmp_path, llm=None)
    rc = cli_mod.main(["--no-ui", "generate", "--config", str(cfg), "--llm-endpoint", ENDPOINT])
    assert rc == 2
