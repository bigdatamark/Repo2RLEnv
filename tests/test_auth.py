# [MODULE DOC] Tests for repo2rlenv.auth token resolution: GitHub token precedence (explicit env
# var, env fallback when gh CLI disabled, None), clone-URL token injection, LLM API key resolution
# (provider defaults, explicit env var, unknown providers, no fall-through, blank = unset) and HF
# token env fallback. Environment is patched with mock.patch.dict.
"""Token resolution paths."""

from __future__ import annotations

import os
from unittest import mock

from repo2rlenv.auth import (
    auth_clone_url,
    resolve_github_token,
    resolve_hf_token,
    resolve_llm_api_key,
)
from repo2rlenv.spec.input import AuthSpec, RepoSpec


def test_explicit_env_wins():
    # [FUNCTION DOC] RepoSpec.auth_token_env wins even when gh CLI use is enabled.
    repo = RepoSpec(url="huggingface/trl", auth_token_env="MY_PRIVATE_PAT")
    auth = AuthSpec(use_gh_cli=True)
    with mock.patch.dict(os.environ, {"MY_PRIVATE_PAT": "explicit-token"}):
        assert resolve_github_token(repo, auth) == "explicit-token"


def test_falls_through_to_env_var_when_gh_disabled():
    # [FUNCTION DOC] With gh disabled, the configured GITHUB_TOKEN env var is used.
    repo = RepoSpec(url="huggingface/trl")
    auth = AuthSpec(use_gh_cli=False, github_token_env="GITHUB_TOKEN")
    with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "env-token"}, clear=False):
        assert resolve_github_token(repo, auth) == "env-token"


def test_returns_none_when_nothing_set():
    # [FUNCTION DOC] No sources -> None.
    repo = RepoSpec(url="huggingface/trl")
    auth = AuthSpec(use_gh_cli=False, github_token_env="ABSENT_VAR")
    with mock.patch.dict(os.environ, {}, clear=True):
        assert resolve_github_token(repo, auth) is None


def test_clone_url_injects_token():
    # [FUNCTION DOC] auth_clone_url inserts 'x-access-token:<token>@' into the https URL.
    url = auth_clone_url("https://github.com/huggingface/trl", "ghp_xxx")
    assert url == "https://x-access-token:ghp_xxx@github.com/huggingface/trl"


def test_clone_url_passthrough_without_token():
    # [FUNCTION DOC] A None token leaves the URL unchanged.
    url = auth_clone_url("https://github.com/huggingface/trl", None)
    assert url == "https://github.com/huggingface/trl"


def test_llm_api_key_resolution_uses_provider_default():
    # [FUNCTION DOC] anthropic resolves from ANTHROPIC_API_KEY.
    with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-xxx"}):
        assert resolve_llm_api_key("anthropic") == "sk-ant-xxx"


def test_llm_api_key_resolution_explicit_env():
    # [FUNCTION DOC] An explicit env var name is honoured.
    with mock.patch.dict(os.environ, {"MY_KEY": "custom"}):
        assert resolve_llm_api_key("anthropic", "MY_KEY") == "custom"


def test_hf_token_falls_back_to_env():
    # [FUNCTION DOC] With use_hf_cli False, the token comes from HF_TOKEN (the cache file is not
    # read).
    with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_xxx"}, clear=True):
        # Patch the cache file path so we don't accidentally pick up real cache
        auth = AuthSpec(use_hf_cli=False)
        assert resolve_hf_token(auth) == "hf_xxx"


def test_llm_api_key_unknown_provider_returns_none_without_raising():
    # [FUNCTION DOC] Providers outside LLM_KEY_ENV_DEFAULTS return None without raising.
    """Providers outside LLM_KEY_ENV_DEFAULTS are LiteLLM's to resolve."""
    with mock.patch.dict(os.environ, {}, clear=True):
        assert resolve_llm_api_key("hosted_vllm") is None
        assert resolve_llm_api_key("ollama") is None
        assert resolve_llm_api_key("bedrock") is None


def test_llm_api_key_explicit_env_works_for_unknown_provider():
    # [FUNCTION DOC] An explicit env var works for unknown providers.
    with mock.patch.dict(os.environ, {"VLLM_KEY": "secret"}, clear=True):
        assert resolve_llm_api_key("hosted_vllm", "VLLM_KEY") == "secret"


def test_llm_api_key_explicit_env_never_falls_through():
    # [FUNCTION DOC] An unset explicit env var returns None rather than falling back to the provider
    # default.
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-personal"}, clear=True):
        assert resolve_llm_api_key("openai", "PROXY_KEY") is None


def test_llm_api_key_blank_counts_as_unset():
    # [FUNCTION DOC] A blank provider env var is treated as unset (None).
    with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=True):
        assert resolve_llm_api_key("anthropic") is None
