# [MODULE DOC] Credential resolution for repository hosts, the HF Hub and LLM providers, plus helper
# to embed a token into a clone URL. Tokens are read from env vars, the `gh` CLI or the HF token
# cache file and are never logged. Depends on spec.input.AuthSpec/RepoSpec; sources.SourceKind is
# imported lazily to avoid a circular import.
"""Token resolution — gh CLI first, env var fallback. No secret ever logged."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from repo2rlenv.spec.input import AuthSpec, RepoSpec


class AuthError(RuntimeError):
    # [CLASS DOC] Authentication failure exception (subclass of RuntimeError). Declared for callers;
    # no function in this module raises it.
    pass


def resolve_github_token(repo: RepoSpec, auth: AuthSpec) -> str | None:
    # [FUNCTION DOC] Resolves a GitHub token in order: (1) env var named by repo.auth_token_env, if
    # set and non-empty; (2) stdout of `gh auth token` when auth.use_gh_cli is true and gh is on
    # PATH (5s timeout, non-zero exit or errors are swallowed); (3) env var named by
    # auth.github_token_env (default GITHUB_TOKEN). Output: token str or None (anonymous). Side
    # effects: may spawn a subprocess; reads env. Raises: nothing.
    """Return a GitHub token following the documented resolution order.

    Order:
      1. repo.auth_token_env (if explicitly set)
      2. `gh auth token` (if auth.use_gh_cli)
      3. $GITHUB_TOKEN
      4. None (anonymous)
    """
    if repo.auth_token_env:
        token = os.environ.get(repo.auth_token_env)
        if token:
            return token

    if auth.use_gh_cli and shutil.which("gh"):
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            pass

    return os.environ.get(auth.github_token_env)


def resolve_hf_token(auth: AuthSpec) -> str | None:
    # [FUNCTION DOC] Resolves a Hugging Face Hub token: if auth.use_hf_cli, reads
    # ~/.cache/huggingface/token (stripped; OSError ignored; empty file skipped); else/then falls
    # back to the env var named by auth.hf_token_env. Output: str or None. Side effects: filesystem
    # read.
    """Return an HF Hub token. huggingface_hub auto-resolves the cache file."""
    if auth.use_hf_cli:
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if token_file.exists():
            try:
                token = token_file.read_text().strip()
                if token:
                    return token
            except OSError:
                pass

    return os.environ.get(auth.hf_token_env)


# Provider → env var for the hosted providers we resolve ourselves, so a missing
# key fails fast with the var named. Everything else (Bedrock, Vertex, self-hosted
# vLLM / Ollama, …) resolves inside LiteLLM — and the self-hosted ones need none.
LLM_KEY_ENV_DEFAULTS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "huggingface": "HF_TOKEN",
    "together": "TOGETHER_API_KEY",
    "groq": "GROQ_API_KEY",
}


def resolve_llm_api_key(provider: str, llm_api_key_env: str | None = None) -> str | None:
    # [FUNCTION DOC] Returns the API key for an LLM provider from the environment. Inputs: provider
    # (e.g. "anthropic", case-insensitive); llm_api_key_env (explicit env var name; if given it is
    # the only var consulted). Without it, uses LLM_KEY_ENV_DEFAULTS; unknown providers return None
    # so LiteLLM can do its own lookup. Empty values count as unset. Output: str or None.
    """Return the API key for `provider` from the environment, or None.

    `llm_api_key_env` names the var explicitly and never falls through: if it
    is unset the result is None, not some other provider's key. Without it,
    only `LLM_KEY_ENV_DEFAULTS` providers are looked up here — for the rest
    LiteLLM does its own (`HOSTED_VLLM_API_KEY`, `OLLAMA_API_KEY`, AWS
    credentials, …). A blank value counts as unset; `.env.example` ships blanks.
    """
    env_name = llm_api_key_env or LLM_KEY_ENV_DEFAULTS.get(provider.lower())
    if not env_name:
        return None
    return os.environ.get(env_name) or None


def auth_clone_url(repo_url: str, token: str | None) -> str:
    # [FUNCTION DOC] Embeds a token into an HTTPS clone URL: github.com URLs become
    # https://x-access-token:TOKEN@..., gitlab.com URLs become https://oauth2:TOKEN@...; any other
    # URL (file://, other hosts) or a falsy token returns repo_url unchanged. Output: str. Side
    # effects: none. The returned URL contains a secret and must not be logged.
    """Inject token into URL for private clone. Token never logged.

    Local (file://) and public clones pass through untouched.
    """
    if not token:
        return repo_url
    if repo_url.startswith("https://github.com/"):
        return repo_url.replace("https://", f"https://x-access-token:{token}@")
    if repo_url.startswith("https://gitlab.com/"):
        return repo_url.replace("https://", f"https://oauth2:{token}@")
    return repo_url


def resolve_repo_token(repo: RepoSpec, auth: AuthSpec) -> str | None:
    # [FUNCTION DOC] Source-aware token lookup keyed on repo.source_kind: LOCAL returns None without
    # shelling out; GITLAB returns the repo.auth_token_env var if set and non-empty, else
    # $GITLAB_TOKEN; anything else (GitHub) delegates to resolve_github_token. Output: str or None.
    """Source-aware token resolution.

    - local: no token (and no `gh` shell-out)
    - gitlab: `repo.auth_token_env` then `$GITLAB_TOKEN`
    - github: the documented `resolve_github_token` chain (unchanged)
    """
    from repo2rlenv.sources import SourceKind

    kind = repo.source_kind
    if kind == SourceKind.LOCAL:
        return None
    if kind == SourceKind.GITLAB:
        if repo.auth_token_env:
            tok = os.environ.get(repo.auth_token_env)
            if tok:
                return tok
        return os.environ.get("GITLAB_TOKEN")
    return resolve_github_token(repo, auth)
