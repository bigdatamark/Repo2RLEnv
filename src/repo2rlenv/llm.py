# [MODULE DOC] LiteLLM wrapper: the single chat-completion entry point across providers. Resolves
# API keys (LLMSpec.api_key_env, provider defaults from auth.LLM_KEY_ENV_DEFAULTS, or a placeholder
# for custom endpoints), builds provider-appropriate kwargs (temperature suppression for reasoning
# models, max_tokens vs max_completion_tokens, json_schema response_format, HF router base), calls
# litellm.completion, records token usage and litellm.completion_cost, and falls back along
# LLMSpec.fallback on transient errors. Side effects: network calls, env var reads, logging.
"""LiteLLM wrapper — single entry point across providers, with cost tracking.

The pipelines call `complete(input, prompt)`; we resolve the API key from
either the LLMSpec hint or the provider-default env var, dispatch, then use
LiteLLM's `completion_cost()` to attach a USD estimate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from repo2rlenv.auth import LLM_KEY_ENV_DEFAULTS, resolve_llm_api_key
from repo2rlenv.spec.input import LLMSpec

logger = logging.getLogger(__name__)

# Sent with `endpoint` to providers whose client insists on *some* key string
# (`openai/<model>` against vLLM, Ollama, llama.cpp — the server ignores it).
# Same value vLLM's docs use.
_PLACEHOLDER_API_KEY = "EMPTY"


# Models that reject any `temperature` value (forced default). Add patterns
# as new releases land. Probed empirically via scripts/probe_llm_routes.py.
_NO_TEMPERATURE_RE = re.compile(
    r"(claude-opus-4-7|claude-opus-4-8|gpt-5(\.|-|$)|gpt-6|o1-|o3-|o4-)",
    re.IGNORECASE,
)
_OPENAI_COMPLETION_LIMIT_RE = re.compile(r"^(gpt-[56](?:[.-]|$)|o[134](?:-|$))")


def _supports_temperature(model: str) -> bool:
    # [FUNCTION DOC] HELPER: True unless the model id matches _NO_TEMPERATURE_RE (Opus 4.7/4.8,
    # GPT-5/6, o1/o3/o4 families), which reject an explicit temperature.
    return _NO_TEMPERATURE_RE.search(model) is None


def completion_token_limit(spec: LLMSpec, max_tokens: int) -> dict[str, int]:
    # [FUNCTION DOC] Return the completion-length kwarg for a spec as a one-entry dict:
    # {"max_completion_tokens": max_tokens} for provider "openai" with a GPT-5/6 or o1/o3/o4 model
    # id (matched from the start of spec.model), otherwise {"max_tokens": max_tokens}. Pure; also
    # used by external Harbor call builders.
    """Provider wire parameters shared with externally orchestrated Harbor calls."""
    name = (
        "max_completion_tokens"
        if spec.provider == "openai" and _OPENAI_COMPLETION_LIMIT_RE.match(spec.model)
        else "max_tokens"
    )
    return {name: max_tokens}


@dataclass(slots=True)
class LLMResponse:
    # [CLASS DOC] Slotted dataclass result of one completion: content (text, "" if none), usage (raw
    # usage dict or None), cost_usd (best-effort USD cost of this call, 0.0 if unknown),
    # prompt_tokens, completion_tokens.
    content: str
    usage: dict | None = None
    cost_usd: float = 0.0  # cost of THIS call, in USD (best-effort)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _is_failover_eligible(exc: BaseException) -> bool:
    # [FUNCTION DOC] HELPER: True if the exception's class name is one of LiteLLM's transient errors
    # (InternalServerError, RateLimitError, ServiceUnavailableError, APIConnectionError, Timeout,
    # APIError), meaning retrying on a fallback model is worthwhile. Matches by name only.
    """True for transient provider errors worth retrying on a fallback model.

    LiteLLM raises specific subclasses; we match on class names so we don't
    have to import the symbols (some live in nested submodules and shift
    between versions).
    """
    name = type(exc).__name__
    # Retry on: 5xx upstream, rate limits, network blips, timeouts.
    # Don't retry on: 4xx bad-request, auth errors, not-found, content filter.
    return name in {
        "InternalServerError",  # 5xx incl. Anthropic 529 Overloaded
        "RateLimitError",  # 429
        "ServiceUnavailableError",
        "APIConnectionError",
        "Timeout",
        "APIError",  # generic upstream error
    }


def _resolve_api_key(spec: LLMSpec) -> str | None:
    # [FUNCTION DOC] HELPER: Decide which API key to pass LiteLLM for a spec. If spec.api_key_env is
    # set: returns that env var's value via resolve_llm_api_key, raising RuntimeError if unset. Else
    # if spec.endpoint is set: never forwards provider-default keys; logs a warning if the
    # provider's default key env var is set; returns the "EMPTY" placeholder for providers listed in
    # LLM_KEY_ENV_DEFAULTS, else None. Else: returns the provider-default key, raising RuntimeError
    # naming the env var when a known provider has none; None for unknown providers (LiteLLM
    # resolves).
    """The API key to hand LiteLLM, or None to let LiteLLM resolve it.

    - `api_key_env` given → its value; raise if unset (never falls through)
    - `endpoint` given → the provider-default key is *not* forwarded to a
      custom server. `LLM_KEY_ENV_DEFAULTS` clients (`openai/`, …) need a key
      string, so they get the placeholder; `hosted_vllm/`, `ollama/`, … get
      None and LiteLLM applies its own `HOSTED_VLLM_API_KEY`-style lookup.
    - otherwise → provider-default key; raise naming the var for
      `LLM_KEY_ENV_DEFAULTS` providers, None (LiteLLM's call) for the rest
    """
    provider = spec.provider.lower()
    if spec.api_key_env:
        key = resolve_llm_api_key(provider, spec.api_key_env)
        if key is None:
            raise RuntimeError(
                f"no API key for provider {spec.provider!r}: ${spec.api_key_env} is unset."
            )
        return key
    if spec.endpoint:
        if provider in LLM_KEY_ENV_DEFAULTS and resolve_llm_api_key(provider):
            key_env = LLM_KEY_ENV_DEFAULTS[provider]
            logger.warning(
                "$%s is set but is not sent to the custom LLM endpoint. "
                "Use --llm-key-env %s (or llm.api_key_env in config) "
                "if this endpoint should receive that key.",
                key_env,
                key_env,
            )
        return _PLACEHOLDER_API_KEY if provider in LLM_KEY_ENV_DEFAULTS else None
    key = resolve_llm_api_key(provider)
    if key is None and provider in LLM_KEY_ENV_DEFAULTS:
        raise RuntimeError(
            f"no API key resolved for provider {spec.provider!r}. Set "
            f"{LLM_KEY_ENV_DEFAULTS[provider]}, or use --llm-endpoint for a self-hosted server."
        )
    return key


def check_provider(spec: LLMSpec) -> None:
    # [FUNCTION DOC] Validate that LiteLLM recognizes every model in the spec's fallback chain by
    # calling litellm.get_llm_provider on each qualified_name. Raises RuntimeError (first line of
    # the underlying error, chained) on the first unknown one. Returns None; no network.
    """Raise early if LiteLLM doesn't know a provider in the spec's fallback chain.

    Cheap next to what follows (clone, image pull, agent loop), so the bootstrap
    entry points call it before any of that starts.
    """
    import litellm  # type: ignore[import-untyped]

    cur: LLMSpec | None = spec
    while cur is not None:
        try:
            litellm.get_llm_provider(cur.qualified_name)
        except Exception as exc:
            first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            raise RuntimeError(
                f"unknown LLM provider in {cur.qualified_name!r}: {first_line}"
            ) from exc
        cur = cur.fallback


def _do_complete(
    spec: LLMSpec,
    *,
    system: str | None,
    user: str,
    max_tokens: int,
    temperature: float,
    response_schema: dict | None = None,
) -> LLMResponse:
    # [FUNCTION DOC] HELPER: One chat completion without fallback.
    # Inputs: spec; system (optional system prompt); user (user message); max_tokens; temperature;
    # response_schema (optional JSON schema, sent as strict json_schema response_format named
    # "repo2rlenv_response"). Builds messages and kwargs (model=qualified_name, num_retries=0,
    # timeout=spec.timeout_sec, token-limit key via completion_token_limit, api_key if resolved,
    # temperature only if supported, api_base=endpoint if set, HF router base URL for provider
    # "huggingface" without an endpoint), calls litellm.completion, and returns an LLMResponse with
    # the first choice's content, usage tokens and completion_cost (failure to cost is logged at
    # debug and yields 0.0). Raises whatever _resolve_api_key or litellm raises. Side effect:
    # network.
    """One non-fallback chat-completion call. Internal helper for `complete()`."""
    import litellm  # type: ignore[import-untyped]

    api_key = _resolve_api_key(spec)

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    kwargs: dict = {
        "model": spec.qualified_name,
        "messages": messages,
        "num_retries": 0,
        "timeout": spec.timeout_sec,
    }
    # Unknown-to-LiteLLM model IDs still need the current OpenAI wire contract.
    # Do not rely on the installed SDK's model catalog to rename this parameter.
    kwargs.update(completion_token_limit(spec, max_tokens))
    if api_key is not None:
        kwargs["api_key"] = api_key
    # Newer reasoning-focused models (Opus 4.7+, GPT-5+) reject `temperature`.
    if _supports_temperature(spec.model):
        kwargs["temperature"] = temperature
    if spec.endpoint:
        kwargs["api_base"] = spec.endpoint
    if response_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "repo2rlenv_response",
                "strict": True,
                "schema": response_schema,
            },
        }

    if spec.provider == "huggingface" and spec.endpoint is None:
        kwargs.setdefault("api_base", "https://router.huggingface.co/v1")

    response = litellm.completion(**kwargs)
    choice = response.choices[0]
    content = choice.message.content or ""

    usage_obj = getattr(response, "usage", None)
    prompt_tokens = 0
    completion_tokens = 0
    if usage_obj is not None:
        prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage_obj, "completion_tokens", 0) or 0

    cost_usd = 0.0
    try:
        cost_usd = float(litellm.completion_cost(completion_response=response))
    except Exception as exc:
        logger.debug("completion_cost failed for %s: %s", spec.qualified_name, exc)

    return LLMResponse(
        content=content,
        usage=dict(usage_obj) if usage_obj else None,
        cost_usd=cost_usd,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def complete(
    spec: LLMSpec,
    *,
    system: str | None = None,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    response_schema: dict | None = None,
    _depth: int = 0,
) -> LLMResponse:
    # [FUNCTION DOC] Public chat-completion call with automatic fallback.
    # Inputs: spec; keyword-only system (default None), user, max_tokens (default 1024), temperature
    # (default 0.7), response_schema (default None), _depth (internal recursion counter, default 0).
    # Calls _do_complete; on an exception that is failover-eligible, when spec.fallback is set and
    # _depth < 3, logs a warning and recursively calls itself with spec.fallback and _depth+1;
    # otherwise re-raises. Returns LLMResponse. Cost of failed attempts is not accumulated.
    """Single chat-completion call with automatic fallback on transient errors.

    Calls `spec.qualified_name`. On 5xx / 429 / network / timeout errors, if
    `spec.fallback` is set, retries with the fallback model recursively (up
    to 3 levels deep, then re-raises). 4xx errors (bad model, auth, etc.)
    are NOT retried — those signal config bugs, not transient failures.

    Honors `LLMSpec.endpoint` for self-hosted backends.
    """
    try:
        return _do_complete(
            spec,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=response_schema,
        )
    except Exception as exc:
        if _depth >= 3 or spec.fallback is None or not _is_failover_eligible(exc):
            raise
        logger.warning(
            "primary LLM %s failed with %s; falling back to %s",
            spec.qualified_name,
            type(exc).__name__,
            spec.fallback.qualified_name,
        )
        return complete(
            spec.fallback,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=response_schema,
            _depth=_depth + 1,
        )
