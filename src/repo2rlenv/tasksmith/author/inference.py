# [MODULE DOC] Fixed inference policy for tasksmith author model calls: output token ceiling
# (MAX_OUTPUT_TOKENS=12000), per-request timeout (MODEL_TIMEOUT_SEC=300s) and Anthropic
# thinking/effort options for models that support adaptive thinking.
from __future__ import annotations

# A design submission can include the complete private verifier plus model reasoning.
# This is an output ceiling; each request still reserves against the same stage budget.
MAX_OUTPUT_TOKENS = 12000
MODEL_TIMEOUT_SEC = 300


def anthropic_options(model: str) -> dict:
    # [FUNCTION DOC] Returns {"thinking": {"type": "adaptive"}, "output_config": {"effort":
    # "medium"}} when model is "anthropic/<name>" and name starts with one of the listed adaptive
    # Claude model prefixes; otherwise an empty dict. Pure.
    adaptive_models = (
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
    )
    if model.startswith("anthropic/") and model.split("/", 1)[1].startswith(adaptive_models):
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}
    return {}


def inference_settings(model: str, *, max_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    # [FUNCTION DOC] Returns a dict describing the inference policy for trace records: model,
    # max_tokens (default MAX_OUTPUT_TOKENS) plus anthropic_options(model). Pure.
    return {"model": model, "max_tokens": max_tokens, **anthropic_options(model)}
