# [MODULE DOC] Bound the evidence passed into model context while full private execution contracts
# stay on disk.
"""Bound model context while retaining full private execution contracts on disk."""

from __future__ import annotations

import json


def bounded_context(value, *, max_chars: int = 80_000) -> str:
    # [FUNCTION DOC] Serialize value as compact JSON evidence no longer than max_chars. Input: value
    # (JSON-compatible data); max_chars (keyword int, default 80000). Shrinking: strings over 4000
    # chars become {prefix (first 2500), suffix (last 1000), omitted_characters}; lists over 12
    # items become {total_items, sample (first 8 shrunk), omitted_items}; dicts keep their first 32
    # entries plus omitted_fields; nesting deeper than 8 becomes an omission marker; other scalars
    # pass through. If the JSON is still too long it is wrapped as {evidence_excerpt: first
    # max_chars//3 characters of the JSON text, omitted_characters}. Output: JSON string (non-ASCII
    # kept). Raises: ValueError if the result still exceeds max_chars (possible when escaping
    # expands the excerpt); TypeError for non-serializable values.
    """Provide bounded evidence samples with explicit omission counts, never a verifier contract."""

    def shrink(item, depth=0):
        # [FUNCTION DOC] HELPER: Recursively shrink one item per the rules of bounded_context.
        # Inputs: item; depth (int, default 0). Output: shrunk JSON-compatible value.
        if depth > 8:
            return {"omitted": "maximum evidence nesting"}
        if isinstance(item, str):
            if len(item) <= 4000:
                return item
            return {
                "prefix": item[:2500],
                "suffix": item[-1000:],
                "omitted_characters": len(item) - 3500,
            }
        if isinstance(item, list):
            if len(item) <= 12:
                return [shrink(entry, depth + 1) for entry in item]
            return {
                "total_items": len(item),
                "sample": [shrink(entry, depth + 1) for entry in item[:8]],
                "omitted_items": len(item) - 8,
            }
        if isinstance(item, dict):
            entries = list(item.items())
            result = {key: shrink(entry, depth + 1) for key, entry in entries[:32]}
            if len(entries) > 32:
                result["omitted_fields"] = len(entries) - 32
            return result
        return item

    text = json.dumps(shrink(value), ensure_ascii=False)
    if len(text) > max_chars:
        # The outer object remains valid JSON and explicitly identifies an excerpt.
        text = json.dumps(
            {
                "evidence_excerpt": text[: max_chars // 3],
                "omitted_characters": len(text) - max_chars // 3,
            },
            ensure_ascii=False,
        )
    if len(text) > max_chars:
        raise ValueError("Authoring evidence still exceeds its bounded context")
    return text
