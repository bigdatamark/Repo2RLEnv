# [MODULE DOC] Mechanical, deterministic corrections around model output that never invent evidence
# or edit semantics: citation error messages, restoration of inline-code backticks in Markdown
# quotes, resolution of compressed or prefix-less JSON test-result quotes to the exact source bytes,
# de-duplication/renaming of proposed probes, and resolution of verifier edit paths that omitted a
# directory when exactly one existing file matches.
"""Mechanical corrections that preserve model-proposed checks and exact edits."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from repo2rlenv.quality.loop.artifacts import edit_path
from repo2rlenv.quality.loop.models import Citation, Repair, Review, SemanticProbe


def _citation_document_matches(citation: Citation, documents: dict[str, str]) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns sorted document paths whose whitespace-normalized text contains
    # the citation's whitespace-normalized quote (empty list for an empty quote).
    quote = " ".join(citation.quote.split())
    if not quote:
        return []
    return sorted(path for path, text in documents.items() if quote in " ".join(text.split()))


def citation_path_error(citation: Citation, documents: dict[str, str]) -> str:
    # [FUNCTION DOC] Returns an explanatory error string for an ungrounded citation: the cited path,
    # the quote (first 250 chars), how many and which supplied documents (up to 8) contain the quote
    # exactly, and instructions to copy a contiguous excerpt. No side effects.
    """Explain unsupported citation paths without guessing or adding evidence."""
    matches = _citation_document_matches(citation, documents)
    listed = matches[:8]
    omitted = (
        f" ({len(matches) - len(listed)} additional paths omitted)" if len(matches) > 8 else ""
    )
    return (
        f"Review citation is not grounded in supplied text: {citation.path}; "
        f"invalid quote={citation.quote[:250]!r}. "
        f"Exact quote matches {len(matches)} supplied document(s): {listed!r}{omitted}. "
        "Copy a short contiguous excerpt from the correct supplied path; "
        "preserve punctuation and serialization. No ellipses or paraphrases."
    )


def _inline_markdown_matches(document: str, quote: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Finds lines in a Markdown document that equal the quote once single
    # backticks are removed. Only for quotes of >=32 chars (also after removing backticks) without
    # newlines, backslashes or double backticks; skips fenced/indented code, lines with
    # double/escaped backticks, and lines with zero or an odd number of backticks. Returns the
    # original line substrings (with backticks) for each match.
    """Match literal words with only single-backtick formatting differences."""
    if len(quote) < 32 or any(value in quote for value in ("\n", "\r", "\\", "``")):
        return []
    requested = quote.replace("`", "")
    if len(requested.strip()) < 32:
        return []
    matches = []
    fence = None
    for line in document.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if marker:
            run, rest = marker.groups()
            if fence is None:
                fence = (run[0], len(run))
            elif run[0] == fence[0] and len(run) >= fence[1] and not rest.strip():
                fence = None
            continue
        if fence or line.startswith(("    ", "\t")) or "``" in line or "\\`" in line:
            continue
        if not line.count("`") or line.count("`") % 2:
            continue
        offsets = [index for index, char in enumerate(line) if char != "`"]
        plain = "".join(line[index] for index in offsets)
        for match in re.finditer(re.escape(requested), plain):
            matches.append(line[offsets[match.start()] : offsets[match.end() - 1] + 1])
    return matches


def resolve_markdown_citations(review: Review, documents: dict[str, str]) -> tuple[Review, list]:
    # [FUNCTION DOC] Returns (deep-copied review, change list) where each .md citation not already
    # grounded is replaced by its unique backtick-restored match (<=1000 chars) from the document.
    # Changes are {path, old, new}. The input review is not mutated.
    """Restore inline-code delimiters in unique Markdown prose excerpts.

    No word, case, punctuation or numeric changes are accepted. Fenced/indented
    code, escaped backticks and ambiguous matches retain strict validation.
    """
    updated = review.model_copy(deep=True)
    changes = []
    for item in [updated.task, updated.verifier, updated.leakage, *updated.issues, *updated.probes]:
        for citation in item.evidence:
            document = documents.get(citation.path, "")
            if not citation.path.endswith(".md") or " ".join(citation.quote.split()) in " ".join(
                document.split()
            ):
                continue
            matches = _inline_markdown_matches(document, citation.quote)
            if len(matches) == 1 and len(matches[0]) <= 1000:
                changes.append({"path": citation.path, "old": citation.quote, "new": matches[0]})
                citation.quote = matches[0]
    return updated, changes


def _matching_json_keys(node, requested: str, expected: str):
    # [FUNCTION DOC] HELPER: Generator over a decoded JSON tree yielding dict keys equal to
    # requested (or, for 'test_*' names, ending in '::name' or '.name') whose value serializes
    # (sorted keys) to expected.
    if isinstance(node, dict):
        for key, candidate in node.items():
            matches = key == requested or (
                requested.startswith("test_") and key.endswith(("::" + requested, "." + requested))
            )
            if matches and json.dumps(candidate, sort_keys=True) == expected:
                yield key
            yield from _matching_json_keys(candidate, requested, expected)
    elif isinstance(node, list):
        for candidate in node:
            yield from _matching_json_keys(candidate, requested, expected)


def resolve_json_citations(review: Review, documents: dict[str, str]) -> tuple[Review, list]:
    # [FUNCTION DOC] Returns (deep-copied review, change list) where each ungrounded .json citation
    # that parses as a single '"key": value' fragment is replaced by the exact source excerpt,
    # provided exactly one matching key with an equal value exists in the document and exactly one
    # raw occurrence (<=1000 chars) is found. Unparsable documents/fragments and ambiguous matches
    # are left unchanged.
    """Resolve unique structured test-result citations, never paraphrased prose.

    Reviewers sometimes omit a test's module prefix or compress JSON formatting.
    Require the same value and one matching key in a complete JSON document, then
    quote the actual source bytes. Ambiguous keys and different values still fail.
    """
    updated = review.model_copy(deep=True)
    citations = [
        citation
        for item in [
            updated.task,
            updated.verifier,
            updated.leakage,
            *updated.issues,
            *updated.probes,
        ]
        for citation in item.evidence
    ]
    changes = []
    for citation in citations:
        document = documents.get(citation.path, "")
        if not citation.path.endswith(".json") or " ".join(citation.quote.split()) in " ".join(
            document.split()
        ):
            continue
        try:
            decoded = json.loads(document)
            fragment = json.loads("{" + citation.quote + "}")
        except ValueError:
            continue
        if not isinstance(fragment, dict) or len(fragment) != 1:
            continue
        requested, value = next(iter(fragment.items()))
        keys = list(_matching_json_keys(decoded, requested, json.dumps(value, sort_keys=True)))
        if len(keys) != 1:
            continue
        excerpts = []
        pattern = r"(?<!\\)" + re.escape(json.dumps(keys[0])) + r"\s*:\s*"
        for match in re.finditer(pattern, document):
            try:
                candidate, length = json.JSONDecoder().raw_decode(document[match.end() :])
            except ValueError:
                continue
            if json.dumps(candidate, sort_keys=True) == json.dumps(value, sort_keys=True):
                excerpts.append(document[match.start() : match.end() + length])
        if len(excerpts) == 1 and len(excerpts[0]) <= 1000:
            changes.append({"path": citation.path, "old": citation.quote, "new": excerpts[0]})
            citation.quote = excerpts[0]
    return updated, changes


def distinct_probes(
    proposed: list[SemanticProbe], retained: list[SemanticProbe]
) -> list[SemanticProbe]:
    # [FUNCTION DOC] Returns proposed probes minus those whose (kind, focus, script) duplicates a
    # retained or earlier proposed probe; a name collision is resolved by truncating to 27 chars and
    # appending an 8-hex hash of the signature. Raises ValueError if the renamed name still
    # collides.
    """Keep existing controls; rename collisions without changing their scripts."""
    known = {probe.name: probe for probe in retained}
    signatures = {(probe.kind, probe.focus, probe.script) for probe in retained}
    result = []
    for probe in proposed:
        signature = (probe.kind, probe.focus, probe.script)
        if signature in signatures:
            continue
        name = probe.name
        if name in known:
            suffix = hashlib.sha256(repr(signature).encode()).hexdigest()[:8]
            name = f"{name[:27]}-{suffix}"
            if name in known:
                raise ValueError("Probe name remains ambiguous after deterministic renaming")
            probe = probe.model_copy(update={"name": name})
        result.append(probe)
        signatures.add(signature)
        known[name] = probe
    return result


def resolve_verifier_paths(task: Path, repair: Repair) -> Repair:
    # [FUNCTION DOC] Returns a copy of the repair where each replacement edit (nonempty old) under
    # tests/ whose path does not exist is redirected to the unique existing file under tests/ with
    # the same trailing path components that contains old exactly once. Other edits are unchanged;
    # may raise ValueError from edit_path for invalid paths.
    """Resolve an omitted directory only with one existing, exact verifier match.

    No fuzzy text replacement, new-file redirection, or source/oracle path
    resolution. Ambiguous requests are left for the normal correction protocol.
    """
    edits = []
    for edit in repair.edits:
        path = Path(edit_path(edit.path))
        if edit.old and path.parts[0] == "tests" and not (task / path).exists():
            suffix = Path(*path.parts[1:]).parts
            matches = []
            for candidate in (task / "tests").rglob(path.name):
                if (
                    not candidate.is_file()
                    or not candidate.resolve().is_relative_to((task / "tests").resolve())
                    or candidate.relative_to(task).parts[-len(suffix) :] != suffix
                ):
                    continue
                try:
                    text = candidate.read_text()
                except UnicodeDecodeError:
                    continue
                if text.count(edit.old) == 1:
                    matches.append(candidate.relative_to(task).as_posix())
            if len(matches) == 1:
                edit = edit.model_copy(update={"path": matches[0]})
        edits.append(edit)
    return repair.model_copy(update={"edits": edits})
