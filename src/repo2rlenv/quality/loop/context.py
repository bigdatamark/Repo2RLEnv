# [MODULE DOC] Bounded evidence packs for the LLM reviewer. EvidenceContext gathers a task's files
# and trial evidence into a character-budgeted set of documents with an inventory, records explicit
# omissions, serves bounded follow-up reads (line ranges or literal searches), serializes the pack
# as JSON, and verifies that every citation in a model Review quotes text actually present in the
# documents shown.
"""Bounded evidence packs, explicit omissions and verifiable model citations."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

from repo2rlenv.quality.loop.artifacts import digest
from repo2rlenv.quality.loop.models import ReadRequest, Review, TrialRecord
from repo2rlenv.quality.loop.protocol import citation_path_error
from repo2rlenv.quality.loop.rollout_evidence import rollout_documents


def _search_excerpts(lines: list[str], query: str, *, maximum: int | None = None) -> str:
    # [FUNCTION DOC] HELPER: Literal substring search over lines returning labelled excerpts. Each
    # hit yields a window from 5 lines before to 25 after (merged with the previous window when
    # overlapping and still <=3000 chars); when a window would exceed 3000 chars (e.g. one huge
    # minified line) it emits column slices of up to 3000 chars around each hit instead. Stops after
    # 8 windows and appends a note. Returns '[No literal matches found]' when nothing matches. If
    # maximum is given and the result is longer, drops trailing windows with an explicit omission
    # note; raises ValueError if not even one window fits.
    """Bound both match count and characters, including minified JSON traces."""
    excerpts = []
    ranges = []
    matched_windows = 0
    for index, line in enumerate(lines):
        match = line.find(query)
        if match < 0:
            continue
        start, end = max(0, index - 5), min(len(lines), index + 26)
        surrounding = "".join(lines[start:end])
        if len(surrounding) <= 3000:
            previous = ranges[-1] if ranges else None
            merged = "".join(lines[previous[0] : end]) if previous else ""
            if previous and start <= previous[1] and len(merged) <= 3000:
                start = previous[0]
                ranges[-1] = (start, end)
                excerpts[-1] = f"[Lines {start + 1}-{end}]\n" + merged
            else:
                excerpts.append(f"[Lines {start + 1}-{end}]\n" + surrounding)
                ranges.append((start, end))
            matched_windows += 1
        else:
            # One line can contain an entire trajectory. Retain literal bytes
            # around each hit and label omissions instead of expanding that line.
            while match >= 0 and matched_windows < 8:
                left = max(0, match - 1000)
                right = min(len(line), left + 3000)
                excerpts.append(
                    f"[Line {index + 1}, columns {left + 1}-{right}; "
                    "surrounding text omitted]\n" + line[left:right]
                )
                ranges.append(None)
                matched_windows += 1
                match = line.find(query, right)
        if matched_windows >= 8:
            break
    note = (
        "\n[Search limited to the first 8 matching windows; use a narrower literal query "
        "or line ranges to inspect further matches.]"
        if matched_windows >= 8
        else ""
    )
    result = ("\n".join(excerpts) or "[No literal matches found]") + note
    if maximum is None or len(result) <= maximum:
        return result
    # Keep complete literal windows. Existing documents and accepted quotations
    # are never shortened to make room; omissions in this new search are explicit.
    for count in range(len(excerpts) - 1, 0, -1):
        bounded = (
            "\n".join(excerpts[:count])
            + (
                f"\n[Context budget: {len(excerpts) - count} additional search window(s) omitted. "
                "Use a narrower literal query or explicit line ranges to inspect them.]"
            )
            + note
        )
        if len(bounded) <= maximum:
            return bounded
    raise ValueError(
        f"Additional reads exceeded context budget: {maximum} characters remain, "
        "insufficient for one complete search window. Request a narrower literal "
        "query or a short explicit line range; existing evidence is unchanged."
    )


class EvidenceContext:
    # [CLASS DOC] Character-bounded evidence pack over a task directory and its TrialRecords.
    # Attributes: documents (key -> text shown to the model), inventory (path/bytes/sha256 of every
    # readable item), omitted (reasons strings), verifier_paths (selected test files), plus private
    # maps of readable paths and in-memory texts.
    def __init__(self, task: Path, trials: list[TrialRecord], *, limit: int):
        # [FUNCTION DOC] Builds the initial pack. Inputs: task dir, trials, limit (total chars).
        # Inventories every task file, includes priority files (instruction, task.toml, Dockerfile,
        # solve.sh, test.sh, contract.json; 12000 chars each), derives instruction symbols and, from
        # tests/contract.json (expected_passes, test_paths), selected test files/symbols. Then
        # includes top-level tests/solution helpers, small whole selected test files (<=24000 bytes)
        # or symbol excerpts of Python files. Initial task share is 35% of limit when trials exist,
        # else 70%. For each trial: verifies the result.json sha256 (ValueError 'Trial result
        # changed since ingestion'), adds a summarized result.json (with exception message tail),
        # probe scripts and rollout_documents as in-memory texts, and selected
        # verifier/agent/artifact files by priority (baseline evidence deprioritized). Raises
        # ValueError if a trial file is a symlink or escapes the trial dir. Then includes the
        # collected trial candidates up to 70% of limit, tail-preserving. Reads files only.
        self.task, self.limit = task, limit
        self.documents: dict[str, str] = {}
        self.inventory: list[dict] = []
        self.omitted: list[str] = []
        self._paths: dict[str, Path] = {}
        self._texts: dict[str, str] = {}
        self._initial_limit = int(limit * (0.35 if trials else 0.7))
        for path in sorted(task.rglob("*")):
            if path.is_file():
                key = path.relative_to(task).as_posix()
                self._paths[key] = path
                self.inventory.append(
                    {"path": key, "bytes": path.stat().st_size, "sha256": digest(path)}
                )
        priority = [
            "instruction.md",
            "task.toml",
            "environment/Dockerfile",
            "solution/solve.sh",
            "tests/test.sh",
            "tests/contract.json",
        ]
        for key in priority:
            if key in self._paths:
                self._include(key, self._paths[key], maximum=12000, tail=key.endswith(".json"))
        instruction = (task / "instruction.md").read_text()
        symbols = set(re.findall(r"\bdef\s+([A-Za-z_]\w*)", instruction))
        symbols.update(re.findall(r"`(?:[A-Za-z_]\w*\.)*([A-Za-z_]\w*)\s*(?:\(|`)", instruction))
        test_symbols: dict[str, set[str]] = {}
        whole_test_files: set[str] = set()
        contract = self._paths.get("tests/contract.json")
        if contract is not None:
            try:
                data = json.loads(contract.read_text())
                methods = {
                    identity.rsplit("::", 1)[-1].split("[", 1)[0]
                    for identity in data.get("expected_passes", [])[:64]
                }
                for selector in data.get("test_paths", []):
                    path, *nodes = selector.split("::")
                    selected = {node.split("[", 1)[0] for node in nodes} or methods
                    test_symbols.setdefault("tests/source/" + path, set()).update(selected)
                    if not nodes and path.endswith(".py"):
                        whole_test_files.add("tests/source/" + path)
                for key in self._paths:
                    if (
                        key in test_symbols
                        or not key.startswith("tests/source/")
                        or not key.endswith(".py")
                    ):
                        continue
                    module = key.removeprefix("tests/source/")[:-3].replace("/", ".")
                    nodes = set()
                    for identity in data.get("expected_passes", [])[:64]:
                        if identity.startswith((module + ".", module + "::")):
                            nodes.update(identity[len(module) :].lstrip(".:").split("::"))
                    if nodes:
                        test_symbols[key] = nodes
            except (ValueError, AttributeError):
                pass
        self.verifier_paths = sorted(key for key in test_symbols if key in self._paths)
        # Selected assertions precede reference patches and generic grading
        # helpers, which can otherwise fill the entire task share. Every file
        # remains in the inventory, even when its initial excerpt does not fit.
        # Generic method names such as test_empty must never select unrelated
        # source functions/classes.
        helpers = {
            key
            for key in self._paths
            if key.startswith(("tests/", "solution/")) and key.count("/") == 1
        }
        ordered = sorted(
            self._paths.items(),
            key=lambda item: (
                0
                if item[0] in test_symbols
                else 1
                if item[0] in helpers
                else 2
                if item[0].startswith(("environment/", "solution/"))
                else 3,
                item[0],
            ),
        )
        for key, path in ordered:
            if key in self.documents:
                continue
            if key in helpers:
                self._include(key, path, maximum=12000, tail=key.endswith(".json"))
                continue
            if path.suffix == ".py" and path.stat().st_size < 2_000_000:
                if key in whole_test_files and path.stat().st_size <= 24000:
                    # Small selected suites include their fixture helpers and
                    # final assertions, avoiding needless reads of omitted tests.
                    # Larger or individually selected suites retain symbol excerpts.
                    self._include(key, path, maximum=24000)
                else:
                    self._include_symbols(key, path, test_symbols.get(key, symbols))
        # Reserve an execution share before reading large repository files. Collect
        # every trial first: a baseline's long test inventory must not crowd out a
        # later counterexample's actual assertion failure.
        candidates = []
        for index, trial in enumerate(trials):
            path = Path(trial.result)
            if digest(path) != trial.result_sha256:
                raise ValueError("Trial result changed since ingestion")
            prefix = f"evidence/{index}-{trial.role}/"
            summary = trial.model_dump(mode="json")
            native = json.loads(path.read_text())
            message = (native.get("exception_info") or {}).get("exception_message")
            if message:
                summary["exception_message_tail"] = str(message)[-6000:]

            if trial.probe:
                script_key = prefix + "probe-script.sh"
                script = summary["probe"].pop("script")
                summary["probe"]["script_path"] = script_key
                self._texts[script_key] = script
                self.inventory.append(
                    {
                        "path": script_key,
                        "bytes": len(script.encode()),
                        "sha256": hashlib.sha256(script.encode()).hexdigest(),
                    }
                )
                candidates.append((2, index, script_key, 8000))
            self.documents[prefix + "result.json"] = json.dumps(summary, indent=2)
            root = path.parent
            if trial.role == "rollout":
                for name, text in rollout_documents(task, root).items():
                    key = prefix + name
                    self._texts[key] = text
                    self.inventory.append(
                        {
                            "path": key,
                            "bytes": len(text.encode()),
                            "sha256": hashlib.sha256(text.encode()).hexdigest(),
                        }
                    )
                    candidates.append((1, index, key, 18000))
            # Failure logs precede trajectories, verbose test inventories and
            # captured source. All remain addressable through bounded read requests.
            selected = [
                (0, "verifier/stdout.txt", 6000),
                (0, "verifier/stderr.txt", 3000),
                (0, "exception.txt", 3000),
                (1, "verifier/test-stdout.txt", 4000),
                (1, "verifier/test-stderr.txt", 3000),
                (1, "agent/exit-code.txt", 100),
                (3, "agent/oracle.txt", 4000),
                (3, "agent/trajectory.json", 12000),
                (4, "verifier/result.json", 3000),
                (4, "verifier/results.xml", 4000),
            ]
            selected += (
                [
                    (5, item.relative_to(root).as_posix(), 3000)
                    for item in sorted((root / "artifacts").rglob("*"))
                ]
                if (root / "artifacts").is_dir()
                else []
            )
            for priority, relative, maximum in selected:
                candidate = root / relative
                if candidate.is_symlink() or not candidate.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Trial evidence cannot contain symlinks")
                if candidate.is_file():
                    key = prefix + relative
                    self._paths[key] = candidate
                    self.inventory.append(
                        {
                            "path": key,
                            "bytes": candidate.stat().st_size,
                            "sha256": digest(candidate),
                        }
                    )
                    # Expected baseline failures are less useful for diagnosis than
                    # oracle/probe/solver failures on the same revision.
                    candidates.append(
                        (priority + (6 if trial.role == "baseline" else 0), index, key, maximum)
                    )
        self._initial_limit = int(limit * 0.7)
        for _, _, key, maximum in sorted(candidates):
            if key in self._texts:
                self._include_text(key, self._texts[key], maximum=maximum, tail=True)
            else:
                self._include(key, self._paths[key], maximum=maximum, tail=True)

    def _include(self, key: str, path: Path, *, maximum: int, tail: bool = False):
        # [FUNCTION DOC] HELPER: Includes a file via _include_text unless it exceeds 2 MB or is not
        # decodable text, in which case it records an omission reason instead.
        if path.stat().st_size > 2_000_000:
            self.omitted.append(key + ": larger than 2 MB")
            return
        try:
            text = path.read_text()
        except UnicodeError:
            self.omitted.append(key + ": binary")
            return
        self._include_text(key, text, maximum=maximum, tail=tail)

    def _include_text(self, key: str, text: str, *, maximum: int, tail: bool):
        # [FUNCTION DOC] HELPER: Adds text under key bounded by min(maximum, remaining initial
        # budget). Records 'context budget' omission if fewer than 100 chars remain; when truncating
        # records a partial-text omission and keeps either head+tail halves (tail=True) or the head.
        # Leave room for model-requested source excerpts instead of filling the
        # initial prompt with unrelated repository documentation and test names.
        remaining = self._initial_limit - sum(len(value) for value in self.documents.values())
        maximum = min(maximum, remaining)
        if maximum < 100:
            self.omitted.append(key + ": context budget")
            return
        if len(text) > maximum:
            self.omitted.append(key + ": partial text; request line ranges if needed")
            text = (
                (text[: maximum // 2] + "\n[... omitted ...]\n" + text[-maximum // 2 :])
                if tail
                else text[:maximum]
            )
        self.documents[key] = text

    def _include_symbols(self, key: str, path: Path, symbols: set[str]):
        # [FUNCTION DOC] HELPER: For a Python file, extracts source of functions/classes whose names
        # match any symbol (len>=3) case- and underscore-insensitively (also 'XTests' classes for
        # symbol X), each up to 400 lines, stopping near 12000 chars, prefixed by the module header
        # (first 60 lines, <=4000 chars). Stores at most 16000 chars only if it fits the initial
        # budget and records a 'symbol excerpts only' omission. Silently ignores unparsable or
        # non-text files.
        if not symbols:
            return
        try:
            text = path.read_text()
            tree = ast.parse(text)
        except (UnicodeError, SyntaxError):
            return
        lines = text.splitlines(keepends=True)
        chunks = []
        covered = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and any(
                symbol.lower().replace("_", "") == node.name.lower().replace("_", "")
                or (
                    isinstance(node, ast.ClassDef)
                    and node.name.endswith("Tests")
                    and symbol.lower().replace("_", "") == node.name[:-5].lower().replace("_", "")
                )
                for symbol in symbols
                if len(symbol) >= 3
            ):
                if any(start <= node.lineno <= end for start, end in covered):
                    continue
                start = max(1, node.lineno - 1)
                end = min(node.end_lineno or node.lineno, start + 399)
                covered.append((start, end))
                chunks.append(f"[Lines {start}-{end}]\n" + "".join(lines[start - 1 : end]))
                if sum(map(len, chunks)) >= 12000:
                    break
        if chunks:
            # Repairs need the module's real imports and aliases as well as the
            # selected function/class. Keep the header bounded and explicit.
            header_end = min(60, len(lines))
            header = "".join(lines[:header_end])[:4000]
            chunks.insert(
                0, f"[Module header, lines 1-{header_end}, at most 4000 chars]\n" + header
            )
        excerpt = "\n".join(chunks)[:16000]
        if excerpt and sum(map(len, self.documents.values())) + len(excerpt) <= self._initial_limit:
            self.documents[key] = excerpt
            self.omitted.append(
                key + ": symbol excerpts only; request ranges/search for other code"
            )

    def add_text(self, key: str, text: str, *, maximum: int):
        # [FUNCTION DOC] Registers controller-supplied text under a new key: stores the full text
        # for later reads, adds an inventory entry, and shows it in documents up to min(maximum,
        # remaining total limit) with an end marker when truncated (recording an omission). Raises
        # ValueError if the key already exists.
        """Register controller context with bounded excerpts and a readable full source."""
        if key in self._paths or key in self._texts or key in self.documents:
            raise ValueError("Evidence document already exists")
        self._texts[key] = text
        self.inventory.append(
            {
                "path": key,
                "bytes": len(text.encode()),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
        available = max(0, min(maximum, self.limit - sum(map(len, self.documents.values()))))
        marker = (
            "\n[Excerpt ends; request a range or literal search in this document for omitted text.]"
        )
        if len(text) <= available:
            self.documents[key] = text
        else:
            if available > len(marker):
                self.documents[key] = text[: available - len(marker)] + marker
            self.omitted.append(
                key + ": controller context excerpt; full text available through reads"
            )

    def read_more(self, requests: list[ReadRequest]):
        # [FUNCTION DOC] Serves the model's ReadRequests, all-or-nothing. Each request names an
        # inventory path and either a literal query (1-200 chars, via _search_excerpts bounded by
        # remaining budget) or a start/end line range. Adds documents keyed '<path>:search=<q>' or
        # '<path>:L<a>-L<b>' (duplicates skipped). Raises ValueError for unknown paths (listing
        # exact basename matches), files over 2 MB, empty results, or exceeding the total limit; on
        # error documents are unchanged.
        additions = {}
        for request in requests:
            if request.path not in self._paths and request.path not in self._texts:
                matches = sorted(
                    (
                        key
                        for key in self._paths.keys() | self._texts.keys()
                        if Path(key).name == Path(request.path).name
                    ),
                    key=lambda key: (key not in self.verifier_paths, key),
                )
                guidance = (
                    " Exact basename matches: " + ", ".join(matches[:8]) + "."
                    if matches
                    else " Use a path from the supplied inventory or search its catalogue."
                )
                raise ValueError(
                    f"Requested file is not in evidence inventory: {request.path}." + guidance
                )
            if request.path in self._texts:
                text = self._texts[request.path]
            else:
                path = self._paths[request.path]
                if path.stat().st_size > 2_000_000:
                    raise ValueError("Requested file exceeds bounded text reader")
                text = path.read_text()
            lines = text.splitlines(keepends=True)
            suffix = (
                f"search={request.query}"
                if request.query is not None
                else f"L{request.start_line}-L{request.end_line}"
            )
            key = f"{request.path}:{suffix}"
            if key in self.documents or key in additions:
                continue
            remaining = (
                self.limit
                - sum(map(len, self.documents.values()))
                - sum(map(len, additions.values()))
            )
            if request.query is not None:
                if not request.query.strip() or len(request.query) > 200:
                    raise ValueError("Search query must have 1-200 characters")
                try:
                    text = _search_excerpts(lines, request.query, maximum=remaining)
                except ValueError as exc:
                    raise ValueError(f"{request.path}: {exc}") from exc
            else:
                text = "".join(lines[request.start_line - 1 : request.end_line])
            if not text:
                raise ValueError("Requested range is empty")
            if len(text) > remaining:
                raise ValueError(
                    f"Additional reads exceeded context budget: {key} requires {len(text)} "
                    f"characters but {remaining} remain. Request a shorter line range or "
                    "literal search; existing evidence is unchanged."
                )
            additions[key] = text
        updated = {**self.documents, **additions}
        if sum(map(len, updated.values())) > self.limit:
            raise ValueError("Additional reads exceeded context budget")
        self.documents = updated

    def _inventory_payload(self) -> dict:
        # [FUNCTION DOC] HELPER: Returns the inventory section of the payload: a flat list of
        # path/bytes if its JSON is <=30000 chars, else grouped by directory if that fits, else
        # stores a full JSONL catalogue as the in-memory document evidence/full-file-inventory.jsonl
        # and returns a summary (counts per directory, first 100). Raises ValueError if the task
        # already has that reserved path. Mutates the private text map in the last case.
        entries = [{"path": item["path"], "bytes": item["bytes"]} for item in self.inventory]
        if len(json.dumps(entries)) <= 30000:
            return {"inventory": entries}
        # Trial captures repeat long directory prefixes for hundreds of modules.
        # Group them losslessly; all full paths remain available to read_more.
        directories: dict[str, dict[str, int]] = {}
        for item in entries:
            prefix, _, name = item["path"].rpartition("/")
            directories.setdefault(prefix, {})[name] = item["bytes"]
        if len(json.dumps(directories)) > 30000:
            # Large repositories still need a bounded first review. Keep every
            # path addressable through a searchable catalogue instead of dropping
            # files or refusing the review before the model can request evidence.
            key = "evidence/full-file-inventory.jsonl"
            if key in self._paths:
                raise ValueError("Task uses the reserved quality inventory path")
            self._texts[key] = "\n".join(json.dumps(item) for item in entries) + "\n"
            counts = [
                {"directory": name, "files": len(items)} for name, items in directories.items()
            ]
            return {
                "inventory_document": key,
                "inventory_files": len(entries),
                "inventory_directories": counts[:100],
                "inventory_directories_omitted": max(0, len(counts) - 100),
                "inventory_format": (
                    "The complete JSONL catalogue is available through read requests: "
                    "one file path and byte count per line. Search it by filename or "
                    "request bounded line ranges, then read the required source file. "
                    "All original file paths remain directly readable."
                ),
            }
        return {
            "inventory_by_directory": directories,
            "inventory_format": (
                "Each directory maps filenames to byte counts. Read a file using "
                "directory/filename (or filename for the empty directory). "
                "This is the complete inventory, without omitted paths."
            ),
        }

    def payload(self, **extra) -> str:
        # [FUNCTION DOC] Serializes the pack as JSON (documents, selected verifier files up to 32,
        # inventory section, non-budget omissions, a budget-omission count, plus any extra keyword
        # fields) and returns it. Raises ValueError if longer than limit + 70000 chars.
        result = json.dumps(
            {
                "documents": self.documents,
                "selected_verifier_files": self.verifier_paths[:32],
                "selected_verifier_files_omitted": max(0, len(self.verifier_paths) - 32),
                # Hashes remain in the local evidence record. They add no useful
                # review context and repeat for every source copy and trial.
                **self._inventory_payload(),
                "omitted": [item for item in self.omitted if not item.endswith(": context budget")],
                "budget_omissions": {
                    "count": sum(item.endswith(": context budget") for item in self.omitted),
                    "note": "Inventory files absent from documents remain available through bounded reads.",
                },
                **extra,
            },
            ensure_ascii=False,
        )
        # Inventory and decision metadata have their own small allowance.
        if len(result) > self.limit + 70000:
            raise ValueError("Evidence pack exceeds context and inventory limits")
        return result

    def validate_review(self, review: Review):
        # [FUNCTION DOC] Checks every citation in a Review (three assessments, issues and probes):
        # the cited path must be a shown document and the quote must appear in it after whitespace
        # normalization. Raises ValueError with protocol.citation_path_error guidance on the first
        # failure.
        citations = [
            citation
            for name in ("task", "verifier", "leakage")
            for citation in getattr(review, name).evidence
        ]
        citations += [
            citation for item in [*review.issues, *review.probes] for citation in item.evidence
        ]
        for citation in citations:
            document = self.documents.get(citation.path, "")
            if not document or " ".join(citation.quote.split()) not in " ".join(document.split()):
                raise ValueError(citation_path_error(citation, self.documents))
