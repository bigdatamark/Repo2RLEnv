# [MODULE DOC] Docs build tool that generates the (gitignored) prompt reference pages under
# docs/pipelines/prompts/ by embedding the exact retained prompt templates, examples and the owned
# source files that assemble model requests, read straight from src/repo2rlenv/pipelines/recipes and
# related packages. It never imports or executes pipeline code: files are read as text (and
# terminal/runner.py is parsed with ast). Module constants: ASSEMBLY (recipe id -> call-site source
# files to embed), GUIDES (recipe id -> walkthrough page name), TERMINAL (recipes that share the
# common terminal builder). Run as a script (--check, --out) or via hooks.on_pre_build.
"""Build the ignored prompt reference from canonical repository sources."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / "src/repo2rlenv/pipelines/recipes"
OUTPUT = ROOT / "docs/pipelines/prompts"
SOURCE_URL = "https://github.com/huggingface/Repo2RLEnv/blob/main/"

# Full call-site code includes dynamic additions, user-message construction and schemas.
ASSEMBLY = {
    "frontiersmith": [
        "frontiersmith/prompts.py",
        "frontiersmith/models.py",
        "frontiersmith/export.py",
        "frontiersmith/pipeline.py",
    ],
    "codemidas": ["codemidas/models.py", "codemidas/pipeline.py", "codemidas/audit.py"],
    "swe_smith": ["swe_smith/issue.py"],
    "seta_seed2synth": ["seta_seed2synth/recipe.py"],
    "seta_evol": ["seta_evol/recipe.py"],
    "swe_gen": ["swe_gen/instruction.py"],
    "swe_flow": ["swe_flow/author.py"],
    "r2e": ["r2e/pipeline.py"],
    "tmax": ["tmax/recipe.py", "tmax/sampler.py"],
    "endless_terminals": ["endless_terminals/recipe.py", "endless_terminals/sampler.py"],
    "terminalworld": ["terminalworld/recipe.py", "terminalworld/materialize.py"],
    "cli_gym": ["cli_gym/models.py", "cli_gym/pipeline.py"],
    "dataarc": ["dataarc/recipe.py"],
    "swe_next": ["history/pipeline.py"],
    "r2e_gym": ["history/pipeline.py"],
    "scaler": ["scaler/families.py"],
}
GUIDES = {
    "frontiersmith": "frontiersmith",
    "codemidas": "codemidas",
    "swe_smith": "repo_mutate",
    "seta_seed2synth": "terminal_synth",
    "seta_evol": "task_evolve",
    "swe_gen": "pr_to_env",
    "swe_flow": "repo_reconstruct",
    "r2e": "r2e",
    "tmax": "tmax",
    "endless_terminals": "endless_terminals",
    "terminalworld": "terminalworld",
    "cli_gym": "env_repair",
    "dataarc": "dataarc",
    "swe_next": "swe_next",
    "r2e_gym": "r2e_gym",
    "scaler": "scaler",
}
TERMINAL = {"seta_seed2synth", "seta_evol", "tmax", "endless_terminals", "terminalworld", "dataarc"}


def block(path: Path, text: str | None = None, *, label: str | None = None) -> str:
    # [FUNCTION DOC] HELPER: Renders one source file as a collapsible Markdown section: "### title",
    # a link to the file on GitHub (SOURCE_URL + repo-relative path), the SHA-256 of the original
    # file bytes, then a <details> block containing the payload in a fenced code block. Inputs: path
    # (file under ROOT; relative_to(ROOT) raises ValueError otherwise); text (payload override,
    # default the file text); label (title override, default file name). Trailing whitespace on each
    # line is stripped. Language tag is python/json/text by suffix. Fence length is max(4, longest
    # all-backtick token + 1) so embedded Markdown fences do not terminate it. Output: Markdown
    # string. Side effects: file reads.
    # KNOWN ISSUE: only whitespace-separated tokens made purely of backticks are considered, so a
    # payload containing a 4+-backtick fence with an attached language tag (e.g. four backticks then
    # "python") is not counted and could close the 4-backtick fence early.
    payload = path.read_text() if text is None else text
    payload = "\n".join(line.rstrip() for line in payload.splitlines())
    relative = path.relative_to(ROOT).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    language = {".py": "python", ".json": "json"}.get(path.suffix, "text")
    # Native templates can themselves contain Markdown fences and HTML tags.
    fence = "`" * max(
        4, max((len(part) for part in payload.split() if set(part) == {"`"}), default=0) + 1
    )
    title = label or path.name
    body = f"{fence}{language}\n{payload.rstrip()}\n{fence}\n"
    return (
        f"### {title}\n\n"
        f"[Source: `{relative}`]({SOURCE_URL}{relative}) · SHA-256 `{digest}`\n\n"
        "Source hash covers the original file; trailing whitespace is omitted below.\n\n"
        f'<details class="example" markdown="1">\n<summary>Read {title}</summary>\n\n'
        + body
        + "\n</details>\n\n"
    )


def prompt_files(recipe: str) -> list[Path]:
    # [FUNCTION DOC] HELPER: Lists the retained prompt assets of a recipe directory: *prompt.md,
    # strategies/*.md, *demonstrations.json, *examples.json, strategies.json, plus
    # design/tests/review/ audit.md for codemidas (added without existence checks). Input: recipe
    # id. Output: sorted list of Paths.
    directory = RECIPES / recipe
    paths = list(directory.glob("*prompt.md")) + list(directory.glob("strategies/*.md"))
    if recipe == "codemidas":
        paths.extend(
            directory / name for name in ("design.md", "tests.md", "review.md", "audit.md")
        )
    for pattern in ("*demonstrations.json", "*examples.json", "strategies.json"):
        paths.extend(directory.glob(pattern))
    return sorted(paths)


def render(recipe: str, title: str) -> str:
    # [FUNCTION DOC] Builds the full prompt reference page for one recipe. Inputs: recipe id (key of
    # ASSEMBLY and GUIDES) and display title. Chooses an intro paragraph (special variants for
    # scaler, which makes zero LLM calls, and codemidas), adds a pointer to shared_terminal.md for
    # TERMINAL recipes and a demonstrations note for swe_flow, then embeds every prompt_files()
    # asset, tmax's DOMAIN_MODULES taxonomy excerpt, every ASSEMBLY source file and, for codemidas,
    # two extra agent modules. Output: Markdown text ending with exactly one newline. Raises:
    # KeyError for unknown recipe; file errors propagate.
    result = (
        f"# {title}: complete prompt reference\n\n"
        f"Read the [pipeline walkthrough](../{GUIDES[recipe]}.md) first. This reference "
        "contains the exact retained templates and the owned code that adds runtime "
        "instructions, substitutes variables, builds user messages and selects output schemas. "
        "Templates alone are not the final request.\n\n"
        "The configured `llm` is used at each model call; roles do not imply different models. "
        "Resolved requests are stored as `*.request.json` beside model receipts in the "
        "campaign, outside learner-visible bundles. See the "
        "[prompt and evidence guide](../prompt_reference.md).\n\n"
    )
    if recipe == "scaler":
        result = (
            f"# {title}: instruction construction\n\n"
            "This recipe makes **zero LLM calls**. Its generator and reference are supplied "
            "programs executed remotely. The code below shows exactly how the concrete "
            "learner instruction is assembled; it is not a system prompt sent to a model. "
            f"See the [walkthrough](../{GUIDES[recipe]}.md).\n\n"
        )
    if recipe == "codemidas":
        result = (
            f"# {title}: complete prompt reference\n\n"
            "Read the [pipeline walkthrough](../codemidas.md) first. The default author "
            "and four audit solvers use GPT-6 Luna; independent reviewers, the adversarial "
            "attempt and curriculum screening use GPT-6 Sol. Models are explicit, with "
            "no provider fallback. Templates below are combined with typed artifacts, "
            "remote observations and tool schemas at the shown call sites. Exact requests "
            "and Responses outputs are retained outside learner-visible bundles.\n\n"
        )
    if recipe in TERMINAL:
        result += (
            "Also read the [shared terminal additions and schemas](shared_terminal.md). "
            "They are part of the request where the walkthrough indicates the common builder.\n\n"
        )
    if recipe == "swe_flow":
        result += "The call site takes the **first two** demonstrations for each stage; full retained files are shown below.\n\n"
    if recipe != "scaler":
        result += "## Retained templates and examples\n\n"
    for path in prompt_files(recipe):
        result += block(path)
    if recipe == "tmax":
        path = RECIPES / "tmax/taxonomy.json"
        result += block(
            path,
            json.dumps(json.loads(path.read_text())["DOMAIN_MODULES"], indent=2),
            label="DOMAIN_MODULES substitutions",
        )
    if recipe == "scaler":
        result += "## Instruction assembly and instance contract\n\n"
    else:
        result += "## Request assembly and output contract\n\n"
        result += (
            "The source excerpts below are read-only documentation. Model calls return "
            "structured JSON; code in the response executes only in the remote stages "
            "shown in the walkthrough.\n\n"
        )
    for relative in ASSEMBLY[recipe]:
        result += block(RECIPES / relative)
    if recipe == "codemidas":
        for relative in ("execution/responses_agent.py", "tasksmith/author/openai_agent.py"):
            result += block(ROOT / "src/repo2rlenv" / relative)
    return result.rstrip() + "\n"


def shared() -> str:
    # [FUNCTION DOC] Builds shared_terminal.md: intro, the _MATERIALIZATION string constant and the
    # feedback_for/run_synthesis function sources extracted from recipes/terminal/runner.py via ast
    # (literal_eval / get_source_segment), followed by terminal/templates.py, terminal/draft.py,
    # quality/draft_review.py and campaigns/structured.py in full. Output: Markdown string.
    result = (
        "# Shared terminal prompts and output schemas\n\n"
        "SETA Seed2Synth and SETA Evol use the common builder. TMax and Endless "
        "Terminals additionally use separate template, initial-test and final-test calls. "
        "DataArc appends the common materialization contract to its own artifact prompt. "
        "TerminalWorld uses its own replay-informed materializer and shares the output "
        "schema, not the common builder request.\n\n"
    )
    path = RECIPES / "terminal/runner.py"
    source = path.read_text()
    nodes = ast.parse(source).body
    for node in nodes:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_MATERIALIZATION" for t in node.targets
        ):
            result += block(
                path, ast.literal_eval(node.value), label="Common materialization instructions"
            )
        elif isinstance(node, ast.FunctionDef) and node.name in {"feedback_for", "run_synthesis"}:
            result += block(path, ast.get_source_segment(source, node), label=node.name)
    for relative in ("terminal/templates.py", "terminal/draft.py"):
        result += block(RECIPES / relative)
    result += block(ROOT / "src/repo2rlenv/quality/draft_review.py")
    result += block(ROOT / "src/repo2rlenv/campaigns/structured.py")
    return result.rstrip() + "\n"


def quality_loop() -> str:
    # [FUNCTION DOC] Builds quality_loop.md: an intro plus the review/repair prompt templates and
    # the models, context, probe and runner modules of src/repo2rlenv/quality/loop, each via
    # block(). Output: Markdown string.
    root = ROOT / "src/repo2rlenv/quality/loop"
    result = (
        "# Harbor review and repair: complete prompt reference\n\n"
        "Read the [component walkthrough](../quality_loop.md) for execution, "
        "evidence and budget boundaries. These are the exact prompts, structured "
        "outputs and owned code that assembles evidence and decides when to repair.\n\n"
    )
    for relative in (
        "prompts/review.md",
        "prompts/repair.md",
        "models.py",
        "context.py",
        "probe_recovery.py",
        "probe_behavior.py",
        "runner.py",
    ):
        result += block(root / relative)
    return result.rstrip() + "\n"


def tasksmith() -> str:
    # [FUNCTION DOC] Builds tasksmith.md: an intro plus the investigate/design prompts and the
    # models, runner and author/artifact modules of src/repo2rlenv/tasksmith. Output: Markdown
    # string.
    root = ROOT / "src/repo2rlenv/tasksmith"
    result = "# Tasksmith: complete prompt reference\n\nRead the [pipeline walkthrough](../tasksmith.md) for the stage diagram, contracts and execution boundaries. These prompts and schemas are generated from the implementation.\n\n"
    for relative in (
        "prompts/investigate.md",
        "prompts/design.md",
        "models.py",
        "runner.py",
        "author/artifact.py",
    ):
        result += block(root / relative)
    return result.rstrip() + "\n"


def pages() -> dict[str, str]:
    # [FUNCTION DOC] Computes every expected page deterministically: reads recipes/catalog.json to
    # get each recipe's title, renders one page per ASSEMBLY recipe plus shared_terminal.md,
    # quality_loop.md and tasksmith.md. Output: dict file name -> Markdown text. Raises: KeyError if
    # a recipe is missing from the catalog.
    """Return deterministic pages without importing or executing pipeline code."""
    catalog = {r["id"]: r for r in json.loads((RECIPES / "catalog.json").read_text())}
    expected = {f"{recipe}.md": render(recipe, catalog[recipe]["title"]) for recipe in ASSEMBLY}
    expected["shared_terminal.md"] = shared()
    expected["quality_loop.md"] = quality_loop()
    expected["tasksmith.md"] = tasksmith()
    return expected


def generate(output: Path = OUTPUT, *, check: bool = False) -> int:
    # [FUNCTION DOC] Writes or checks the prompt pages. Inputs: output directory (default OUTPUT);
    # check flag (keyword-only). Computes which files are missing or differ. With check=True raises
    # ValueError if any differ, writing nothing; otherwise creates the directory and rewrites only
    # changed files. Output: total number of expected pages. Stale extra files in the directory are
    # not removed.
    expected = {output / name: text for name, text in pages().items()}
    changed = [
        path for path, text in expected.items() if not path.exists() or path.read_text() != text
    ]
    if check:
        if changed:
            raise ValueError(
                "Generated prompt docs differ; run python docs/_tools/generate_prompt_reference.py"
            )
        return len(expected)
    output.mkdir(parents=True, exist_ok=True)
    for path in changed:
        path.write_text(expected[path])
    return len(expected)


def main() -> None:
    # [FUNCTION DOC] CLI entry point: parses --check and --out, calls generate(), prints a count
    # summary to stdout. Raises: ValueError from generate in check mode.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check already generated references")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    count = generate(args.out, check=args.check)
    print(f"{count} prompt references {'checked' if args.check else 'generated'}")


if __name__ == "__main__":
    main()
