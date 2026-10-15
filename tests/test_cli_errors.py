# [MODULE DOC] Tests for the CLI error boundary in cli.main: failures across command families
# yield exit
# code 2, a JSON {error, message} record on stdout under --json, diagnostics on stderr
# (tracebacks only with --verbose), and logging never pollutes the JSON stream.
"""User-facing failures preserve parseable output across command families."""

from __future__ import annotations

import json
import logging

import pytest

from repo2rlenv.cli import main


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    # [FUNCTION DOC] Autouse fixture disabling cli._load_dotenv_if_present so a local .env
    # cannot affect tests. Inputs: monkeypatch.
    monkeypatch.setattr("repo2rlenv.cli._load_dotenv_if_present", lambda: None)


@pytest.mark.parametrize(
    "args",
    [
        ["generate", "--config", "missing.yaml"],
        ["tasksmith", "show", "missing.json"],
        ["quality", "show", "missing.json"],
        ["tasks", "show", "missing-task"],
        ["generate", "--pipeline-opt", "invalid"],
        ["generate", "--max-spend-usd", "invalid"],
        ["pipelines", "describe", "terminal_synth", "--recipe", "missing"],
        ["pipelines", "describe"],
    ],
)
def test_errors_are_json_records(args, capsys, tmp_path, monkeypatch):
    # [FUNCTION DOC] Parametrized bad invocations return 2 with exactly {error, message} JSON
    # on stdout and no traceback on stderr. Inputs: args, capsys, tmp_path (cwd), monkeypatch.
    monkeypatch.chdir(tmp_path)
    assert main([*args, "--json"]) == 2
    result = capsys.readouterr()
    assert set(json.loads(result.out)) == {"error", "message"}
    assert "Traceback" not in result.err


@pytest.mark.parametrize(
    "error", [ImportError("Install repo2rlenv[tasksmith]"), RuntimeError("Provider unavailable")]
)
@pytest.mark.parametrize("verbose", [False, True])
def test_dependency_and_provider_errors(monkeypatch, capsys, error, verbose):
    # [FUNCTION DOC] ImportError/RuntimeError raised by cmd_generate become JSON records; a
    # traceback appears on stderr only with --verbose. Inputs: monkeypatch, capsys, error,
    # verbose.
    def fail(args):
        # [FUNCTION DOC] HELPER: Replacement cmd_generate that raises the parametrized
        # error.
        raise error

    monkeypatch.setattr("repo2rlenv.cli.cmd_generate", fail)
    assert main((["--verbose"] if verbose else []) + ["generate", "--json"]) == 2
    result = capsys.readouterr()
    assert json.loads(result.out) == {"error": type(error).__name__, "message": str(error)}
    assert ("Traceback" in result.err) == verbose


def test_human_error_does_not_pollute_stdout(monkeypatch, capsys, tmp_path):
    # [FUNCTION DOC] Without --json, a missing config returns 2, leaves stdout empty and names
    # FileNotFoundError on stderr. Inputs: monkeypatch, capsys, tmp_path.
    monkeypatch.chdir(tmp_path)
    assert main(["generate", "--config", "missing.yaml"]) == 2
    result = capsys.readouterr()
    assert result.out == ""
    assert "FileNotFoundError" in result.err


def test_logs_stay_out_of_json_stream(monkeypatch, capsys):
    # [FUNCTION DOC] A warning logged before failure must not corrupt the JSON stdout record.
    # Inputs: monkeypatch, capsys.
    def fail(args):
        # [FUNCTION DOC] HELPER: Replacement cmd_generate that logs a warning then raises
        # RuntimeError.
        logging.getLogger("repo2rlenv").warning("Diagnostic before failure")
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr("repo2rlenv.cli.cmd_generate", fail)
    assert main(["generate", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "RuntimeError"


def test_invalid_llm_config_preserves_json_error_contract(tmp_path, capsys):
    # [FUNCTION DOC] A config with no llm block plus --llm-endpoint yields a JSON
    # ValidationError record with exit 2. Inputs: tmp_path (writes config.json), capsys.
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "repo": {"url": "example/project"},
                "pipeline": {"name": "pr_diff"},
                "output": {
                    "destination": str(tmp_path / "tasks"),
                    "org": "example",
                    "dataset_name": "test",
                },
            }
        )
    )
    assert (
        main(
            [
                "generate",
                "--config",
                str(config),
                "--llm-endpoint",
                "http://localhost:8000/v1",
                "--json",
            ]
        )
        == 2
    )
    output = capsys.readouterr()
    assert json.loads(output.out)["error"] == "ValidationError"
    assert "Traceback" not in output.err
