# [MODULE DOC] Tests for tasksmith bootstrap resumption: tasksmith/matrix_runner cached CPU
# reports, the GPU batch stop-on-first-failure in tasksmith/cli._bootstrap, and
# bootstrap_inputs.apply_cpu_bootstrap validation of cached snapshot evidence (expiry, ref binding,
# provider) while preserving explicit option limits.
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from repo2rlenv.campaigns.budget import BudgetLedger
from repo2rlenv.tasksmith import matrix_runner


def test_cpu_cached_report_requires_same_runtime_and_terminated_worker(tmp_path, monkeypatch):
    # [FUNCTION DOC] Runs run_cpu_matrix with provision_worker stubbed to raise (first run must
    # fail), then writes a completed report and a terminated worker receipt and asserts the cached
    # report is returned; a running receipt raises ValueError 'Incomplete bootstrap'; a changed
    # runtime wheel hash raises 'inputs changed'. Inputs: tmp_path, monkeypatch. Side effects:
    # files and a budget ledger under tmp_path.
    config = tmp_path / "config.json"
    config.write_text(json.dumps([{"name": "peft", "ref": "a" * 40}]))
    output, campaign = tmp_path / "matrix/cpu", tmp_path / "campaign"
    BudgetLedger(campaign / "budget.sqlite3", limit_usd="50")
    monkeypatch.setattr(matrix_runner, "check_runtime_wheel", lambda _: "runtime-a")

    def offline(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Stub provision_worker that raises RuntimeError('fixture stops
        # before provider creation').
        raise RuntimeError("fixture stops before provider creation")

    monkeypatch.setattr(matrix_runner, "provision_worker", offline)
    with pytest.raises(RuntimeError, match="fixture stops"):
        matrix_runner.run_cpu_matrix(config, output, campaign, tmp_path / "runtime.whl")
    (output / "report.json").write_text(json.dumps({"status": "completed"}))
    (output / "workers").mkdir()
    receipt = output / "workers/worker.json"
    receipt.write_text(json.dumps({"state": "terminated"}))
    assert matrix_runner.run_cpu_matrix(config, output, campaign, tmp_path / "runtime.whl") == {
        "status": "completed"
    }
    receipt.write_text(json.dumps({"state": "running"}))
    with pytest.raises(ValueError, match="Incomplete bootstrap"):
        matrix_runner.run_cpu_matrix(config, output, campaign, tmp_path / "runtime.whl")
    receipt.write_text(json.dumps({"state": "terminated"}))
    monkeypatch.setattr(matrix_runner, "check_runtime_wheel", lambda _: "runtime-b")
    with pytest.raises(ValueError, match="inputs changed"):
        matrix_runner.run_cpu_matrix(config, output, campaign, tmp_path / "runtime.whl")


def test_gpu_batch_stops_after_first_failed_attempt(tmp_path, monkeypatch):
    # [FUNCTION DOC] Stubs run_gpu_repository to record names and report failure; asserts
    # _bootstrap on a two-repo GPU matrix returns exit code 1 after attempting only the first repo.
    from repo2rlenv.tasksmith.cli import _bootstrap

    config = tmp_path / "config.json"
    config.write_text(json.dumps([{"name": name, "ref": "a" * 40} for name in ["peft", "trl"]]))
    seen = []

    def failed(spec, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Stub run_gpu_repository; records spec.name and returns a failed
        # GPU result dict.
        seen.append(spec.name)
        return {"source": spec.model_dump(), "resource": "gpu", "status": "failed"}

    monkeypatch.setattr(matrix_runner, "run_gpu_repository", failed)
    args = SimpleNamespace(
        matrix=config,
        output=tmp_path / "output",
        campaign=tmp_path,
        resource="gpu",
        env_file=None,
        json=True,
    )
    assert _bootstrap(args) == 1
    assert seen == ["peft"]


@pytest.mark.parametrize("invalid", ["expired", "ref", "provider"])
def test_bootstrap_options_reject_unusable_cache_evidence(tmp_path, invalid):
    # [FUNCTION DOC] Parametrized over expired snapshot, mismatched hint ref, and daytona provider;
    # writes a CPU report and asserts apply_cpu_bootstrap raises ValueError matching 'expired',
    # 'bound' or 'Modal' respectively.
    from datetime import UTC, datetime, timedelta

    from repo2rlenv.tasksmith.bootstrap_inputs import apply_cpu_bootstrap
    from repo2rlenv.tasksmith.models import Options

    created = datetime.now(UTC) - timedelta(days=2) if invalid == "expired" else datetime.now(UTC)
    report = {
        "resource": "cpu",
        "status": "completed",
        "snapshot_id": "im-fixture",
        "snapshot_created_at": created.isoformat(),
        "snapshot_ttl_sec": 86400,
        "repositories": [
            {
                "status": "ready",
                "source": {"name": "peft", "ref": "a" * 40},
                "hint": {
                    "ref": ("b" if invalid == "ref" else "a") * 40,
                    "base_image": "python:3.12-slim",
                    "dependencies": ["pytest==8.4.2"],
                    "scope": "repository smoke only",
                },
            }
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    options = Options(provider="daytona" if invalid == "provider" else "modal")
    with pytest.raises(
        ValueError, match={"expired": "expired", "ref": "bound", "provider": "Modal"}[invalid]
    ):
        apply_cpu_bootstrap(options, path)


def test_bootstrap_options_keep_explicit_model_and_spend_limits(tmp_path):
    # [FUNCTION DOC] Asserts apply_cpu_bootstrap returns a new Options keeping max_spend_usd and
    # author_turns, setting worker_snapshot and a bootstrap hint keyed by the peft GitHub URL,
    # without mutating the input options.
    from datetime import UTC, datetime

    from repo2rlenv.tasksmith.bootstrap_inputs import apply_cpu_bootstrap
    from repo2rlenv.tasksmith.models import Options

    report = {
        "resource": "cpu",
        "status": "completed",
        "snapshot_id": "im-fixture",
        "snapshot_created_at": datetime.now(UTC).isoformat(),
        "snapshot_ttl_sec": 86400,
        "repositories": [
            {
                "status": "ready",
                "source": {"name": "peft", "ref": "a" * 40},
                "hint": {
                    "ref": "a" * 40,
                    "base_image": "python:3.12-slim",
                    "dependencies": ["pytest==8.4.2"],
                    "scope": "repository smoke only",
                },
            }
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    options = Options(max_spend_usd="7.00", author_turns=4)
    updated = apply_cpu_bootstrap(options, path)
    assert updated.max_spend_usd == "7.00" and updated.author_turns == 4
    assert updated.worker_snapshot == "im-fixture"
    assert updated.bootstrap_hints["https://github.com/huggingface/peft"].ref == "a" * 40
    assert options.worker_snapshot is None
