# [MODULE DOC] Tests for tasksmith.build_logs and worker build-failure handling: failed Docker builds keep
# complete private stdout/stderr plus a sha256 receipt and surface the causal exception even when
# buried mid-log; dependency-image cache hits skip building; timeouts keep partial streams and
# redact credentials/commands; oversized streams are bounded while retaining the middle exception;
# authenticated URLs are redacted; public image build failures are recorded before bootstrap
# raises. subprocess.run is mocked, so no Docker runs.
"""Build failures retain causal evidence without executing Docker or target code."""

from __future__ import annotations

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from repo2rlenv.execution.artifacts import unpack_evidence
from repo2rlenv.spec.recipe_options import PythonRepositoryProfile
from repo2rlenv.tasksmith import build_logs, worker

CAUSE = "ValueError: Asset exceeds download allowance: hf-internal-testing/dummy-gemma"


def failed_build_output():
    # [FUNCTION DOC] HELPER: Returns a synthetic, very long Docker build stderr string: many progress lines, a
    # traceback ending in CAUSE, then a 'failed to solve' line with a huge echoed inline command,
    # so CAUSE is not in the last 6000 chars.
    return (
        "#6 installing dependencies\n" * 600
        + "#7 0.21 Traceback (most recent call last):\n"
        + '  File "<string>", line 24, in fetch_assets\n'
        + "#7 0.22 "
        + CAUSE
        + '\nERROR: failed to solve: process "/bin/sh -c python -c '
        + "echoed_inline_source_" * 2000
        + '" did not complete successfully: exit code: 1\n'
    )


def mock_docker(monkeypatch, *, stderr="", returncode=0, present=False):
    # [FUNCTION DOC] HELPER: Patches worker.subprocess.run with a fake. 'docker build' returns the given
    # returncode with stdout 'build progress' and the given stderr; 'docker image inspect' with
    # --format returns 'sha256:image', without it returns 0 if present else 1; any other argv
    # fails an assertion. Inputs: monkeypatch; keyword stderr (''), returncode (0), present
    # (False). Output: the list that records (argv, kwargs) for each call.
    calls = []

    def run(argv, **kwargs):
        # [FUNCTION DOC] HELPER: the fake subprocess.run described in mock_docker; records each call and returns
        # a CompletedProcess.
        calls.append((argv, kwargs))
        if argv[:2] == ["docker", "build"]:
            return subprocess.CompletedProcess(argv, returncode, "build progress\n", stderr)
        assert argv[:3] == ["docker", "image", "inspect"]
        if "--format" in argv:
            return subprocess.CompletedProcess(argv, 0, "sha256:image\n", "")
        return subprocess.CompletedProcess(argv, 0 if present else 1, "", "")

    monkeypatch.setattr(worker.subprocess, "run", run)
    return calls


def test_failed_dependency_build_transfers_complete_private_logs_and_cause(tmp_path, monkeypatch):
    # [FUNCTION DOC] Runs worker.main for a 'dependencies' stage request with a failing build, unpacks the
    # evidence tarball and asserts stage-result.json is failed with CAUSE in error and in a
    # truncated (<6000 chars) log excerpt, full stdout/stderr files are kept, the log receipt
    # marks stderr complete with the correct sha256, and no dependency-cache.json is written. Sets
    # REPO2RLENV_REMOTE_WORKER and sys.argv.
    stderr = failed_build_output()
    assert CAUSE not in stderr[-6000:]  # The previous error policy lost this exception.
    mock_docker(monkeypatch, stderr=stderr, returncode=1)
    config = tmp_path / "request.json"
    config.write_text(
        json.dumps(
            {
                "stage": "dependencies",
                "hint": {
                    "ref": "a" * 40,
                    "base_image": "python:3.12-slim",
                    "dependencies": ["pytest==8.4.2"],
                    "scope": "Read-only fixture, no target execution.",
                },
            }
        )
    )
    output = tmp_path / "artifact"
    monkeypatch.setenv("REPO2RLENV_REMOTE_WORKER", "1")
    monkeypatch.setattr("sys.argv", ["worker", str(config), str(output)])
    worker.main()
    received = unpack_evidence(
        output.with_suffix(".tar.gz"), tmp_path / "received", root_name="artifact"
    )
    result = json.loads((received / "stage-result.json").read_text())
    assert result["status"] == "failed"
    assert CAUSE in result["error"]
    assert CAUSE in result["logs"]["dependency-build.stderr"]
    assert len(result["logs"]["dependency-build.stderr"]) < 6000
    assert (received / "dependency-build.stderr").read_text() == stderr
    assert (received / "dependency-build.stdout").read_text() == "build progress\n"
    receipt = json.loads((received / "dependency-build.logs.json").read_text())
    assert receipt["streams"]["stderr"]["complete"]
    assert receipt["streams"]["stderr"]["sha256"] == hashlib.sha256(stderr.encode()).hexdigest()
    assert not (received / "dependency-cache.json").exists()


@pytest.mark.parametrize("present", [False, True])
def test_successful_dependency_build_preserves_recipe_cache_and_commands(
    tmp_path, monkeypatch, present
):
    # [FUNCTION DOC] Parametrized on whether the image is already present. Asserts build_dependency_image's
    # cache_hit, image_id and key (sha256 of recipe + image id), that docker build runs only on a
    # miss with argv [docker build -t IMAGE context] and timeout 600, and that log files are
    # written only on a miss.
    calls = mock_docker(monkeypatch, stderr="build warning\n", present=present)
    result = worker.build_dependency_image("python:3.12-slim", ["pytest==8.4.2"], tmp_path)
    assert result["cache_hit"] is present
    assert result["image_id"] == "sha256:image"
    assert result["key"] == hashlib.sha256((result["recipe"] + "sha256:image").encode()).hexdigest()
    builds = [(argv, kwargs) for argv, kwargs in calls if argv[:2] == ["docker", "build"]]
    assert len(builds) == (0 if present else 1)
    if present:
        assert not (tmp_path / "dependency-build.logs.json").exists()
    else:
        assert builds[0][0] == [
            "docker",
            "build",
            "-t",
            result["image"],
            str(tmp_path / "dependency-context"),
        ]
        assert builds[0][1]["timeout"] == 600
        assert (tmp_path / "dependency-build.stdout").read_text() == "build progress\n"
        assert (tmp_path / "dependency-build.stderr").read_text() == "build warning\n"


def test_timeout_keeps_partial_streams_and_does_not_echo_command_or_credentials(
    tmp_path, monkeypatch
):
    # [FUNCTION DOC] Makes subprocess.run raise TimeoutExpired with partial bytes including a secret from an env
    # var. Expects worker.run to raise ValueError 'timed out after 600s' that contains CAUSE but
    # neither the command arguments nor the secret; the saved stdout is kept and stderr contains
    # [REDACTED] with the secret absent in all written files.
    secret = "fixture-model-api-key-should-not-be-recorded"
    monkeypatch.setenv("TEST_API_KEY", secret)
    stderr = (CAUSE + "\nAuthorization: Bearer " + secret + "\n").encode()

    def timed_out(argv, **kwargs):
        # [FUNCTION DOC] HELPER: fake subprocess.run that always raises TimeoutExpired with partial stdout/stderr
        # bytes.
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"], b"partial stdout", stderr)

    monkeypatch.setattr(worker.subprocess, "run", timed_out)
    with pytest.raises(ValueError, match="timed out after 600s") as failure:
        worker.run(
            ["docker", "build", "sensitive-command-argument"],
            timeout=600,
            build_log=tmp_path / "public-build",
        )
    assert CAUSE in str(failure.value)
    assert "sensitive-command-argument" not in str(failure.value)
    for artifact in tmp_path.iterdir():
        assert secret not in artifact.read_text()
    assert secret not in str(failure.value)
    assert (tmp_path / "public-build.stdout").read_text() == "partial stdout"
    assert "[REDACTED]" in (tmp_path / "public-build.stderr").read_text()


def test_oversized_stream_has_explicit_bound_and_retains_middle_exception(tmp_path, monkeypatch):
    # [FUNCTION DOC] Lowers build_logs.MAX_STREAM_BYTES to 16384 and saves a large stderr. Asserts the retained
    # file is within the bound, contains CAUSE and an 'omitted middle' marker, the returned detail
    # contains CAUSE, and the receipt is incomplete with redacted_bytes equal to the full size and
    # retained_bytes equal to the file size.
    monkeypatch.setattr(build_logs, "MAX_STREAM_BYTES", 16_384)
    text = "leading progress\n" * 3000 + CAUSE + "\n" + "trailing progress\n" * 3000
    detail = build_logs.save_build_logs(tmp_path / "dependency-build", "", text)
    retained = (tmp_path / "dependency-build.stderr").read_bytes()
    assert len(retained) <= 16_384
    assert CAUSE.encode() in retained
    assert CAUSE in detail
    assert b"omitted middle" in retained
    receipt = json.loads((tmp_path / "dependency-build.logs.json").read_text())["streams"]["stderr"]
    assert not receipt["complete"]
    assert receipt["redacted_bytes"] == len(text.encode())
    assert receipt["retained_bytes"] == len(retained)


def test_public_build_log_redacts_authenticated_urls(tmp_path):
    # [FUNCTION DOC] Saves stdout with a user:password URL and asserts the credential is absent from the detail
    # and replaced by [REDACTED]@host in the saved stdout file.
    raw = "Fetching https://build-user:credential-value@example.org/package.whl\n"
    detail = build_logs.save_build_logs(tmp_path / "public-build", raw, "")
    assert "credential-value" not in detail
    assert (tmp_path / "public-build.stdout").read_text() == (
        "Fetching https://[REDACTED]@example.org/package.whl\n"
    )


def test_public_image_failure_is_recorded_before_bootstrap_returns(tmp_path, monkeypatch):
    # [FUNCTION DOC] Stubs worker source materialization, dependency image, snapshot and test_image, and makes
    # the docker build fail. Expects worker.bootstrap to raise ValueError mentioning the asset
    # allowance cause, and asserts public-build.stderr contains CAUSE and public-build.logs.json
    # exists.
    base = tmp_path / "base"
    base.mkdir()
    monkeypatch.setattr(worker, "materialize_source", lambda *args: base)
    monkeypatch.setattr(worker, "dependency_image", lambda *args: {})
    monkeypatch.setattr(
        worker, "bootstrap_snapshot", lambda *args: (SimpleNamespace(image_digest="merged"), base)
    )
    monkeypatch.setattr(
        worker,
        "test_image",
        lambda *args: SimpleNamespace(
            returncode=0, passed=["native"], statuses={"native": "passed"}
        ),
    )
    mock_docker(monkeypatch, stderr=failed_build_output(), returncode=1)
    profile = SimpleNamespace(
        options=PythonRepositoryProfile(source_paths=["src"], test_paths=["tests"])
    )
    with pytest.raises(ValueError, match="Asset exceeds download allowance"):
        worker.bootstrap(
            {"repo": "https://github.com/org/repo", "head": "a" * 40},
            profile,
            tmp_path,
            checkout=tmp_path / "checkout",
        )
    assert CAUSE in (tmp_path / "public-build.stderr").read_text()
    assert (tmp_path / "public-build.logs.json").is_file()
