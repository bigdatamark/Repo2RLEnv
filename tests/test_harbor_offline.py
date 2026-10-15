# [MODULE DOC] Tests for execution/harbor_offline.py:OfflineDockerEnvironment, the single-container
# Harbor environment for remote workers: no-network contract enforcement across phases, Harbor
# version pinning, refusal of local execution, and capability flags. Skipped when harbor is not
# installed.
from __future__ import annotations

import pytest
import yaml

pytest.importorskip("harbor")

from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths

from repo2rlenv.execution.harbor_offline import OfflineDockerEnvironment


@pytest.mark.parametrize(
    ("startup", "phase"),
    [("no-network", "no-network"), ("public", "no-network"), ("no-network", "public")],
)
def test_real_harbor_constructor_preserves_offline_contract(tmp_path, monkeypatch, startup, phase):
    # [FUNCTION DOC] Parametrized over startup/phase network modes. Sets
    # REPO2RLENV_REMOTE_WORKER=1, makes subprocess.run fail the test if called, writes a
    # Dockerfile, and constructs the environment with Harbor's real models. Any public mode must
    # raise ValueError 'every execution phase'; all-no-network must disable egress control and add
    # a compose overlay setting services.main.network_mode to none. Inputs: tmp_path, monkeypatch,
    # startup, phase. Side effects: env var, files under tmp_path.
    monkeypatch.setenv("REPO2RLENV_REMOTE_WORKER", "1")

    def no_local_process(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Replacement for subprocess.run that fails the test if any local
        # process is started.
        pytest.fail("Constructing an offline environment must not probe or start Docker")

    monkeypatch.setattr("subprocess.run", no_local_process)
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM python:3.12-slim\n")
    options = {
        "environment_dir": environment_dir,
        "environment_name": "offline-contract",
        "session_id": "offline-contract__env",
        "trial_paths": TrialPaths(trial_dir=tmp_path / "trial"),
        "task_env_config": EnvironmentConfig(),
        "network_policy": NetworkPolicy(network_mode=startup),
        "phase_network_policies": [NetworkPolicy(network_mode=phase)],
    }
    if startup != "no-network" or phase != "no-network":
        with pytest.raises(ValueError, match="every execution phase"):
            OfflineDockerEnvironment(**options)
    else:
        environment = OfflineDockerEnvironment(**options)
        assert not environment._enable_egress_control
        overlay = environment._docker_compose_paths[-1]
        assert yaml.safe_load(overlay.read_text())["services"]["main"]["network_mode"] == "none"


def test_untested_harbor_version_is_rejected_before_initialization(monkeypatch):
    # [FUNCTION DOC] Patches the module's version lookup to return 999.0.0 and asserts construction
    # raises RuntimeError 'tested Harbor'.
    monkeypatch.setenv("REPO2RLENV_REMOTE_WORKER", "1")
    monkeypatch.setattr("repo2rlenv.execution.harbor_offline.version", lambda _: "999.0.0")
    with pytest.raises(RuntimeError, match="tested Harbor"):
        OfflineDockerEnvironment()


def test_adapter_refuses_local_execution_before_touching_docker(monkeypatch):
    # [FUNCTION DOC] Unsets REPO2RLENV_REMOTE_WORKER and asserts construction raises RuntimeError
    # 'remote worker'.
    monkeypatch.delenv("REPO2RLENV_REMOTE_WORKER", raising=False)
    with pytest.raises(RuntimeError, match="remote worker"):
        OfflineDockerEnvironment()


@pytest.mark.parametrize("mode", ["public", "allowlist"])
def test_offline_policy_never_silently_weakens_a_requested_network_contract(mode):
    # [FUNCTION DOC] Parametrized over public/allowlist. On an uninitialized instance asserts
    # validate_network_policy_support rejects the mode with ValueError, accepts no-network, and
    # capabilities report disable_internet but no dynamic_network_policy.
    environment = object.__new__(OfflineDockerEnvironment)
    with pytest.raises(ValueError, match="every execution phase"):
        environment.validate_network_policy_support(NetworkPolicy(network_mode=mode))
    environment.validate_network_policy_support(NetworkPolicy(network_mode="no-network"))
    assert environment.capabilities.disable_internet
    assert not environment.capabilities.dynamic_network_policy
