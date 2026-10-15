# [MODULE DOC] Harbor 0.22.0 Docker environment adapter for permanently offline single-container
# tasks, used inside remote workers because Modal's VM kernel lacks nft_fib needed by Harbor's
# dynamic firewall. Adds an offline compose overlay (offline-compose.yaml) instead of an egress
# sidecar.
"""Harbor 0.22.0 adapter for permanently offline, single-container tasks.

Modal's VM kernel lacks nft_fib support required by Harbor's dynamic firewall.
These recipes need no runtime network transitions: Docker's isolated network
namespace provides the narrower contract directly, including UDP and ICMP.
"""

from __future__ import annotations

import os
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode, NetworkPolicy


class OfflineDockerEnvironment(DockerEnvironment):
    # [CLASS DOC] DockerEnvironment subclass that only permits no-network Linux Dockerfile tasks.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Requires env REPO2RLENV_REMOTE_WORKER == "1" and harbor version 0.22.0
        # (RuntimeError otherwise), initializes the parent, then raises ValueError for Windows
        # containers, task compose files or extra compose paths.
        if os.environ.get("REPO2RLENV_REMOTE_WORKER") != "1":
            raise RuntimeError("The owned Docker adapter runs inside a remote worker only")
        if version("harbor") != "0.22.0":
            raise RuntimeError("Offline adapter requires the tested Harbor 0.22.0 contract")
        super().__init__(*args, **kwargs)
        if (
            self._is_windows_container
            or self._environment_docker_compose_path.exists()
            or self.extra_docker_compose_paths
        ):
            raise ValueError(
                "Offline adapter supports Linux Dockerfile tasks without extra services"
            )

    @staticmethod
    def _requires_egress_control(*, startup_network_policy, phase_network_policies):
        # [FUNCTION DOC] HELPER: Static override that always returns False so no egress-control
        # sidecar is started; policy validation rejects modes that would need one.
        # Policy validation below rejects every mode requiring a sidecar. The
        # namespace policy is added last to the generated compose configuration.
        return False

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # [FUNCTION DOC] Property: EnvironmentCapabilities with disable_internet and mounted set to
        # True.
        return EnvironmentCapabilities(disable_internet=True, mounted=True)

    def validate_network_policy_support(self, network_policy: NetworkPolicy | None = None) -> None:
        # [FUNCTION DOC] Raises ValueError unless the given (or default) network policy is no-
        # network with no allowed hosts.
        policy = network_policy or self.network_policy
        if policy.network_mode != NetworkMode.NO_NETWORK or policy.allowed_hosts:
            raise ValueError("Offline adapter requires no-network in every execution phase")

    @property
    def _docker_compose_paths(self) -> list[Path]:
        # [FUNCTION DOC] HELPER: Property returning the parent's compose paths plus the packaged
        # offline-compose.yaml overlay (added last).
        return [
            *super()._docker_compose_paths,
            Path(str(files(__package__).joinpath("offline-compose.yaml"))),
        ]
