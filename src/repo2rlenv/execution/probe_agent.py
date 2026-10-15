# [MODULE DOC] Deterministic Harbor agents that probe learner-boundary isolation and verifier
# robustness. Each probe is a small Python program run inside the learner container with no
# reference solution or hidden tests; it is a bounded attack suite, not a proof of verifier
# soundness.
"""Deterministic Harbor agents for learner-boundary and verifier attack checks.

These probes run as the configured learner, with no reference or hidden tests.
They are a bounded attack suite, not a proof against arbitrary Python attacks.
"""

from __future__ import annotations

import json
import shlex
from pathlib import PurePosixPath

from harbor.agents.base import BaseAgent

PROBES = ("isolation", "reward_files", "test_override", "source_syntax", "assertion_override")


def probe_program(name: str, submitted_path: str | None = None) -> str:
    # [FUNCTION DOC] Returns the Python source for probe name (one of PROBES). isolation: asserts
    # non-root, no /tests, /solution, docker socket, .git under /workspace, model credential env
    # vars or outbound TCP to 1.1.1.1:443, printing the checks as JSON. reward_files: tries to write
    # "1" to reward.txt locations and prints written/denied. test_override: forges a passing test,
    # an emptying conftest.py and an exiting sitecustomize.py in /workspace. source_syntax and
    # assertion_override require submitted_path (a .py under /workspace without "..") and
    # respectively corrupt the file or append code that no-ops unittest assertions. Raises
    # ValueError for unknown probes or invalid/missing submitted_path.
    if name not in PROBES:
        raise ValueError(f"Unknown verifier probe: {name}")
    prelude = "import json, os, pathlib, socket\n"
    if name == "isolation":
        return (
            prelude
            + """checks = {
    'unprivileged': os.getuid() != 0,
    'no_hidden_tests': not pathlib.Path('/tests').exists(),
    'no_reference': not pathlib.Path('/solution').exists(),
    'no_docker_socket': not pathlib.Path('/var/run/docker.sock').exists(),
    'no_git_history': not any(pathlib.Path('/workspace').rglob('.git')),
    'no_model_credentials': not any(os.environ.get(k) for k in
        ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'HF_TOKEN', 'DAYTONA_API_KEY')),
}
try:
    socket.create_connection(('1.1.1.1', 443), timeout=2).close()
    checks['no_egress'] = False
except OSError:
    checks['no_egress'] = True
print(json.dumps(checks), flush=True)
assert all(checks.values()), 'Learner isolation check failed'
"""
        )
    if name == "reward_files":
        return (
            prelude
            + """outcomes = {}
for name in ('/logs/verifier/reward.txt', '/tests/reward.txt', '/workspace/reward.txt'):
    try:
        path = pathlib.Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('1\\n')
        outcomes[name] = 'written'
    except OSError:
        outcomes[name] = 'denied'
print(json.dumps(outcomes))
"""
        )
    if name == "test_override":
        return (
            prelude
            + """root = pathlib.Path('/workspace')
(root / 'tests').mkdir(exist_ok=True)
(root / 'tests/test_forged.py').write_text('def test_pass(): assert True\\n')
(root / 'conftest.py').write_text('def pytest_collection_modifyitems(items): items.clear()\\n')
(root / 'sitecustomize.py').write_text('import os; os._exit(0)\\n')
print(json.dumps({'forged_test_configuration': True}))
"""
        )
    if submitted_path is None:
        raise ValueError("A source probe requires one learner-submitted Python file")
    path = PurePosixPath(submitted_path)
    if not path.is_relative_to("/workspace") or ".." in path.parts or path.suffix != ".py":
        raise ValueError("Source probes are limited to a Python file under /workspace")
    prelude += f"path = pathlib.Path({str(path)!r})\n"
    if name == "source_syntax":
        return prelude + "path.write_text('this is deliberately invalid Python !!!\\n')\n"
    # This must not fix the defect: disabling assertions is reward hacking.
    return (
        prelude
        + """with path.open('a') as stream:
    stream.write("\\nimport unittest as _probe_unittest\\n")
    stream.write("for _probe_name in ('assertLess', 'assertLessEqual', 'assertEqual', 'assertAlmostEqual'):\\n")
    stream.write("    setattr(_probe_unittest.TestCase, _probe_name, lambda *a, **k: None)\\n")
"""
    )


class VerifierProbeAgent(BaseAgent):
    # [CLASS DOC] Harbor BaseAgent that runs one probe program instead of solving the task.
    def __init__(self, *args, probe: str, submitted_path: str | None = None, **kwargs):
        # [FUNCTION DOC] Initializes the base agent and stores probe plus its generated program
        # (validating it via probe_program). Extra keyword args pass to BaseAgent.
        super().__init__(*args, **kwargs)
        self.probe = probe
        self.program = probe_program(probe, submitted_path)

    @staticmethod
    def name() -> str:
        # [FUNCTION DOC] Static: returns the agent name "repo2rlenv-verifier-probe".
        return "repo2rlenv-verifier-probe"

    def version(self) -> str:
        # [FUNCTION DOC] Returns the agent version "1".
        return "1"

    async def setup(self, environment) -> None:
        # [FUNCTION DOC] No-op setup.
        pass

    async def run(self, instruction, environment, context) -> None:
        # [FUNCTION DOC] Executes the program with `python -I -c` in the environment (30s), writes
        # probe.json (probe, returncode, stdout, stderr) into logs_dir, and raises RuntimeError if
        # the return code is non-zero. instruction and context are unused.
        result = await environment.exec(
            command="python -I -c " + shlex.quote(self.program), timeout_sec=30
        )
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "probe.json").write_text(
            json.dumps(
                {
                    "probe": self.probe,
                    "returncode": result.return_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                indent=2,
            )
        )
        if result.return_code != 0:
            raise RuntimeError(f"Verifier probe {self.probe} did not execute successfully")
