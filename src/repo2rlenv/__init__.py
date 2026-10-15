# [PACKAGE DOC] Top-level package for Repo2RLEnv (PyPI: repo2rlenv), which turns a repository into a
# verifiable RL dataset in Harbor task format. Exposes only __version__, read from installed package
# metadata via importlib.metadata; falls back to "0.0.0+unknown" when the distribution is not
# installed (PackageNotFoundError). No other side effects on import. Subpackages (pipelines,
# bootstrap, emitter, ui, ...) are imported explicitly by callers.
"""Repo2RLEnv — turn any repository into an RL environment for training and evaluation."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    # Single source of truth: read whatever pyproject.toml declared at install time.
    __version__ = _pkg_version("repo2rlenv")
except PackageNotFoundError:
    # Running from source without `pip install -e .` — rare, but don't crash.
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
