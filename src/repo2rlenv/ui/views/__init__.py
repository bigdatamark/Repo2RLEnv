# [PACKAGE DOC] Package repo2rlenv.ui.views: long-running redrawing Live views; re-exports
# BootstrapView (bootstrap agent loop) and GenerationView (synthesis pipeline progress).
"""Long-running live views built on the UI primitives."""

from repo2rlenv.ui.views.bootstrap import BootstrapView
from repo2rlenv.ui.views.generation import GenerationView

__all__ = ["BootstrapView", "GenerationView"]
