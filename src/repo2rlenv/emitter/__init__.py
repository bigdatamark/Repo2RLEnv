# [PACKAGE DOC] Emitter package: writers that turn generated tasks into Harbor task directories. Re-
# exports HarborTask and write_harbor_task from emitter.harbor (the patch-based native task API).
# Sibling modules: bundle (general TaskBundle with content-hash identity and atomic emission, used
# by owned recipes) and evaluation (advisory evaluation label metadata).
"""Task emitters — currently only Harbor."""

from repo2rlenv.emitter.harbor import HarborTask, write_harbor_task

__all__ = ["HarborTask", "write_harbor_task"]
