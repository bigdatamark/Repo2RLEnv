# [PACKAGE DOC] Remote execution package: provider workers (Modal, Daytona) implementing the
# RemoteWorker contract, worker lifecycle/receipts, a durable remote job supervisor, Harbor trial
# dispatch and recovery, offline Harbor environments, verifier probe agents, a Responses-API agent,
# owned-runtime wheel transfer and Python repository bootstrap/test helpers. Harbor stays the task
# execution contract; this package only moves work and evidence to and from remote workers. The
# package init itself defines nothing.
"""Remote generation workers; Harbor remains the task execution contract."""

from __future__ import annotations
