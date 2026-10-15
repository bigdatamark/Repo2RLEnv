# [MODULE DOC] Task-count allocation for parallel campaign batches, independent of spend limits:
# decides how many new tasks a recipe may still dispatch for a source given a global collection
# target, a per-source target and unfinished in-flight batches.
"""Task-count allocation for parallel campaign batches, separate from spend limits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveBatch:
    # [CLASS DOC] Frozen dataclass describing an in-flight generation batch. Fields: recipe, source
    # (source identifier), target (number of tasks the batch intends to produce, must be >= 1),
    # exported_ids (frozenset of task IDs it has exported so far).
    recipe: str
    source: str
    target: int
    exported_ids: frozenset[str]


def available_task_slots(
    recipe: str,
    source: str,
    inventory: dict[str, str],
    active: list[ActiveBatch],
    *,
    target: int = 100,
    source_target: int = 25,
) -> int:
    # [FUNCTION DOC] Returns how many more tasks may be dispatched for (recipe, source). Inputs:
    # recipe; source; inventory (mapping task_id -> source of already exported tasks, a single
    # snapshot); active (list of ActiveBatch); target (keyword, collection total, default 100);
    # source_target (keyword, per-source cap, default 25).
    # Only batches with the same recipe count. A batch's outstanding allocation is its target minus
    # the number of its exported IDs present in the inventory (floored at 0). Result = max(0,
    # min(target - len(inventory) - outstanding over all relevant batches, source_target - inventory
    # entries for this source - outstanding over relevant batches of this source)).
    # Raises ValueError if target, source_target or any batch target is < 1. Pure function; callers
    # must serialize allocation and dispatch. Note: the inventory is not filtered by recipe, so
    # exports of other recipes count against target.
    """Reserve unfinished batches against one inventory snapshot before dispatch.

    A run receipt can advance after the inventory was read. Only exports visible
    in that inventory reduce its outstanding allocation; otherwise a racing
    export could free a slot before being counted toward the collection target.
    The caller must serialize allocation and dispatch across campaign workers.
    """
    if target < 1 or source_target < 1 or any(batch.target < 1 for batch in active):
        raise ValueError("Task allocation targets must be positive")
    relevant = [batch for batch in active if batch.recipe == recipe]
    outstanding = sum(
        max(0, batch.target - len(batch.exported_ids.intersection(inventory))) for batch in relevant
    )
    source_outstanding = sum(
        max(0, batch.target - len(batch.exported_ids.intersection(inventory)))
        for batch in relevant
        if batch.source == source
    )
    return max(
        0,
        min(
            target - len(inventory) - outstanding,
            source_target
            - sum(value == source for value in inventory.values())
            - source_outstanding,
        ),
    )
