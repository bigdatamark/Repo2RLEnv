# [MODULE DOC] Tests for campaigns.scheduling.available_task_slots with ActiveBatch: in-flight
# allocations, receipts vs. inventory, per-recipe independence and partial batch release.
# Appears to use a per-recipe/source target of 40 tasks (inferred from the assertions).
# KNOWN ISSUE: this module lacks 'from __future__ import annotations' (CLAUDE.md invariant 7).
from repo2rlenv.campaigns.scheduling import ActiveBatch, available_task_slots


def test_inflight_allocations_prevent_parallel_batches_exceeding_collection_target():
    # [FUNCTION DOC] With 60 existing tasks and a 25-slot active batch, a second batch gets 15
    # slots, and a third gets 0 once the second is active. Inputs: none. Output: none. Raises:
    # AssertionError on failure.
    inventory = {f"old-{i}": "old" for i in range(60)}
    active = [ActiveBatch("r2e", "first", 25, frozenset())]
    assert available_task_slots("r2e", "second", inventory, active) == 15
    active.append(ActiveBatch("r2e", "second", 15, frozenset()))
    assert available_task_slots("r2e", "third", inventory, active) == 0


def test_receipt_ahead_of_inventory_does_not_free_unaccounted_slots():
    # [FUNCTION DOC] Receipt ids already counted by an active batch do not free extra slots
    # when they later appear in inventory. Inputs: none. Output: none. Raises: AssertionError
    # on failure.
    inventory = {f"old-{i}": "old" for i in range(60)}
    active = [ActiveBatch("r2e", "first", 25, frozenset({"new-1", "new-2"}))]
    assert available_task_slots("r2e", "second", inventory, active) == 15
    inventory["new-1"] = "first"
    assert available_task_slots("r2e", "second", inventory, active) == 15


def test_source_cap_and_other_recipes_are_independent():
    # [FUNCTION DOC] A different recipe's active batch does not consume r2e slots; an r2e
    # source capped at 25 with 20 existing gives 5, then 0. Inputs: none. Output: none.
    # Raises: AssertionError on failure.
    inventory = {f"old-{i}": "first" for i in range(20)}
    active = [ActiveBatch("swe_gen", "first", 25, frozenset())]
    assert available_task_slots("r2e", "first", inventory, active) == 5
    active.append(ActiveBatch("r2e", "first", 5, frozenset()))
    assert available_task_slots("r2e", "first", inventory, active) == 0


def test_partial_finished_batch_releases_only_unfilled_slots():
    # [FUNCTION DOC] Tasks a batch already produced are counted once (via inventory), so a
    # partial batch only holds its unfilled slots. Inputs: none. Output: none. Raises:
    # AssertionError on failure.
    inventory = {f"old-{i}": "old" for i in range(60)}
    inventory.update({f"new-{i}": "first" for i in range(10)})
    active = [ActiveBatch("r2e", "first", 25, frozenset(f"new-{i}" for i in range(10)))]
    assert available_task_slots("r2e", "second", inventory, active) == 15
    assert available_task_slots("r2e", "second", inventory, []) == 25
