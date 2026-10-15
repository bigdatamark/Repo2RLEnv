# [MODULE DOC] Initial-state preflight for terminal tasks: before checking that a reference solution
# completes a task, verify that a separately generated set of initial-state tests passes on the
# fresh, untouched task image (using Harbor's "nop" agent). Returns repair feedback for the author
# when it fails.
"""Verify separately generated initial-state tests on a fresh task before checking completion."""

from __future__ import annotations

import ast

from repo2rlenv.execution.harbor import run_trial
from repo2rlenv.pipelines.recipes.terminal.draft import TerminalDraft, emit_draft
from repo2rlenv.pipelines.recipes.terminal.runner import feedback_for


def initial_state(
    *,
    worker,
    draft,
    design,
    candidate,
    name,
    recipe,
    lineage,
    org,
    timeout_sec,
    resume,
    python,
    trial_id,
    attempt,
    agent_user,
):
    # [FUNCTION DOC] Check that design.initial_tests pass on the unmodified task environment. Inputs
    # (all keyword-only): worker (execution worker for run_trial); draft (TerminalDraft); design
    # (object with initial_tests pytest source); candidate (Path working dir); name, org, recipe,
    # lineage, timeout_sec, resume, agent_user (forwarded to emit_draft); python, trial_id
    # (forwarded to run_trial); attempt (int, used to name initial-<attempt> and initial-
    # trial-<attempt> dirs). Behavior: collects top-level test_ function names from initial_tests,
    # builds a copy of the draft with those tests and equal weights 1/n, validates it as
    # TerminalDraft, emits it as task "<name>-initial", and runs a trial with agent "nop". Output:
    # None when the trial completed with reward 1; otherwise the feedback dict from
    # feedback_for(result, agent="oracle") with agent set to "initial_state" and a required_repair
    # message about repairing setup fixtures. Raises: pydantic ValidationError if the test count is
    # outside 5-10 or contract is violated; ZeroDivisionError if initial_tests has no test
    # functions; errors from emit_draft/run_trial. KNOWN ISSUE: zero initial tests causes
    # ZeroDivisionError (1/len(tests)) rather than a feedback dict, and fewer than 5 or more than 10
    # initial tests raise ValidationError because TerminalDraft requires 5-10 weights.
    tests = [
        node.name
        for node in ast.parse(design.initial_tests).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    data = draft.model_dump()
    data.update(
        tests_python=design.initial_tests,
        weights=[{"name": name, "weight": 1 / len(tests)} for name in tests],
    )
    check = TerminalDraft.model_validate(data)
    task = emit_draft(
        check,
        candidate / f"initial-{attempt}",
        name=name + "-initial",
        org=org,
        recipe=recipe,
        lineage=lineage,
        timeout_sec=timeout_sec,
        resume=resume,
        agent_user=agent_user,
    )
    result = run_trial(
        worker,
        task,
        candidate / f"initial-trial-{attempt}",
        trial_id=trial_id,
        agent="nop",
        python=python,
        resume=resume,
    )
    if result.completed and result.reward == 1:
        return None
    feedback = feedback_for(result, agent="oracle")
    feedback.update(
        agent="initial_state",
        required_repair="The fresh image does not satisfy the initial-state tests. "
        "Repair the setup fixtures; the solver must begin with the inputs described in the task.",
    )
    return feedback
