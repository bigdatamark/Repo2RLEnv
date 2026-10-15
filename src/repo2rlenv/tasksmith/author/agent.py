# [MODULE DOC] Author-agent dispatch and the shared shell tool schema. SHELL_TOOL is the function-
# tool definition for running a bounded command (timeout 1-120s) in the remote builder sandbox,
# never on the controller. run_agent routes to the OpenAI runtime or an external agent engine.
"""Only explicit controller tools are available to the author runtime."""

from repo2rlenv.tasksmith.author.external_agent import run_external_agent

SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Run a bounded command in the remote builder sandbox. Never runs on the controller. Inspect repository files and existing tests; do not modify the pinned checkout.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_sec": {"type": "integer", "minimum": 1, "maximum": 120},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(*, runtime, **kwargs):
    # [FUNCTION DOC] Async dispatcher: when runtime == "openai", lazily imports and awaits
    # run_openai_agent with kwargs; otherwise awaits run_external_agent(engine=runtime, **kwargs).
    # Keyword-only inputs; kwargs are the runtime's arguments (model, system, prompt, budget, tools,
    # handlers, trace, max_turns, max_cost). Output: whatever the runtime returns. Exceptions from
    # the runtime propagate.
    if runtime == "openai":
        from repo2rlenv.tasksmith.author.openai_agent import run_openai_agent

        return await run_openai_agent(**kwargs)
    return await run_external_agent(engine=runtime, **kwargs)
