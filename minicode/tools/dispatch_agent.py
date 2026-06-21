"""The ``dispatch_agent`` tool: the parent agent's handle to spawn an isolated
read-only exploration sub-agent (see :mod:`minicode.subagent`).

Intent recognition is model-driven, exactly like Claude Code's Task tool: this
tool is advertised to the main model with a description that explains *when* to
delegate. The model decides to call it; we just run the sub-agent and hand back
its concise summary as the tool result.
"""

from __future__ import annotations

from typing import Any, Callable

from minicode.subagent import SubAgentTracker, run_explore_subagent
from minicode.tooling import ToolDefinition, ToolResult
from minicode.types import ModelAdapter

DISPATCH_DESCRIPTION = (
    "Delegate an open-ended codebase exploration/search question to an isolated, "
    "read-only sub-agent. The sub-agent investigates on its own (grep/list/read "
    "across many files) and returns a SINGLE concise summary, keeping your own "
    "context clean. "
    "Use this INSTEAD of reading many files yourself when the user asks a broad "
    "'where/how/which-files' question, e.g. 'where is auth handled and how does it "
    "flow?', 'find everywhere X is used', 'how is the build configured?'. "
    "Do NOT use it for edits, for running commands, or when you already know the "
    "exact file to open — use the file tools directly in those cases. "
    "Pass a self-contained 'task' string; the sub-agent has no access to this "
    "conversation."
)

DISPATCH_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "A self-contained exploration question for the sub-agent.",
        },
        "agent_type": {
            "type": "string",
            "enum": ["explore"],
            "description": "Sub-agent kind (currently only 'explore').",
        },
    },
    "required": ["task"],
}


def create_dispatch_agent_tool(
    *,
    cwd: str,
    runtime: dict | None = None,
    tracker: SubAgentTracker | None = None,
    make_model: Callable[[Any], ModelAdapter] | None = None,
) -> ToolDefinition:
    """Build the dispatch tool, closing over the model runtime + live tracker."""

    def _validate(input_data: dict) -> dict:
        task = input_data.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task is required")
        return {
            "task": task.strip(),
            "agent_type": input_data.get("agent_type", "explore"),
        }

    def _run(parsed: dict, context) -> ToolResult:
        if runtime is None and make_model is None:
            return ToolResult(
                ok=False, output="dispatch_agent requires a configured model."
            )
        try:
            summary = run_explore_subagent(
                task=parsed["task"],
                cwd=context.cwd,
                runtime=runtime,
                permissions=context.permissions,
                tracker=tracker,
                agent_type=parsed["agent_type"],
                make_model=make_model,
            )
        except Exception as error:  # noqa: BLE001
            return ToolResult(ok=False, output=f"sub-agent failed: {error}")
        return ToolResult(ok=True, output=summary)

    return ToolDefinition(
        name="dispatch_agent",
        description=DISPATCH_DESCRIPTION,
        input_schema=DISPATCH_INPUT_SCHEMA,
        validator=_validate,
        run=_run,
    )
