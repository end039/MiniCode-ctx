"""Explore/search sub-agent.

An isolated, read-only agent the main agent can *dispatch* to do broad codebase
exploration without bloating its own context. Mirrors Claude Code's Task/Explore
agent: the sub-agent burns its own context reading many files, then returns only
a concise conclusion to the parent — a "context firewall".

This is the wired-up successor to the role-definition-only ``sub_agents.py``: it
actually runs a bounded agent loop (reusing :func:`run_agent_turn`) over an
isolated message list with a read-only tool registry, and returns the final text.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from minicode.tooling import ToolRegistry
from minicode.types import ModelAdapter

# How many model-call rounds the sub-agent may take before it must conclude.
# Each round can issue several tool calls, so this is generous in practice
# (~2-3 reads/round → well over a hundred file reads at the default). Tunable via
# MINI_CODE_SUBAGENT_MAX_STEPS — coding models have 256k+ context, so a thorough
# exploration should not be cut short the way the old default of 12 did.
DEFAULT_SUBAGENT_MAX_STEPS = 40


def _subagent_max_steps() -> int:
    raw = os.environ.get("MINI_CODE_SUBAGENT_MAX_STEPS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_SUBAGENT_MAX_STEPS


# run_agent_turn appends this fallback as the last assistant message when the
# step budget is exhausted. We detect it to force a clean wrap-up so the parent
# never receives the bare step-limit notice as the sub-agent's "conclusion".
_STEP_LIMIT_MARKER = "maximum tool step limit"

WRAPUP_NUDGE = (
    "You have reached your exploration budget and must STOP searching now. Do not "
    "call any more tools. Based on everything you have already read, write your "
    "<final> answer to the original question: what you found, the key files as "
    "path:line, and how the pieces connect."
)

EXPLORE_SYSTEM_PROMPT = (
    "You are an exploration sub-agent running in an ISOLATED context. Your ONLY "
    "job is to investigate the codebase and report back a concise, self-contained "
    "answer to the parent agent.\n"
    "Rules:\n"
    "- You are READ-ONLY: use grep_files, list_files, file_tree, read_file, and the "
    "code-intelligence tools. You cannot edit, write, or run commands.\n"
    "- Be efficient: search first to locate relevant files, then read only the parts "
    "that matter. Do not read whole large files when a grep + targeted read suffices.\n"
    "- You have NO user to ask; never wait for input. Investigate, then conclude.\n"
    "- As soon as you can answer the question well, STOP searching and conclude — "
    "do not keep reading for exhaustive completeness.\n"
    "- Finish with a <final> answer that stands on its own: state what you found, "
    "cite key files as path:line, and explain how the pieces connect. Keep it tight "
    "— the parent sees ONLY your final answer, not your intermediate reads."
)


# ---------------------------------------------------------------------------
# Live tracker (drives the TUI's bottom-bar sub-agent count)
# ---------------------------------------------------------------------------


@dataclass
class SubAgentRecord:
    """One in-flight sub-agent."""
    id: str
    agent_type: str
    task: str
    started_at: float = field(default_factory=time.time)
    status: str = "running"


class SubAgentTracker:
    """Thread-safe registry of in-flight sub-agents.

    Only *dispatched* sub-agents are registered here. The background-memory
    thread is never registered, so it is excluded from the count by construction
    (which is exactly what the TUI indicator wants).

    ``on_change`` is invoked (best-effort) on every start/finish so the UI can
    re-render the live count. It is safe to set it to a throttled ``rerender``.
    """

    def __init__(self, on_change: Callable[[], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, SubAgentRecord] = {}
        self.on_change = on_change

    def start(self, agent_type: str, task: str) -> str:
        agent_id = f"sub-{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._active[agent_id] = SubAgentRecord(
                id=agent_id, agent_type=agent_type, task=task
            )
        self._notify()
        return agent_id

    def finish(self, agent_id: str, status: str = "completed") -> None:
        with self._lock:
            self._active.pop(agent_id, None)
        self._notify()

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def active_records(self) -> list[SubAgentRecord]:
        with self._lock:
            return list(self._active.values())

    def _notify(self) -> None:
        if self.on_change is not None:
            try:
                self.on_change()
            except Exception:  # noqa: BLE001 — UI callback must never break a run
                pass


# ---------------------------------------------------------------------------
# Read-only tool registry for the sub-agent
# ---------------------------------------------------------------------------


def build_readonly_registry(cwd: str, runtime: dict | None = None) -> ToolRegistry:
    """A registry with only read/search tools — no edit/write/run, no nested
    dispatch (so sub-agents cannot recurse)."""
    # Lazy imports: importing the package ``minicode.tools`` here would re-enter
    # its ``__init__`` (which imports this module's dispatch tool). Import the
    # individual tool modules directly to avoid that cycle.
    from minicode.tools.read_file import read_file_tool
    from minicode.tools.list_files import list_files_tool
    from minicode.tools.grep_files import grep_files_tool
    from minicode.tools.file_tree import file_tree_tool
    from minicode.tools.code_nav import (
        find_symbols_tool,
        find_references_tool,
        get_ast_info_tool,
    )

    return ToolRegistry(
        [
            grep_files_tool,
            list_files_tool,
            file_tree_tool,
            read_file_tool,
            find_symbols_tool,
            find_references_tool,
            get_ast_info_tool,
        ]
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_explore_subagent(
    *,
    task: str,
    cwd: str,
    runtime: dict | None = None,
    permissions: Any = None,
    tracker: SubAgentTracker | None = None,
    agent_type: str = "explore",
    make_model: Callable[[ToolRegistry], ModelAdapter] | None = None,
    max_steps: int | None = None,
    container: str | None = None,
) -> str:
    """Run one isolated read-only exploration and return its final answer.

    The sub-agent gets a fresh message list and its own model adapter advertising
    only read-only tools, so it cannot touch the parent's conversation or files.
    ``make_model`` lets callers (and tests) inject a model; by default an
    :class:`AnthropicModelAdapter` is built from ``runtime``. ``max_steps`` of
    None resolves to :func:`_subagent_max_steps` (env-tunable, generous default).

    If the sub-agent exhausts its step budget, it is forced to write a final
    summary from what it has already read — the parent never receives the bare
    step-limit notice as a "conclusion".
    """
    # Lazy import to keep module import order simple (agent_loop never imports us).
    from minicode.agent_loop import run_agent_turn

    if max_steps is None:
        max_steps = _subagent_max_steps()

    sub_tools = build_readonly_registry(cwd, runtime)

    def _build_model(registry: ToolRegistry) -> ModelAdapter:
        if make_model is not None:
            return make_model(registry)
        from minicode.anthropic_adapter import AnthropicModelAdapter

        if runtime is None:
            raise RuntimeError("no runtime configured for the sub-agent model")
        return AnthropicModelAdapter(runtime, registry)

    sub_model = _build_model(sub_tools)

    sub_messages: list[dict[str, Any]] = [
        {"role": "system", "content": EXPLORE_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    def _last_assistant(messages: list[dict[str, Any]]) -> str:
        return next(
            (
                m.get("content", "")
                for m in reversed(messages)
                if m.get("role") == "assistant"
            ),
            "",
        )

    agent_id = tracker.start(agent_type, task) if tracker is not None else None
    try:
        result_messages = run_agent_turn(
            model=sub_model,
            tools=sub_tools,
            messages=sub_messages,
            cwd=cwd,
            permissions=permissions,
            max_steps=max_steps,
            tool_container=container,
        )

        final = _last_assistant(result_messages)

        # Budget exhausted before concluding → force a no-tool wrap-up so the
        # parent gets real findings instead of the step-limit fallback. An empty
        # tool registry makes the model answer with text (it cannot tool-call).
        if _STEP_LIMIT_MARKER in (final or ""):
            wrap_model = _build_model(ToolRegistry([]))
            wrap_messages = result_messages + [
                {"role": "user", "content": WRAPUP_NUDGE}
            ]
            wrap_out = run_agent_turn(
                model=wrap_model,
                tools=ToolRegistry([]),
                messages=wrap_messages,
                cwd=cwd,
                permissions=permissions,
                max_steps=2,
                tool_container=container,
            )
            wrap_final = _last_assistant(wrap_out)
            if wrap_final and _STEP_LIMIT_MARKER not in wrap_final:
                final = wrap_final
    finally:
        if tracker is not None and agent_id is not None:
            tracker.finish(agent_id)

    tool_calls = sum(
        1 for m in result_messages if m.get("role") == "assistant_tool_call"
    )
    final = (final or "").strip() or "(sub-agent produced no conclusion)"
    header = f"[explore sub-agent · {tool_calls} read/search step(s)]\n"
    return header + final
