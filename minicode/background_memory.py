"""Background memory extraction for MiniCode.

A *background* (cheap) LLM continuously distills durable, reusable facts from the
conversation and writes them into the layered memory files managed by
:class:`minicode.memory.MemoryManager` (project-scope and user-scope).

Design constraints (per project owner):

* The background model only ever sees **low-signal, high-level** inputs:
  1. the existing memory files (so it can avoid duplicates and pick a scope),
  2. the user's questions,
  3. the main agent's per-turn *summary* output — i.e. the "what I did" text the
     user sees, **never** code, tool arguments, or raw tool output.
* Extra lightweight signals we add (metadata only, no large text — easy to turn
  off): the *names* of tools used in the turn and the project name. These are
  part of "what was done" and never leak code/output.

Extraction runs on a daemon worker thread so it never blocks the main agent.
Turns are buffered and processed in batches (default every 3 turns), plus a final
flush on :meth:`close`. Memory writes are guarded by a lock.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from minicode.logging_config import get_logger
from minicode.memory import MemoryManager, MemoryScope
from minicode.types import ModelAdapter

logger = get_logger("background_memory")

_SENTINEL = object()

DEFAULT_EXTRACT_EVERY_N_TURNS = 3
MAX_FACTS_PER_BATCH = 5
MAX_SUMMARY_CHARS = 1_500  # clip an over-long agent summary defensively

_CURATOR_SYSTEM = (
    "You are a background memory curator for a terminal coding agent. You distill "
    "durable, reusable facts from high-level conversation summaries and maintain "
    "long-term memory. You never see source code, tool arguments, or raw tool "
    "output — only short summaries of what the agent did."
)

_CURATOR_INSTRUCTION = """\
You maintain two memory files:
- PROJECT memory: facts specific to THIS codebase/project (architecture, conventions,
  decisions, where things live, recurring tasks).
- USER memory: durable facts about the USER and how they like to work, that apply
  ACROSS projects (preferences, workflow, tools, language).

## Existing memory
{existing_memory}

## Recent conversation (summaries only)
{recent_turns}

## Task
Extract at most {max_facts} NEW, durable facts worth remembering long-term. Rules:
- Skip anything ephemeral, one-off, or already covered by existing memory.
- Each fact: one concise factual sentence, self-contained, no code.
- Choose scope "project" or "user" deliberately.
- If nothing is worth saving, return an empty array.

Output ONLY a JSON array, no prose. Schema:
[{{"scope": "project"|"user", "category": "<short noun>", "content": "<one sentence>"}}]
"""


@dataclass
class TurnSummary:
    """One turn's low-signal inputs for the curator."""

    user_input: str
    agent_output: str
    tools_used: list[str] = field(default_factory=list)
    index: int = 0


class BackgroundMemoryExtractor:
    """Continuously extracts durable facts into layered memory, off the hot path."""

    def __init__(
        self,
        *,
        model_adapter: Optional[ModelAdapter],
        memory_manager: MemoryManager,
        workspace: str = ".",
        extract_every_n_turns: int = DEFAULT_EXTRACT_EVERY_N_TURNS,
        enabled: bool = True,
        project_name: Optional[str] = None,
        include_tool_names: bool = True,
        on_write: Optional[Callable[[list[dict[str, Any]]], None]] = None,
    ) -> None:
        self.model_adapter = model_adapter
        self.memory = memory_manager
        self.workspace = workspace
        self.extract_every_n_turns = max(1, extract_every_n_turns)
        self.enabled = enabled and model_adapter is not None
        self.project_name = project_name or _basename(workspace)
        self.include_tool_names = include_tool_names
        # Called (from the worker thread) with the list of facts just written.
        self.on_write = on_write

        self._pending: list[TurnSummary] = []
        self._turn_count = 0
        self._buf_lock = threading.Lock()   # guards _pending / _turn_count
        self._mem_lock = threading.Lock()   # guards memory writes
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None

        if self.enabled:
            self._worker = threading.Thread(
                target=self._run, name="bg-memory", daemon=True
            )
            self._worker.start()

    # -- public API ---------------------------------------------------------

    def record_turn(
        self,
        user_input: str,
        agent_output: str,
        tools_used: Optional[list[str]] = None,
    ) -> None:
        """Buffer one turn; may enqueue a batch for background extraction."""
        if not self.enabled:
            return
        if not (user_input or "").strip() or not (agent_output or "").strip():
            return
        summary = (agent_output or "").strip()
        if len(summary) > MAX_SUMMARY_CHARS:
            summary = summary[:MAX_SUMMARY_CHARS] + " …(truncated)"
        with self._buf_lock:
            self._turn_count += 1
            self._pending.append(
                TurnSummary(
                    user_input=(user_input or "").strip(),
                    agent_output=summary,
                    tools_used=list(dict.fromkeys(tools_used or [])),
                    index=self._turn_count,
                )
            )
            if len(self._pending) >= self.extract_every_n_turns:
                self._enqueue_pending_locked()

    def flush(self, *, block: bool = True, timeout: Optional[float] = None) -> None:
        """Enqueue any buffered turns; optionally wait for the queue to drain."""
        if not self.enabled:
            return
        with self._buf_lock:
            self._enqueue_pending_locked()
        if block:
            _join_with_timeout(self._queue, timeout)

    def close(self, timeout: float = 60.0) -> None:
        """Stop the worker, then flush any remaining turns *synchronously*.

        The final flush runs in the calling thread (not the daemon worker) so the
        last turns are guaranteed to land before the process exits — a daemon
        thread would otherwise be killed mid-write on a slow API call.
        """
        if not self.enabled:
            return
        # Let the worker finish anything already queued, then stop it.
        self._queue.put(_SENTINEL)
        if self._worker is not None:
            self._worker.join(timeout)
        # Process sub-threshold leftovers synchronously to guarantee a flush.
        with self._buf_lock:
            remaining = list(self._pending)
            self._pending.clear()
        if remaining:
            try:
                self._process(remaining)
            except Exception as error:  # noqa: BLE001 - best-effort final flush
                logger.warning("final memory flush failed: %s", error)

    # -- worker -------------------------------------------------------------

    def _enqueue_pending_locked(self) -> None:
        """Move buffered turns into a job. Caller must hold ``_buf_lock``."""
        if self._pending:
            self._queue.put(list(self._pending))
            self._pending.clear()

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is _SENTINEL:
                    break
                self._process(job)
            except Exception as error:  # noqa: BLE001 - background best-effort
                logger.warning("background memory extraction failed: %s", error)
            finally:
                self._queue.task_done()

    # -- extraction (synchronous; unit-testable) ----------------------------

    def _process(self, turns: list[TurnSummary]) -> list[dict[str, Any]]:
        """Run one extraction pass over *turns*; returns the facts written."""
        if self.model_adapter is None or not turns:
            return []
        prompt = self._build_prompt(turns)
        step = self.model_adapter.next(
            [
                {"role": "system", "content": _CURATOR_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        )
        facts = _parse_facts(getattr(step, "content", "") or "")
        return self._apply(facts)

    def _build_prompt(self, turns: list[TurnSummary]) -> str:
        return _CURATOR_INSTRUCTION.format(
            existing_memory=self._render_existing_memory(),
            recent_turns=self._render_turns(turns),
            max_facts=MAX_FACTS_PER_BATCH,
        )

    def _render_existing_memory(self) -> str:
        parts: list[str] = []
        with self._mem_lock:
            for scope in (MemoryScope.PROJECT, MemoryScope.USER):
                mem = self.memory.memories.get(scope)
                if mem and mem.entries:
                    parts.append(mem.format_as_markdown(include_header=True))
        return "\n\n".join(parts) if parts else "(memory is currently empty)"

    def _render_turns(self, turns: list[TurnSummary]) -> str:
        lines: list[str] = [f"(project: {self.project_name})"]
        for t in turns:
            lines.append(f"\n### Turn {t.index}")
            lines.append(f"User asked: {t.user_input}")
            lines.append(f"Agent did (summary): {t.agent_output}")
            if self.include_tool_names and t.tools_used:
                lines.append(f"Tools used: {', '.join(t.tools_used)}")
        return "\n".join(lines)

    def _apply(self, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        written: list[dict[str, Any]] = []
        for fact in facts[:MAX_FACTS_PER_BATCH]:
            content = str(fact.get("content", "")).strip()
            if not content:
                continue
            scope = (
                MemoryScope.USER
                if str(fact.get("scope", "")).lower() == "user"
                else MemoryScope.PROJECT
            )
            category = str(fact.get("category", "general")).strip().lower() or "general"
            with self._mem_lock:
                if self._is_duplicate(scope, content):
                    continue
                self.memory.add_entry(scope, category, content, tags=["auto"])
            written.append({"scope": scope.value, "category": category, "content": content})
            logger.info("memory[%s/%s] += %s", scope.value, category, content)
        if written and self.on_write is not None:
            try:
                self.on_write(written)
            except Exception as error:  # noqa: BLE001 - UI callback is best-effort
                logger.warning("memory on_write callback failed: %s", error)
        return written

    def _is_duplicate(self, scope: MemoryScope, content: str) -> bool:
        norm = _normalize(content)
        mem = self.memory.memories.get(scope)
        if not mem:
            return False
        for entry in mem.entries:
            existing = _normalize(entry.content)
            if existing == norm or existing in norm or norm in existing:
                return True
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _basename(path: str) -> str:
    cleaned = (path or "").rstrip("/\\")
    return cleaned.replace("\\", "/").rsplit("/", 1)[-1] or "project"


def _parse_facts(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array of fact objects from a model response, robustly."""
    if not text:
        return []
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    blob = text[start : end + 1]
    try:
        data = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    facts: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict) and str(item.get("content", "")).strip():
            facts.append(item)
    return facts


def _join_with_timeout(q: "queue.Queue[Any]", timeout: Optional[float]) -> None:
    """Best-effort ``Queue.join`` with an optional timeout."""
    if timeout is None:
        q.join()
        return
    deadline = time.time() + timeout
    # Queue has no timed join; poll unfinished_tasks under its mutex.
    with q.all_tasks_done:  # type: ignore[attr-defined]
        while q.unfinished_tasks:  # type: ignore[attr-defined]
            remaining = deadline - time.time()
            if remaining <= 0:
                return
            q.all_tasks_done.wait(remaining)  # type: ignore[attr-defined]
