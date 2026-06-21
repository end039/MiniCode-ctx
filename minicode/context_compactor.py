"""Layered, intra-turn context compaction for MiniCode.

This module is modeled on the TypeScript MiniCode reference (``src/compact/*``)
and is intended to be called **before every model call inside the agent loop**,
so that a single long turn whose context balloons can be compacted *mid-turn* —
not only at turn boundaries.

Tiers (selected by context-window utilization):

1. ``microcompact`` (>= 0.50): cheap, no-LLM. Clears the *bodies* of OLD tool
   results (keeping the most recent few). Reversible — originals are archived in
   ``self._cleared`` and can be restored with :meth:`restore_cleared`.
2. ``auto_compact`` (>= 0.85): LLM summarization. Folds the *previous summary*
   plus the older conversation into a fresh summary, while keeping the most
   recent ``MAX_KEEP_TOKENS`` of conversation verbatim. The replaced span is
   archived in ``self.archive`` so the turn can be restored ("rewind").

Mapping to the three design points discussed in roadmap.md (Q3):

* (a) Compaction runs *before* the LLM answers — :meth:`before_model_call` is
  invoked at the top of each agent-loop step, ahead of ``model.next(...)``.
* (b) The cheap tier runs on *every* step (true intra-turn compaction). The
  expensive LLM tier may also run intra-turn, but is rate-limited by a
  re-growth gate (:data:`AUTOCOMPACT_REGROWTH_FACTOR`) so we do not summarize on
  every step.
* (c) We always keep the most recent N tokens verbatim and summarize
  ``[previous summary + older messages] -> new summary``, archiving the replaced
  span so the original context can be restored.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from minicode.context_manager import (
    DEFAULT_CONTEXT_WINDOWS,
    estimate_message_tokens,
    estimate_messages_tokens,
)
from minicode.logging_config import get_logger
from minicode.types import ChatMessage, ModelAdapter

logger = get_logger("context_compactor")


# ---------------------------------------------------------------------------
# Thresholds (mirror TS src/compact/constants.ts)
# ---------------------------------------------------------------------------

MICROCOMPACT_UTILIZATION = 0.50   # clear old tool-result bodies above this
AUTOCOMPACT_UTILIZATION = 0.85    # run the LLM summarizer above this
BLOCKED_UTILIZATION = 0.95        # informational: context is critically full

KEEP_RECENT_TOOL_RESULTS = 3      # microcompact keeps this many recent results
MIN_KEEP_MESSAGES = 6             # auto-compact always keeps at least this many
FORCE_KEEP_MESSAGES = 2           # manual /compact keeps only this many recent
MIN_KEEP_TOKENS = 10_000          # informational floor for the kept window
MAX_KEEP_TOKENS = 40_000          # auto-compact keeps roughly this many recent tokens

MAX_AUTOCOMPACT_FAILURES = 3      # disable the LLM tier after this many failures
# Re-run the LLM summarizer intra-turn only after the kept context has grown by
# this factor since the last summary. Prevents summarizing on every step while
# still allowing a single ballooning turn to be compacted more than once.
AUTOCOMPACT_REGROWTH_FACTOR = 1.15

CLEAR_MARKER = (
    "[old tool output cleared to save context — rerun the tool if you need it again]"
)
SUMMARY_MARKER = "[CONVERSATION SUMMARY]"

# Context windows for models not present in the base table (e.g. DeepSeek).
_EXTRA_CONTEXT_WINDOWS = {
    "deepseek-v4-flash": 128_000,
    "deepseek-chat": 128_000,
    "deepseek-reasoner": 128_000,
}

_SUMMARY_SYSTEM = (
    "You compress coding-agent conversations into dense, faithful summaries."
)
_SUMMARY_INSTRUCTION = (
    "Summarize the conversation below so an AI coding agent can continue with no "
    "loss of essential context. Preserve: the user's goals and constraints, "
    "decisions already made, files/functions touched, concrete facts learned from "
    "tool output, and any unfinished work or next steps. Be concise and factual. "
    "Do not call tools. Output only the summary.\n\n<conversation>\n{body}\n</conversation>"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def context_window_for(model: str) -> int:
    """Resolve the context-window size (tokens) for *model*.

    ``MINI_CODE_CONTEXT_WINDOW`` overrides everything when set.
    """
    override = os.environ.get("MINI_CODE_CONTEXT_WINDOW", "").strip()
    if override.isdigit():
        return int(override)
    if model in DEFAULT_CONTEXT_WINDOWS:
        return DEFAULT_CONTEXT_WINDOWS[model]
    if model in _EXTRA_CONTEXT_WINDOWS:
        return _EXTRA_CONTEXT_WINDOWS[model]
    return DEFAULT_CONTEXT_WINDOWS["default"]


def _is_summary(message: dict[str, Any]) -> bool:
    return message.get("role") == "system" and str(
        message.get("content", "")
    ).startswith(SUMMARY_MARKER)


def utilization(messages: list[dict[str, Any]], window: int) -> float:
    """Fraction of the context window the messages currently occupy."""
    if window <= 0:
        return 0.0
    return estimate_messages_tokens(messages) / window


def _render_for_summary(messages: list[dict[str, Any]]) -> str:
    """Flatten messages into plain text for the summarizer prompt."""
    parts: list[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "") or ""
        if _is_summary(m):
            parts.append(f"[Previous Summary]: {content[len(SUMMARY_MARKER):].strip()}")
        elif role == "system":
            continue  # the live system prompt is not part of the summary input
        elif role == "user":
            parts.append(f"[User]: {content}")
        elif role in ("assistant", "assistant_progress"):
            parts.append(f"[Assistant]: {content}")
        elif role == "assistant_tool_call":
            try:
                rendered_input = json.dumps(m.get("input"), ensure_ascii=False)
            except (TypeError, ValueError):
                rendered_input = str(m.get("input"))
            parts.append(f"[Tool Call: {m.get('toolName')}]: {rendered_input}")
        elif role == "tool_result":
            body = content if len(content) <= 500 else content[:500] + "... (truncated)"
            err = " ERROR" if m.get("isError") else ""
            parts.append(f"[Tool Result: {m.get('toolName')}{err}]: {body}")
    return "\n\n".join(parts)


def _align_boundary(convo: list[dict[str, Any]], boundary: int) -> int:
    """Never let the *kept* region start on an orphan ``tool_result``.

    Tool calls and their results are appended adjacently by the agent loop, so a
    cut that lands on a ``tool_result`` would orphan it (its ``assistant_tool_call``
    got compressed). The Anthropic API rejects an orphan ``tool_result``; advance
    past the result run so the kept region begins cleanly.
    """
    n = len(convo)
    b = boundary
    while 0 < b < n and convo[b].get("role") == "tool_result":
        b += 1
    return b


def _find_retention_boundary(convo: list[dict[str, Any]], window: int) -> int:
    """Return the index in *convo* before which messages get compressed.

    Scans from the tail accumulating tokens, keeping recent messages up to
    ``MAX_KEEP_TOKENS`` but always keeping at least ``MIN_KEEP_MESSAGES``.
    """
    token_sum = 0
    boundary = len(convo)
    for i in range(len(convo) - 1, -1, -1):
        t = estimate_message_tokens(convo[i])
        if token_sum + t > MAX_KEEP_TOKENS:
            break
        token_sum += t
        boundary = i
    # Always keep at least MIN_KEEP_MESSAGES of the most recent conversation.
    min_keep_boundary = max(0, len(convo) - MIN_KEEP_MESSAGES)
    boundary = min(boundary, min_keep_boundary)
    return _align_boundary(convo, boundary)


# ---------------------------------------------------------------------------
# Compactor
# ---------------------------------------------------------------------------

@dataclass
class ContextCompactor:
    """Stateful, multi-tier context compactor.

    Construct one per agent turn (or reuse across turns). Cross-turn summary
    state lives inside the message list itself (a ``system`` message tagged with
    :data:`SUMMARY_MARKER`), so folding works even with a fresh compactor.
    """

    model_adapter: Optional[ModelAdapter] = None
    model_name: str = "default"
    window: int = 0
    disabled: bool = False
    failures: int = 0
    _last_autocompact_tokens: int = 0
    # Real token count reported by the provider for the last call; when > 0 it
    # replaces the heuristic estimate for utilization (design point: use real usage).
    real_total_tokens: int = 0
    # toolUseId -> original tool-result content (for restore_cleared)
    _cleared: dict[str, str] = field(default_factory=dict)
    # archived spans replaced by auto_compact (for rewind/restore)
    archive: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.window <= 0:
            self.window = context_window_for(self.model_name)

    def set_real_tokens(self, total_tokens: int) -> None:
        """Record the provider-reported context size of the last call."""
        if total_tokens and total_tokens > 0:
            self.real_total_tokens = int(total_tokens)

    def current_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Best available token count: provider-reported if known, else estimate."""
        return self.real_total_tokens or estimate_messages_tokens(messages)

    def current_utilization(self, messages: list[dict[str, Any]]) -> float:
        if self.window <= 0:
            return 0.0
        return self.current_tokens(messages) / self.window

    # -- public entry point: call at the top of every agent-loop step --------

    def before_model_call(
        self,
        messages: list[dict[str, Any]],
        *,
        step: int = 0,
        on_event: Optional[Callable[[str, float], None]] = None,
    ) -> list[dict[str, Any]]:
        """Compact *messages* (if needed) right before ``model.next(messages)``.

        Returns a possibly-compacted message list. Never raises — compaction is
        best-effort and falls back to the input on any failure.
        """
        msgs = messages
        util = self.current_utilization(msgs)

        # Tier 1: micro-compact every step (cheap, no LLM, reversible).
        if util >= MICROCOMPACT_UTILIZATION:
            compacted = self.microcompact(msgs)
            if compacted is not msgs:
                msgs = compacted
                util = self.current_utilization(msgs)
                if on_event:
                    on_event("microcompact", util)

        # Tier 2: LLM summarization when still critical (intra-turn allowed,
        # but rate-limited by the re-growth gate so we don't summarize every step).
        if (
            not self.disabled
            and self.model_adapter is not None
            and util >= AUTOCOMPACT_UTILIZATION
            and self._should_autocompact(msgs)
        ):
            if on_event:
                on_event("autocompact_start", util)  # surfaces "compacting…" in the UI
            compacted = self.auto_compact(msgs)
            if compacted is not None:
                msgs = compacted
                # Real token count is now stale; fall back to the estimate.
                self.real_total_tokens = 0
                if on_event:
                    on_event("autocompact", self.current_utilization(msgs))
            elif on_event:
                on_event("autocompact_skipped", util)

        return msgs

    def _should_autocompact(self, messages: list[dict[str, Any]]) -> bool:
        tokens = self.current_tokens(messages)
        if self._last_autocompact_tokens == 0:
            return True
        return tokens >= self._last_autocompact_tokens * AUTOCOMPACT_REGROWTH_FACTOR

    # -- tier 1: micro-compact ----------------------------------------------

    def microcompact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Clear the bodies of all but the most recent tool results.

        The messages are kept (so every ``tool_use`` still has a matching
        ``tool_result`` for the API); only their ``content`` is replaced with
        :data:`CLEAR_MARKER`. Originals are archived for :meth:`restore_cleared`.
        """
        result_indices = [
            i
            for i, m in enumerate(messages)
            if m.get("role") == "tool_result" and m.get("content") != CLEAR_MARKER
        ]
        if len(result_indices) <= KEEP_RECENT_TOOL_RESULTS:
            return messages

        to_clear = set(result_indices[: len(result_indices) - KEEP_RECENT_TOOL_RESULTS])
        out: list[dict[str, Any]] = []
        changed = False
        for i, m in enumerate(messages):
            if i in to_clear:
                tool_id = m.get("toolUseId")
                if tool_id is not None and tool_id not in self._cleared:
                    self._cleared[tool_id] = m.get("content", "")
                cleared = dict(m)
                cleared["content"] = CLEAR_MARKER
                out.append(cleared)
                changed = True
            else:
                out.append(m)
        return out if changed else messages

    # -- tier 2: LLM auto-compact -------------------------------------------

    def auto_compact(
        self, messages: list[dict[str, Any]], *, force: bool = False
    ) -> Optional[list[dict[str, Any]]]:
        """Summarize older messages into one summary, keeping recent ones.

        Returns the new message list, or ``None`` if nothing was compacted.
        Implements design point (c): folds any previous summary plus the older
        conversation into a fresh summary and keeps the most recent N tokens.

        ``force=True`` (manual ``/compact``) ignores the size guards and keeps
        only the last :data:`FORCE_KEEP_MESSAGES`, so it compacts even a small
        conversation as long as there is something to summarize.
        """
        if self.model_adapter is None:
            return None
        try:
            system_prompts = [
                m for m in messages if m.get("role") == "system" and not _is_summary(m)
            ]
            prior_summaries = [m for m in messages if _is_summary(m)]
            convo = [m for m in messages if m.get("role") != "system"]

            if force:
                # Need at least one message to compress plus the kept tail.
                if len(convo) < FORCE_KEEP_MESSAGES + 1:
                    return None
                boundary = _align_boundary(convo, len(convo) - FORCE_KEEP_MESSAGES)
            else:
                if len(convo) <= MIN_KEEP_MESSAGES:
                    return None
                boundary = _find_retention_boundary(convo, self.window)
            if boundary <= 0:
                return None
            to_compress = convo[:boundary]
            to_keep = convo[boundary:]
            if not to_compress:
                return None

            body = _render_for_summary(prior_summaries + to_compress)
            summary_text = self._summarize(body)
            if not summary_text:
                self._note_failure()
                return None

            summary_msg = {
                "role": "system",
                "content": f"{SUMMARY_MARKER}\n{summary_text}",
            }
            new_messages = system_prompts + [summary_msg] + to_keep

            # Archive the replaced span so the turn can be restored ("rewind").
            self.archive.append(
                {"replaced": prior_summaries + to_compress, "summary": summary_msg}
            )
            self.failures = 0
            self._last_autocompact_tokens = estimate_messages_tokens(new_messages)
            logger.info(
                "auto-compact: summarized %d msgs, kept %d (%.0f%% -> %.0f%%)",
                len(to_compress),
                len(to_keep),
                utilization(messages, self.window) * 100,
                utilization(new_messages, self.window) * 100,
            )
            return new_messages
        except Exception as error:  # noqa: BLE001 - compaction is best-effort
            logger.warning("auto-compact failed: %s", error)
            self._note_failure()
            return None

    def _summarize(self, body: str) -> Optional[str]:
        assert self.model_adapter is not None
        step = self.model_adapter.next(
            [
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user", "content": _SUMMARY_INSTRUCTION.format(body=body)},
            ]
        )
        text = (getattr(step, "content", "") or "").strip()
        return text or None

    def _note_failure(self) -> None:
        self.failures += 1
        if self.failures >= MAX_AUTOCOMPACT_FAILURES:
            self.disabled = True
            logger.warning(
                "auto-compact disabled after %d consecutive failures", self.failures
            )

    # -- restore (rewind support) -------------------------------------------

    def restore_cleared(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Re-expand any micro-compacted tool results from the archive."""
        out: list[dict[str, Any]] = []
        for m in messages:
            if (
                m.get("role") == "tool_result"
                and m.get("content") == CLEAR_MARKER
                and m.get("toolUseId") in self._cleared
            ):
                restored = dict(m)
                restored["content"] = self._cleared[m["toolUseId"]]
                out.append(restored)
            else:
                out.append(m)
        return out
