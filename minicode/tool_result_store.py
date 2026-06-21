"""Spill oversized tool results to disk (mirrors TS src/utils/tool-result-storage.ts).

When a tool produces a very large output, keeping it verbatim in the conversation
wastes the context window. Instead we write the full output to a file and replace
the in-context content with a short head preview plus the file path, so the agent
can re-read the full output on demand.

The TUI still shows the full output to the user — only the copy sent to the model
is shrunk.
"""

from __future__ import annotations

import os
from pathlib import Path

from minicode.config import MINI_CODE_DIR

PERSISTED_OUTPUT_TAG = "<persisted-output>"

# Defaults mirror the TS reference (50k threshold, 2k head preview).
DEFAULT_MAX_RESULT_CHARS = 50_000
DEFAULT_PREVIEW_CHARS = 2_000


def _max_result_chars() -> int:
    raw = os.environ.get("MINI_CODE_MAX_TOOL_RESULT_CHARS", "").strip()
    return int(raw) if raw.isdigit() else DEFAULT_MAX_RESULT_CHARS


def _results_dir() -> Path:
    return MINI_CODE_DIR / "tool-results"


def _sanitize(tool_use_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (tool_use_id or ""))
    return safe or "result"


def _format_chars(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M chars"
    if n >= 1_000:
        return f"{round(n / 1_000)}K chars"
    return f"{n} chars"


def _preview(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    head = content[:limit]
    last_nl = head.rfind("\n")
    cut = last_nl if last_nl > limit * 0.5 else limit
    return content[:cut]


def maybe_persist_tool_result(content: str, tool_use_id: str) -> str:
    """Return *content* unchanged, or a preview + path if it is too large.

    Already-persisted content (carrying :data:`PERSISTED_OUTPUT_TAG`) is returned
    as-is so repeated passes are idempotent.
    """
    if not isinstance(content, str):
        return content
    if content.startswith(PERSISTED_OUTPUT_TAG):
        return content
    threshold = _max_result_chars()
    if len(content) <= threshold:
        return content

    try:
        results_dir = _results_dir()
        results_dir.mkdir(parents=True, exist_ok=True)
        filepath = results_dir / f"{_sanitize(tool_use_id)}.txt"
        filepath.write_text(content, encoding="utf-8")
    except OSError:
        # If we can't spill, keep the original content rather than lose it.
        return content

    # Keep the preview no larger than the threshold so spilling always shrinks.
    preview = _preview(content, min(DEFAULT_PREVIEW_CHARS, threshold))
    return (
        f"{PERSISTED_OUTPUT_TAG}\n"
        f"Output too large ({_format_chars(len(content))}). Full output saved to: {filepath}\n\n"
        f"Preview (first {_format_chars(len(preview))}):\n{preview}"
    )
