"""Tests for the TUI-facing features: tool-result spill, usage meter,
command gating, and the /resume session picker (no network)."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import minicode.tool_result_store as trs  # noqa: E402
import minicode.tty_app as tty  # noqa: E402
from minicode.session import SessionMetadata  # noqa: E402
from minicode.tui.chrome import REVERSE  # noqa: E402


class TestToolResultSpill(unittest.TestCase):
    def setUp(self):
        self._orig = trs.MINI_CODE_DIR
        self._tmp = tempfile.TemporaryDirectory()
        trs.MINI_CODE_DIR = Path(self._tmp.name)
        os.environ["MINI_CODE_MAX_TOOL_RESULT_CHARS"] = "100"

    def tearDown(self):
        trs.MINI_CODE_DIR = self._orig
        os.environ.pop("MINI_CODE_MAX_TOOL_RESULT_CHARS", None)
        self._tmp.cleanup()

    def test_small_output_unchanged(self):
        out = trs.maybe_persist_tool_result("short", "tid1")
        self.assertEqual(out, "short")

    def test_large_output_spilled_to_disk(self):
        big = "X" * 500
        out = trs.maybe_persist_tool_result(big, "tid2")
        self.assertTrue(out.startswith(trs.PERSISTED_OUTPUT_TAG))
        self.assertIn("Full output saved to:", out)
        # the file exists and holds the full content
        spilled = Path(self._tmp.name) / "tool-results" / "tid2.txt"
        self.assertTrue(spilled.exists())
        self.assertEqual(spilled.read_text(), big)
        # in-context copy is much smaller than the original
        self.assertLess(len(out), len(big))

    def test_idempotent(self):
        big = "Y" * 500
        once = trs.maybe_persist_tool_result(big, "tid3")
        twice = trs.maybe_persist_tool_result(once, "tid3")
        self.assertEqual(once, twice)


class TestContextMeter(unittest.TestCase):
    def test_no_usage_yet(self):
        s = tty.ScreenState(context_used=0, context_max=256_000)
        self.assertEqual(tty._format_context_meter(s), "ctx --/256k")

    def test_with_usage(self):
        s = tty.ScreenState(context_used=12_000, context_max=256_000)
        self.assertEqual(tty._format_context_meter(s), "ctx 12k/256k 5%")

    def test_small_window_for_testing(self):
        s = tty.ScreenState(context_used=4_096, context_max=4_096)
        self.assertEqual(tty._format_context_meter(s), "ctx 4.1k/4k 100%")


class TestCommandGating(unittest.TestCase):
    def test_hides_unimplemented_and_shows_new(self):
        names = {c.name for c in tty._get_visible_commands("/")}
        # newly added / fixed commands are visible
        for shown in ("/compact", "/resume", "/memory", "/help", "/context"):
            self.assertIn(shown, names)
        # declared-but-unimplemented commands are hidden
        for hidden in ("/cost", "/clear", "/history", "/tasks", "/retry", "/transcript-save"):
            self.assertNotIn(hidden, names)

    def test_prefix_filtering_respects_gate(self):
        # "/c" matches /context, /config, /config-paths, /clear, /cmd, /compact, /cost
        names = {c.name for c in tty._get_visible_commands("/c")}
        self.assertIn("/compact", names)
        self.assertNotIn("/clear", names)
        self.assertNotIn("/cost", names)


class TestSessionPicker(unittest.TestCase):
    def _meta(self, sid, first):
        now = time.time()
        return SessionMetadata(
            session_id=sid, created_at=now, updated_at=now,
            first_message=first, message_count=3, workspace="/x",
        )

    def test_renders_and_highlights_selection(self):
        picker = tty.SessionPicker(
            sessions=[self._meta("aaaaaaaa", "first task"),
                      self._meta("bbbbbbbb", "second task")],
            index=1,
        )
        out = tty._render_session_picker(picker)
        self.assertIn("aaaaaaaa", out)
        self.assertIn("bbbbbbbb", out)
        # the selected (index 1) row is reverse-highlighted
        self.assertIn(REVERSE, out)
        highlighted_line = [ln for ln in out.splitlines() if REVERSE in ln][0]
        self.assertIn("bbbbbbbb", highlighted_line)

    def test_empty(self):
        out = tty._render_session_picker(tty.SessionPicker(sessions=[], index=0))
        self.assertIn("No saved sessions", out)


class TestThinkingRoundTrip(unittest.TestCase):
    def test_thinking_on_by_default(self):
        # Default: thinking stays ON (round-trip handles the echo-back).
        from minicode.anthropic_adapter import _should_disable_thinking
        self.assertFalse(
            _should_disable_thinking({"baseUrl": "https://api.deepseek.com/anthropic"})
        )

    def test_env_can_force_off(self):
        from minicode.anthropic_adapter import _should_disable_thinking
        os.environ["MINI_CODE_EXTENDED_THINKING"] = "0"
        try:
            self.assertTrue(
                _should_disable_thinking({"baseUrl": "https://api.deepseek.com/anthropic"})
            )
        finally:
            os.environ.pop("MINI_CODE_EXTENDED_THINKING", None)

    def test_thinking_blocks_round_trip_into_request(self):
        # assistant_thinking messages re-emit their raw blocks before the turn.
        from minicode.anthropic_adapter import _to_anthropic_messages
        blocks = [{"type": "thinking", "thinking": "let me reason", "signature": "sig"}]
        msgs = [
            {"role": "user", "content": "do it"},
            {"role": "assistant_thinking", "blocks": blocks},
            {"role": "assistant_tool_call", "toolUseId": "t1", "toolName": "ls", "input": {}},
            {"role": "tool_result", "toolUseId": "t1", "toolName": "ls", "content": "a", "isError": False},
        ]
        _system, converted = _to_anthropic_messages(msgs)
        # the thinking block and the tool_use merge into one assistant message,
        # thinking first.
        assistant = next(m for m in converted if m["role"] == "assistant")
        self.assertEqual(assistant["content"][0]["type"], "thinking")
        self.assertEqual(assistant["content"][0]["signature"], "sig")
        self.assertEqual(assistant["content"][1]["type"], "tool_use")


class TestSubAgentIndicator(unittest.TestCase):
    def test_hidden_when_no_tracker_or_idle(self):
        from minicode.subagent import SubAgentTracker
        # No tracker -> empty
        s = tty.ScreenState()
        self.assertEqual(tty._format_subagent_indicator(s), "")
        # Tracker with nothing active -> empty
        s.subagents = SubAgentTracker()
        self.assertEqual(tty._format_subagent_indicator(s), "")

    def test_shown_and_highlighted_when_active(self):
        from minicode.subagent import SubAgentTracker
        tracker = SubAgentTracker()
        tracker.start("explore", "where is auth handled and how does it flow")
        s = tty.ScreenState(subagents=tracker)
        out = tty._format_subagent_indicator(s)
        self.assertIn("1 subagent running", out)
        self.assertIn("explore:", out)
        self.assertIn(REVERSE, out)  # reverse-video highlight

    def test_counts_multiple(self):
        from minicode.subagent import SubAgentTracker
        tracker = SubAgentTracker()
        tracker.start("explore", "task one")
        tracker.start("explore", "task two")
        s = tty.ScreenState(subagents=tracker)
        out = tty._format_subagent_indicator(s)
        self.assertIn("2 subagents running", out)
        self.assertIn("+1 more", out)


class TestForceCompact(unittest.TestCase):
    class _Fake:
        def next(self, messages):
            from minicode.types import AgentStep
            return AgentStep(type="assistant", content="SUMMARY")

    def test_force_compacts_small_conversation(self):
        from minicode.context_compactor import ContextCompactor
        c = ContextCompactor(model_adapter=self._Fake(), model_name="x", window=1000)
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2 latest"},
        ]
        self.assertIsNone(c.auto_compact(msgs))             # normal: too small -> no-op
        forced = c.auto_compact(msgs, force=True)            # force: compacts anyway
        self.assertIsNotNone(forced)
        self.assertTrue(any(m.get("content") == "a2 latest" for m in forced))


if __name__ == "__main__":
    unittest.main(verbosity=2)
