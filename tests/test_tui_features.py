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


if __name__ == "__main__":
    unittest.main(verbosity=2)
