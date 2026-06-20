"""Unit tests for minicode.context_compactor (no network required)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import minicode.context_compactor as cc  # noqa: E402
from minicode.context_compactor import (  # noqa: E402
    CLEAR_MARKER,
    SUMMARY_MARKER,
    ContextCompactor,
)
from minicode.types import AgentStep  # noqa: E402


class FakeAdapter:
    """Records summarization calls and returns a canned summary."""

    def __init__(self, summary: str = "FOLDED-SUMMARY") -> None:
        self.summary = summary
        self.calls = 0
        self.last_messages = None

    def next(self, messages):
        self.calls += 1
        self.last_messages = messages
        return AgentStep(type="assistant", content=self.summary)


def system(c):
    return {"role": "system", "content": c}


def user(c):
    return {"role": "user", "content": c}


def asst(c):
    return {"role": "assistant", "content": c}


def call(tid, inp=None):
    return {
        "role": "assistant_tool_call",
        "toolUseId": tid,
        "toolName": "run_command",
        "input": inp or {"command": "ls"},
    }


def tr(tid, content, err=False):
    return {
        "role": "tool_result",
        "toolUseId": tid,
        "toolName": "run_command",
        "content": content,
        "isError": err,
    }


class TestContextWindow(unittest.TestCase):
    def test_known_unknown(self):
        self.assertEqual(cc.context_window_for("deepseek-v4-flash"), 128_000)
        self.assertEqual(
            cc.context_window_for("nope-unknown"),
            cc.DEFAULT_CONTEXT_WINDOWS["default"],
        )

    def test_env_override(self):
        os.environ["MINI_CODE_CONTEXT_WINDOW"] = "4096"
        try:
            self.assertEqual(cc.context_window_for("deepseek-v4-flash"), 4096)
        finally:
            del os.environ["MINI_CODE_CONTEXT_WINDOW"]


class TestMicrocompact(unittest.TestCase):
    def test_clears_old_keeps_recent_and_is_reversible(self):
        comp = ContextCompactor(model_name="deepseek-v4-flash")
        msgs = [system("sys")]
        for i in range(5):
            msgs.append(call(f"t{i}"))
            msgs.append(tr(f"t{i}", f"OUTPUT-{i} " * 50))

        out = comp.microcompact(msgs)

        cleared = [m for m in out if m.get("content") == CLEAR_MARKER]
        kept = [
            m
            for m in out
            if m.get("role") == "tool_result" and m.get("content") != CLEAR_MARKER
        ]
        self.assertEqual(len(cleared), 2)  # 5 results, keep recent 3 -> clear 2
        self.assertEqual(len(kept), 3)
        # every tool_use still has a matching tool_result message
        self.assertEqual(
            sum(1 for m in out if m.get("role") == "tool_result"), 5
        )

        restored = comp.restore_cleared(out)
        self.assertFalse(any(m.get("content") == CLEAR_MARKER for m in restored))

    def test_noop_when_few_results(self):
        comp = ContextCompactor(model_name="deepseek-v4-flash")
        msgs = [system("s"), call("a"), tr("a", "x"), call("b"), tr("b", "y")]
        self.assertIs(comp.microcompact(msgs), msgs)


class TestAutoCompact(unittest.TestCase):
    def setUp(self):
        self._max, self._min = cc.MAX_KEEP_TOKENS, cc.MIN_KEEP_MESSAGES
        cc.MAX_KEEP_TOKENS = 12  # force older messages to be compressed
        cc.MIN_KEEP_MESSAGES = 2

    def tearDown(self):
        cc.MAX_KEEP_TOKENS, cc.MIN_KEEP_MESSAGES = self._max, self._min

    def test_folds_prior_summary_and_keeps_recent(self):
        fake = FakeAdapter()
        comp = ContextCompactor(
            model_adapter=fake, model_name="deepseek-v4-flash", window=1000
        )
        msgs = [
            system("LIVE SYSTEM PROMPT"),
            system(SUMMARY_MARKER + "\nearlier summary text"),
            user("u1"), asst("a1"),
            user("u2"), asst("a2"),
            user("u3"), asst("a3 latest reply"),
        ]

        new = comp.auto_compact(msgs)

        self.assertIsNotNone(new)
        self.assertEqual(fake.calls, 1)
        summaries = [
            m for m in new if str(m.get("content", "")).startswith(SUMMARY_MARKER)
        ]
        self.assertEqual(len(summaries), 1)  # exactly one summary, not two
        self.assertIn("FOLDED-SUMMARY", summaries[0]["content"])
        # live system prompt is preserved verbatim
        self.assertTrue(
            any(m.get("content") == "LIVE SYSTEM PROMPT" for m in new)
        )
        # the prior summary was folded into the summarizer input (point c)
        summarizer_input = fake.last_messages[1]["content"]
        self.assertIn("Previous Summary", summarizer_input)
        self.assertIn("earlier summary text", summarizer_input)
        # the most recent message is kept verbatim
        self.assertTrue(any(m.get("content") == "a3 latest reply" for m in new))
        # the replaced span is archived for restore/rewind
        self.assertEqual(len(comp.archive), 1)

    def test_kept_region_never_starts_on_orphan_tool_result(self):
        fake = FakeAdapter()
        comp = ContextCompactor(
            model_adapter=fake, model_name="deepseek-v4-flash", window=1000
        )
        msgs = [system("S")]
        for i in range(6):
            msgs.append(user(f"u{i}"))
            msgs.append(call(f"t{i}"))
            msgs.append(tr(f"t{i}", f"result {i}"))

        new = comp.auto_compact(msgs)

        self.assertIsNotNone(new)
        non_system = [m for m in new if m.get("role") != "system"]
        self.assertNotEqual(non_system[0].get("role"), "tool_result")


class TestBeforeModelCall(unittest.TestCase):
    def test_microcompact_triggers_by_utilization(self):
        # tiny window so a handful of tool results crosses the 50% threshold
        comp = ContextCompactor(model_name="deepseek-v4-flash", window=200)
        msgs = [system("s")]
        for i in range(5):
            msgs.append(call(f"t{i}"))
            msgs.append(tr(f"t{i}", "X" * 400))

        out = comp.before_model_call(msgs, step=1)

        self.assertTrue(any(m.get("content") == CLEAR_MARKER for m in out))

    def test_below_threshold_is_untouched(self):
        comp = ContextCompactor(model_name="deepseek-v4-flash", window=1_000_000)
        msgs = [system("s"), user("hi"), asst("hello")]
        self.assertIs(comp.before_model_call(msgs, step=1), msgs)

    def test_autocompact_regrowth_gate(self):
        comp = ContextCompactor(model_name="x", window=1000)
        msgs = [user("a" * 100)]
        self.assertTrue(comp._should_autocompact(msgs))  # first time always allowed
        comp._last_autocompact_tokens = cc.estimate_messages_tokens(msgs)
        self.assertFalse(comp._should_autocompact(msgs))  # no growth -> blocked
        grown = msgs + [user("b" * 2000)]
        self.assertTrue(comp._should_autocompact(grown))  # grew enough -> allowed


if __name__ == "__main__":
    unittest.main(verbosity=2)
