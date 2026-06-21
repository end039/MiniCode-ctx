"""Tests for the explore/search sub-agent: tracker counting, the read-only
registry, the runner's isolation, and the dispatch_agent tool (no network)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minicode.subagent import (  # noqa: E402
    SubAgentTracker,
    build_readonly_registry,
    run_explore_subagent,
)
from minicode.tools.dispatch_agent import create_dispatch_agent_tool  # noqa: E402
from minicode.tooling import ToolContext  # noqa: E402
from minicode.types import AgentStep  # noqa: E402


class _FinalModel:
    """A model that immediately returns a final answer (no tool calls)."""

    def __init__(self, answer: str = "auth lives in auth.py:10") -> None:
        self.answer = answer
        self.calls = 0

    def next(self, messages):
        self.calls += 1
        return AgentStep(type="assistant", content=self.answer)


class TestSubAgentTracker(unittest.TestCase):
    def test_start_finish_counting(self):
        events: list[int] = []
        t = SubAgentTracker(on_change=lambda: events.append(1))
        self.assertEqual(t.active_count(), 0)
        aid = t.start("explore", "find auth")
        self.assertEqual(t.active_count(), 1)
        self.assertEqual(t.active_records()[0].task, "find auth")
        t.finish(aid)
        self.assertEqual(t.active_count(), 0)
        # on_change fired on both start and finish
        self.assertEqual(len(events), 2)

    def test_on_change_errors_are_swallowed(self):
        def boom() -> None:
            raise RuntimeError("ui blew up")

        t = SubAgentTracker(on_change=boom)
        aid = t.start("explore", "x")  # must not raise
        t.finish(aid)
        self.assertEqual(t.active_count(), 0)


class TestReadOnlyRegistry(unittest.TestCase):
    def test_only_read_tools_no_write_no_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = build_readonly_registry(tmp)
            names = {t.name for t in reg.list()}
            for present in ("read_file", "grep_files", "list_files", "file_tree"):
                self.assertIn(present, names)
            for absent in ("write_file", "edit_file", "patch_file", "run_command",
                           "modify_file", "dispatch_agent"):
                self.assertNotIn(absent, names)


class TestRunExploreSubAgent(unittest.TestCase):
    def test_returns_final_and_tracks_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = SubAgentTracker()
            seen_during_run: list[int] = []

            class _Probe(_FinalModel):
                def next(self, messages):
                    # The tracker should report 1 active while we're running.
                    seen_during_run.append(tracker.active_count())
                    return super().next(messages)

            out = run_explore_subagent(
                task="where is auth handled?",
                cwd=tmp,
                tracker=tracker,
                make_model=lambda reg: _Probe(),
            )
            self.assertIn("auth lives in auth.py:10", out)
            self.assertIn("explore sub-agent", out)
            self.assertEqual(seen_during_run, [1])     # live during the run
            self.assertEqual(tracker.active_count(), 0)  # cleared afterwards

    def test_subagent_model_sees_only_readonly_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            captured = {}

            def _make(reg):
                captured["names"] = {t.name for t in reg.list()}
                return _FinalModel()

            run_explore_subagent(task="t", cwd=tmp, make_model=_make)
            self.assertIn("grep_files", captured["names"])
            self.assertNotIn("write_file", captured["names"])

    def test_finishes_even_when_model_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = SubAgentTracker()

            class _Boom:
                def next(self, messages):
                    raise ValueError("kaboom")

            # run_agent_turn catches model errors and returns a fallback
            # assistant message, so the run completes and the tracker clears.
            out = run_explore_subagent(
                task="t", cwd=tmp, tracker=tracker, make_model=lambda reg: _Boom()
            )
            self.assertEqual(tracker.active_count(), 0)
            self.assertIn("error", out.lower())


class TestDispatchTool(unittest.TestCase):
    def test_validate_requires_task(self):
        tool = create_dispatch_agent_tool(cwd=".", runtime={"model": "x"})
        with self.assertRaises(ValueError):
            tool.validator({"task": "   "})
        parsed = tool.validator({"task": "  find auth  "})
        self.assertEqual(parsed["task"], "find auth")
        self.assertEqual(parsed["agent_type"], "explore")

    def test_run_returns_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = SubAgentTracker()
            tool = create_dispatch_agent_tool(
                cwd=tmp,
                runtime={"model": "x"},
                tracker=tracker,
                make_model=lambda reg: _FinalModel("found in cli.py:3"),
            )
            parsed = tool.validator({"task": "where is the CLI entry?"})
            res = tool.run(parsed, ToolContext(cwd=tmp, permissions=None))
            self.assertTrue(res.ok)
            self.assertIn("found in cli.py:3", res.output)
            self.assertEqual(tracker.active_count(), 0)

    def test_run_without_model_is_graceful(self):
        tool = create_dispatch_agent_tool(cwd=".", runtime=None, make_model=None)
        parsed = tool.validator({"task": "anything"})
        res = tool.run(parsed, ToolContext(cwd=".", permissions=None))
        self.assertFalse(res.ok)
        self.assertIn("requires a configured model", res.output)


class TestStepBudget(unittest.TestCase):
    def test_default_is_generous(self):
        from minicode.subagent import _subagent_max_steps, DEFAULT_SUBAGENT_MAX_STEPS
        self.assertGreaterEqual(DEFAULT_SUBAGENT_MAX_STEPS, 30)  # not the old 12
        self.assertEqual(_subagent_max_steps(), DEFAULT_SUBAGENT_MAX_STEPS)

    def test_env_overrides_max_steps(self):
        from minicode.subagent import _subagent_max_steps
        os.environ["MINI_CODE_SUBAGENT_MAX_STEPS"] = "7"
        try:
            self.assertEqual(_subagent_max_steps(), 7)
        finally:
            os.environ.pop("MINI_CODE_SUBAGENT_MAX_STEPS", None)

    def test_env_ignores_garbage(self):
        from minicode.subagent import _subagent_max_steps, DEFAULT_SUBAGENT_MAX_STEPS
        os.environ["MINI_CODE_SUBAGENT_MAX_STEPS"] = "0"
        try:
            self.assertEqual(_subagent_max_steps(), DEFAULT_SUBAGENT_MAX_STEPS)
        finally:
            os.environ.pop("MINI_CODE_SUBAGENT_MAX_STEPS", None)

    def test_wrapup_forces_summary_on_exhaustion(self):
        # Explore phase never concludes (always tool-calls) → hits the step limit;
        # the wrap-up phase (empty registry) must produce a real summary instead
        # of the "maximum tool step limit" fallback leaking to the parent.
        with tempfile.TemporaryDirectory() as tmp:
            class _ToolLoop:
                def next(self, messages):
                    return AgentStep(
                        type="tool_calls",
                        calls=[{"id": "t", "toolName": "grep_files",
                                "input": {"pattern": "zzz"}}],
                        content="",
                    )

            def _make(reg):
                # Empty registry ⇒ wrap-up phase; otherwise the explore phase.
                if len(reg.list()) == 0:
                    return _FinalModel("WRAP-UP: auth handled in auth.py:1")
                return _ToolLoop()

            out = run_explore_subagent(task="t", cwd=tmp, make_model=_make, max_steps=2)
            self.assertIn("WRAP-UP", out)
            self.assertNotIn("maximum tool step limit", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
