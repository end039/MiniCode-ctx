"""Unit tests for minicode.background_memory (no network required)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import minicode.memory as memory_module  # noqa: E402
from minicode.background_memory import (  # noqa: E402
    BackgroundMemoryExtractor,
    TurnSummary,
    _parse_facts,
)
from minicode.memory import MemoryManager, MemoryScope  # noqa: E402
from minicode.types import AgentStep  # noqa: E402


class FakeAdapter:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0
        self.last = None

    def next(self, messages):
        self.calls += 1
        self.last = messages
        return AgentStep(type="assistant", content=self.content)


def turn(u="q", a="did something", tools=None, i=1):
    return TurnSummary(user_input=u, agent_output=a, tools_used=tools or [], index=i)


class _Base(unittest.TestCase):
    def setUp(self):
        self._orig_dir = memory_module.MINI_CODE_DIR
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        # Redirect USER-scope memory into the temp dir (not the real home).
        memory_module.MINI_CODE_DIR = tmp / "userhome"
        self.proj = tmp / "proj"
        self.proj.mkdir(parents=True, exist_ok=True)
        self.mgr = MemoryManager(workspace=str(self.proj))

    def tearDown(self):
        memory_module.MINI_CODE_DIR = self._orig_dir
        self._tmp.cleanup()

    def extractor(self, content, **kw):
        return BackgroundMemoryExtractor(
            model_adapter=FakeAdapter(content),
            memory_manager=self.mgr,
            workspace=str(self.proj),
            enabled=kw.pop("enabled", False),
            **kw,
        )


class TestParseFacts(unittest.TestCase):
    def test_extracts_array_with_surrounding_prose(self):
        txt = 'Sure:\n[{"scope":"project","category":"arch","content":"Uses urllib"}]\nDone.'
        facts = _parse_facts(txt)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["scope"], "project")

    def test_invalid_or_empty(self):
        self.assertEqual(_parse_facts("no json here"), [])
        self.assertEqual(_parse_facts('{"not": "a list"}'), [])
        self.assertEqual(_parse_facts("[]"), [])

    def test_skips_items_without_content(self):
        facts = _parse_facts('[{"scope":"user","content":""},{"content":"keep me"}]')
        self.assertEqual(facts, [{"content": "keep me"}])


class TestExtraction(_Base):
    def test_writes_project_and_user_scopes(self):
        content = json.dumps(
            [
                {
                    "scope": "project",
                    "category": "architecture",
                    "content": "The adapter posts to baseUrl + /v1/messages",
                },
                {
                    "scope": "user",
                    "category": "workflow",
                    "content": "User develops in WSL for docker compatibility",
                },
            ]
        )
        ext = self.extractor(content)
        written = ext._process([turn("how does the adapter call the API?",
                                     "Traced the adapter; it hits the messages endpoint.",
                                     ["read_file"])])
        self.assertEqual(ext.model_adapter.calls, 1)
        self.assertEqual(len(written), 2)
        proj = self.mgr.memories[MemoryScope.PROJECT].entries
        usr = self.mgr.memories[MemoryScope.USER].entries
        self.assertTrue(any("v1/messages" in e.content for e in proj))
        self.assertTrue(any("WSL" in e.content for e in usr))
        # MEMORY.md files materialised on disk
        self.assertTrue((self.proj / ".mini-code-memory" / "MEMORY.md").exists())

    def test_deduplicates_against_existing(self):
        self.mgr.add_entry(
            MemoryScope.PROJECT, "arch", "The project has zero runtime dependencies"
        )
        content = json.dumps(
            [{"scope": "project", "category": "arch",
              "content": "the project has zero runtime dependencies"}]
        )
        ext = self.extractor(content)
        written = ext._process([turn()])
        self.assertEqual(written, [])
        matches = [
            e for e in self.mgr.memories[MemoryScope.PROJECT].entries
            if "zero runtime" in e.content.lower()
        ]
        self.assertEqual(len(matches), 1)

    def test_prompt_carries_only_summary_inputs(self):
        ext = self.extractor("[]")
        prompt = ext._build_prompt([turn("USER_QUESTION", "AGENT_SUMMARY", ["edit_file"])])
        self.assertIn("USER_QUESTION", prompt)
        self.assertIn("AGENT_SUMMARY", prompt)
        self.assertIn("edit_file", prompt)          # tool *name* only
        self.assertIn("Existing memory", prompt)
        self.assertIn("Recent conversation", prompt)


class TestThreadedPath(_Base):
    def test_record_turn_then_flush_writes_memory(self):
        content = json.dumps(
            [{"scope": "project", "category": "x", "content": "A fact from the threaded path"}]
        )
        ext = self.extractor(content, enabled=True, extract_every_n_turns=1)
        try:
            ext.record_turn("a question?", "did the thing", ["ls"])
            ext.flush(block=True, timeout=5)
        finally:
            ext.close(timeout=5)
        self.assertTrue(
            any(
                "threaded path" in e.content
                for e in self.mgr.memories[MemoryScope.PROJECT].entries
            )
        )

    def test_disabled_records_nothing(self):
        ext = self.extractor("[]", enabled=False)
        ext.record_turn("q", "a", ["ls"])  # no-op when disabled
        self.assertEqual(len(ext._pending), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
