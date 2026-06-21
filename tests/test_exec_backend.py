"""Tests for Docker-aware execution (minicode.exec_backend) and the file/command
tools' container branches. Uses a fake docker CLI (in-memory FS) so no Docker
daemon is required; a separate script does the real python:3.11 e2e."""

from __future__ import annotations

import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import minicode.exec_backend as eb  # noqa: E402
from minicode.exec_backend import ContainerBackend, maybe_container_backend  # noqa: E402
from minicode.tooling import ToolContext  # noqa: E402
from minicode.tools.read_file import read_file_tool  # noqa: E402
from minicode.tools.write_file import write_file_tool  # noqa: E402
from minicode.tools.edit_file import edit_file_tool  # noqa: E402
from minicode.tools.run_command import run_command_tool  # noqa: E402


class FakeDocker:
    """Interprets the docker-exec argv forms our backend emits, over an
    in-memory filesystem. Returns (code, stdout, stderr)."""

    def __init__(self):
        self.files: dict[str, str] = {}
        self.shell_cmds: list[str] = []
        self.canned: dict[str, tuple[int, str, str]] = {}

    def __call__(self, argv, timeout=60, stdin=None):
        a = list(argv[2:])  # drop ["docker", "exec"]
        while a and a[0] in ("-w", "-i"):
            a = a[2:] if a[0] == "-w" else a[1:]
        rest = a[1:]  # drop container id
        if rest[:2] == ["cat", "--"]:
            p = rest[2]
            return (0, self.files[p], "") if p in self.files else (1, "", f"cat: {p}: No such file")
        if rest[:1] == ["mkdir"]:
            return (0, "", "")
        if rest[:2] == ["sh", "-c"]:  # write via: cat > <quoted-path>
            target = shlex.split(rest[2])[-1]
            self.files[target] = stdin or ""
            return (0, "", "")
        if rest[:2] == ["test", "-e"]:
            return (0 if rest[2] in self.files else 1, "", "")
        if rest[:2] == ["sh", "-lc"]:
            cmd = rest[2]
            self.shell_cmds.append(cmd)
            return self.canned.get(cmd, (0, f"ran: {cmd}", ""))
        return (0, "", "")


def run_tool(tool, inp, ctx):
    """Invoke a tool the way the registry does: validate, then run."""
    return tool.run(tool.validator(inp), ctx)


class _DockerPatch(unittest.TestCase):
    def setUp(self):
        self.fake = FakeDocker()
        self._orig = eb._run_docker
        eb._run_docker = self.fake

    def tearDown(self):
        eb._run_docker = self._orig


class TestResolve(unittest.TestCase):
    def test_paths(self):
        b = ContainerBackend("c")
        self.assertEqual(b.resolve("/testbed", "src/x.py"), "/testbed/src/x.py")
        self.assertEqual(b.resolve("/testbed", "/etc/hosts"), "/etc/hosts")
        self.assertEqual(b.resolve("/testbed", "a/../b"), "/testbed/b")


class TestBackendIO(_DockerPatch):
    def test_write_read_exists_roundtrip(self):
        b = ContainerBackend("c")
        self.assertFalse(b.exists("/testbed/f.txt"))
        b.write_text("/testbed/f.txt", "hello")
        self.assertTrue(b.exists("/testbed/f.txt"))
        self.assertEqual(b.read_text("/testbed/f.txt"), "hello")

    def test_read_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            ContainerBackend("c").read_text("/nope")

    def test_maybe_backend_none_without_container(self):
        self.assertIsNone(maybe_container_backend(ToolContext(cwd="/x")))
        self.assertIsNotNone(maybe_container_backend(ToolContext(cwd="/x", container="c")))


class TestToolsInContainer(_DockerPatch):
    def _ctx(self):
        return ToolContext(cwd="/testbed", permissions=None, container="c")

    def test_write_then_read_file(self):
        run_tool(write_file_tool, {"path": "app/x.py", "content": "print(1)\n"}, self._ctx())
        self.assertEqual(self.fake.files["/testbed/app/x.py"], "print(1)\n")
        out = run_tool(read_file_tool, {"path": "app/x.py"}, self._ctx())
        self.assertTrue(out.ok)
        self.assertIn("FILE: app/x.py", out.output)
        self.assertIn("print(1)", out.output)

    def test_read_missing_file_in_container(self):
        out = run_tool(read_file_tool, {"path": "nope.py"}, self._ctx())
        self.assertFalse(out.ok)
        self.assertIn("not found in container", out.output)

    def test_edit_file_in_container(self):
        self.fake.files["/testbed/m.py"] = "a = 1\nb = 2\n"
        res = run_tool(
            edit_file_tool, {"path": "m.py", "old": "a = 1", "new": "a = 99"}, self._ctx()
        )
        self.assertTrue(res.ok)
        self.assertEqual(self.fake.files["/testbed/m.py"], "a = 99\nb = 2\n")

    def test_run_command_in_container(self):
        self.fake.canned["pytest -q"] = (0, "1 passed", "")
        res = run_tool(run_command_tool, {"command": "pytest -q"}, self._ctx())
        self.assertTrue(res.ok)
        self.assertIn("1 passed", res.output)
        self.assertIn("pytest -q", self.fake.shell_cmds)

    def test_run_command_nonzero_is_error(self):
        self.fake.canned["false"] = (1, "", "boom")
        res = run_tool(run_command_tool, {"command": "false"}, self._ctx())
        self.assertFalse(res.ok)

    def test_run_command_timeout_is_error(self):
        self.fake.canned["sleep 999"] = (124, "", "timed out after 600s")
        res = run_tool(run_command_tool, {"command": "sleep 999"}, self._ctx())
        self.assertFalse(res.ok)
        self.assertIn("timed out", res.output)

    def test_run_command_with_args_quoting(self):
        # args path: command + quoted args joined into one shell string
        self.fake.canned["python -c 'print(2)'"] = (0, "2", "")
        res = run_tool(
            run_command_tool, {"command": "python", "args": ["-c", "print(2)"]}, self._ctx()
        )
        self.assertTrue(res.ok)
        self.assertIn("python -c 'print(2)'", self.fake.shell_cmds)


class TestHostPathUnchanged(unittest.TestCase):
    """No container ⇒ tools must still hit the real host filesystem."""

    def test_read_write_on_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = ToolContext(cwd=tmp, permissions=None)  # container=None
            run_tool(write_file_tool, {"path": "h.txt", "content": "host-data"}, ctx)
            self.assertEqual((Path(tmp) / "h.txt").read_text(), "host-data")
            out = run_tool(read_file_tool, {"path": "h.txt"}, ctx)
            self.assertIn("host-data", out.output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
