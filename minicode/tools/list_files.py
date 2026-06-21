from __future__ import annotations

import posixpath
import shlex
from pathlib import Path

from minicode.exec_backend import maybe_container_backend
from minicode.tooling import ToolDefinition, ToolResult
from minicode.workspace import resolve_tool_path


def _validate(input_data: dict) -> dict:
    if "path" in input_data and not isinstance(input_data["path"], str):
        raise ValueError("path must be a string")
    return {"path": input_data.get("path", ".")}


def _run_in_container(input_data: dict, context, backend) -> ToolResult:
    target = backend.resolve(context.cwd, input_data["path"])
    if not backend.exists(target):
        return ToolResult(ok=False, output=f"Path does not exist: {input_data['path']}")
    is_dir, _o, _e = backend.run_shell(f"test -d {shlex.quote(target)}", context.cwd, 15)
    if is_dir != 0:  # a file
        return ToolResult(ok=True, output=f"file {posixpath.basename(target)}")
    # -1 one per line, -A include hidden (not . ..), -p mark dirs with trailing /
    code, out, err = backend.run_shell(f"ls -1Ap -- {shlex.quote(target)}", context.cwd, 30)
    if code != 0:
        return ToolResult(ok=False, output=err.strip() or "ls failed in container")
    names = [n for n in out.splitlines() if n]
    lines = [
        f"{'dir ' if n.endswith('/') else 'file'} {n[:-1] if n.endswith('/') else n}"
        for n in names[:200]
    ]
    return ToolResult(ok=True, output="\n".join(lines) if lines else "(empty)")


def _run(input_data: dict, context) -> ToolResult:
    backend = maybe_container_backend(context)
    if backend is not None:
        return _run_in_container(input_data, context, backend)

    target = resolve_tool_path(context, input_data["path"], "list")
    if not target.exists():
        return ToolResult(ok=False, output=f"Path does not exist: {input_data['path']}")
    if target.is_file():
        return ToolResult(ok=True, output=f"file {Path(input_data['path']).name}")

    entries = sorted(Path(target).iterdir(), key=lambda item: item.name.lower())
    lines = []
    for entry in entries:
        lines.append(f"{'dir ' if entry.is_dir() else 'file'} {entry.name}")
    return ToolResult(ok=True, output="\n".join(lines[:200]) if lines else "(empty)")


list_files_tool = ToolDefinition(
    name="list_files",
    description="List files and directories relative to the workspace root.",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
    validator=_validate,
    run=_run,
)
