from __future__ import annotations

import re
import shlex
from pathlib import Path

from minicode.exec_backend import maybe_container_backend
from minicode.tooling import ToolDefinition, ToolResult
from minicode.workspace import resolve_tool_path

# Cap on grep output lines (container branch) to avoid flooding the context.
_GREP_MAX_LINES = 500


def _validate(input_data: dict) -> dict:
    pattern = input_data.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern is required")
    return {
        "pattern": pattern,
        "path": input_data.get("path", "."),
    }


def _run_in_container(input_data: dict, context, backend) -> ToolResult:
    target = backend.resolve(context.cwd, input_data["path"])
    # -r recursive, -n line numbers, -I skip binary, -E extended regex.
    cmd = f"grep -rnI -E -- {shlex.quote(input_data['pattern'])} {shlex.quote(target)}"
    code, out, err = backend.run_shell(cmd, context.cwd, 60)
    if code == 1:  # grep: no matches
        return ToolResult(ok=True, output="No matches found.")
    if code not in (0, 1):
        return ToolResult(ok=False, output=err.strip() or "grep failed in container")
    lines = [ln for ln in out.splitlines() if ln]
    if not lines:
        return ToolResult(ok=True, output="No matches found.")
    if len(lines) > _GREP_MAX_LINES:
        shown = "\n".join(lines[:_GREP_MAX_LINES])
        return ToolResult(
            ok=True,
            output=f"{shown}\n\n⚠️ Results truncated at {_GREP_MAX_LINES} lines.",
        )
    return ToolResult(ok=True, output="\n".join(lines))


def _run(input_data: dict, context) -> ToolResult:
    backend = maybe_container_backend(context)
    if backend is not None:
        return _run_in_container(input_data, context, backend)

    root = resolve_tool_path(context, input_data["path"], "search")
    regex = re.compile(input_data["pattern"])
    results: list[str] = []
    skipped = 0
    file_count = 0
    
    # 跳过常见大目录
    SKIP_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', '.tox', 'dist', 'build'}
    MAX_FILES = 5000

    try:
        all_files = sorted(root.rglob("*"))
    except PermissionError:
        return ToolResult(ok=False, output=f"Permission denied: {root}")
    except OSError as e:
        return ToolResult(ok=False, output=f"Cannot read directory: {e}")

    for file_path in all_files:
        # 跳过大目录
        if any(part in SKIP_DIRS for part in file_path.parts):
            skipped += 1
            continue
            
        # 限制文件数量
        if file_count >= MAX_FILES:
            output = "\n".join(results) if results else "No matches found."
            output += f"\n\n⚠️ Results truncated at {MAX_FILES} files. Try a more specific path."
            return ToolResult(ok=True, output=output)
        
        file_count += 1
        
        if not file_path.is_file():
            continue
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            skipped += 1
            continue
        except OSError:
            skipped += 1
            continue
        for index, line in enumerate(lines, start=1):
            if regex.search(line):
                results.append(f"{file_path.relative_to(Path(context.cwd)).as_posix()}:{index}:{line}")
    
    output = "\n".join(results) if results else "No matches found."
    if skipped > 0:
        output += f"\n({skipped} file(s) skipped)"
    return ToolResult(ok=True, output=output)


grep_files_tool = ToolDefinition(
    name="grep_files",
    description="Search UTF-8 text files under the workspace using a regex pattern.",
    input_schema={"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]},
    validator=_validate,
    run=_run,
)

