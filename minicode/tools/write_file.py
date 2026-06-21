from __future__ import annotations

from minicode.exec_backend import maybe_container_backend
from minicode.file_review import apply_reviewed_file_change
from minicode.tooling import ToolDefinition
from minicode.workspace import resolve_tool_path


def _validate(input_data: dict) -> dict:
    path = input_data.get("path")
    content = input_data.get("content")
    if not isinstance(path, str) or not path:
        raise ValueError("path is required")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    return {"path": path, "content": content}


def _run(input_data: dict, context):
    backend = maybe_container_backend(context)
    if backend is not None:
        target = backend.resolve(context.cwd, input_data["path"])
    else:
        target = resolve_tool_path(context, input_data["path"], "write")
    return apply_reviewed_file_change(
        context, input_data["path"], target, input_data["content"], backend
    )


write_file_tool = ToolDefinition(
    name="write_file",
    description="Write a UTF-8 text file relative to the workspace root.",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    validator=_validate,
    run=_run,
)

