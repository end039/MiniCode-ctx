"""Execution backend abstraction: run commands and do file IO either on the
host (default) or *inside a Docker container*.

This is what makes MiniCode "Docker-aware". When a tool runs with
``ToolContext.container`` set, its filesystem reads/writes and command execution
are routed through ``docker exec`` / ``docker cp`` against that container instead
of the host. The repository being worked on stays inside the container (e.g. a
SWE-bench image's ``/testbed``) — we never bind-mount it onto the host.

Host behaviour is intentionally left untouched: tools keep their existing
host code path and only branch into a :class:`ContainerBackend` when one is
present (see :func:`maybe_container_backend`). The container itself is the
sandbox, so we do not re-apply the host's workspace-escape guard inside it.
"""

from __future__ import annotations

import posixpath
import shlex
import subprocess
from typing import Protocol

# Generous default; build/test commands inside a container can take minutes.
CONTAINER_EXEC_TIMEOUT = 600


def _run_docker(
    argv: list[str], *, timeout: int = 60, stdin: str | None = None
) -> tuple[int, str, str]:
    """Run a docker CLI command on the host, returning (code, stdout, stderr)."""
    try:
        completed = subprocess.run(  # noqa: S603
            argv,
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
        return completed.returncode, completed.stdout or "", completed.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except FileNotFoundError:
        return 127, "", "docker CLI not found on host"


class ExecBackend(Protocol):
    is_container: bool

    def resolve(self, cwd: str, path: str) -> str: ...
    def run_shell(self, command: str, cwd: str, timeout: int) -> tuple[int, str, str]: ...
    def read_text(self, path: str) -> str: ...
    def write_text(self, path: str, content: str) -> None: ...
    def exists(self, path: str) -> bool: ...


class ContainerBackend:
    """Routes execution and file IO into a running container via the docker CLI."""

    is_container = True

    def __init__(self, container: str) -> None:
        self.cid = container

    # -- path handling (POSIX inside the container) -----------------------
    def resolve(self, cwd: str, path: str) -> str:
        if not posixpath.isabs(path):
            path = posixpath.join(cwd or "/", path)
        return posixpath.normpath(path)

    # -- command execution ------------------------------------------------
    def run_shell(self, command: str, cwd: str, timeout: int) -> tuple[int, str, str]:
        argv = ["docker", "exec"]
        if cwd:
            argv += ["-w", cwd]
        argv += [self.cid, "sh", "-lc", command]
        return _run_docker(argv, timeout=timeout)

    # -- file IO ----------------------------------------------------------
    def read_text(self, path: str) -> str:
        code, out, err = _run_docker(["docker", "exec", self.cid, "cat", "--", path])
        if code != 0:
            raise FileNotFoundError(err.strip() or path)
        return out

    def write_text(self, path: str, content: str) -> None:
        parent = posixpath.dirname(path)
        if parent:
            _run_docker(["docker", "exec", self.cid, "mkdir", "-p", "--", parent])
        # Write via stdin so arbitrary content needs no shell escaping.
        code, _out, err = _run_docker(
            ["docker", "exec", "-i", self.cid, "sh", "-c", f"cat > {shlex.quote(path)}"],
            stdin=content,
        )
        if code != 0:
            raise OSError(err.strip() or f"failed to write {path} in container")

    def exists(self, path: str) -> bool:
        code, _out, _err = _run_docker(["docker", "exec", self.cid, "test", "-e", path])
        return code == 0


def maybe_container_backend(context) -> ContainerBackend | None:
    """Return a :class:`ContainerBackend` if the context targets a container,
    else ``None`` (meaning: use the existing host code path unchanged)."""
    cid = getattr(context, "container", None)
    return ContainerBackend(cid) if cid else None
