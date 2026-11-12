"""Run functions in a Vercel sandbox, in an ``ai.contrib.sandbox`` runner.

Needs the ``sandbox`` extra: ``pip install 'ai[sandbox]'``.
"""

from __future__ import annotations

import base64
import dataclasses
import importlib.metadata
import os
import posixpath
import shlex
import stat
import subprocess
import sys
import tomllib
from collections.abc import AsyncIterator, Iterable
from contextlib import AsyncExitStack, asynccontextmanager
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
from anyio.abc import ByteReceiveStream, ByteSendStream
from anyio.streams.buffered import BufferedByteReceiveStream
from vercel.sandbox import SandboxApiError, SandboxStdinError

if TYPE_CHECKING:
    from vercel.sandbox import Command, InteractiveSession, Sandbox

from . import runner as runner_
from .runner import Process, Runner
from .util import protected

type StrPath = str | PathLike[str]


async def sync_dir(box: Sandbox, src_dir: StrPath, target_dir: str) -> None:
    """Copy a local directory to a sandbox.

    Skips .venv*, .env*, .git, .tau.
    """
    await sync_paths(box, src_dir, target_dir, _walk(Path(src_dir)))


def _walk(src: Path) -> list[Path]:
    paths = []
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not _skipped(d)]
        for name in files:
            if not _skipped(name):
                paths.append(Path(root, name).relative_to(src))
    return paths


async def sync_paths(
    box: Sandbox, src_dir: StrPath, target_dir: str, paths: Iterable[StrPath]
) -> None:
    """Copy the given paths, relative to ``src_dir``, to a sandbox.

    Paths that aren't regular files are skipped.
    """
    src = Path(src_dir)
    await box.fs.mkdir(target_dir)
    async with box.fs.batch(cwd=target_dir) as batch:
        for rel in paths:
            path = src / rel
            if not path.is_file():
                continue
            batch.write_bytes(
                Path(rel).as_posix(),
                path.read_bytes(),
                mode=stat.S_IMODE(path.stat().st_mode),
            )


def project_paths(src_dir: StrPath) -> list[Path]:
    """List a project's files: what git lists, else what ``sync_dir`` copies.

    Git lists tracked files and untracked ones it doesn't ignore; of
    those, what ``sync_dir`` would skip is still skipped.  The paths are
    relative to ``src_dir``, for ``sync_paths``.
    """
    try:
        listed = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            cwd=src_dir,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return _walk(Path(src_dir))
    return [
        Path(p)
        for p in os.fsdecode(listed).split("\0")
        if p and not any(_skipped(part) for part in Path(p).parts)
    ]


async def sync_workspace(
    box: Sandbox, src_dir: StrPath, target_dir: str
) -> dict[str, str]:
    """Copy a uv project to a sandbox, with its local path sources.

    Each path source goes where its relative path puts it from
    ``target_dir``, with the files ``project_paths`` lists.  That leaves out
    ``.git``, so for a source versioned by ``uv-dynamic-versioning``, the
    returned env sets the version to the one installed here; only one
    source can be.
    """
    await sync_dir(box, src_dir, target_dir)
    env: dict[str, str] = {}
    for name, path in _path_sources(Path(src_dir)):
        if path.is_absolute():
            raise ValueError(f"path source {name} is absolute: {path}")
        remote = posixpath.normpath(posixpath.join(target_dir, path.as_posix()))
        if remote == os.pardir or remote.startswith(os.pardir + "/"):
            raise ValueError(f"path source {name} is outside the sandbox")
        local = Path(src_dir, path)
        await sync_paths(box, local, remote, project_paths(local))
        if "uv-dynamic-versioning" in _pyproject(local).get("tool", {}):
            if _VERSION_BYPASS in env:
                raise ValueError(
                    "more than one path source uses uv-dynamic-versioning"
                )
            env[_VERSION_BYPASS] = importlib.metadata.version(name)
    return env


_VERSION_BYPASS = "UV_DYNAMIC_VERSIONING_BYPASS"


def _pyproject(project_dir: Path) -> dict[str, Any]:
    with open(project_dir / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _path_sources(project_dir: Path) -> list[tuple[str, Path]]:
    sources = _pyproject(project_dir).get("tool", {}).get("uv", {})
    found = []
    for name, entry in sources.get("sources", {}).items():
        for source in entry if isinstance(entry, list) else [entry]:
            if "path" in source:
                found.append((name, Path(source["path"])))
    return found


def _skipped(name: str) -> bool:
    return name.startswith((".venv", ".env")) or name in (".git", ".tau")


def _remote_main(main_path: StrPath | None, src_dir: StrPath) -> str | None:
    """Where ``main_path``, or by default ``__main__``, is in the copy.

    By default, a ``__main__`` outside ``src_dir``, such as an installed
    script, isn't loaded.
    """
    explicit = main_path is not None
    if main_path is None:
        main_path = getattr(sys.modules["__main__"], "__file__", None)
        if main_path is None:
            return None
    remote_main = Path(os.path.relpath(main_path, src_dir))
    if remote_main.parts[0] == os.pardir:
        if explicit:
            raise ValueError(f"{main_path} is not inside {src_dir}")
        return None
    if any(_skipped(part) for part in remote_main.parts):
        if explicit:
            raise ValueError(f"{main_path} is not copied to the sandbox")
        return None
    return remote_main.as_posix()


# Sandbox command output doesn't yet pass arbitrary bytes through intact, so
# the runner's stdout is piped through this, which base64-encodes each chunk
# onto its own line.
_BASE64_ENCODER = """\
import base64, sys
while data := sys.stdin.buffer.read1():
    sys.stdout.buffer.write(base64.b64encode(data) + b"\\n")
    sys.stdout.buffer.flush()
"""


def _uv_args() -> list[str]:
    runner = ["python", "-m", runner_._MODULE]
    return ["uv", "run", "--python", _python_version(), *runner]


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _runner_script() -> str:
    runner = shlex.join(_uv_args())
    encoder = shlex.join(
        [
            *["uv", "run", "--no-project", "--python", _python_version()],
            *["python", "-c", _BASE64_ENCODER],
        ]
    )
    return f"set -o pipefail; {runner} | {encoder}"


class Base64ByteReceiveStream(ByteReceiveStream):
    """Decode newline-terminated base64 lines from a byte stream."""

    def __init__(self, stream: ByteReceiveStream) -> None:
        self._stream = stream
        self._lines = b""
        self._data = b""
        self._eof = False

    async def receive(self, max_bytes: int = 65536) -> bytes:
        while not self._data:
            line, newline, rest = self._lines.partition(b"\n")
            if newline:
                self._lines = rest
            elif self._eof:
                self._lines = b""
                if not line.strip():
                    raise anyio.EndOfStream
            else:
                try:
                    self._lines += await self._stream.receive()
                except anyio.EndOfStream:
                    self._eof = True
                continue
            self._data = base64.b64decode(line.strip(), validate=True)
        data, self._data = self._data[:max_bytes], self._data[max_bytes:]
        return data

    async def aclose(self) -> None:
        await self._stream.aclose()


@dataclasses.dataclass
class _CommandProcess:
    stdin: ByteSendStream
    stdout: ByteReceiveStream
    cmd: Command

    async def exit_message(self) -> str:
        returncode = await self.cmd.wait()
        stderr = await self.cmd.stderr.read() if self.cmd.stderr else b""
        text = stderr.decode(errors="replace")
        return f"runner exited with {returncode}\n{text}"


@asynccontextmanager
async def sandbox_process(
    box: Sandbox,
    *,
    src_dir: StrPath = ".",
    target_dir: str = "workspace",
    env: dict[str, str] | None = None,
) -> AsyncIterator[Process]:
    """Start a runner in a sandbox on a copy of ``src_dir``, with uv.

    The copy is made by ``sync_workspace``.
    """
    sync_env = await sync_workspace(box, src_dir, target_dir)
    cmd = await box.create_command(
        "bash",
        ["-c", _runner_script()],
        cwd=target_dir,
        env={**sync_env, **(env or {})},
        stdin=subprocess.PIPE,
    )
    assert cmd.stdin is not None and cmd.stdout is not None
    stdout = Base64ByteReceiveStream(cmd.stdout)
    try:
        yield _CommandProcess(cmd.stdin, stdout, cmd)
    except BaseException:
        await protected(cmd.kill())
        raise
    try:
        await cmd.stdin.aclose()
        await cmd.wait()
    except (SandboxApiError, SandboxStdinError) as exc:
        # The sandbox may have stopped, as at its execution time limit.
        if not _session_gone(exc):
            raise
    await stdout.aclose()
    if cmd.stderr is not None:
        await cmd.stderr.aclose()


def sandbox_runner(
    box: Sandbox,
    main_path: StrPath | None = None,
    *,
    src_dir: StrPath = ".",
    target_dir: str = "workspace",
    env: dict[str, str] | None = None,
) -> Runner:
    """Make a sandbox runner, running a copy of ``src_dir`` with ``uv run``.

    ``src_dir`` is a uv project that depends on ``ai``.  ``main_path``
    is a local path inside it, loaded as ``__main__`` in the runner so
    that functions defined there can be sent; it defaults to the local
    ``__main__``, when that's inside ``src_dir``.
    """
    process = sandbox_process(
        box, src_dir=src_dir, target_dir=target_dir, env=env
    )
    return Runner(process, _remote_main(main_path, src_dir))


def _session_gone(exc: SandboxApiError | SandboxStdinError) -> bool:
    """Whether ``exc`` says the sandbox's session is gone (HTTP 410)."""
    cause = exc.cause if isinstance(exc, SandboxStdinError) else exc
    return isinstance(cause, SandboxApiError) and cause.status_code == 410


_READY = b"RUN_SYNC_READY\n"


@dataclasses.dataclass
class _PtyProcess:
    stdin: ByteSendStream
    stdout: ByteReceiveStream
    pty: InteractiveSession
    box: Sandbox
    stderr_path: str

    async def exit_message(self) -> str:
        returncode = await self.pty.wait()
        stderr = await self.box.fs.read_text(self.stderr_path)
        return f"runner exited with {returncode}\n{stderr}"


@asynccontextmanager
async def pty_sandbox_process(
    box: Sandbox, *, src_dir: StrPath = ".", target_dir: str = "workspace"
) -> AsyncIterator[Process]:
    """Like ``sandbox_process``, but on a PTY in raw mode, so replies are raw.

    The runner's stderr goes to a file in the sandbox instead of the terminal.
    """
    sync_env = await sync_workspace(box, src_dir, target_dir)
    env_args = [f"{k}={v}" for k, v in sync_env.items()]
    command = shlex.join(["env", *env_args, *_uv_args()])
    stderr_path = f"/tmp/run_sync-{os.urandom(4).hex()}.err"
    script = (
        f"cd {shlex.quote(target_dir)} && stty raw -echo"
        f" && printf {shlex.quote(_READY.decode())}"
        f" && exec {command} 2>{stderr_path}"
    )
    stack = AsyncExitStack()
    try:
        pty = await stack.enter_async_context(
            box.open_interactive("sh", ["-c", script])
        )
        stdout = BufferedByteReceiveStream(pty.stream)
        await stdout.receive_until(_READY, 65536)
        yield _PtyProcess(pty.stream, stdout, pty, box, stderr_path)
    except BaseException:
        await protected(stack.aclose())
        raise
    await stack.aclose()


def pty_sandbox_runner(
    box: Sandbox,
    main_path: StrPath | None = None,
    *,
    src_dir: StrPath = ".",
    target_dir: str = "workspace",
) -> Runner:
    """Like ``sandbox_runner``, but with ``pty_sandbox_process``."""
    process = pty_sandbox_process(box, src_dir=src_dir, target_dir=target_dir)
    return Runner(process, _remote_main(main_path, src_dir))
