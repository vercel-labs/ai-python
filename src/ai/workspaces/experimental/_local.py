"""A directory on this machine."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from . import _base, _gateway, _pty
from . import errors as errors_

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_TERM_GRACE = 3.0


class LocalProcess(_base.Process):
    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc

    @property
    def pid(self) -> int | None:
        return self._proc.pid

    async def write(self, data: str) -> None:
        if self._proc.stdin is None:
            raise errors_.WorkspaceError(
                "process was not spawned with a writable stdin"
            )
        self._proc.stdin.write(data.encode())
        await self._proc.stdin.drain()

    async def readline(self) -> str:
        if self._proc.stdout is None:
            raise errors_.WorkspaceError(
                "process was not spawned with a readable stdout"
            )
        return (await self._proc.stdout.readline()).decode()

    async def read_stderr(self) -> str:
        if self._proc.stderr is None:
            return ""
        return (await self._proc.stderr.read()).decode()

    async def terminate(self) -> None:
        """Stop the group: SIGTERM, then SIGKILL, bounded."""
        if self._proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(self._proc.pid), 15)
        try:
            async with asyncio.timeout(_TERM_GRACE):
                await self._proc.wait()
        except TimeoutError:
            await self.kill()

    async def detach(self) -> None:
        """Terminate: a local child cannot outlive its client meaningfully."""
        await self.terminate()

    async def kill(self) -> None:
        if self._proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(self._proc.pid), 9)
        with contextlib.suppress(Exception):
            await self._proc.wait()

    async def wait(self) -> int | None:
        return await self._proc.wait()


class Local(_base.Workspace):
    kind = "local"

    def __init__(
        self,
        path: str | Path,
        *,
        env: Mapping[str, str] | None = None,
        gateway: _gateway.Gateway | None = None,
    ) -> None:
        self._root = Path(path).expanduser().resolve()
        self.env = dict(env or {})
        # No firewall on your own machine: a gateway here means its
        # credential goes in the harness's environment, as the real key.
        self.gateway = gateway
        self._children: list[LocalProcess] = []

    @property
    def path(self) -> str:
        return str(self._root)

    @property
    def coords(self) -> _base.WorkspaceCoords:
        return _base.WorkspaceCoords(provider="local", location=str(self._root))

    async def open(self) -> None:
        if self._root.exists() and not self._root.is_dir():
            raise errors_.WorkspaceError(
                f"{self._root} exists and is not a directory"
            )
        self._root.mkdir(parents=True, exist_ok=True)

    async def close(self) -> None:
        # The workspace owns what it spawned; nothing outlives it.
        for child in list(self._children):
            with contextlib.suppress(Exception):
                await child.terminate()
        self._children.clear()

    def _resolve(self, path: str) -> Path:
        target = Path(path)
        return target if target.is_absolute() else self._root / target

    async def spawn(
        self, argv: Sequence[str], *, env: Mapping[str, str] | None = None
    ) -> _base.Process:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self._root,
                env={**os.environ, **self.env, **(env or {})},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Own the whole tree, so teardown reaches grandchildren too.
                start_new_session=True,
            )
        except OSError as exc:
            # A program that never started is not a process to stream from.
            raise errors_.WorkspaceError(
                f"cannot spawn {argv[0]!r}: {exc}"
            ) from exc
        child = LocalProcess(proc)
        self._children.append(child)
        return child

    async def exec(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _base.ExecResult:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self._root,
                env={**os.environ, **self.env, **(env or {})},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            # Shell semantics: a command that cannot be executed is exit 127,
            # not an exception. Probes read the exit code; raising OSError
            # here would leak a Python detail into "is this installed?".
            return _base.ExecResult(exit_code=127, stderr=f"{argv[0]}: {exc}")
        try:
            async with asyncio.timeout(timeout):
                stdout, stderr = await proc.communicate()
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), 9)
            raise
        return _base.ExecResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )

    async def read_text(self, path: str) -> str:
        return await asyncio.to_thread(self._resolve(path).read_text)

    async def write_text(self, path: str, data: str) -> None:
        target = self._resolve(path)

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(data)

        await asyncio.to_thread(_write)

    async def read_bytes(self, path: str) -> bytes:
        return await asyncio.to_thread(self._resolve(path).read_bytes)

    async def _copy_in(
        self,
        source: Path,
        dest: str,
        ignore: Sequence[str],
        *,
        gitignore: bool,
    ) -> int:
        target = self._resolve(dest)
        return await asyncio.to_thread(
            _base.copy_tree, source, target, tuple(ignore), gitignore=gitignore
        )

    async def _copy_out(
        self,
        source: str,
        dest: Path,
        ignore: Sequence[str],
        *,
        gitignore: bool,
    ) -> int:
        root = self._resolve(source)
        if not root.is_dir():
            raise FileNotFoundError(
                f"{source} is not a directory in this workspace"
            )
        return await asyncio.to_thread(
            _base.copy_tree, root, dest, tuple(ignore), gitignore=gitignore
        )

    async def exists(self, path: str) -> bool:
        return await asyncio.to_thread(self._resolve(path).exists)

    async def home(self) -> str:
        return str(Path.home())

    # -- pseudo-terminals -------------------------------------------------
    #
    # The same holder the sandbox uses, run as a local subprocess: one
    # implementation of persistence, scrollback and resize, and `attach`
    # works identically here. Named ptys live under ~/.ai-python/ptys so a
    # later process can find them; unnamed ones use a private socket and a
    # holder that ends the program when this connection drops.

    def _pty_dir(self) -> Path:
        d = Path.home() / ".ai-python" / "ptys"
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def pty(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        name: str | None = None,
        size: tuple[int, int] | None = None,
    ) -> _pty.Pty:
        cols, rows = size or _pty.DEFAULT_SIZE
        if name is not None:
            sock = self._pty_dir() / f"{name}.sock"
            if _holder_pid(sock) is not None:
                raise errors_.WorkspaceError(
                    f"a pty named {name!r} is already running here; "
                    "attach(name) instead"
                )
            flags: list[str] = []
        else:
            sock = Path(tempfile.mkdtemp(prefix="harness-pty-")) / "pty.sock"
            flags = ["--transient"]
        holder = str(files("ai.workspaces.experimental") / "_pty_holder.py")
        await asyncio.create_subprocess_exec(
            sys.executable,
            # -P: keep the script's own directory off sys.path, where this
            # package's tty.py would shadow the stdlib module pty imports.
            "-P",
            holder,
            str(sock),
            *flags,
            "--cols",
            str(cols),
            "--rows",
            str(rows),
            "--",
            *argv,
            cwd=self.path,
            env={**os.environ, **self.env, **(env or {})},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,  # its life is the program's, not ours
        )
        for _ in range(100):
            if sock.exists():
                break
            await asyncio.sleep(0.05)
        else:
            raise errors_.WorkspaceError(
                f"the pty holder for {argv[0]!r} never opened its socket"
            )
        return await self._connect_pty(sock, name)

    async def attach(self, name: str) -> _pty.Pty:
        sock = self._pty_dir() / f"{name}.sock"
        if not sock.exists():
            raise errors_.WorkspaceError(
                f"no pty named {name!r} is running here"
            )
        return await self._connect_pty(sock, name)

    async def ptys(self) -> list[_pty.PtyInfo]:
        found: list[_pty.PtyInfo] = []
        for sock in sorted(self._pty_dir().glob("*.sock")):
            pid = _holder_pid(sock)
            if pid is not None:
                found.append(
                    _pty.PtyInfo(
                        name=sock.stem,
                        pid=pid,
                        attached=Path(f"{sock}.client").exists(),
                    )
                )
        return found

    async def _connect_pty(self, sock: Path, name: str | None) -> _pty.Pty:
        reader, writer = await asyncio.open_unix_connection(str(sock))
        pid_text = await asyncio.to_thread(
            lambda: Path(f"{sock}.pid").read_text()
            if Path(f"{sock}.pid").exists()
            else ""
        )
        pid = int(pid_text) if pid_text.strip().isdigit() else None

        async def send_bytes(data: bytes) -> None:
            writer.write(data)
            await writer.drain()

        async def recv_bytes() -> bytes:
            return await reader.read(65536)

        async def close_channel() -> None:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

        return _pty.FramedPty(
            name=name,
            send_bytes=send_bytes,
            recv_bytes=recv_bytes,
            close_channel=close_channel,
            pid=pid,
        )

    async def reachable_url(self, port: int) -> str:
        return f"http://127.0.0.1:{port}"


def _holder_pid(sock: Path) -> int | None:
    """Return the program's pid if its holder still serves `sock`.

    Read from the pid file the holder writes beside its socket and removes
    when the program ends — never by connecting, which would take the pty
    from whoever is attached (the holder serves the newest client).
    """
    try:
        text = Path(f"{sock}.pid").read_text().strip()
    except OSError:
        return None
    if not text.isdigit() or not sock.exists():
        return None
    try:
        os.kill(int(text), 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    return int(text)
