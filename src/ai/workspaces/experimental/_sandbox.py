"""A Vercel Sandbox microVM.

Same contract as `Local`, different machine. The mapping is mechanical
because the sandbox SDK already exposes a duplex process and a filesystem:
`create_process` gives a writable stdin and readable stdout, which is
exactly what an adapter needs to tunnel a harness CLI's stdio.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import io
import json
import os
import secrets
import shlex
import subprocess
import tarfile
import tempfile
import time
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ... import errors as ai_errors
from ...providers import _optional
from . import _base, _gateway, _pty
from . import errors as errors_

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vercel import sandbox as vercel_sandbox

DEFAULT_WORKDIR = "/vercel/sandbox"
# What `vercel env pull` writes, and what the SDK reads from it.
DEFAULT_ENV_FILE = ".env.local"
# Credentials for reaching Vercel — not configuration for the work, so
# these are kept out of the workspace environment and never shipped to the VM.
AUTH_KEYS = (
    "VERCEL_OIDC_TOKEN",
    "VERCEL_TOKEN",
    "VERCEL_TEAM_ID",
    "VERCEL_PROJECT_ID",
)
#: What installing a harness CLI inside the VM reaches (`npm install -g`).
#: Allowed without any header rewrite when a gateway locks egress down;
#: everything not listed is refused. Add to it with `allow_hosts=`.
INSTALL_HOSTS: tuple[str, ...] = ("registry.npmjs.org",)
#: Where the workspace remembers which hosts it injects credentials for,
#: so a reconnect by name answers `injects_credentials_for` truthfully.
EGRESS_RECORD = ".ai-python/egress.json"
# Printed by the conduit shell once its terminal is in raw mode and it is
# about to become the FIFO writer.
CONDUIT_READY = "__harness_conduit_ready__"
CONDUIT_READY_TIMEOUT = 30.0
PTY_SOCKET_WAIT = 10.0
"""Seconds the in-VM client waits for a new holder to bind its socket."""

# The conduit: a pump that opens the FIFO, announces itself only once it
# owns BOTH ends, then copies the PTY to the FIFO.
#
# A shell doing `echo READY; exec cat > fifo` is not equivalent: the marker
# is printed before the exec, so bytes can arrive while nothing is reading
# the terminal yet and be lost. Measured — it fails roughly every time
# without an arbitrary sleep after the marker. Here the process that will
# do the copying is the one that says it is ready.
_PUMP = (
    "import os, sys\n"
    "fd = os.open(sys.argv[1], os.O_WRONLY)\n"
    # The marker arrives as an ARGUMENT, never as a literal in this source.
    # Embedded, a SyntaxError traceback would echo the line containing it
    # and the readiness check would match the crash. (It did.)
    "sys.stdout.write(sys.argv[2] + chr(10))\n"
    "sys.stdout.flush()\n"
    "while True:\n"
    "    chunk = os.read(0, 65536)\n"
    "    if not chunk:\n"
    "        break\n"
    "    os.write(fd, chunk)\n"
)
# Exit code reported for a process whose sandbox was torn down under it.
TERMINATED_WITH_WORKSPACE = -1
# How long a program gets to exit on EOF before it is killed, and how long
# a kill gets to take effect before close() gives up and says so.
EXIT_GRACE = 10.0
KILL_GRACE = 3.0
# Supervisor that launches a program with a durable, closeable stdin.
#
# The shell holds the FIFO's write end (fd 3) and WAITS on the program, so:
#  * a conduit detaching never delivers EOF — the program keeps running;
#  * any stop signal to the shell (TERM/INT/HUP via the trap, or KILL via
#    the kernel closing fd 3) closes the write end, the program sees EOF on
#    stdin and exits on its own terms — a graceful stop, not a SIGTERM the
#    CLIs are measured to shrug off;
#  * the shell lives exactly as long as the program: no holder to orphan,
#    and it removes the FIFO and pid file when the program ends;
#  * the program is started with fd 3 CLOSED (`3>&-`): a background child
#    inherits every open fd, so without this it would hold its own
#    stdin's write end and never see EOF — measured: even `cat` then
#    outlived the supervisor closing fd 3 and had to be killed;
#  * the program's OS pid is published to <fifo>.pid, which is what makes
#    liveness checkable and an escalation targetable from outside.
_SUPERVISE = (
    'F={fifo}; mkfifo -m 600 "$F" || exit 97; exec 3<>"$F"; '
    '{program} <"$F" 3>&- & P=$!; echo "$P" >"$F.pid"; echo $$ >"$F.sup"; '
    "trap 'exec 3>&-' TERM INT HUP; S=0; "
    'while kill -0 "$P" 2>/dev/null; do wait "$P"; S=$?; done; '
    'rm -f "$F" "$F.pid" "$F.sup"; exit $S'
)
# How much one upload request may carry.
BATCH_BYTES = 8 * 1024 * 1024
BATCH_FILES = 400


def _gone(exc: BaseException) -> bool:
    """Whether this error means the sandbox itself is no longer there."""
    return "sandbox_stopped" in str(exc) or "no longer available" in str(exc)


class SandboxProcess(_base.Process):
    """A process in the microVM.

    Its BODY runs under `create_process`, which survives the client
    disconnecting and can be re-attached by id. Its STDIN is a FIFO, kept
    open by a holder so that conduits attaching and detaching never deliver
    EOF to the program. Input arrives over an interactive PTY session.
    """

    def __init__(
        self,
        proc: vercel_sandbox.Process,
        sandbox: vercel_sandbox.Sandbox | None = None,
        fifo: str | None = None,
    ) -> None:
        self._proc = proc
        self._sandbox = sandbox
        self._fifo = fifo
        self._conduit: Any = None
        self._conduit_scope: Any = None
        self._os_pid: int | None = None
        self._streams_closed = False

    @property
    def pid(self) -> int | None:
        # A sandbox process is addressed by its own id, not an OS pid this
        # machine could ever signal. Its pid INSIDE the VM is `os_pid()`.
        return None

    async def os_pid(self) -> int | None:
        """Return the program's pid inside the VM, published by the supervisor.

        None when the program has already ended (the supervisor removed the
        pid file) or this process was not launched through `spawn`.
        """
        if self._os_pid is not None:
            return self._os_pid
        if self._sandbox is None or self._fifo is None:
            return None
        text = await self._run(["cat", f"{self._fifo}.pid"])
        self._os_pid = int(text) if text.strip().isdigit() else None
        return self._os_pid

    async def _run(self, argv: list[str], *, timeout: float = 10) -> str:
        """Run one program in the VM, bounded; its stdout, or "" on any failure.

        Plain argv, never a shell string: what these helpers need is a
        handful of single programs (cat, kill, rm), and a shell buys
        nothing but quoting hazards.
        """
        if self._sandbox is None:
            return ""
        try:
            async with asyncio.timeout(timeout + 5):
                done = await self._sandbox.run_process(
                    argv[0],
                    argv[1:],
                    kill_after=timeout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
            return (await _text(done.stdout)).strip()
        except Exception:
            return ""

    async def _ok(self, argv: list[str], *, timeout: float = 10) -> bool:
        """Run one program in the VM, bounded; whether it exited 0."""
        if self._sandbox is None:
            return False
        try:
            async with asyncio.timeout(timeout + 5):
                done = await self._sandbox.run_process(
                    argv[0],
                    argv[1:],
                    kill_after=timeout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            return done.returncode == 0
        except Exception:
            return False

    async def _alive(self) -> bool:
        pid = await self.os_pid()
        if pid is None:
            return False
        return await self._ok(["kill", "-0", str(pid)], timeout=5)

    async def write(self, data: str) -> None:
        await self._ensure_conduit()
        assert self._conduit is not None
        await self._conduit.send(data.encode())

    async def _ensure_conduit(self) -> None:
        """Attach the input conduit on first use.

        Lazily, because opening it eagerly deadlocks a program that never
        reads stdin: `cat > fifo` blocks until a reader opens the other end,
        and a short-lived process has already exited by then.
        """
        if self._conduit is not None:
            return
        if self._sandbox is None or self._fifo is None:
            raise errors_.WorkspaceError(
                "this process was started without an input conduit"
            )
        # `stty raw -echo` matters: without it the PTY echoes everything we
        # send back at us and caps input lines at ~4 KiB, silently
        # corrupting any protocol with larger frames.
        self._conduit_scope = self._sandbox.open_interactive(
            "/bin/sh",
            [
                "-c",
                "stty raw -echo; exec python3 -c "
                + f"{shlex.quote(_PUMP)} {shlex.quote(self._fifo)} "
                + shlex.quote(CONDUIT_READY),
            ],
        )
        self._conduit = await self._conduit_scope.__aenter__()
        await self._await_ready()

    async def _await_ready(self) -> None:
        """Wait for the conduit shell to announce itself.

        `open_interactive` returns as soon as the PTY exists, which is
        before the shell has exec'd into `cat`. Bytes sent in that window go
        to the shell instead of the FIFO and are lost. A sleep would paper
        over it; the marker makes it deterministic.
        """
        seen = b""
        try:
            async with asyncio.timeout(CONDUIT_READY_TIMEOUT):
                while CONDUIT_READY.encode() not in seen:
                    chunk = await self._conduit.receive()
                    if not chunk:
                        break
                    seen += chunk
        except TimeoutError as exc:
            raise errors_.WorkspaceError(
                "the sandbox input conduit never became ready"
            ) from exc

    @property
    def id(self) -> str:
        """The sandbox-side process id — what a later client re-attaches to."""
        return self._proc.id

    async def detach(self) -> None:
        """Let go of the program without ending it.

        Drops the input conduit AND closes our output readers. The readers
        are ours, not the program's: closing them affects nothing in the VM,
        while leaving them open hands a half-iterated HTTP stream to garbage
        collection at interpreter exit — which is what printed
        `aclose(): asynchronous generator is already running` after
        perfectly successful runs. Callers stop their reader tasks first
        (the SDK client's disconnect, the JSON-RPC peer's close), so by the
        time this runs the streams are idle and close cleanly.
        """
        if self._conduit_scope is not None:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(10):
                    await self._conduit_scope.__aexit__(None, None, None)
            self._conduit_scope = None
            self._conduit = None
        await self._close_streams()

    async def readline(self) -> str:
        assert self._proc.stdout is not None, "spawned with stdout piped"
        try:
            return await self._proc.stdout.readline()
        except Exception as exc:
            raise errors_.WorkspaceError(
                "the sandbox is gone; its process output is unreadable"
                if _gone(exc)
                else f"reading sandbox process output failed: {exc}"
            ) from exc

    async def read_stderr(self) -> str:
        assert self._proc.stderr is not None, "spawned with stderr piped"
        try:
            return await self._proc.stderr.read()
        except Exception as exc:
            if _gone(exc):
                return ""
            raise errors_.WorkspaceError(
                f"reading sandbox stderr failed: {exc}"
            ) from exc

    async def terminate(self) -> None:
        """End the program — for real, and say so if it will not.

        The ladder, every rung bounded so this can never hang:
        1. drop our conduit and close our readers;
        2. stop the supervisor, which closes the FIFO's write end — the
           program sees EOF on stdin and exits gracefully (EXIT_GRACE);
        3. still alive: SIGKILL the program by pid, and anything else that
           holds the FIFO (a stale conduit pump), then wait KILL_GRACE;
        4. still alive: raise. A close() that returns means the process is
           gone; one that lies would be worse than one that fails.
        The FIFO and pid file are removed on every path.
        """
        await self.detach()
        pid = await self.os_pid()
        with contextlib.suppress(Exception):
            async with asyncio.timeout(10):
                await self._proc.terminate()
        supervisor = await self._run(["cat", f"{self._fifo}.sup"])
        if supervisor.isdigit():
            # Closes the FIFO's write end via the trap: EOF, a graceful exit.
            await self._run(["kill", "-TERM", supervisor])
        if pid is not None and not await self._gone_within(EXIT_GRACE):
            await self._run(["kill", "-KILL", str(pid)])
            # Anything else still holding the FIFO — a stale conduit pump.
            await self._run(["pkill", "-KILL", "-f", str(self._fifo)])
            if not await self._gone_within(KILL_GRACE):
                raise errors_.WorkspaceError(
                    f"sandbox process {pid} survived EOF and SIGKILL; it is "
                    "still running"
                )
        if self._fifo is not None:
            await self._run(
                [
                    "rm",
                    "-f",
                    self._fifo,
                    f"{self._fifo}.pid",
                    f"{self._fifo}.sup",
                ],
                timeout=5,
            )

    async def _gone_within(self, seconds: float) -> bool:
        """Poll the program's liveness inside the VM until it is gone."""
        deadline = asyncio.get_running_loop().time() + seconds
        while True:
            if not await self._alive():
                return True
            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(0.5)

    async def _close_streams(self) -> None:
        """Close the output readers before the process goes away.

        Left to garbage collection, the SDK's HTTP byte stream is finalized
        while still iterating and Python reports `aclose(): asynchronous
        generator is already running` on stderr — after a perfectly
        successful run. Closing them here is deterministic and keeps that
        noise off a working program.
        """
        if self._streams_closed:
            return
        self._streams_closed = True
        for stream in (self._proc.stdout, self._proc.stderr):
            if stream is not None and hasattr(stream, "aclose"):
                # Bounded: aclose() on a stream some other task is still
                # iterating blocks until that iteration ends — possibly never.
                with contextlib.suppress(Exception):
                    async with asyncio.timeout(10):
                        await stream.aclose()

    async def kill(self) -> None:
        with contextlib.suppress(Exception):
            await self._proc.kill()

    async def wait(self) -> int | None:
        """Wait for the process to end.

        A stopped sandbox answers HTTP 410 for everything inside it. That is
        not an error to hand the caller — the workspace was torn down, so
        the process is definitively over. Anything else is a real failure
        and surfaces as one.
        """
        try:
            return await self._proc.wait()
        except Exception as exc:
            if _gone(exc):
                return TERMINATED_WITH_WORKSPACE
            raise errors_.WorkspaceError(
                f"waiting on sandbox process failed: {exc}"
            ) from exc


class VercelSandbox(_base.Workspace):
    kind = "sandbox"
    # Ephemeral and ours: installing a harness here modifies nothing the
    # user owns.
    owner = "provider"
    # Duplex after all, via the layout in `spawn`: a durable body reading a
    # FIFO, plus an interactive PTY session as the input conduit.
    duplex_spawn = True

    def __init__(
        self,
        *,
        token: str | None = None,
        team_id: str | None = None,
        project_id: str | None = None,
        workdir: str = DEFAULT_WORKDIR,
        snapshot: str | None = None,
        env: Mapping[str, str] | None = None,
        forward_project_env: bool = False,
        env_file: str | Path | None = None,
        project_root: str | Path | None = None,
        execution_time_limit: float | None = None,
        ports: list[int] | None = None,
        name: str | None = None,
        keep: bool | None = None,
        gateway: _gateway.Gateway | None = None,
        allow_hosts: Sequence[str] = (),
    ) -> None:
        """Create a sandbox, or reconnect to one by `name`.

        `name` reconnects to a sandbox that already exists instead of creating
        one — the idiom of `psutil.Process(pid)`: the object FOR an existing
        thing. Everything inside it is then reachable: its files, the harness's
        own session store, processes still running.

        `keep` decides what `close()` does. A sandbox you created is stopped
        on close unless keep=True; one you reconnected to is left running
        unless keep=False — you attached to something, you did not make it.

        `gateway` is where the harnesses in this VM reach their model, and
        it turns the VM's egress into an allowlist: requests to the gateway
        host leave with the credential written into them by the sandbox
        firewall, `INSTALL_HOSTS` and `allow_hosts` are reachable as they
        are, and everything else is refused. The key itself never enters
        the VM — the harnesses are handed a placeholder. There is no flag to
        put the real key in a sandbox; `env=` is yours for anything else.
        Egress is fixed when a VM is created: reconnecting by name to one
        made without a gateway and asking for one is refused.

        Needs the ``sandbox`` extra with interactive PTY support, and raises
        ``ai.errors.InstallationError`` here, before anything boots, when it
        is missing.
        """
        sdk = _optional.import_optional_sdk(
            "vercel.sandbox", extra="sandbox", feature="VercelSandbox"
        )
        # vercel-sandbox 0.7.0 on PyPI has no stdin channel, and the build
        # that has one reports the same version, so check the feature itself.
        if not hasattr(sdk.Sandbox, "open_interactive"):
            raise ai_errors.InstallationError(
                "the installed vercel-sandbox has no interactive PTY support "
                "(`Sandbox.open_interactive`), which VercelSandbox needs to "
                "drive a harness; install a vercel-sandbox that includes "
                "https://github.com/vercel/vercel-py/pull/415"
            )
        self._credentials = {
            k: v
            for k, v in (
                ("token", token),
                ("team_id", team_id),
                ("project_id", project_id),
            )
            if v is not None
        }
        self._workdir = workdir
        self._snapshot = snapshot
        # `.env.local` is read to AUTHENTICATE — nothing more. After
        # `vercel link` + `vercel env pull` the Vercel credentials in that
        # file let us create a sandbox without being passed anything.
        #
        # The project's other variables stay on your machine. Sending the
        # contents of a secrets file to another host is not something a
        # constructor should do quietly; say `env=` for what the work
        # needs, or `forward_project_env=True` to send the lot deliberately.
        from_file = _read_env_file(env_file, project_root)
        for key in AUTH_KEYS:
            if key in from_file and key not in os.environ:
                os.environ[key] = from_file[key]
        project_env = (
            {k: v for k, v in from_file.items() if k not in AUTH_KEYS}
            if forward_project_env
            else {}
        )
        # The harness inside the VM cannot log in interactively; its
        # credentials arrive as environment variables or not at all. Set on
        # the sandbox itself, so they exist from boot — before anything is
        # installed or spawned.
        self.env = {**project_env, **dict(env or {})}
        self.gateway = gateway
        self._allow_hosts = tuple(allow_hosts)
        self._brokered_hosts: set[str] = set()
        self._execution_time_limit = execution_time_limit
        self._name = name
        self._keep = keep if keep is not None else name is not None
        self._ports = list(ports or [])
        self._sandbox: vercel_sandbox.Sandbox | None = None
        self._children: list[SandboxProcess] = []

    @property
    def path(self) -> str:
        return self._workdir

    @property
    def coords(self) -> _base.WorkspaceCoords:
        if self._sandbox is None:
            raise errors_.WorkspaceError(
                "sandbox is not open; no coordinates to name yet"
            )
        return _base.WorkspaceCoords(
            provider="sandbox",
            location=self._sandbox.name or self._workdir,
            options={"workdir": self._workdir},
        )

    async def open(self) -> None:
        if self._sandbox is not None:
            return
        from vercel.sandbox import create_sandbox, get_sandbox  # noqa: PLC0415

        if self._name is not None:
            try:
                # Same credentials create_sandbox gets: env-based auth works
                # without them, but a caller who passed token=/team_id= must
                # reach the same account on reconnect.
                credentials: dict[str, Any] = dict(self._credentials)
                self._sandbox = await get_sandbox(
                    name=self._name, **credentials
                )
            except Exception as exc:
                if _looks_like_auth_failure(exc):
                    raise errors_.NotAuthenticatedError(
                        "vercel-sandbox", str(exc)[:200], _auth_hint()
                    ) from exc
                if (
                    getattr(exc, "status_code", None) == 404
                    or getattr(exc, "code", None) == "not_found"
                ):
                    # The platform's own answer, not a failure to reach it.
                    raise errors_.WorkspaceGoneError(
                        f"no sandbox named {self._name!r} exists: {exc}"
                    ) from exc
                raise errors_.WorkspaceError(
                    f"no sandbox named {self._name!r} is reachable: {exc}"
                ) from exc
            status = getattr(self._sandbox, "status", "") or ""
            status = str(getattr(status, "value", status))
            if status and status not in ("running", "pending"):
                self._sandbox = None
                raise errors_.WorkspaceGoneError(
                    f"sandbox {self._name!r} is {status}, not running; its "
                    "files and "
                    + "processes are gone. Open a new workspace and resume the "
                    "session there."
                )
            # Egress was decided when the VM was made; read back what it is.
            self._brokered_hosts = await self._read_egress_record()
            if (
                self.gateway is not None
                and self.gateway.host not in self._brokered_hosts
            ):
                raise errors_.WorkspaceError(
                    f"sandbox {self._name!r} was created without credential "
                    "injection for "
                    + f"{self.gateway.host}: a harness opened here would carry "
                    "the real key "
                    + "into the VM. Open a new VercelSandbox(gateway=...) "
                    "instead."
                )
            return
        options: dict[str, Any] = dict(self._credentials)
        if self.env:
            options["env"] = dict(self.env)
        if self._execution_time_limit is not None:
            options["execution_time_limit"] = self._execution_time_limit
        if self._ports:
            options["ports"] = self._ports
        if self._snapshot is not None:
            from vercel.sandbox import SnapshotSource  # noqa: PLC0415

            options["source"] = SnapshotSource(snapshot_id=self._snapshot)
        if self.gateway is not None:
            options["network_policy"] = egress_policy(
                self.gateway, (*INSTALL_HOSTS, *self._allow_hosts)
            )
        try:
            self._sandbox = await create_sandbox(**options)
        except Exception as exc:
            if _looks_like_auth_failure(exc):
                raise errors_.NotAuthenticatedError(
                    "vercel-sandbox", str(exc)[:200], _auth_hint()
                ) from exc
            raise
        await self._sandbox.fs.mkdir(self._workdir)
        if self.gateway is not None:
            self._brokered_hosts = {self.gateway.host}
            await self._write_egress_record()

    def injects_credentials_for(self, host: str) -> bool:
        return host in self._brokered_hosts

    async def _write_egress_record(self) -> None:
        home = await self.home()
        fs = self._live().fs
        await fs.mkdir(
            EGRESS_RECORD.rsplit("/", 1)[0], cwd=home, recursive=True
        )
        await fs.write_text(
            EGRESS_RECORD, json.dumps(sorted(self._brokered_hosts)), cwd=home
        )

    async def _read_egress_record(self) -> set[str]:
        from vercel.sandbox import SandboxPathNotFoundError  # noqa: PLC0415

        try:
            text = await self._live().fs.read_text(
                EGRESS_RECORD, cwd=await self.home()
            )
        except SandboxPathNotFoundError:
            return set()
        return set(json.loads(text))

    async def close(self) -> None:
        if self._sandbox is None:
            return
        if self._keep:
            # Leave everything running: detach our conduits, keep the VM and
            # its processes. This is how a conversation is handed to a VM and
            # the laptop closed — the harness keeps working; the session
            # store keeps filling; a later VercelSandbox(name=...) finds it.
            await self._release_ptys()
            for child in list(self._children):
                with contextlib.suppress(Exception):
                    await child.detach()
            self._children.clear()
            self._forget_vm()
            return
        # Stopping the VM ends every process in it at once, so there is no
        # reason to climb the per-process ladder here: just let go of our
        # conduits and readers, then stop.
        await self._release_ptys()
        for child in list(self._children):
            with contextlib.suppress(Exception):
                await child.detach()
        self._children.clear()
        with contextlib.suppress(Exception):
            async with asyncio.timeout(60):
                await self._sandbox.stop()
        self._forget_vm()

    @property
    def name(self) -> str | None:
        """The platform's name for this sandbox. None until open.

        What VercelSandbox(name=...) reconnects to.
        """
        return (
            getattr(self._sandbox, "name", None)
            if self._sandbox is not None
            else self._name
        )

    def _live(self) -> vercel_sandbox.Sandbox:
        if self._sandbox is None:
            raise errors_.WorkspaceError("sandbox is not open")
        return self._sandbox

    def _forget_vm(self) -> None:
        """Let go of this VM and of everything learned about it.

        Reopening the same object may reach a DIFFERENT VM (no name means a new
        one), whose home and scripts are its own: nothing cached may carry over.
        """
        self._sandbox = None
        self._home = None
        self._pty_tools_cache = None

    async def spawn(
        self, argv: Sequence[str], *, env: Mapping[str, str] | None = None
    ) -> _base.Process:
        """Start a process with duplex stdio.

        The platform's process API has no stdin, so input reaches the program
        through a FIFO that an interactive PTY session feeds.
        """
        sandbox = self._live()
        fifo = f"/tmp/harness-{secrets.token_hex(6)}.in"
        program = " ".join(shlex.quote(part) for part in argv)
        # See _SUPERVISE: the shell holds the FIFO's write end and waits on
        # the program, so detaching never delivers EOF, stopping the shell
        # always does, and nothing outlives the program.
        launch = _SUPERVISE.format(fifo=shlex.quote(fifo), program=program)
        proc = await sandbox.create_process(
            "/bin/sh",
            ["-c", launch],
            cwd=self._workdir,
            env={**self.env, **(env or {})},
        )
        child = SandboxProcess(proc, sandbox, fifo)
        self._children.append(child)
        return child

    async def exec(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _base.ExecResult:
        command, *args = argv
        done = await self._live().run_process(
            command,
            args,
            cwd=self._workdir,
            env={**self.env, **(env or {})},
            kill_after=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return _base.ExecResult(
            exit_code=done.returncode if done.returncode is not None else -1,
            stdout=await _text(done.stdout),
            stderr=await _text(done.stderr),
        )

    async def read_text(self, path: str) -> str:
        from vercel.sandbox import SandboxPathNotFoundError  # noqa: PLC0415

        try:
            return await self._live().fs.read_text(path, cwd=self._workdir)
        except SandboxPathNotFoundError as exc:
            # The contract is the stdlib's: a missing file raises the error
            # every caller already handles, not a provider-specific one.
            raise FileNotFoundError(path) from exc

    # -- pseudo-terminals -------------------------------------------------
    #
    # The holder (`_pty_holder.py`) runs in the VM and owns the program's
    # pty; the client (`_pty_client.py`) runs on the platform's interactive
    # PTY and is the wire to its socket. Both are uploaded once per open.
    # Frames pass through the platform PTY untouched because the client's
    # shell sets it raw first — the same trick the stdin conduit uses.

    _pty_tools_cache: tuple[str, str, str] | None = None
    _ptys: list[_pty.Pty] | None = None

    async def _release_ptys(self) -> None:
        """Drop our end of every pty still open here, in THIS task.

        A pty a caller forgot to close would otherwise have its platform
        session finalized by garbage collection, in whatever task runs the
        finalizer — anyio refuses to exit a scope from a different task and
        prints a traceback. Detaching here is quiet and correct: a named
        program keeps running; an unnamed one ends with our connection.
        """
        for pty in list(self._ptys or []):
            with contextlib.suppress(Exception):
                await pty.detach()
        self._ptys = []

    _home: str | None = None

    async def home(self) -> str:
        """Return HOME in the VM, asked once and remembered.

        HOME does not change under a running sandbox, and every store lookup and
        lock path starts from it (measured: up to eight identical round trips in
        one listing).
        """
        if self._home is None:
            self._home = await super().home()
        return self._home

    async def _pty_tools(self) -> tuple[str, str, str]:
        """Install the holder and client scripts, and the socket directory.

        Written fresh once per connection so the VM never runs a stale copy,
        in ONE batched request (the batch creates the directories; a keep
        file makes the socket directory): measured, writing them one at a
        time cost five round trips, about a second, before every attach.
        """
        if self._pty_tools_cache is None:
            bin_dir, sock_dir = (
                f"{await self.home()}/.ai-python/bin",
                await self._pty_dir(),
            )
            async with self._live().fs.batch() as batch:
                for script in ("_pty_holder.py", "_pty_client.py"):
                    batch.write_bytes(
                        f"{bin_dir}/{script}",
                        (
                            files("ai.workspaces.experimental") / script
                        ).read_bytes(),
                    )
                batch.write_bytes(f"{sock_dir}/.keep", b"")
            self._pty_tools_cache = (
                f"{bin_dir}/_pty_holder.py",
                f"{bin_dir}/_pty_client.py",
                sock_dir,
            )
        return self._pty_tools_cache

    async def _pty_dir(self) -> str:
        return f"{await self.home()}/.ai-python/ptys"

    async def pty(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        name: str | None = None,
        size: tuple[int, int] | None = None,
    ) -> _pty.Pty:
        holder, _, sock_dir = await self._pty_tools()
        cols, rows = size or _pty.DEFAULT_SIZE
        if name is not None:
            sock = f"{sock_dir}/{name}.sock"
            if await self._pty_alive(sock):
                raise errors_.WorkspaceError(
                    f"a pty named {name!r} is already running here; "
                    "attach(name) instead"
                )
            flags: list[str] = []
        else:
            sock = f"/tmp/harness-pty-{secrets.token_hex(6)}.sock"
            flags = ["--transient"]
        await self._live().create_process(
            "python3",
            [
                holder,
                sock,
                *flags,
                "--cols",
                str(cols),
                "--rows",
                str(rows),
                "--",
                *argv,
            ],
            cwd=self._workdir,
            env={**self.env, **(env or {})},
        )
        # The client waits for the holder's socket inside the VM; no polling
        # from here (measured: each look was a 0.5s round trip).
        try:
            return await self._connect_pty(sock, name, wait=PTY_SOCKET_WAIT)
        except errors_.WorkspaceError as exc:
            raise errors_.WorkspaceError(
                f"the pty holder for {argv[0]!r} never opened its socket in "
                "the sandbox"
            ) from exc

    async def attach(self, name: str) -> _pty.Pty:
        sock = f"{await self._pty_dir()}/{name}.sock"
        pid = await self._holder_pid(sock)
        if pid is None:
            raise errors_.WorkspaceError(
                f"no pty named {name!r} is running in this sandbox"
            )
        return await self._connect_pty(sock, name, pid=pid)

    async def ptys(self) -> list[_pty.PtyInfo]:
        # Listing needs no scripts in the VM: only the directory's contents.
        sock_dir = await self._pty_dir()
        listing = await self.exec(["ls", sock_dir])
        entries = set(listing.stdout.split())
        found: list[_pty.PtyInfo] = []
        for entry in sorted(entries):
            if not entry.endswith(".sock"):
                continue
            pid = await self._holder_pid(f"{sock_dir}/{entry}")
            if pid is not None:
                found.append(
                    _pty.PtyInfo(
                        name=entry[: -len(".sock")],
                        pid=pid,
                        attached=f"{entry}.client" in entries,
                    )
                )
        return found

    async def _pty_alive(self, sock: str) -> bool:
        return await self._holder_pid(sock) is not None

    async def _holder_pid(self, sock: str) -> int | None:
        """Return the pid of a live holder.

        The holder writes the program's pid next to its socket and removes both
        when the program ends; a pid that no longer answers is a holder that
        died without cleaning up. Never probed by connecting: that would take
        the pty from whoever is attached.
        """
        try:
            pid = (await self.read_text(f"{sock}.pid")).strip()
        except FileNotFoundError:
            return None
        if not pid.isdigit():
            return None
        return (
            int(pid)
            if (await self.exec(["kill", "-0", pid])).exit_code == 0
            else None
        )

    async def _connect_pty(
        self,
        sock: str,
        name: str | None,
        *,
        pid: int | None = None,
        wait: float = 0,
    ) -> _pty.Pty:
        _, client, _ = await self._pty_tools()
        marker = "__harness_pty_ready__"
        scope = self._live().open_interactive(
            "/bin/sh",
            [
                "-c",
                "stty raw -echo; exec python3 "
                + shlex.join([client, sock, marker, str(wait)]),
            ],
        )
        conduit = await scope.__aenter__()
        seen = b""
        try:
            async with asyncio.timeout(CONDUIT_READY_TIMEOUT):
                while marker.encode() not in seen:
                    chunk = await conduit.receive()
                    if not chunk:
                        break
                    seen += chunk
        except TimeoutError as exc:
            with contextlib.suppress(Exception):
                await scope.__aexit__(None, None, None)
            raise errors_.WorkspaceError(
                "the sandbox pty client never became ready"
            ) from exc
        if marker.encode() not in seen:
            # The client ended before connecting: no holder on that socket.
            with contextlib.suppress(Exception):
                await scope.__aexit__(None, None, None)
            raise errors_.WorkspaceError(
                f"no pty holder answered on {sock}: "
                f"{seen.decode(errors='replace').strip()[-200:]}"
            )
        # Frames may already follow the marker (a named pty replays its
        # scrollback on attach); hand them to the codec rather than drop them.
        leftover = seen.partition(marker.encode())[2].lstrip(b"\r\n")
        if pid is None:
            # A pty just started: its holder wrote the pid once it forked.
            try:
                pid_text = (await self.read_text(f"{sock}.pid")).strip()
            except FileNotFoundError:
                pid_text = ""
            pid = int(pid_text) if pid_text.isdigit() else None

        async def send_bytes(data: bytes) -> None:
            try:
                await conduit.send(data)
            except Exception as exc:
                # The platform closed the interactive session under us. Say
                # so in the SDK's vocabulary rather than leaking anyio's.
                raise errors_.WorkspaceError(
                    "the sandbox terminal connection dropped; attach again "
                    + f"(a named pty keeps running): {type(exc).__name__}"
                ) from exc

        async def recv_bytes() -> bytes:
            nonlocal leftover
            if leftover:
                data, leftover = leftover, b""
                return data
            try:
                return await conduit.receive()
            except Exception:
                return b""  # end of stream: the codec settles receive()/wait()

        async def close_channel() -> None:
            await scope.__aexit__(None, None, None)

        pty = _pty.FramedPty(
            name=name,
            send_bytes=send_bytes,
            recv_bytes=recv_bytes,
            close_channel=close_channel,
            pid=pid,
        )
        if self._ptys is None:
            self._ptys = []
        self._ptys.append(pty)
        return pty

    async def write_text(self, path: str, data: str) -> None:
        fs = self._live().fs
        parent = "/".join(path.split("/")[:-1])
        if parent:
            await fs.mkdir(parent, cwd=self._workdir, recursive=True)
        await fs.write_text(path, data, cwd=self._workdir)

    async def read_bytes(self, path: str) -> bytes:
        from vercel.sandbox import SandboxPathNotFoundError  # noqa: PLC0415

        try:
            return await self._live().fs.read_bytes(path, cwd=self._workdir)
        except SandboxPathNotFoundError as exc:
            raise FileNotFoundError(path) from exc

    async def _copy_in(
        self,
        source: Path,
        dest: str,
        ignore: Sequence[str],
        *,
        gitignore: bool,
    ) -> int:
        """Copy a local tree in, batched.

        One request per batch rather than one per file: measured over 200
        files, batching took 0.2s where per-file writes took 75.5s. Traffic
        goes over the same authenticated API as everything else — nothing
        is exposed to reach the VM.
        """
        root, patterns = source, tuple(ignore)
        fs = self._live().fs
        pending: list[tuple[str, bytes]] = []
        staged = 0
        count = 0

        async def flush() -> None:
            nonlocal pending, staged
            if not pending:
                return
            async with fs.batch() as batch:
                for path, payload in pending:
                    batch.write_bytes(path, payload)
            pending, staged = [], 0

        for local, relative in _base.walk_uploadable(
            root, patterns, gitignore=gitignore
        ):
            payload = local.read_bytes()
            pending.append((f"{self._workdir}/{dest}/{relative}", payload))
            staged += len(payload)
            count += 1
            # Bound each request so one large tree does not become one
            # enormous upload that fails whole.
            if staged >= BATCH_BYTES or len(pending) >= BATCH_FILES:
                await flush()
        await flush()
        return count

    async def _copy_out(
        self,
        source: str,
        dest: Path,
        ignore: Sequence[str],
        *,
        gitignore: bool,
    ) -> int:
        """Copy a tree out as ONE archive.

        There is no batched read — `fs.batch()` only writes — so reading a
        tree file by file would cost a round trip each. Tarring it on the
        far side and reading one file costs one, whatever the file count.

        The `ignore` globs go to tar so `node_modules` never crosses the
        wire — the slash-free ones only, which tar reads exactly as we do
        (a whole path component) once told not to let `*` match a slash.
        The tree is filtered again as it lands, and `.gitignore` is read
        there, on the extracted files: tar saves bandwidth, it never decides.
        """
        archive = f"/tmp/harness-out-{secrets.token_hex(6)}.tgz"
        excludes = [
            f"--exclude={pattern}" for pattern in ignore if "/" not in pattern
        ]
        command = [
            "tar",
            "czf",
            archive,
            "-C",
            source,
            "--no-wildcards-match-slash",
        ]
        made = await self.exec(
            ["sh", "-c", shlex.join([*command, *excludes, "."])]
        )
        if made.exit_code != 0:
            raise errors_.WorkspaceError(
                f"could not archive {source!r} in the sandbox: "
                f"{made.stderr.strip()[:200]}"
            )
        payload = await self.read_bytes(archive)
        await self.exec(["rm", "-f", archive])

        def _extract() -> int:
            with tempfile.TemporaryDirectory(prefix="harness-out-") as staging:
                with tarfile.open(
                    fileobj=io.BytesIO(payload), mode="r:gz"
                ) as tar:
                    tar.extractall(staging, filter="data")
                return _base.copy_tree(
                    Path(staging), dest, tuple(ignore), gitignore=gitignore
                )

        return await asyncio.to_thread(_extract)

    async def exists(self, path: str) -> bool:
        return await self._live().fs.exists(path, cwd=self._workdir)

    async def reachable_url(self, port: int) -> str:
        sandbox = self._live()
        if port not in self._ports:
            # Exposing a port is an API call; do it on demand rather than
            # making every caller declare ports up front.
            await sandbox.update(ports=sorted({*self._ports, port}))
            self._ports.append(port)
        for route in getattr(sandbox, "routes", None) or []:
            if getattr(route, "port", None) == port:
                return str(route.url)
        raise errors_.WorkspaceError(
            f"sandbox exposed port {port} but published no route for it"
        )


async def _text(stream: Any) -> str:
    if stream is None:
        return ""
    if isinstance(stream, str):
        return stream
    if callable(stream):
        result = stream()
        if inspect.isawaitable(result):
            result = await result
        return str(result)
    return str(stream)


def egress_policy(gateway: _gateway.Gateway, hosts: Sequence[str]) -> Any:
    """Build the sandbox firewall's view of a gateway.

    Deny everything, allow the gateway host with `Authorization` rewritten to
    the real credential on the way out, allow `hosts` untouched. The credential
    exists in this policy and on the host that built it — nowhere the VM can
    read.
    """
    from vercel.sandbox import (  # noqa: PLC0415
        NetworkPolicy,
        NetworkPolicyRule,
        NetworkPolicyTransform,
    )

    inject = NetworkPolicyRule(
        transform=(
            NetworkPolicyTransform(
                headers={"authorization": f"Bearer {gateway.credential}"}
            ),
        )
    )
    allow: dict[str, tuple[Any, ...]] = {gateway.host: (inject,)}
    for host in hosts:
        allow.setdefault(host, ())
    return NetworkPolicy(mode="custom", allow=allow)


def _read_env_file(
    env_file: str | Path | None, project_root: str | Path | None
) -> dict[str, str]:
    """Parse `.env.local`, searching upward from the project root.

    `vercel env pull` writes the file at the root of the linked project,
    which is not necessarily where you are running from. A missing file is
    not an error — plenty of people pass credentials another way.
    """
    if env_file is not None:
        path = Path(env_file).expanduser()
        return _parse_env(path) if path.is_file() else {}
    start = Path(project_root or Path.cwd()).expanduser().resolve()
    for directory in (start, *start.parents):
        candidate = directory / DEFAULT_ENV_FILE
        if candidate.is_file():
            return _parse_env(candidate)
        if (directory / ".git").exists():
            break  # do not wander out of the repository
    return {}


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def oidc_token_expiry(token: str | None) -> float | None:
    """Return the `exp` claim of a Vercel OIDC JWT, or None.

    As a Unix timestamp.

    Decoded, not verified: this is for telling a caller WHY the platform
    said 403, not for trusting the token.
    """
    if not token or token.count(".") != 2:
        return None
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return None
    exp = claims.get("exp")
    return float(exp) if isinstance(exp, int | float) else None


def _looks_like_auth_failure(exc: BaseException) -> bool:
    """Return whether `exc` is a credentials failure.

    A platform denial, or the SDK's own pre-flight credentials error
    (`SandboxCredentialsError`: a token it cannot derive a team/project from).
    Both mean the same thing to a caller: fix your credentials.
    """
    text = str(exc).lower()
    return (
        type(exc).__name__ == "SandboxCredentialsError"
        or "403" in text
        or "401" in text
        or "not authorized" in text
        or "forbidden" in text
        or "could not determine" in text
    )


def _auth_hint() -> str:
    exp = oidc_token_expiry(os.environ.get("VERCEL_OIDC_TOKEN"))
    if exp is not None and exp < time.time():
        when = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(exp))
        return (
            f"VERCEL_OIDC_TOKEN expired at {when}. It is short-lived; refresh "
            "it "
            "with `vercel env pull` (or set VERCEL_TOKEN + VERCEL_TEAM_ID + "
            "VERCEL_PROJECT_ID for a long-lived credential)."
        )
    return (
        "Sandbox creation needs VERCEL_OIDC_TOKEN (from `vercel env pull`) or "
        "VERCEL_TOKEN + VERCEL_TEAM_ID + VERCEL_PROJECT_ID, with access to the "
        "team."
    )
