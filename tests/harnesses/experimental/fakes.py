"""Offline doubles for the harness core: an adapter, a process, a workspace.

The e2e suite drives real CLIs. These drive the SESSION state machine and
the JSON-RPC peer without one, so ordering, cancellation and concurrency
bugs show up in milliseconds, not in a sandbox an hour later.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from ai.harnesses.experimental import _approval, _capabilities, _harness
from ai.types import events
from ai.types import messages as m
from ai.workspaces.experimental import Local, _base


class Pause:
    """A point in a scripted turn where the fake adapter waits.

    `reached` is set when the turn gets there; the turn goes on once
    `release` is set, or ends as interrupted when the adapter is stopped.
    """

    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self.release = asyncio.Event()


class Run:
    """Run a callable at this point of a scripted turn (e.g. a steer)."""

    def __init__(self, fn: Callable[[], Awaitable[None]]) -> None:
        self.fn = fn


Step = events.AgentEvent | Pause | Run


def text(chunk: str, block: str = "t") -> list[Step]:
    return [
        events.TextStart(block_id=block),
        events.TextDelta(block_id=block, chunk=chunk),
        events.TextEnd(block_id=block),
    ]


def tool(
    call_id: str, name: str = "Bash", args: str = "{}", result: Any = "ok"
) -> list[Step]:
    call = m.ToolCallPart(tool_call_id=call_id, tool_name=name, tool_args=args)
    part = m.ToolResultPart(tool_call_id=call_id, tool_name=name, result=result)
    return [
        events.ToolStart(tool_call_id=call_id, tool_name=name),
        events.ToolDelta(tool_call_id=call_id, chunk=args),
        events.ToolEnd(tool_call_id=call_id, tool_call=call),
        events.ToolCallResult(
            message=m.Message(role="tool", parts=[part]), results=[part]
        ),
    ]


def reply(*chunks: str) -> list[Step]:
    """A whole plain-text turn, the way an adapter ends one."""
    steps: list[Step] = [events.StreamStart()]
    for i, chunk in enumerate(chunks):
        steps += text(chunk, block=f"t{i}")
    return steps


class FakeAdapter:
    """A scripted `base.Adapter`.

    `script(prompt)` returns the steps of the turn for that prompt; the
    default echoes the prompt. Like the real adapters, the turn ends with a
    `StreamEnd` whose message carries every part the turn produced — unless
    the script ends it early (stopped, or raising).
    """

    kind = "fake"

    def __init__(
        self,
        script: Callable[[str], list[Step]] | None = None,
        *,
        capabilities: _capabilities.Capabilities | None = None,
        honors_stop: bool = True,
    ) -> None:
        self.script = script or reply
        self.capabilities = capabilities or _capabilities.Capabilities(
            steer=True,
            stop=True,
            resume=True,
            fork=True,
            history=True,
            images=True,
            documents=True,
        )
        self.honors_stop = honors_stop
        self.version: str | None = "fake 1.0"
        self.verified_version: str | None = None
        self.process_ids: list[int] = []
        self.approve: Any = None
        self.approval_timeout = 300.0
        self.on_approval_error: Any = None
        self.live_history: Any = lambda session_id: []
        self.stops: list[str] = []
        self.steers: list[tuple[str, str]] = []
        self.closed_sessions: list[str] = []
        self.turns_started: list[str] = []
        self.prompts: list[list[Any]] = []
        self.histories: list[list[m.Message] | None] = []
        self.turns_finalized: list[str] = []
        self._stopped: dict[str, asyncio.Event] = {}
        self._ids = itertools.count(1)

    async def start(self, workspace: Any, options: dict[str, Any]) -> None:
        pass

    async def close(self) -> None:
        pass

    async def detach(self) -> None:
        pass

    async def new_session(self, history: list[m.Message] | None = None) -> str:
        self.histories.append(history)
        return f"s{next(self._ids)}"

    async def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)

    async def stop(self, session_id: str) -> None:
        self.stops.append(session_id)
        self._stopped.setdefault(session_id, asyncio.Event()).set()

    async def steer(self, session_id: str, text: str) -> None:
        self.steers.append((session_id, text))

    async def turn(
        self, session_id: str, given: list[Any]
    ) -> AsyncIterator[events.AgentEvent]:
        self.prompts.append(given)
        prompt = "\n\n".join(p.text for p in given if isinstance(p, m.TextPart))
        self.turns_started.append(prompt)
        stopped = self._stopped[session_id] = asyncio.Event()
        parts: list[Any] = []
        try:
            for step in self.script(prompt):
                if isinstance(step, Pause):
                    step.reached.set()
                    if not self.honors_stop:
                        await step.release.wait()
                        continue
                    release = asyncio.ensure_future(step.release.wait())
                    stop = asyncio.ensure_future(stopped.wait())
                    try:
                        await asyncio.wait(
                            {release, stop},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        release.cancel()
                        stop.cancel()
                    if stopped.is_set():
                        yield events.StreamEnd(
                            message=m.Message(role="assistant", parts=parts),
                            finish_reason="cancelled",
                        )
                        return
                    continue
                if isinstance(step, Run):
                    await step.fn()
                    continue
                if isinstance(step, events.TextDelta):
                    parts.append(m.TextPart(text=step.chunk))
                elif isinstance(step, events.ToolEnd):
                    parts.append(step.tool_call)
                elif isinstance(step, events.StreamEnd):
                    yield step
                    return
                yield step
            yield events.StreamEnd(
                message=m.Message(role="assistant", parts=parts),
                finish_reason="stop",
            )
        finally:
            self.turns_finalized.append(prompt)


class AskingAdapter(FakeAdapter):
    """A `FakeAdapter` that asks for approval the way the real clients do.

    From tasks spawned by a reader started when the harness OPENED (codex's
    JSON-RPC pump; claude's read loop is started at connect), so the hook
    runs with whatever context that reader had unless the harness fixes it.
    `ask(name)` is a script step that asks and waits for the decision;
    decisions land in `decisions`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.decisions: list[_approval.Decision] = []
        self._asks: asyncio.Queue[
            tuple[_approval.ApprovalContext, asyncio.Future[Any]]
        ] = asyncio.Queue()
        self._reader: asyncio.Task[None] | None = None
        self._answers: set[asyncio.Task[None]] = set()
        self._workspace: Any = None
        self._session_id = ""

    async def start(self, workspace: Any, options: dict[str, Any]) -> None:
        self._workspace = workspace
        self._reader = asyncio.create_task(self._read())

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()

    async def _read(self) -> None:
        while True:
            request, answer = await self._asks.get()
            task = asyncio.create_task(self._answer(request, answer))
            self._answers.add(task)
            task.add_done_callback(self._answers.discard)

    async def _answer(
        self, request: _approval.ApprovalContext, answer: asyncio.Future[Any]
    ) -> None:
        try:
            answer.set_result(await self.approve(request))
        except BaseException as exc:
            answer.set_exception(exc)

    def ask(self, name: str = "Bash", call_id: str = "c1") -> Run:
        async def asking() -> None:
            answer: asyncio.Future[Any] = (
                asyncio.get_running_loop().create_future()
            )
            request = _approval.ApprovalContext(
                call=_approval.ToolCall(id=call_id, name=name),
                workspace=self._workspace,
                harness=self.kind,
                session_id=self._session_id,
            )
            await self._asks.put((request, answer))
            with contextlib.suppress(Exception):
                self.decisions.append(await answer)

        return Run(asking)

    def turn(
        self, session_id: str, given: list[Any]
    ) -> AsyncIterator[events.AgentEvent]:
        self._session_id = session_id
        return super().turn(session_id, given)


class FakeWorkspace(_base.Workspace):
    """Just enough workspace for a Harness: nothing runs anywhere."""

    kind = "local"
    duplex_spawn = True

    def __init__(self) -> None:
        self.env: dict[str, str] = {}

    @property
    def path(self) -> str:
        return "/fake"

    @property
    def coords(self) -> _base.WorkspaceCoords:
        return _base.WorkspaceCoords(provider="local", location=self.path)

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def spawn(self, argv: Any, *, env: Any = None) -> Any:
        raise NotImplementedError

    async def exec(
        self, argv: Any, *, env: Any = None, timeout: float | None = None
    ) -> _base.ExecResult:
        raise NotImplementedError

    async def pty(self, argv: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def attach(self, name: str) -> Any:
        raise NotImplementedError

    async def ptys(self) -> list[Any]:
        return []

    async def read_text(self, path: str) -> str:
        raise FileNotFoundError(path)

    async def write_text(self, path: str, data: str) -> None:
        raise NotImplementedError

    async def read_bytes(self, path: str) -> bytes:
        raise FileNotFoundError(path)

    async def exists(self, path: str) -> bool:
        return False

    async def _copy_in(self, *args: Any, **kwargs: Any) -> int:
        raise NotImplementedError

    async def _copy_out(self, *args: Any, **kwargs: Any) -> int:
        raise NotImplementedError

    async def reachable_url(self, port: int) -> str:
        raise NotImplementedError


async def open_harness(
    adapter: FakeAdapter | None = None, **options: Any
) -> tuple[_harness.Harness, FakeAdapter]:
    adapter = adapter or FakeAdapter()
    harness = _harness.Harness(
        cast("Any", adapter),
        workspace=FakeWorkspace(),
        **options,
    )
    await harness.open()
    return harness, adapter


class FakeProcess:
    """In-memory duplex stdio for `JsonRpcPeer`.

    `answer(message)` is called for every message the client writes —
    requests, notifications, and its responses to server requests. For a
    request, a non-None return is sent back as the result. `server_says()`
    pushes a line from the "server": a notification or a request.
    """

    def __init__(
        self, answer: Callable[[dict[str, Any]], Any] | None = None
    ) -> None:
        self._lines: asyncio.Queue[str] = asyncio.Queue()
        self.answer = answer or (lambda message: {})
        self.written: list[dict[str, Any]] = []
        self.pid: int | None = None

    def server_says(self, message: dict[str, Any]) -> None:
        self._lines.put_nowait(json.dumps(message) + "\n")

    async def write(self, data: str) -> None:
        message = json.loads(data)
        self.written.append(message)
        result = self.answer(message)
        if "method" in message and "id" in message and result is not None:
            self.server_says({"id": message["id"], "result": result})

    async def readline(self) -> str:
        return await self._lines.get()

    async def read_stderr(self) -> str:
        return ""

    async def terminate(self) -> None:
        self._lines.put_nowait("")

    async def kill(self) -> None:
        self._lines.put_nowait("")

    async def os_pid(self) -> int | None:
        return self.pid

    async def detach(self) -> None:
        pass

    async def wait(self) -> int | None:
        return 0


class LockWorkspace(Local):
    """A real local workspace whose HOME is a temp dir, so lock files never
    touch the user's own home. `before_exec` runs ahead of every command —
    the place to inject an interleaving."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self._home = root / "home"
        self._home.mkdir(parents=True, exist_ok=True)
        self.before_exec: Callable[[list[str]], Awaitable[None]] | None = None
        self.execs: list[list[str]] = []
        self.reads: list[str] = []

    async def home(self) -> str:
        return str(self._home)

    async def exec(self, argv: Any, **kwargs: Any) -> Any:
        self.execs.append(list(argv))
        if self.before_exec is not None:
            await self.before_exec(list(argv))
        return await super().exec(argv, **kwargs)

    async def read_text(self, path: str) -> str:
        self.reads.append(path)
        return await super().read_text(path)
