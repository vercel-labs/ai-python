"""`Worker` and `WorkerProvider`: a CLI session in the shape of a model.

An `ai.Model` is an id and a provider, and `ai.stream` calls
`model.provider.stream(model, messages)`. A worker has the same shape:

- `Worker` is plain data: the settings of one CLI session and its state,
  `session_id` and `messages`. It matches the history that a model call
  gets with that state.
- `WorkerProvider` is the CLI behind it: where it is and how it runs. Its
  `stream` makes one model call on the worker's open session.

Direct control is `Session`: `worker.session()` makes one.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Self

import pydantic

from ..types import events as events_
from ..types import messages as messages_
from ..types import tools as tools_
from . import _errors as errors
from . import _session as session_
from . import _wire

if TYPE_CHECKING:
    from ..models.core import params as params_


class WorkerProvider(pydantic.BaseModel):
    """The CLI behind a worker: where it is and how it runs.

    Workers can share one. A concrete provider (`CodexProvider`, ...) adds
    the private wire (`_wire.Wire`) that speaks its CLI's protocol.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str
    """The CLI's name, for telemetry."""
    executable: str | None = None
    """The CLI command or path. None: the usual command on PATH."""
    env: dict[str, str] = pydantic.Field(default_factory=dict)
    """Added to the CLI's environment: keys, base URLs, config dirs."""

    def _make_wire(self, worker: Any) -> _wire.Wire:
        """Return a new wire for `worker`. Concrete providers implement it."""
        raise NotImplementedError

    async def stream(
        self,
        worker: Worker,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
    ) -> AsyncGenerator[events_.Event]:
        """Answer `messages` with one assistant message, as a model does.

        Use `ai.stream(worker, messages)` or `ai.Agent.run(worker, ...)`,
        inside `async with worker.session()`. The message ends with
        `finish_reason="tool_call"` when the CLI calls `ai` tools from
        `tools=`; the CLI waits for the results, which come with the next
        call. To cancel the stream or to stop reading it interrupts the
        turn. `params` has no effect: use worker settings.
        """
        if output_type is not None:
            raise errors.UnsupportedError(
                "workers do not support output_type yet"
            )
        session = worker._session
        if session is None:
            raise errors.WorkerError(
                "the worker has no open session: "
                "use `async with worker.session()`"
            )
        if session._stream_lock.locked():
            raise errors.WorkerError("a worker runs one stream() at a time")
        async with session._stream_lock:
            seeds = not worker.messages
            new = worker._sync(messages)
            seeds = seeds and bool(worker.messages)
            if tools is not None:
                worker.tools = [t for t in tools if t.kind == "function"]
            run = await session._start_for_stream(seeds=seeds)
            feed = session.events()
            ended = False
            try:
                for message in new:
                    await session.send(message)
                started = False
                async for event in feed:
                    if isinstance(event, session_.MessageEvent):
                        continue
                    if isinstance(event, events_.StreamStart):
                        started = True
                    if not started:
                        continue  # the end of a message that began earlier
                    if isinstance(event, events_.StreamEnd):
                        ended = True
                        error = run.failures.pop(event.message.id, None)
                        if error is not None:
                            raise error
                        yield event
                        return
                    yield event
                raise errors.WorkerError("the worker closed during stream()")
            except (asyncio.CancelledError, GeneratorExit) as exc:
                if not ended:
                    # Shielded: a second cancel must not cut the interrupt.
                    task = asyncio.ensure_future(session.interrupt())
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        if isinstance(exc, GeneratorExit):
                            raise
                    except Exception:
                        pass
                raise
            finally:
                await feed.aclose()

    async def generate(
        self,
        worker: Worker,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
    ) -> messages_.Message:
        """Not implemented: `ai.experimental_generate` drains `stream()`."""
        raise NotImplementedError


class Worker(pydantic.BaseModel):
    """One session of a coding-agent CLI, in the shape of an `ai.Model`.

    Open a session, then use the worker as a model or control it directly::

        async with worker.session() as session:
            async with ai.stream(worker, messages) as s: ...
            await session.send(ai.user_message("And the tests?"))

    The fields are plain data. `model_dump()` saves the worker, and
    `Worker.model_validate(data)` makes the correct subclass again; it
    continues the same CLI session when a session starts. `model_copy()`
    makes a worker with no open session. Two open workers on one
    `session_id` are two writers on one CLI session: set `session_id=None`
    on the copy to start a new session that is seeded from `messages`.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    worker_class_id: str
    provider: WorkerProvider
    """The CLI that runs the session."""
    model_name: str | None = None
    """The CLI's model. None: the CLI default."""
    cwd: str = pydantic.Field(default_factory=os.getcwd)
    """Where the CLI runs."""
    instructions: str | None = None
    """Added to the CLI's system prompt when a session starts."""
    tools: list[tools_.Tool] = pydantic.Field(default_factory=list)
    """`ai` function tools that the CLI can call through the MCP bridge."""
    session_id: str | None = None
    """The CLI's id for the session. Set it to continue a session."""
    messages: list[messages_.Message] = pydantic.Field(default_factory=list)
    """The conversation, in `ai` messages. The session keeps it current."""

    _classes: ClassVar[dict[str, type[Worker]]] = {}
    _session: session_.Session | None = pydantic.PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        if type(self) is Worker:
            raise TypeError(
                "Worker must be subclassed, for example workers.Codex"
            )
        super().__init__(**data)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:  # noqa: PLW3201
        super().__pydantic_init_subclass__(**kwargs)
        class_id = cls.model_fields["worker_class_id"].default
        if isinstance(class_id, str):
            existing = Worker._classes.get(class_id)
            if (
                existing is not None
                and existing.__qualname__ != cls.__qualname__
            ):
                raise RuntimeError(f"duplicate worker class id: {class_id!r}")
            Worker._classes[class_id] = cls

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _load_registered_worker(
        cls, data: Any, handler: pydantic.ValidatorFunctionWrapHandler
    ) -> Any:
        if cls is Worker and isinstance(data, dict):
            worker_class = Worker._classes.get(str(data.get("worker_class_id")))
            if worker_class is None:
                raise ValueError(
                    f"unknown worker_class_id: {data.get('worker_class_id')!r}"
                )
            return worker_class.model_validate(data)
        return handler(data)

    def __copy__(self) -> Self:
        copied = super().__copy__()
        copied._session = None
        return copied

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        copied = super().__deepcopy__(memo)
        copied._session = None
        return copied

    @property
    def id(self) -> str:
        return self.model_name or "default"

    def session(self) -> session_.Session:
        """Return a session of the CLI: `async with worker.session()`."""
        return session_.Session(self)

    def _sync(
        self, messages: list[messages_.Message]
    ) -> list[messages_.Message]:
        """Match a whole history with the conversation; return the new part.

        First use: the history up to its last user messages becomes the
        conversation. After that, the history must continue the
        conversation.
        """
        if not self.messages:
            # Without a session_id the past seeds the new CLI session; with
            # one, the CLI has it already.
            start = len(messages)
            while start and messages[start - 1].role == "user":
                start -= 1
            if start == len(messages):
                raise errors.TranscriptMismatchError(
                    "the last message must be a user message: a worker "
                    "cannot continue tool calls that it did not make"
                )
            self.messages = list(messages[:start])
            return list(messages[start:])
        skipped = {m.id for m in self.messages if session_.is_interrupted(m)}
        known = [m.id for m in self.messages if m.id not in skipped]
        given = [m for m in messages if m.id not in skipped]
        if [m.id for m in given[: len(known)]] != known:
            raise errors.TranscriptMismatchError(
                "the messages do not continue this worker's "
                "conversation; use worker.messages as the history, "
                "or a new worker"
            )
        new = given[len(known) :]
        if not new:
            raise ValueError("stream(): no new message to send")
        for message in new:
            if message.role not in ("user", "tool"):
                raise errors.TranscriptMismatchError(
                    f"a new {message.role} message cannot go to the "
                    "CLI: only user and tool messages can"
                )
        return new
