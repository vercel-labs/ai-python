"""`ExternalHarness` and its provider: a CLI in the shape of a model.

An `ai.Model` is an id and a provider, and `ai.stream` calls
`model.provider.stream(model, messages)`. An external harness has the same
shape:

- `ExternalHarness` is plain data: the settings of the CLI, and `session_id`,
  the CLI session that its model calls continue.
- `ExternalHarnessProvider` is the CLI behind it: where it is and how it runs.
  Its `stream` makes one model call.

The state and direct control are `Session`: `harness.session()` makes one.
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


class ExternalHarnessProvider(pydantic.BaseModel):
    """The CLI behind an external harness: where it is and how it runs.

    External harnesses can share one. A concrete provider (`CodexProvider`, ...)
    adds the private wire (`_wire.Wire`) that speaks its CLI's protocol.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str
    """The CLI's name, for telemetry."""
    executable: str | None = None
    """The CLI command or path. None: the usual command on PATH."""
    env: dict[str, str] = pydantic.Field(default_factory=dict)
    """Added to the CLI's environment: keys, base URLs, config dirs."""

    def _make_wire(self, harness: Any) -> _wire.Wire:
        """Return a new wire for `harness`. Concrete providers implement it."""
        raise NotImplementedError

    async def stream(
        self,
        harness: ExternalHarness,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
    ) -> AsyncGenerator[events_.Event]:
        """Answer `messages` with one assistant message, as a model does.

        Use `ai.stream(harness, messages)` or `ai.Agent.run(harness, ...)`. In
        an open session (`async with harness.session()`), the call runs on the
        live CLI. Without one, the call starts the CLI, continues
        `harness.session_id`, and stops the CLI when the turn ends. An external
        harness with no `session_id` starts a new session, seeded with the past
        in `messages` (lossy), and remembers its id.

        The message ends with `finish_reason="tool_call"` when the CLI calls
        `ai` tools from `tools=`; the CLI waits for the results, which come with
        the next call. To cancel the stream or to stop reading it interrupts the
        turn. `params` has no effect: use external harness settings.
        """
        if output_type is not None:
            raise errors.UnsupportedError(
                "external harnesses do not support output_type yet"
            )
        pin = harness.session_id
        # A session of this external harness that is open here: the live CLI.
        session = next(
            (
                s
                for s in reversed(session_._OPEN.get())
                if s._open
                and (s.harness is harness or (pin is not None and s.id == pin))
            ),
            None,
        )
        stale = None
        if session is None:
            # No block: the session that the last call made, if it is still
            # the external harness's session.
            session = harness._session
            if session is None or session.id != pin:
                stale, session = session, session_.Session(harness)
        if session._stream_lock.locked():
            raise errors.ExternalHarnessError(
                "an external harness runs one stream() at a time"
            )
        seeds = not session.messages
        new = session._sync(messages)
        seeds = seeds and bool(session.messages)
        async with session._stream_lock:
            if not session._open:
                harness._session = session
            if stale is not None:
                # Its CLI can wait in a turn that nobody continues.
                await stale._stop_cli()
            if tools is not None:
                harness.tools = [t for t in tools if t.kind == "function"]
            run = await session._start_for_stream(seeds=seeds)
            queue: asyncio.Queue[Any] = asyncio.Queue()
            session._stream_queues.add(queue)
            ended = waits = False
            try:
                for message in new:
                    await session._send(message)
                started = False
                while True:
                    event = await queue.get()
                    if event is session_._END:
                        raise errors.ExternalHarnessError(
                            "the external harness closed during stream()"
                        )
                    if isinstance(event, BaseException):
                        raise event
                    if isinstance(event, session_.MessageEvent):
                        continue
                    if isinstance(event, events_.StreamStart):
                        started = True
                    if not started:
                        continue  # the end of a message that began earlier
                    if isinstance(event, events_.StreamEnd):
                        ended = True
                        waits = event.finish_reason == "tool_call"
                        error = run.failures.pop(event.message.id, None)
                        if error is not None:
                            raise error
                        yield event
                        return
                    yield event
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
                session._stream_queues.discard(queue)
                if not session._open and not waits:
                    # Nobody holds the session open: the CLI runs for one
                    # turn. After `tool_call` it waits for the results.
                    await session._stop_cli()

    async def generate(
        self,
        harness: ExternalHarness,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
    ) -> messages_.Message:
        """Not implemented: `ai.experimental_generate` drains `stream()`."""
        raise NotImplementedError


class ExternalHarness(pydantic.BaseModel):
    """A coding-agent CLI, in the shape of an `ai.Model`.

    Use it where a model goes. As with a model, give the whole history::

        async with ai.stream(harness, messages) as s: ...
        async with ai.Agent(tools=[...]).run(harness, messages) as run: ...

    Each call starts the CLI, continues the session `session_id`, and stops the
    CLI when the turn ends. An external harness with no `session_id` starts a
    new session and remembers its id. In a session block the same calls run on
    one live CLI, and the session gives direct control::

        async with harness.session() as session:
            async with ai.stream(harness, messages) as s: ...
            await session.send(ai.user_message("And the tests?"))

    The fields are plain data. `model_dump()` saves the external harness with
    its `session_id`, and `ExternalHarness.model_validate(data)` makes the
    correct subclass again; it continues the same CLI session. Two external
    harnesses with one `session_id` are two writers on one CLI session: for a
    new conversation, use `harness.model_copy(update={"session_id": None})`.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    harness_class_id: str
    provider: ExternalHarnessProvider
    """The CLI that runs the sessions."""
    model_name: str | None = None
    """The CLI's model. None: the CLI default."""
    cwd: str = pydantic.Field(default_factory=os.getcwd)
    """Where the CLI runs."""
    instructions: str | None = None
    """Added to the CLI's system prompt when a session starts."""
    tools: list[tools_.Tool] = pydantic.Field(default_factory=list)
    """`ai` function tools that the CLI can call through the MCP bridge."""
    session_id: str | None = None
    """The CLI's id for the session that the model calls continue.

    None: the first new session of the external harness sets it.
    """

    _classes: ClassVar[dict[str, type[ExternalHarness]]] = {}
    _session: session_.Session | None = pydantic.PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        if type(self) is ExternalHarness:
            raise TypeError(
                "ExternalHarness must be subclassed, "
                "for example ai.external.Codex"
            )
        super().__init__(**data)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:  # noqa: PLW3201
        super().__pydantic_init_subclass__(**kwargs)
        class_id = cls.model_fields["harness_class_id"].default
        if isinstance(class_id, str):
            existing = ExternalHarness._classes.get(class_id)
            if (
                existing is not None
                and existing.__qualname__ != cls.__qualname__
            ):
                raise RuntimeError(
                    f"duplicate external harness class id: {class_id!r}"
                )
            ExternalHarness._classes[class_id] = cls

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _load_registered_harness(
        cls, data: Any, handler: pydantic.ValidatorFunctionWrapHandler
    ) -> Any:
        if cls is ExternalHarness and isinstance(data, dict):
            harness_class = ExternalHarness._classes.get(
                str(data.get("harness_class_id"))
            )
            if harness_class is None:
                raise ValueError(
                    "unknown harness_class_id: "
                    f"{data.get('harness_class_id')!r}"
                )
            return harness_class.model_validate(data)
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

    def session(
        self,
        session_id: str | None = None,
        *,
        messages: list[messages_.Message] | None = None,
    ) -> session_.Session:
        """Return a session of the CLI.

        No argument: the external harness's session (`harness.session_id`), or a
        new session if the external harness has none. `session_id`: another
        session that the CLI has with this id. `messages`: the conversation so
        far. A new session starts with it (lossy); a session with an id has it
        already.
        """
        return session_.Session(self, session_id, messages)
