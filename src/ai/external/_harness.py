"""`ExternalHarness` and its provider: a CLI in the shape of a model.

An `ai.Model` is an id and a provider, and `ai.stream` calls
`model.provider.stream(model, messages)`. An external harness has the same
shape:

- `ExternalHarness` is the settings of the CLI. It comes with one `Session`,
  `harness.session`, and `async with harness` opens it.
- `ExternalHarnessProvider` is the CLI behind it: where it is and how it runs.
  Its `stream` makes one model call.

The state and direct control are in the session.
"""

from __future__ import annotations

import asyncio
import dataclasses
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

        Use `ai.stream(harness, messages)` or `ai.Agent.run(harness, ...)` in
        the block of the external harness (`async with harness`): the call
        runs on its live CLI. Outside the block, the call raises. A new
        session starts with the past in `messages` (lossy).

        The message ends with `finish_reason="tool_call"` when the CLI calls
        `ai` tools from `tools=`; the CLI waits for the results, which come with
        the next call. To cancel the stream or to stop reading it interrupts the
        turn. `params` has no effect: use external harness settings.
        """
        if output_type is not None:
            raise errors.UnsupportedError(
                "external harnesses do not support output_type yet"
            )
        session = harness.session
        if not session._open:
            raise errors.ExternalHarnessError(
                "a model call of an external harness needs an open external "
                "harness: use `async with harness`"
            )
        if session._stream_lock.locked():
            raise errors.ExternalHarnessError(
                "an external harness runs one stream() at a time"
            )
        seeds = not session.messages
        new = session._sync(messages)
        seeds = seeds and bool(session.messages)
        async with session._stream_lock:
            if tools is not None:
                harness.tools = [t for t in tools if t.kind == "function"]
            run = await session._start_for_stream(seeds=seeds)
            queue: asyncio.Queue[Any] = asyncio.Queue()
            session._stream_queues.add(queue)
            ended = False
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


@dataclasses.dataclass(kw_only=True, eq=False)
class ExternalHarness:
    """A coding-agent CLI, in the shape of an `ai.Model`.

    One external harness is one conversation. It comes with one session,
    `harness.session`, and it has one live CLI at a time. For two
    conversations, make two external harnesses.

    Open it, and use it where a model goes. As with a model, give the whole
    history::

        async with ai.external.Codex(cwd=".") as harness:
            async with ai.stream(harness, messages) as s: ...
            await harness.session.send(ai.user_message("And the tests?"))

    The block holds the live CLI: the model calls in it (`ai.stream`,
    `ai.Agent.run`) run on that CLI, and the session gives direct control. A
    model call outside the block raises. An external harness that closed
    opens again, and continues its session.

    The fields are the settings of the CLI. The conversation is in the
    session: `harness.session.id` and `harness.session.messages`. To continue
    a conversation later, save `harness.session.id` and give it to a new
    external harness as `resume`.

    A copy or a pickle is a closed external harness with the same settings
    and the same conversation, not the live CLI. `dataclasses.replace(harness)`
    gives the same settings with a new session.
    """

    harness_class_id: ClassVar[str]
    """The name of the kind of CLI, for telemetry and errors."""

    provider: ExternalHarnessProvider
    """The CLI that runs the session."""
    model_name: str | None = None
    """The CLI's model. None: the CLI default."""
    cwd: str = dataclasses.field(default_factory=os.getcwd)
    """Where the CLI runs."""
    instructions: str | None = None
    """Added to the CLI's system prompt when the session starts."""
    tools: list[tools_.Tool] = dataclasses.field(default_factory=list)
    """`ai` function tools that the CLI can call through the MCP bridge."""

    resume: dataclasses.InitVar[str | None] = None
    """The id of a session that the CLI has: the session continues it.

    None: a new session. Only the constructor takes it; read the id of the
    session from `harness.session.id`.
    """
    history: dataclasses.InitVar[list[messages_.Message] | None] = None
    """The conversation so far. A new session starts with it (lossy).

    With `resume`, the CLI has it already and nothing goes to the CLI. Only
    the constructor takes it; read the conversation from
    `harness.session.messages`.
    """

    session: session_.Session = dataclasses.field(init=False, repr=False)
    """The session of this external harness: its state and direct control."""

    def __post_init__(
        self, resume: str | None, history: list[messages_.Message] | None
    ) -> None:
        if type(self) is ExternalHarness:
            raise TypeError(
                "ExternalHarness must be subclassed, "
                "for example ai.external.Codex"
            )
        self.session = session_.Session(self, resume, history)

    def __getstate__(self) -> dict[str, Any]:
        # For `copy` and `pickle`: the settings and the conversation, not the
        # live CLI.
        session = self.session
        return {**self.__dict__, "session": (session.id, session.messages)}

    def __setstate__(self, state: dict[str, Any]) -> None:
        session_id, messages = state["session"]
        self.__dict__.update(state)
        self.session = session_.Session(self, session_id, messages)

    async def __aenter__(self) -> Self:
        """Open the external harness: its model calls and its session work.

        The CLI starts with the first `session.send()` or the first model
        call, not here.
        """
        self.session._enter()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Stop the CLI and close. The state of the session stays."""
        await self.session._close()

    @property
    def id(self) -> str:
        return self.model_name or "default"
