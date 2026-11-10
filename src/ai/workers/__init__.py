"""Coding-agent CLIs as workers.

Experimental: not part of the stable API, may change or be removed.

A worker is a CLI in the shape of a model: plain data and a provider. Use
it where a model goes, with the whole history::

    import ai
    from ai import workers

    worker = workers.Codex(cwd=".")
    messages = [ai.user_message("Fix the tests")]
    async with ai.stream(worker, messages) as s:
        async for event in s:
            ...

    # ai tools reach the CLI through a local MCP bridge
    async with ai.Agent(tools=[search_docs]).run(worker, messages) as run:
        ...

Each call starts the CLI, continues the session `worker.session_id`, and
stops the CLI when the turn ends. A worker with no `session_id` starts a new
session and remembers its id. Open a session to keep the CLI alive between
turns and to control it::

    async with worker.session() as session:
        await session.send(ai.user_message("Fix the tests"))
        async for event in session.events():
            ...  # session.send() again to steer, session.interrupt() to stop

Model calls in the block run on that live session. `worker.session()`
gives the worker's session; `worker.session(session_id)` continues another
one, and `worker.session(messages=history)` makes one that starts with a
past. The CLI's own tools come as `ai.events.BuiltinTool*` events. The state
is plain data: `session.id` is the CLI's id for the session, and
`session.messages` holds the conversation in `ai` messages. The provider
says where the CLI is: `workers.CodexProvider(executable=..., env=...)`.
"""

from ._errors import (
    ExecutableMissingError,
    NotAuthenticatedError,
    TranscriptMismatchError,
    UnsupportedError,
    WorkerError,
)
from ._session import (
    MessageEvent,
    Session,
    WorkerEvent,
    WorkerMessageSpanData,
    WorkerSessionSpanData,
)
from ._worker import Worker, WorkerProvider
from .claude_code import ClaudeCode, ClaudeCodeProvider
from .codex import Codex, CodexProvider
from .pi import Pi, PiProvider

__all__ = [
    "ClaudeCode",
    "ClaudeCodeProvider",
    "Codex",
    "CodexProvider",
    "ExecutableMissingError",
    "MessageEvent",
    "NotAuthenticatedError",
    "Pi",
    "PiProvider",
    "Session",
    "TranscriptMismatchError",
    "UnsupportedError",
    "Worker",
    "WorkerError",
    "WorkerEvent",
    "WorkerMessageSpanData",
    "WorkerProvider",
    "WorkerSessionSpanData",
]
