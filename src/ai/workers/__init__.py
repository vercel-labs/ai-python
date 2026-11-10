"""Coding-agent CLIs as workers.

Experimental: not part of the stable API, may change or be removed.

A worker is one session of a CLI, in the shape of a model: plain data and
a provider. Open a session, and use the worker as a model::

    import ai
    from ai import workers

    worker = workers.Codex(cwd=".")
    async with worker.session():
        messages = [ai.user_message("Fix the tests")]
        async with ai.stream(worker, messages) as s:
            async for event in s:
                ...

        # ai tools reach the CLI through a local MCP bridge
        async with ai.Agent(tools=[search_docs]).run(worker, messages) as run:
            ...

Or control the session directly::

    async with worker.session() as session:
        feed = session.events()
        await session.send(ai.user_message("Fix the tests"))
        async for event in feed:
            ...  # session.send() again to steer, session.interrupt() to stop

Both work in one session. The CLI's own tools come as
`ai.events.BuiltinTool*` events. The state is plain data: `worker.messages`
holds the conversation in `ai` messages, and `worker.model_dump()` /
`workers.Worker.model_validate(data)` save and continue it. The provider
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
    is_interrupted,
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
    "is_interrupted",
]
