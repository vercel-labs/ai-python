"""Coding-agent CLIs as workers.

Experimental: not part of the stable API, may change or be removed.

A worker is one session of a CLI. Use it as a model::

    import ai
    from ai import workers

    worker = workers.Codex(cwd=".")
    async with ai.stream(worker, [ai.user_message("Fix the tests")]) as s:
        async for event in s:
            ...

    # ai tools reach the CLI through a local MCP bridge
    async with ai.Agent(tools=[search_docs]).run(worker, messages) as run:
        ...

Or control it directly::

    async with workers.Codex(cwd=".") as worker:
        feed = worker.events()
        await worker.send("Fix the tests")
        async for event in feed:
            ...  # call worker.send() again to steer, worker.interrupt() to stop

The CLI's own tools come as `ai.events.BuiltinTool*` events. The state is
plain data: `worker.messages` holds the conversation in `ai` messages, and
`worker.model_dump()` / `workers.Worker.model_validate(data)` save and
continue it.
"""

from ._errors import (
    ExecutableMissingError,
    NotAuthenticatedError,
    TranscriptMismatchError,
    UnsupportedError,
    WorkerError,
)
from ._worker import (
    MessageEvent,
    Worker,
    WorkerEvent,
    WorkerMessageSpanData,
    WorkerSessionSpanData,
    is_interrupted,
)
from .claude_code import ClaudeCode
from .codex import Codex
from .pi import Pi

__all__ = [
    "ClaudeCode",
    "Codex",
    "ExecutableMissingError",
    "MessageEvent",
    "NotAuthenticatedError",
    "Pi",
    "TranscriptMismatchError",
    "UnsupportedError",
    "Worker",
    "WorkerError",
    "WorkerEvent",
    "WorkerMessageSpanData",
    "WorkerSessionSpanData",
    "is_interrupted",
]
