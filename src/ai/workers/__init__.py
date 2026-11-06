"""Coding-agent CLIs as workers.

Experimental: not part of the stable API, may change or be removed.

A worker goes where a model goes, and it keeps its conversation::

    import ai
    from ai import workers

    async with workers.Codex(model="gpt-5.5") as worker:
        async with ai.stream(worker, [ai.user_message("Fix the tests")]) as s:
            async for event in s:
                ...

        # an ai.Agent's tools reach the CLI through MCP
        async with ai.Agent(tools=[search_docs]).run(worker, messages) as run:
            ...

The CLI's own tools come back as `ai.events.BuiltinTool*` events.

A worker is a pydantic model: `worker.model_dump()` saves the conversation,
`workers.Worker.model_validate(data)` continues it, `worker.model_copy()` is
a new worker on the same settings, and `Codex(session_id=...)` resumes
a conversation the CLI has stored.
"""

from ._adapters.base import SessionInfo
from ._errors import (
    ExecutableMissingError,
    NotAuthenticatedError,
    TranscriptMismatchError,
    WorkerError,
)
from ._kinds import Codex
from ._worker import Worker

__all__ = [
    "Codex",
    "ExecutableMissingError",
    "NotAuthenticatedError",
    "SessionInfo",
    "TranscriptMismatchError",
    "Worker",
    "WorkerError",
]
