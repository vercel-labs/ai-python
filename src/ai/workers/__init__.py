"""Coding-agent CLI sessions: Claude Code, Codex and Pi.

Experimental: not part of the stable API, may change or be removed.

Use a worker as a model, continuing from its current message mirror::

    import ai
    from ai import workers

    async with workers.ClaudeCode(model_name="sonnet") as worker:
        async with ai.stream(worker, [ai.user_message("Fix the tests")]) as s:
            async for event in s:
                ...
        reply = await ai.experimental_generate(
            worker, [*worker.messages, ai.user_message("Summarize the fix.")]
        )

Or use ``await worker.send(prompt)`` and ``worker.events()`` directly.
``SessionEvent`` includes model events (``StreamStart``/``StreamEnd``),
``UserMessageAccepted`` and ``HookEvent``. The feed spans turns until close;
there is exactly one reader, including ``stream()`` or ``ai.Agent.run()``.
Explicitly close a reader when leaving it early (``contextlib.aclosing``).

With ``approve=True``, direct consumers answer ``PendingHookEvent`` outside
Agent using ``ai.resolve_hook(event.hook, {"granted": False},
registry=worker.hook_registry)``. Bare model streaming cannot deliver these
approvals: use the direct feed or Agent. Python tools run only through
``ai.Agent(tools=[...]).run(worker, messages)`` and reach the CLI via MCP;
CLI-owned tools appear as ``ai.events.BuiltinTool*`` events.

``send()`` waits for transport acceptance, not an answer or durable storage.
The reader updates ``messages``; once the turn and reader stop, ``sync()``
replaces that mirror from the authoritative CLI store (also after TUI use).
``interrupt()`` requests a clean turn end. ``await worker.fork()`` returns
an open native branch with a new session id; close it when done. Copying
preserves the same session id and is not a fork.

Dump/validate with ``model_dump()`` / ``Worker.model_validate()`` to resume
settings and the mirror; native storage is still required. Migrate old
``kind``/``model`` keys to ``worker_class_id``/``model_name`` and remove the
old ``fork`` field; no legacy aliases exist. ``executable`` remains a string
(PATH command or filesystem path), or None for the CLI default. A mismatched
history raises ``TranscriptMismatchError`` instead of silently reseeding.
Importing history into a fresh worker is lossy, unlike a native fork.

Concrete classes own private CLI transport hooks; adapter objects are gone.
See WORKERS.md for examples, migration details and protocol limitations.
"""

from ._adapter import SessionInfo
from ._errors import (
    ExecutableMissingError,
    NotAuthenticatedError,
    TranscriptMismatchError,
    UnsupportedError,
    WorkerError,
)
from ._worker import SessionEvent, UserMessageAccepted, Worker
from .claude_code import ClaudeCode
from .codex import Codex
from .pi import Pi

__all__ = [
    "ClaudeCode",
    "Codex",
    "ExecutableMissingError",
    "NotAuthenticatedError",
    "Pi",
    "SessionEvent",
    "SessionInfo",
    "TranscriptMismatchError",
    "UnsupportedError",
    "UserMessageAccepted",
    "Worker",
    "WorkerError",
]
