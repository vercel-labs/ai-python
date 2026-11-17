"""Claude Code, Codex, Fx and Pi coding-agent CLIs as external harnesses.

Experimental: not part of the stable API, may change or be removed.

An external harness is a CLI in the shape of a model: plain data and a provider.
Use it where a model goes, in a session block, with the whole history::

    import ai

    harness = ai.external.Codex(cwd=".")
    messages = [ai.user_message("Fix the tests")]
    async with harness.session():
        async with ai.stream(harness, messages) as s:
            async for event in s:
                ...

    # ai tools reach the CLI through a local MCP bridge
    async with harness.session():
        async with ai.Agent(tools=[search_docs]).run(harness, messages) as run:
            ...

The block holds one live CLI, and the model calls in it run on that CLI. A model
call outside a session block raises. The session also gives direct control::

    async with harness.session() as session:
        await session.send(ai.user_message("Fix the tests"))
        async for event in session.events():
            ...  # session.send() again to steer, session.interrupt() to stop

`harness.session()` starts a new session; `harness.session(session_id)`
continues one that the CLI has, and `harness.session(messages=history)` makes
one that starts with a past. The CLI's own tools come as
`ai.events.BuiltinTool*` events. The state is in the session, as plain data:
`session.id` is the CLI's id for the session, and `session.messages` holds the
conversation in `ai` messages. The external harness holds only settings. The
provider says where the CLI is:
`ai.external.CodexProvider(executable=..., env=...)`.
"""

from ._errors import (
    ExecutableMissingError,
    ExternalHarnessError,
    NotAuthenticatedError,
    TranscriptMismatchError,
    UnsupportedError,
)
from ._harness import ExternalHarness, ExternalHarnessProvider
from ._session import (
    ExternalHarnessEvent,
    ExternalHarnessMessageSpanData,
    ExternalHarnessSessionSpanData,
    MessageEvent,
    Session,
)
from .claude_code import ClaudeCode, ClaudeCodeProvider
from .codex import Codex, CodexProvider
from .fx import Fx, FxProvider
from .pi import Pi, PiProvider

__all__ = [
    "ClaudeCode",
    "ClaudeCodeProvider",
    "Codex",
    "CodexProvider",
    "ExecutableMissingError",
    "ExternalHarness",
    "ExternalHarnessError",
    "ExternalHarnessEvent",
    "ExternalHarnessMessageSpanData",
    "ExternalHarnessProvider",
    "ExternalHarnessSessionSpanData",
    "Fx",
    "FxProvider",
    "MessageEvent",
    "NotAuthenticatedError",
    "Pi",
    "PiProvider",
    "Session",
    "TranscriptMismatchError",
    "UnsupportedError",
]
