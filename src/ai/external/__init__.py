"""Claude Code, Codex, Fx and Pi coding-agent CLIs as external harnesses.

Experimental: not part of the stable API, may change or be removed.

An external harness is a CLI in the shape of a model: settings and a provider.
Open it, and use it where a model goes, with the whole history::

    import ai

    messages = [ai.user_message("Fix the tests")]
    async with ai.external.Codex(cwd=".") as harness:
        async with ai.stream(harness, messages) as s:
            async for event in s:
                ...

    # ai tools reach the CLI through a local MCP bridge
    async with ai.external.Codex(cwd=".") as harness:
        async with ai.Agent(tools=[search_docs]).run(harness, messages) as run:
            ...

The block holds one live CLI, and the model calls in it run on that CLI. A model
call outside the block raises. One external harness is one conversation: it
comes with one session, `harness.session`, which has the state and gives direct
control::

    async with ai.external.Codex(cwd=".") as harness:
        await harness.session.send(ai.user_message("Fix the tests"))
        async for event in harness.session.events():
            ...  # send() again to steer, harness.session.interrupt() to stop

`Codex()` has a new session; `Codex(resume=session_id)` continues one that the
CLI has, and `Codex(history=messages)` has a new one that starts with a past.
The CLI's own tools come as `ai.events.BuiltinTool*` events. The state is plain
data: `harness.session.id` is the CLI's id for the session, and
`harness.session.messages` holds the conversation in `ai` messages. The provider
says where the CLI is: `ai.external.CodexProvider(executable=..., env=...)`.
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
