"""The tool-approval hook's vocabulary.

The hook sees the harness's OWN tool call — native name, native input — and
decides. This SDK deliberately does NOT classify calls into capabilities:
`Bash("ls")` and `Bash("rm -rf /")` are the same tool, `mcp__x__y` means
nothing to us, and a subagent can call anything. Inventing a taxonomy would
be inventing certainty.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...types import messages
from ...workspaces.experimental import _base


class ToolCall(BaseModel):
    """One call the harness is asking about, as the harness described it."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    """The native tool name, verbatim, taken from the harness's own
    callback — never from display text an agent can influence."""
    input: dict[str, Any] = Field(default_factory=dict)
    """Native arguments, unnormalized."""
    title: str | None = None
    """Harness display copy. Never authoritative; do not key policy on it."""
    kind: str | None = None
    """A hint the HARNESS volunteered, passed through untouched.

    May be absent, may be wrong. Never inferred by this SDK.
    """
    agent_id: str | None = None
    """Set when a subagent made the call."""
    hints: dict[str, Any] = Field(default_factory=dict)
    """Extras the harness supplied (e.g. the path that triggered the ask)."""
    raw: dict[str, Any] = Field(default_factory=dict)
    """The untouched native payload, always available as the escape hatch."""


class ApprovalContext(BaseModel):
    """What the hook gets to decide with.

    Carries no control verbs by design: the agent is blocked waiting on this
    decision, so calling back into the session would deadlock. Talking to the
    agent is `Deny(reason)`; stopping it is `Deny(reason, stop=True)`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    call: ToolCall
    history: list[messages.Message] = Field(default_factory=list)
    workspace: _base.Workspace
    harness: str
    session_id: str
    prompt: str | None = None


class Allow(BaseModel):
    """Let the call run."""

    model_config = ConfigDict(frozen=True)

    decision: Literal["allow"] = "allow"
    remember: bool = False
    """Persist this decision for the rest of the session.

    Neither harness supports it yet (`Capabilities.remember_decisions` is
    False on both), so asking raises UnsupportedError on the turn rather
    than silently asking again or widening authority.
    """
    input: dict[str, Any] | None = None
    """Rewrite the call's arguments before it runs.

    Claude supports this; codex does not, and asking there raises
    UnsupportedError rather than silently running the original.
    """


class Deny(BaseModel):
    """Block the call.

    The reason is delivered to the agent as guidance — the same channel as
    `steer`.
    """

    model_config = ConfigDict(frozen=True)

    decision: Literal["deny"] = "deny"
    reason: str = ""
    stop: bool = False
    """Also end the turn, rather than letting the agent try something else."""

    def __init__(self, reason: str = "", /, **data: Any) -> None:
        # `Deny("why")` and `Deny(reason="why")` both work; the keyword wins.
        super().__init__(**{"reason": reason, **data})


Decision = Allow | Deny

# Resolve forward references eagerly. A model that is "not fully defined"
# raises only when it is first CONSTRUCTED — which here is inside the
# harness's permission callback, where the failure surfaced as an opaque
# "internal system error" from the agent and never reached the caller.
ApprovalContext.model_rebuild()
