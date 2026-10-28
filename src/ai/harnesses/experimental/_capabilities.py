"""What a harness can actually do, declared before anything runs.

Absence is advertised here and raises at the call site. A capability that is
quietly ignored is indistinguishable from one that works — the most
expensive lie this SDK could tell, and the one the previous version told.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

ApprovalCoverage = Literal["all", "policy", "none"]


class Capabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    run: bool = True
    stream: bool = True
    steer: bool = False
    stop: bool = False
    resume: bool = False
    """Continue a conversation this process did not create — by id, or from
    a handle in another process. Named for what the harness CLIs call it."""
    fork: bool = False
    """Branch a conversation instead of writing into it — the safe way to
    pick up one a human still has open."""
    history: bool = False
    """Read a stored transcript.

    When true it is FULL fidelity: tool calls and results included, never a
    display summary.
    """
    rewrite_tool_input: bool = False
    deny_reason: bool = False
    """Whether `Deny(reason)` actually reaches the agent.

    Claude's permission result carries a message; Codex's approval response
    carries only a decision — the protocol has no field for a reason — so
    there the text is dropped and the agent only learns it was refused.
    """
    remember_decisions: bool = False
    """Whether `Allow(remember=True)` can persist a decision for the session.

    False where the harness offers no faithful mechanism — asking for one there
    raises rather than doing nothing.
    """
    images: bool = False
    """Images (`FilePart` with an `image/*` type) in a prompt."""
    documents: bool = False
    """PDFs (`FilePart` with `application/pdf`) in a prompt. Text files
    (`text/*`) are taken by every harness, as text."""

    approval: ApprovalCoverage = "none"
    """What the hook actually sees:

    - ``all`` — every tool call is offered, read-only ones included.
      Claude Code: the hook runs as a PreToolUse hook (measured: Read, a
      read-only Bash and Write all reach it).
    - ``policy`` — the HARNESS's own policy decides which calls need
      approval, and only those reach the hook. Codex: under its strictest
      policy (`untrusted`), every shell command and file change was asked
      about when measured, `cat` and `ls` included, but its other item kinds
      (MCP tools, web search) are not verified, so it does not claim all.
    - ``none`` — no approval channel exists.

    Below ``all``, an approval hook is a GATE, not an audit log. To observe
    every tool call, read the event stream (`ToolStart`/`ToolEnd`), which
    sees all of them.
    """

    @property
    def approve(self) -> bool:
        return self.approval != "none"
