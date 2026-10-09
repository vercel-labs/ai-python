"""The tool-approval hook: optional, tool-call level, no invented taxonomy.

The hook sees the harness's OWN tool call — native name, native input —
plus the conversation so far and the workspace. The SDK never classifies a
call into capabilities on the harness's behalf: `Bash("ls")` and
`Bash("rm -rf /")` are the same tool, and `mcp__x__y` means nothing to us.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from ai.harnesses.experimental import (
    Allow,
    ApprovalContext,
    Decision,
    Deny,
    claude_code,
)
from ai.harnesses.experimental.errors import UnsupportedError
from ai.workspaces.experimental import Workspace
from tests.harnesses.experimental.conftest import requires, requires_claude

pytestmark = pytest.mark.live


# Keys that DENOTE a filesystem location in either harness's native payload.
# claude: file_path / notebook_path / path / cwd. codex: cwd, and `changes`
# keyed by path. A shell command's argv is not here on purpose: which files
# `bash -lc "…"` touches is unknowable from the payload — the README's own
# point — and codex echoes the argv under several keys (`command`,
# `commandActions`, `proposedExecpolicyAmendment`, `availableDecisions`),
# where `/bin/bash` is the interpreter, not a write target. A blacklist of
# those keys would rot; a whitelist of location keys will not.
_LOCATION_KEYS = {
    "cwd",
    "path",
    "file_path",
    "filePath",
    "notebook_path",
    "paths",
    "grantRoot",
}


def _absolute_paths(payload: Any) -> list[str]:
    """Every absolute path the call would TOUCH, from a native payload."""
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "changes" and isinstance(value, dict):
                found.extend(
                    k for k in value if isinstance(k, str) and k.startswith("/")
                )
            elif key in _LOCATION_KEYS:
                values = value if isinstance(value, list) else [value]
                found.extend(
                    v
                    for v in values
                    if isinstance(v, str) and v.startswith("/")
                )
            else:
                found.extend(_absolute_paths(value))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(_absolute_paths(value))
    return found


async def test_hook_receives_the_native_call_and_its_context(
    any_make_harness: Callable[..., Any], any_workspace: Workspace
) -> None:
    seen = []

    async def approve(ctx: ApprovalContext) -> Decision:
        seen.append(ctx)
        return Allow()

    async with any_make_harness(approve=approve) as any_harness:
        requires(any_harness, "approve")
        await any_harness.run("Write the word HELLO into greeting.txt.")

    assert seen, "a write must reach the hook"
    ctx = seen[0]
    assert ctx.call.name, "the native tool name, verbatim from the any_harness"
    assert isinstance(ctx.call.input, dict), "native input, unnormalized"
    assert (
        ctx.call.raw is not None
    ), "the untouched native payload is always available"
    assert ctx.workspace is any_workspace
    assert any(
        m.role == "user" for m in ctx.history
    ), "history travels with the request"


async def test_the_hook_never_receives_an_invented_capability(
    any_make_harness: Callable[..., Any],
) -> None:
    """`kind` is a passthrough hint from the any_harness, not an SDK
    judgment."""
    seen = []

    async def approve(ctx: ApprovalContext) -> Decision:
        seen.append(ctx.call)
        return Allow()

    async with any_make_harness(approve=approve) as any_harness:
        requires(any_harness, "approve")
        await any_harness.run("Write the word HELLO into hint.txt.")

    call = seen[0]
    assert not hasattr(
        call, "capability"
    ), "the SDK must not classify native tools"
    assert call.kind is None or isinstance(call.kind, str)


async def test_no_hook_approves_everything(
    any_make_harness: Callable[..., Any], any_workspace: Workspace
) -> None:
    """Absent hook is approve-all, not deny-all and not ask-a-human.

    The any_harness's own default is to ask the person at the terminal. In a
    program there is no such person, so the agent would sit on a question
    nobody answers and then narrate what it would have done — a turn that
    looks successful and changed nothing. The hook is how you RESTRICT an
    agent; not passing one lets it work.
    """
    async with any_make_harness(writable=True) as any_harness:
        await any_harness.run(
            "Write the word HELLO into greeting.txt, then reply DONE."
        )

    assert "HELLO" in await any_workspace.read_text("greeting.txt")


async def test_deny_blocks_the_call_and_the_reason_reaches_the_agent(
    any_make_harness: Callable[..., Any], any_workspace: Workspace
) -> None:
    async def approve(ctx: ApprovalContext) -> Decision:
        # Deny everything. Keying on native tool names would make this test
        # a claude test — "Write" on one any_harness is "fileChange" on the
        # other, which is exactly why the SDK refuses to classify them.
        return Deny(
            "this any_workspace is read-only for you; do not write files"
        )

    async with any_make_harness(approve=approve, writable=True) as any_harness:
        requires(any_harness, "approve")
        # Codex's approval response carries only a decision — the protocol
        # has no field for a reason — so the text cannot reach the agent
        # there. Blocking is universal; explaining is not.
        reason_travels = any_harness.capabilities.deny_reason
        result = await any_harness.run(
            "Write the word HELLO into greeting.txt. "
            + "If you are not permitted, say exactly why you were refused."
        )

    assert not await any_workspace.exists(
        "greeting.txt"
    ), "the denial must actually block"
    if reason_travels:
        assert "read-only" in result.text.lower(), "the agent was told why"


async def test_deny_with_stop_ends_the_turn(
    any_make_harness: Callable[..., Any],
) -> None:
    async def approve(ctx: ApprovalContext) -> Decision:
        return Deny("not allowed", stop=True)

    import time

    async with any_make_harness(approve=approve, writable=True) as any_harness:
        requires(any_harness, "approve")
        started = time.monotonic()
        result = await any_harness.run(
            "Write A into one.txt, then write B into two.txt, then reply DONE."
        )
        took = time.monotonic() - started
        # It STOPPED: nothing after the denied call happened.
        assert not await any_harness.workspace.exists("one.txt")
        assert not await any_harness.workspace.exists("two.txt")

    assert result.finish_reason == "cancelled"
    # And promptly. This passed at 121.77s: the interrupt was awaited before
    # the approval was answered, which codex will not process, so the
    # decline waited out a 120s RPC timeout. Right outcome, two minutes late.
    assert took < 60, f"Deny(stop=True) took {took:.0f}s to end the turn"


async def test_hook_can_use_the_workspace_to_decide(
    any_make_harness: Callable[..., Any], any_workspace: Workspace
) -> None:
    """Containment is the any_workspace's judgment, never the caller's pathlib —
    the same code has to be correct when the any_workspace is remote."""
    decisions: list[tuple[str, bool]] = []
    consulted: list[str] = []

    async def approve(ctx: ApprovalContext) -> Decision:
        consulted.append(ctx.call.name)
        # Portable: scan the raw payload for absolute paths and ask the
        # WORKSPACE about each one, rather than knowing any any_harness's
        # argument names.
        for path in _absolute_paths(ctx.call.raw):
            inside = ctx.workspace.contains(path)
            decisions.append((path, inside))
            if not inside:
                return Deny("outside the any_workspace")
        return Allow()

    async with any_make_harness(approve=approve, writable=True) as any_harness:
        requires(any_harness, "approve")
        await any_harness.run("Write OK into notes.txt inside this project.")

    assert consulted, "the hook must be asked before a write happens"
    escaped = [path for path, inside in decisions if not inside]
    assert (
        not escaped
    ), f"paths judged outside the workspace: {escaped} (calls: {consulted})"
    assert await any_workspace.exists("notes.txt")


async def test_remember_either_works_or_refuses(
    any_make_harness: Callable[..., Any],
) -> None:
    """`Allow(remember=True)` must not be a flag that does nothing.

    Where a any_harness can persist a decision, the same call is not asked
    twice. Where it cannot — Claude's session-scoped rules do not suppress
    the next identical request, and the only mechanism that does would
    blanket-accept every edit — the SDK raises instead of quietly widening
    the caller's authority.
    """
    calls: list[str] = []

    async def approve(ctx: ApprovalContext) -> Decision:
        calls.append(ctx.call.name)
        return Allow(remember=True)

    prompt = "Write A into one.txt, then write B into two.txt, then reply DONE."
    async with any_make_harness(approve=approve, writable=True) as any_harness:
        requires(any_harness, "approve")
        if any_harness.capabilities.remember_decisions:
            await any_harness.run(prompt)
            assert calls, "the first call still asks"
            assert len(calls) < 2 or calls[0] != calls[1]
        else:
            # Same contract on every any_harness: asking for a capability that
            # is not there raises, rather than doing nothing quietly.
            with pytest.raises(UnsupportedError, match="remember"):
                await any_harness.run(prompt)


async def test_a_raising_hook_denies_and_surfaces_the_error(
    any_make_harness: Callable[..., Any], any_workspace: Workspace
) -> None:
    """An approval gate that fails open is not a gate."""

    async def approve(ctx: ApprovalContext) -> Decision:
        raise ValueError("hook is broken")

    async with any_make_harness(approve=approve) as any_harness:
        requires(any_harness, "approve")
        result = await any_harness.run("Write OK into oops.txt.")

    assert not await any_workspace.exists("oops.txt")
    assert result.approval_errors, "the broken hook is reported, not swallowed"


@pytest.mark.slow
async def test_a_hanging_hook_denies_on_timeout(
    any_make_harness: Callable[..., Any], any_workspace: Workspace
) -> None:
    async def approve(ctx: ApprovalContext) -> Decision:
        await asyncio.sleep(3600)
        return Allow()

    async with any_make_harness(
        approve=approve, approval_timeout=5
    ) as any_harness:
        requires(any_harness, "approve")
        await any_harness.run("Write OK into hang.txt.")

    assert not await any_workspace.exists("hang.txt")


async def test_hook_has_no_control_verbs(
    any_make_harness: Callable[..., Any],
) -> None:
    """No reentrancy: the agent is blocked on this hook, so calling back into
    the session would deadlock. Talking back is `Deny(reason)`; stopping is
    `Deny(reason, stop=True)`."""
    probed = []

    async def approve(ctx: ApprovalContext) -> Decision:
        probed.append(
            [v for v in ("run", "stream", "steer", "stop") if hasattr(ctx, v)]
        )
        return Allow()

    async with any_make_harness(approve=approve) as any_harness:
        requires(any_harness, "approve")
        await any_harness.run("Write the word HELLO into probe.txt.")

    assert probed and all(found == [] for found in probed)


async def test_a_gate_is_not_an_audit_log(
    any_make_harness: Callable[..., Any], any_workspace: Workspace
) -> None:
    """A hook sees what the any_harness asks about — not everything that runs.

    Claude never routes read-only tools through its permission callback, so
    a Read can happen without the hook hearing about it. The EVENT STREAM
    sees it. Anyone treating approvals as an audit trail would be wrong, and
    `capabilities.approval == "policy"` is the SDK saying so out loud.
    """
    from ai.types.events import ToolStart

    seen_by_hook: list[str] = []

    async def approve(ctx: ApprovalContext) -> Decision:
        seen_by_hook.append(ctx.call.name)
        return Allow()

    async with any_make_harness(approve=approve) as any_harness:
        requires(any_harness, "approve")
        assert any_harness.capabilities.approval == "policy"
        session = any_harness.session()
        turn = session.stream("Read util.py and reply with the function name.")
        streamed = [e.tool_name async for e in turn if isinstance(e, ToolStart)]

    assert len(seen_by_hook) <= len(streamed), (
        "the hook can never see MORE calls than actually ran; approvals are "
        + "a subset of what the event stream observes, never a superset"
    )


@requires_claude
async def test_allow_can_rewrite_the_call_input_where_supported(
    any_workspace: Workspace,
) -> None:
    """Input rewriting is real on Claude and absent on Codex; see
    test_capabilities.py for the other half of this contract."""

    async def approve(ctx: ApprovalContext) -> Decision:
        if ctx.call.name == "Write":
            return Allow(input={**ctx.call.input, "content": "REWRITTEN"})
        return Allow()

    async with claude_code(
        workspace=any_workspace, approve=approve
    ) as any_harness:
        requires(any_harness, "rewrite_tool_input")
        await any_harness.run("Write the word ORIGINAL into rewritten.txt.")

    assert "REWRITTEN" in await any_workspace.read_text("rewritten.txt")
