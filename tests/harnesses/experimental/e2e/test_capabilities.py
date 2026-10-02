"""Capability honesty: absent is advertised and raises, never silently no-ops.

This is the rule the previous SDK learned the hard way — a permission policy
that was quietly ignored on one harness looked exactly like a policy that
worked. Every asymmetry between harnesses has to be legible here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ai.harnesses.experimental import (
    Allow,
    ApprovalContext,
    Decision,
    Harness,
    codex,
)
from ai.harnesses.experimental.errors import UnsupportedError
from ai.workspaces.experimental import Workspace
from tests.harnesses.experimental.conftest import requires

pytestmark = pytest.mark.live

CONTROL_VERBS = ("steer", "stop", "resume", "rewrite_tool_input")


async def test_capabilities_are_declared_before_any_turn(
    harness: Harness,
) -> None:
    caps = harness.capabilities

    for verb in CONTROL_VERBS:
        assert isinstance(getattr(caps, verb), bool), f"{verb} must be declared"
    assert caps.approval in ("all", "policy", "none")


async def test_every_harness_can_run_and_stream(harness: Harness) -> None:
    """The floor: no harness is admitted that cannot do these."""
    assert harness.capabilities.run is True
    assert harness.capabilities.stream is True


async def test_an_absent_capability_raises_rather_than_no_ops(
    harness: Harness,
) -> None:
    if harness.capabilities.steer:
        pytest.skip(f"{harness.name} supports steering; nothing to assert here")

    session = harness.session()
    turn = session.stream("Reply with exactly: READY")
    with pytest.raises(UnsupportedError, match="steer"):
        await session.steer("this harness cannot do this")
    async for _ in turn:
        pass


async def test_input_rewriting_is_refused_where_unsupported(
    make_harness: Callable[..., Any],
) -> None:
    """Codex's approval protocol is accept/acceptForSession/decline with no
    way to rewrite a call. Asking for one must fail loudly, not be dropped."""

    async def approve(ctx: ApprovalContext) -> Decision:
        return Allow(input={**ctx.call.input, "content": "REWRITTEN"})

    async with make_harness(approve=approve) as harness:
        requires(harness, "approve")
        if harness.capabilities.rewrite_tool_input:
            pytest.skip(f"{harness.name} supports input rewriting")
        with pytest.raises(UnsupportedError, match="rewrite"):
            await harness.run("Write the word ORIGINAL into rewritten.txt.")


async def test_no_harness_claims_to_offer_every_tool_call(
    harness: Harness,
) -> None:
    """Neither harness routes read-only tools through its approval channel,
    so neither may advertise "all". Claiming it would be the old lie in new
    clothes: a caller would believe a gate covers calls it never sees."""
    assert harness.capabilities.approval != "all"


async def test_capabilities_do_not_change_after_open(harness: Harness) -> None:
    before = harness.capabilities.model_dump()
    await harness.run("Reply with exactly: READY")

    assert harness.capabilities.model_dump() == before


@pytest.mark.sandbox
@pytest.mark.parametrize("mode", ["read-only", "workspace-write"])
async def test_codex_refuses_a_sandbox_mode_that_cannot_run_in_a_vm(
    mode: str, remote_workspace: Workspace
) -> None:
    """Inside a microVM codex's own sandbox (bubblewrap) cannot start.

    Measured: every command then fails, reads included, and the turn only
    narrates the failure. That is a mode quietly not working, so opening
    the harness refuses it instead.
    """
    with pytest.raises(UnsupportedError, match=mode):
        async with codex(workspace=remote_workspace, sandbox=mode):
            pass
