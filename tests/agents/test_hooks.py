"""Hooks: live resolution, cancellation, pre-registration, schema validation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pydantic
import pytest

import ai
from ai.types import events as agent_events_

from ..conftest import MOCK_MODEL, mock_llm, text_msg


class Confirmation(pydantic.BaseModel):
    approved: bool
    reason: str = ""


# -- resolve_hook() with live future (long-running mode) -------------------


async def test_resolve_live_future() -> None:
    """In long-running mode, resolve_hook() unblocks the awaiting coroutine."""
    resolved_value: Confirmation | None = None
    my_agent = ai.agent()

    @my_agent.loop
    async def custom(context: ai.Context) -> AsyncGenerator[ai.events.Event]:
        nonlocal resolved_value
        async for event in ai.models.stream(context.model, context.messages):
            yield event
        result = await ai.hook("confirm_1", payload=Confirmation)
        resolved_value = result

    mock_llm([[text_msg("OK")]])

    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            if not isinstance(event, agent_events_.HookEvent):
                continue
            # When we see the pending hook, resolve it.
            if event.hook.status == "pending":
                ai.resolve_hook("confirm_1", {"approved": True, "reason": "looks good"})

    assert resolved_value is not None
    assert resolved_value.approved is True
    assert resolved_value.reason == "looks good"


# -- cancel_hook() --------------------------------------------------------


async def test_cancel_live_hook() -> None:
    """cancel_hook() cancels the future, causing CancelledError in graph."""
    was_cancelled = False
    my_agent = ai.agent()

    @my_agent.loop
    async def custom(context: ai.Context) -> AsyncGenerator[ai.events.Event]:
        nonlocal was_cancelled
        async for event in ai.models.stream(context.model, context.messages):
            yield event
        try:
            await ai.hook("cancel_me", payload=Confirmation)
        except asyncio.CancelledError:
            was_cancelled = True

    mock_llm([[text_msg("OK")]])

    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            if not isinstance(event, agent_events_.HookEvent):
                continue
            if event.hook.status == "pending":
                await ai.cancel_hook("cancel_me", reason="denied")

    assert was_cancelled


# -- cancel_hook() on non-existent label raises ----------------------------


async def test_cancel_nonexistent_raises() -> None:
    with pytest.raises(ValueError, match="No pending hook"):
        await ai.cancel_hook("does_not_exist_xyz")


# -- Pre-registration (serverless re-entry) --------------------------------


async def test_pre_registered_resolution_consumed() -> None:
    """Pre-registered resolution is consumed by hook() without suspending."""
    resolved_value: Confirmation | None = None
    my_agent = ai.agent()

    @my_agent.loop
    async def custom(context: ai.Context) -> AsyncGenerator[ai.events.Event]:
        nonlocal resolved_value
        async for event in ai.models.stream(context.model, context.messages):
            yield event
        resolved_value = await ai.hook("pre_reg_1", payload=Confirmation)

    # Pre-register BEFORE run.
    ai.resolve_hook("pre_reg_1", {"approved": True})

    mock_llm([[text_msg("OK")]])
    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for _msg in stream:
            pass

    assert resolved_value is not None
    assert resolved_value.approved is True


# -- Schema validation on resolve -----------------------------------------


def test_resolve_validates_schema() -> None:
    """resolve_hook() with invalid data raises from pydantic validation."""
    # 'approved' is required bool, passing string should raise.
    with pytest.raises(pydantic.ValidationError):
        ai.resolve_hook(
            "schema_test",
            {"approved": "not_a_bool"},
            payload=Confirmation,
        )


# -- Resolved hook emits message -------------------------------------------


async def test_resolved_hook_emits_message() -> None:
    """After resolution, a 'resolved' HookPart message is emitted."""
    my_agent = ai.agent()

    @my_agent.loop
    async def custom(context: ai.Context) -> AsyncGenerator[ai.events.Event]:
        async for event in ai.models.stream(context.model, context.messages):
            yield event
        await ai.hook("emit_test", payload=Confirmation)

    mock_llm([[text_msg("OK")]])

    hooks: list[ai.messages.HookPart[Any]] = []
    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            if not isinstance(event, agent_events_.HookEvent):
                continue
            hooks.append(event.hook)
            if event.hook.status == "pending":
                ai.resolve_hook("emit_test", {"approved": False})

    resolved = [h for h in hooks if h.status == "resolved"]
    assert len(resolved) == 1
    assert resolved[0].resolution == {"approved": False}


# -- Hook metadata surfaces in pending message -----------------------------


async def test_hook_metadata_in_pending() -> None:
    my_agent = ai.agent()

    @my_agent.loop
    async def custom(context: ai.Context) -> AsyncGenerator[ai.events.Event]:
        async for event in ai.models.stream(context.model, context.messages):
            yield event
        await ai.hook(
            "meta_test",
            payload=Confirmation,
            metadata={"tool": "rm -rf", "path": "/"},
            interrupt_loop=True,
        )

    mock_llm([[text_msg("OK")]])
    hooks: list[ai.messages.HookPart[Any]] = []
    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            if isinstance(event, agent_events_.HookEvent):
                hooks.append(event.hook)

    assert len(hooks) >= 1
    assert hooks[0].metadata == {"tool": "rm -rf", "path": "/"}
