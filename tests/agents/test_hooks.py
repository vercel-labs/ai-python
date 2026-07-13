"""Hooks: live resolution, cancellation, pre-registration, schema validation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pydantic
import pytest

import ai
from ai.types import events as agent_events_

from ..conftest import MOCK_MODEL, Recorder, mock_llm, text_msg


class Confirmation(pydantic.BaseModel):
    approved: bool
    reason: str = ""


# -- resolve_hook() with live future (long-running mode) -------------------


async def test_resolve_live_future() -> None:
    """In long-running mode, resolve_hook() unblocks the awaiting coroutine."""
    resolved_value: Confirmation | None = None

    class MyAgent(ai.Agent):
        async def loop(
            self, context: ai.Context
        ) -> AsyncGenerator[ai.events.Event]:
            nonlocal resolved_value
            async with ai.models.stream(context=context) as stream:
                async for event in stream:
                    yield event
            result = await ai.hook("confirm_1", payload=Confirmation)
            resolved_value = result

    my_agent = MyAgent()

    mock_llm([[text_msg("OK")]])

    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            if not isinstance(event, agent_events_.HookEvent):
                continue
            # When we see the deferred hook, resolve it.
            if event.hook.status == "pending":
                ai.resolve_hook(
                    "confirm_1", {"approved": True, "reason": "looks good"}
                )

    assert resolved_value is not None
    assert resolved_value.approved is True
    assert resolved_value.reason == "looks good"


# -- cancel_hook() --------------------------------------------------------


async def test_cancel_live_hook() -> None:
    """cancel_hook() cancels the future, causing CancelledError in graph."""
    was_cancelled = False

    class MyAgent(ai.Agent):
        async def loop(
            self, context: ai.Context
        ) -> AsyncGenerator[ai.events.Event]:
            nonlocal was_cancelled
            async with ai.models.stream(context=context) as stream:
                async for event in stream:
                    yield event
            try:
                await ai.hook("cancel_me", payload=Confirmation)
            except asyncio.CancelledError:
                was_cancelled = True

    my_agent = MyAgent()

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
    # Outside a run there is no current Runtime at all.
    with pytest.raises(LookupError):
        await ai.cancel_hook("does_not_exist_xyz")

    mock_llm([[text_msg("OK")]])
    async with ai.Agent().run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        with pytest.raises(ValueError, match="No deferred hook"):
            await ai.cancel_hook("does_not_exist_xyz")
        async for _msg in stream:
            pass


# -- Pre-registration (serverless re-entry) --------------------------------


async def test_pre_registered_resolution_consumed() -> None:
    """Pre-registered resolution is consumed by hook() without suspending."""
    resolved_value: Confirmation | None = None

    class MyAgent(ai.Agent):
        async def loop(
            self, context: ai.Context
        ) -> AsyncGenerator[ai.events.Event]:
            nonlocal resolved_value
            async with ai.models.stream(context=context) as stream:
                async for event in stream:
                    yield event
            resolved_value = await ai.hook("pre_reg_1", payload=Confirmation)

    my_agent = MyAgent()

    mock_llm([[text_msg("OK")]])
    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        # Pre-register before iterating: the loop hasn't started yet.
        ai.resolve_hook("pre_reg_1", {"approved": True})
        async for _msg in stream:
            pass

    assert resolved_value is not None
    assert resolved_value.approved is True


# -- Explicit HookRegistry ---------------------------------------------------


async def test_explicit_registry_isolates_live_hook() -> None:
    """A hook created in an explicit registry is invisible to the run's."""
    reg = ai.HookRegistry()
    resolved_value: Confirmation | None = None

    class MyAgent(ai.Agent):
        async def loop(
            self, context: ai.Context
        ) -> AsyncGenerator[ai.events.Event]:
            nonlocal resolved_value
            async with ai.models.stream(context=context) as stream:
                async for event in stream:
                    yield event
            resolved_value = await ai.hook(
                "iso_1", payload=Confirmation, registry=reg
            )

    my_agent = MyAgent()

    mock_llm([[text_msg("OK")]])

    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            if not isinstance(event, agent_events_.HookEvent):
                continue
            if event.hook.status == "pending":
                # The run's own registry doesn't know about this hook.
                with pytest.raises(ValueError, match="No deferred hook"):
                    await ai.cancel_hook("iso_1")
                ai.resolve_hook("iso_1", {"approved": True}, registry=reg)

    assert resolved_value is not None
    assert resolved_value.approved is True


async def test_run_with_explicit_registry() -> None:
    """A caller-owned registry can pre-register before the run starts."""
    reg = ai.HookRegistry()
    resolved_value: Confirmation | None = None

    class MyAgent(ai.Agent):
        async def loop(
            self, context: ai.Context
        ) -> AsyncGenerator[ai.events.Event]:
            nonlocal resolved_value
            async with ai.models.stream(context=context) as stream:
                async for event in stream:
                    yield event
            resolved_value = await ai.hook("pre_reg_rt", payload=Confirmation)

    my_agent = MyAgent()

    ai.resolve_hook("pre_reg_rt", {"approved": True}, registry=reg)

    mock_llm([[text_msg("OK")]])
    async with my_agent.run(
        MOCK_MODEL, [ai.user_message("go")], hook_registry=reg
    ) as stream:
        assert stream.hook_registry is reg
        async for _msg in stream:
            pass

    assert resolved_value is not None
    assert resolved_value.approved is True


# -- Nested runs ------------------------------------------------------------


async def test_nested_run_joins_enclosing_registry() -> None:
    """A nested run reuses the enclosing run's registry, so its hooks
    are resolvable from the outermost consumer."""
    resolved_value: Confirmation | None = None

    class InnerAgent(ai.Agent):
        async def loop(
            self, context: ai.Context
        ) -> AsyncGenerator[ai.events.Event]:
            nonlocal resolved_value
            async with ai.models.stream(context=context) as stream:
                async for event in stream:
                    yield event
            resolved_value = await ai.hook("nested_1", payload=Confirmation)

    class OuterAgent(ai.Agent):
        async def loop(
            self, context: ai.Context
        ) -> AsyncGenerator[agent_events_.AgentEvent]:
            inner = InnerAgent()
            async with inner.run(MOCK_MODEL, [ai.user_message("sub")]) as s:
                assert s.hook_registry is context_registry
                async for event in s:
                    yield event

    mock_llm([[text_msg("OK")]])

    async with OuterAgent().run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        context_registry = stream.hook_registry
        async for event in stream:
            if (
                isinstance(event, agent_events_.HookEvent)
                and event.hook.status == "pending"
            ):
                ai.resolve_hook("nested_1", {"approved": True})

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

    class MyAgent(ai.Agent):
        async def loop(
            self, context: ai.Context
        ) -> AsyncGenerator[ai.events.Event]:
            async with ai.models.stream(context=context) as stream:
                async for event in stream:
                    yield event
            await ai.hook("emit_test", payload=Confirmation)

    my_agent = MyAgent()

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


# -- Hook metadata surfaces in deferred message -----------------------------


async def test_hook_metadata_in_deferred() -> None:
    class MyAgent(ai.Agent):
        async def loop(
            self, context: ai.Context
        ) -> AsyncGenerator[ai.events.Event]:
            async with ai.models.stream(context=context) as stream:
                async for event in stream:
                    yield event
            try:
                await ai.hook(
                    "meta_test",
                    payload=Confirmation,
                    metadata={"tool": "rm -rf", "path": "/"},
                )
            except ai.agents.hooks.HookDeferredException:
                return

    my_agent = MyAgent()

    mock_llm([[text_msg("OK")]])
    hooks: list[ai.messages.HookPart[Any]] = []
    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            if isinstance(event, agent_events_.HookEvent):
                hooks.append(event.hook)
                if event.hook.status == "pending":
                    ai.defer_hook(event.hook)

    assert len(hooks) >= 1
    assert hooks[0].metadata == {"tool": "rm -rf", "path": "/"}


# -- telemetry spans --------------------------------------------------------


class _HookAgent(ai.Agent):
    """One model turn, then suspend on a hook."""

    label = "confirm_span"

    async def loop(
        self, context: ai.Context
    ) -> AsyncGenerator[agent_events_.AgentEvent]:
        async with ai.models.stream(context=context) as stream:
            async for event in stream:
                yield event
        await ai.hook(self.label, payload=Confirmation)


def _hook_spans(recorder: Recorder) -> list[ai.telemetry.Span]:
    return [s for s in recorder.ended if s.name == "hook"]


async def test_live_hook_span(recorder: Recorder) -> None:
    mock_llm([[text_msg("OK")]])
    my_agent = _HookAgent()
    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            if (
                isinstance(event, agent_events_.HookEvent)
                and event.hook.status == "pending"
            ):
                ai.resolve_hook(my_agent.label, {"approved": True})

    (hook_span,) = _hook_spans(recorder)
    assert isinstance(hook_span.data, ai.telemetry.HookSpanData)
    assert hook_span.data.status == "resolved"
    assert hook_span.data.resolution == {"approved": True}
    assert hook_span.data.tool_call_id is None
    assert not hook_span.replay
    # The suspension's timeline: deferred once resolvers can see it,
    # resolved when the external input arrived.
    deferred, resolved = hook_span.span_events
    assert deferred.name == ai.telemetry.HOOK_DEFERRED
    assert resolved.name == ai.telemetry.HOOK_RESOLVED
    assert deferred.time_ns <= resolved.time_ns


async def test_cancelled_hook_span(recorder: Recorder) -> None:
    class _CancelHookAgent(ai.Agent):
        label = "cancel_span"

        async def loop(
            self, context: ai.Context
        ) -> AsyncGenerator[agent_events_.AgentEvent]:
            async with ai.models.stream(context=context) as stream:
                async for event in stream:
                    yield event
            try:
                await ai.hook(self.label, payload=Confirmation)
            except asyncio.CancelledError:
                pass

    mock_llm([[text_msg("OK")]])
    my_agent = _CancelHookAgent()
    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            if (
                isinstance(event, agent_events_.HookEvent)
                and event.hook.status == "pending"
            ):
                await ai.cancel_hook(my_agent.label, reason="denied")

    (hook_span,) = _hook_spans(recorder)
    assert isinstance(hook_span.data, ai.telemetry.HookSpanData)
    assert hook_span.data.status == "cancelled"
    assert hook_span.data.resolution is None
    deferred, cancelled = hook_span.span_events
    assert deferred.name == ai.telemetry.HOOK_DEFERRED
    assert cancelled.name == ai.telemetry.HOOK_CANCELLED
    assert cancelled.attributes == {"reason": "denied"}


async def test_pre_registered_hook_is_replay_span(recorder: Recorder) -> None:
    my_agent = _HookAgent()
    mock_llm([[text_msg("OK")]])
    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        ai.resolve_hook(
            my_agent.label, {"approved": True}, payload=Confirmation
        )
        async for _ in stream:
            pass

    (hook_span,) = _hook_spans(recorder)
    assert hook_span.replay
    assert isinstance(hook_span.data, ai.telemetry.HookSpanData)
    assert hook_span.data.status == "resolved"
    # Pre-registration validated against the payload type, so the
    # recorded resolution is the full model dump, defaults included.
    assert hook_span.data.resolution == {"approved": True, "reason": ""}
    # A replayed suspension gets no synthetic timeline.
    assert hook_span.span_events == []
