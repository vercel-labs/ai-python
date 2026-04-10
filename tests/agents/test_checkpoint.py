"""Checkpoint replay, hook cancellation/resolution, serialization."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pydantic

import ai
from ai.agents.checkpoint import Checkpoint, HookEvent, StepEvent, ToolEvent

from ..conftest import MOCK_MODEL, mock_llm, text_msg, tool_call_msg


class Approval(pydantic.BaseModel):
    granted: bool


# -- Replay ----------------------------------------------------------------


async def test_step_replay_skips_llm() -> None:
    my_agent = ai.agent()

    # First run: LLM is called.
    provider1 = ai.EventLogProvider()
    llm1 = mock_llm([[text_msg("Hi there!")]])
    async for _msg in my_agent.run(
        MOCK_MODEL,
        [ai.system_message("test"), ai.user_message("hello")],
        durability=provider1,
    ):
        pass
    assert llm1.call_count == 1
    cp = provider1.checkpoint()

    # Replay: LLM is NOT called.
    llm2 = mock_llm([])
    async for _msg in my_agent.run(
        MOCK_MODEL,
        [ai.system_message("test"), ai.user_message("hello")],
        checkpoint=cp,
    ):
        pass
    assert llm2.call_count == 0


async def test_tool_replay_skips_execution() -> None:
    execution_count = 0

    @ai.tool
    async def counting_tool(x: int) -> int:
        """Counts calls."""
        nonlocal execution_count
        execution_count += 1
        return x + 1

    my_agent = ai.agent(tools=[counting_tool])

    # First run: tool should execute.
    mock_llm(
        [
            [tool_call_msg(tc_id="tc-1", name="counting_tool", args='{"x": 5}')],
            [text_msg("Done", id="msg-2")],
        ]
    )
    provider1 = ai.EventLogProvider()
    async for _msg in my_agent.run(
        MOCK_MODEL,
        [ai.system_message("t"), ai.user_message("go")],
        durability=provider1,
    ):
        pass
    assert execution_count == 1
    cp = provider1.checkpoint()
    assert cp.tools[0].result == 6

    # Replay: tool should NOT execute.
    execution_count = 0
    mock_llm([])
    async for _msg in my_agent.run(
        MOCK_MODEL,
        [ai.system_message("t"), ai.user_message("go")],
        checkpoint=cp,
    ):
        pass
    assert execution_count == 0


# -- Hooks -----------------------------------------------------------------


async def test_hook_cancellation_pending() -> None:
    my_agent = ai.agent()

    @my_agent.loop
    async def custom(context: ai.Context) -> AsyncGenerator[ai.Message]:
        async for msg in await ai.models.stream(context.model, context.messages):
            yield msg
        await ai.hook(
            "my_approval",
            payload=Approval,
            metadata={"tool": "test"},
            interrupt_loop=True,
        )

    mock_llm([[text_msg("OK")]])
    msgs: list[ai.Message] = []
    async for msg in my_agent.run(
        MOCK_MODEL, [ai.system_message("t"), ai.user_message("go")]
    ):
        msgs.append(msg)

    hook_msgs = [m for m in msgs if any(isinstance(p, ai.HookPart) for p in m.parts)]
    assert len(hook_msgs) >= 1
    assert hook_msgs[0].parts[0].status == "pending"  # type: ignore[union-attr]


async def test_hook_resolution_on_reentry() -> None:
    my_agent = ai.agent()

    @my_agent.loop
    async def custom(context: ai.Context) -> AsyncGenerator[ai.Message]:
        async for msg in await ai.models.stream(context.model, context.messages):
            yield msg
        await ai.hook(
            "my_approval",
            payload=Approval,
            interrupt_loop=True,
        )

    mock_llm([[text_msg("OK")]])
    provider1 = ai.EventLogProvider()
    async for _msg in my_agent.run(
        MOCK_MODEL,
        [ai.system_message("t"), ai.user_message("go")],
        durability=provider1,
    ):
        pass
    cp = provider1.checkpoint()

    # Pre-register resolution, then replay.
    ai.resolve_hook("my_approval", {"granted": True})
    mock_llm([])
    provider2 = ai.EventLogProvider(cp)
    async for _msg in my_agent.run(
        MOCK_MODEL,
        [ai.system_message("t"), ai.user_message("go")],
        durability=provider2,
    ):
        pass
    cp2 = provider2.checkpoint()
    assert any(h.label == "my_approval" for h in cp2.hooks)


# -- Serialization ---------------------------------------------------------


def test_checkpoint_serialization_roundtrip() -> None:
    cp = Checkpoint(
        steps=[
            StepEvent(
                index=0,
                message=ai.Message(
                    id="m1",
                    role="assistant",
                    parts=[ai.TextPart(text="hi")],
                ),
            )
        ],
        tools=[ToolEvent(tool_call_id="tc-1", tool_name="test", result=42)],
        hooks=[HookEvent(label="h1", resolution={"granted": True})],
    )
    cp2 = Checkpoint.model_validate(cp.model_dump())
    assert cp2.steps[0].index == 0
    assert cp2.tools[0].result == 42
    assert cp2.hooks[0].label == "h1"
