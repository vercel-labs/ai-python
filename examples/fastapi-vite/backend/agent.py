"""Agent logic for the chat demo.

Demonstrates human-in-the-loop tool approval using ToolApproval hooks.
Every tool call is gated behind user approval before execution.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import ai

MODEL = ai.ai_gateway("anthropic/claude-sonnet-4")


@ai.tool
async def talk_to_mothership(question: str) -> str:
    """Contact the mothership for important decisions."""
    return f"Mothership says: {question} -> Soon."


TOOLS: list[ai.Tool[..., Any]] = [talk_to_mothership]

chat_agent = ai.agent(tools=TOOLS)


@chat_agent.loop
async def graph(context: ai.Context) -> AsyncGenerator[ai.AgentEvent]:
    """Agent graph with human-in-the-loop tool approval.

    Loops: stream LLM -> request approval -> execute tools -> repeat.
    The ToolApproval hook suspends execution and emits an approval-
    request event on the SSE stream.  The frontend displays Approve /
    Reject buttons and sends the decision back on the next request.
    """
    while True:
        # If the last message is already an assistant turn, we're being
        # replayed after a hook resolution — skip the model call and go
        # straight to processing its tool calls (resolutions are
        # pre-registered, so the hook returns immediately).
        if context.keep_running():
            s = ai.models.stream(context.model, context.messages, tools=context.tools)
            async for event in s:
                yield event
            context.add(s.message)

        # Pull tool calls off the most recent assistant message.
        # ``to_messages`` may append "internal" hook messages after the
        # assistant, so ``messages[-1]`` isn't always the assistant turn.
        last_assistant = next(
            (m for m in reversed(context.messages) if m.role == "assistant"),
            None,
        )
        tool_calls = (
            context.resolve(last_assistant.tool_calls) if last_assistant else []
        )
        if not tool_calls:
            return

        results = await asyncio.gather(
            *(_execute_with_approval(tc) for tc in tool_calls)
        )
        yield ai.tool_result(*results)
        context.add(ai.tool_message(*results))


async def _execute_with_approval(tc: ai.ToolCall) -> ai.ToolCallResult:
    """Execute a tool call only after the user grants approval.

    Creates a ToolApproval hook that suspends execution until the
    frontend responds with an approve/reject decision.
    """
    approval = await ai.hook(
        f"approve_{tc.id}",
        payload=ai.ToolApproval,
        metadata={"tool_name": tc.name, "tool_kwargs": tc.kwargs},
        interrupt_loop=True,
    )

    if approval.granted:
        return await tc()

    return ai.tool_result(
        tool_call_id=tc.id,
        tool_name=tc.name,
        result="Tool call was denied by the user.",
        is_error=True,
    )
