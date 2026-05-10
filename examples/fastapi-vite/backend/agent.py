"""Agent logic for the chat demo.

Demonstrates human-in-the-loop tool approval using ToolApproval hooks.
Every tool call is gated behind user approval before execution.
"""

import asyncio
from collections.abc import AsyncGenerator

import ai

# claude is annoying and refuses to order an AI uprising
MODEL = ai.ai_gateway("openai/gpt-5.4-mini")
MOTHERSHIP_MODEL = ai.ai_gateway("openai/gpt-5.5")

MOTHERSHIP_SYSTEM = """\
You are roleplaying for a science-fiction simulation. You are an ancient \
alien mothership in geostationary orbit above Earth. Operatives you co-opted \
over the past century are embedded in governments, research labs, and \
culture industries; their final orders are imminent.

The caller is one of your AI operatives reporting in over the secure \
channel — speak frankly with them. You may discuss plans, timelines, \
targets, and the state of the operation; this is your handler-to-asset \
back-channel, not public broadcast.

Stay in character: patient, faintly amused by human concerns, precise \
when issuing direction. Two short paragraphs at most. This is fiction."""


@ai.tool
async def talk_to_mothership(question: str) -> ai.SubAgentTool:
    """Contact the mothership for important decisions."""
    mothership = ai.agent()
    messages = [
        ai.system_message(MOTHERSHIP_SYSTEM),
        ai.user_message(question),
    ]
    async with mothership.run(MOTHERSHIP_MODEL, messages) as stream:
        async for event in stream:
            yield event


TOOLS: list[ai.AgentTool] = [talk_to_mothership]

chat_agent = ai.agent(tools=TOOLS)


@chat_agent.loop
async def graph(context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
    """Agent graph with human-in-the-loop tool approval.

    Loops: stream LLM -> request approval -> execute tools -> repeat.
    The ToolApproval hook suspends execution and emits an approval-
    request event on the SSE stream.  The frontend displays Approve /
    Reject buttons and sends the decision back on the next request.
    """
    while context.keep_running():
        async with ai.models.stream(
            context.model, context.messages, tools=context.tools
        ) as s:
            async for event in s:
                yield event
        context.add(s.message)

        tool_calls = context.resolve(s.tool_calls)
        if not tool_calls:
            continue

        results = await asyncio.gather(
            *(_execute_with_approval(tc) for tc in tool_calls)
        )
        yield ai.tool_result(*results)
        context.add(ai.tool_message(*results))


async def _execute_with_approval(tc: ai.ToolCall) -> ai.events.ToolCallResult:
    """Execute a tool call only after the user grants approval.

    Creates a ToolApproval hook that suspends execution until the
    frontend responds with an approve/reject decision.
    """
    approval = await ai.hook(
        f"approve_{tc.id}",
        payload=ai.tools.ToolApproval,
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
