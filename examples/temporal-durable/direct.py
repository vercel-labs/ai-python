"""Direct composition: a custom loop where every I/O call is a Temporal activity.

No DurabilityProvider is involved. The user writes a plain ``@agent.loop``
that replaces ``models.stream()`` and tool execution with Temporal
``execute_activity()`` calls. Temporal's event history provides durability.

This is the "lego bricks" approach: the framework gives you ``Agent``,
``Context``, ``@tool`` (for schema extraction), and the message types.
You compose them with Temporal yourself.
"""

from __future__ import annotations

import asyncio
import datetime
from collections.abc import AsyncGenerator
from typing import Any

import temporalio.common
import temporalio.workflow

with temporalio.workflow.unsafe.imports_passed_through():
    import activities

    import ai


# ── Tools ────────────────────────────────────────────────────────
#
# Defined with @tool so the agent can extract JSON schemas for the
# LLM.  The bodies are never called inside the workflow — execution
# goes through Temporal activities instead.


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    raise RuntimeError("should not be called inside workflow")


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    raise RuntimeError("should not be called inside workflow")


# ── Agent with custom loop ───────────────────────────────────────

weather_agent = ai.agent(tools=[get_weather, get_population])


@weather_agent.loop
async def temporal_loop(context: ai.Context) -> AsyncGenerator[ai.Message]:
    """Agent loop where every I/O call is a durable Temporal activity."""
    tool_schemas = [
        {
            "name": t.name,
            "description": t.description,
            "param_schema": t.param_schema,
        }
        for t in context.tools
    ]

    while True:
        # LLM call via activity.
        result = await temporalio.workflow.execute_activity(
            activities.llm_call_activity,
            activities.LLMCallParams(
                messages=[m.model_dump() for m in context.messages],
                tool_schemas=tool_schemas,
            ),
            start_to_close_timeout=datetime.timedelta(minutes=5),
            retry_policy=temporalio.common.RetryPolicy(maximum_attempts=3),
        )
        msg = ai.Message.model_validate(result.message)
        yield msg

        if not msg.tool_calls:
            break

        # Tool calls via activities (parallel).
        tool_call_parts = msg.tool_calls

        async def _run_tool(tc: Any) -> ai.Message:
            dispatch_result = await temporalio.workflow.execute_activity(
                activities.tool_dispatch_activity,
                activities.ToolDispatchParams(
                    tool_name=tc.tool_name,
                    tool_args=tc.tool_args,
                ),
                start_to_close_timeout=datetime.timedelta(minutes=2),
            )
            return ai.Message(
                role="tool",
                parts=[
                    ai.ToolResultPart(
                        tool_call_id=tc.tool_call_id,
                        tool_name=tc.tool_name,
                        result=dispatch_result.result,
                        is_error=dispatch_result.is_error,
                    )
                ],
            )

        tasks = [asyncio.ensure_future(_run_tool(tc)) for tc in tool_call_parts]
        results = await asyncio.gather(*tasks)
        yield ai.tool_message(*results)


# ── Workflow ─────────────────────────────────────────────────────


@temporalio.workflow.defn
class DirectWorkflow:
    @temporalio.workflow.run
    async def run(self, user_query: str) -> str:
        model = ai.model("ai-gateway", "anthropic/claude-sonnet-4")
        messages: list[ai.Message] = [
            ai.system_message(
                "Answer questions using the weather and population tools."
            ),
            ai.user_message(user_query),
        ]

        final_text = ""
        async for msg in weather_agent.run(model, messages):
            if msg.text:
                final_text = msg.text
        return final_text
