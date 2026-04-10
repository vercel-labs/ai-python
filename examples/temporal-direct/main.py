"""Durable agent with a custom loop — every I/O call is a Temporal activity.

No middleware is involved. A custom ``@agent.loop`` replaces the default loop
and calls Temporal activities for every LLM call and tool execution. Temporal's
event history provides durability — on replay, activities return cached results
without re-executing.

This is the "lego bricks" approach: the framework gives you ``Agent``,
``Context``, ``@tool``, and the message types. You compose them yourself.

Prerequisites:
    1. Temporal dev server:  temporal server start-dev
    2. AI_GATEWAY_API_KEY environment variable set

Usage:
    uv run python main.py
    uv run python main.py "What is the weather in Tokyo?"
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import sys
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import temporalio.activity
import temporalio.client
import temporalio.common
import temporalio.worker
import temporalio.workflow

with temporalio.workflow.unsafe.imports_passed_through():
    import ai


# ── Tool definitions ─────────────────────────────────────────────
#
# Declared with @ai.tool so the framework can extract JSON schemas
# for the LLM. The bodies are never called — execution goes through
# Temporal activities instead.


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    raise RuntimeError("not called — routed to activity")


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    raise RuntimeError("not called — routed to activity")


# ── Temporal activities ──────────────────────────────────────────
#
# Each tool has a corresponding activity. The LLM call is also an
# activity. This is the key idea: all I/O is a Temporal activity,
# so the workflow is fully deterministic and replayable.


@temporalio.activity.defn
async def get_weather_activity(city: str) -> str:
    return f"Sunny, 72F in {city}"


@temporalio.activity.defn
async def get_population_activity(city: str) -> int:
    return {"new york": 8_336_817, "los angeles": 3_979_576}.get(
        city.lower(), 1_000_000
    )


# Map tool names to their activities for dispatch in the loop.
TOOL_ACTIVITIES: dict[str, Any] = {
    "get_weather": get_weather_activity,
    "get_population": get_population_activity,
}


@dataclasses.dataclass
class LLMParams:
    messages: list[dict[str, Any]]
    tool_schemas: list[dict[str, Any]]


@dataclasses.dataclass
class LLMResult:
    message: dict[str, Any]


@temporalio.activity.defn
async def llm_call_activity(params: LLMParams) -> LLMResult:
    """Call the LLM, drain the stream, return the final message."""
    model = ai.model("ai-gateway", "anthropic/claude-sonnet-4")
    messages = [ai.Message.model_validate(m) for m in params.messages]
    tools = [ai.ToolSchema(return_type=None, **t) for t in params.tool_schemas]

    s = await ai.models.stream(model, messages, tools=tools)
    result = await ai.models.buffer(s)
    return LLMResult(message=result.model_dump())


# ── Agent with custom loop ───────────────────────────────────────
#
# The loop replaces ai.models.stream() and tool execution with
# Temporal activity calls. The structure mirrors the default loop:
# call LLM → yield message → execute tool calls → repeat.

weather_agent = ai.agent(tools=[get_weather, get_population])


@weather_agent.loop
async def temporal_loop(context: ai.Context) -> AsyncGenerator[ai.Message]:
    tool_schemas = [
        {"name": t.name, "description": t.description, "param_schema": t.param_schema}
        for t in context.tools
    ]

    while True:
        # 1. LLM call via activity
        result = await temporalio.workflow.execute_activity(
            llm_call_activity,
            LLMParams(
                messages=[m.model_dump() for m in context.messages],
                tool_schemas=tool_schemas,
            ),
            start_to_close_timeout=datetime.timedelta(minutes=5),
            retry_policy=temporalio.common.RetryPolicy(maximum_attempts=3),
        )
        msg = ai.Message.model_validate(result.message)
        yield msg

        # 2. No tool calls → done
        if not msg.tool_calls:
            break

        # 3. Execute each tool call as a Temporal activity (parallel)
        async def run_tool(tc: ai.ToolCallPart) -> ai.ToolResultPart:
            import json

            activity_fn = TOOL_ACTIVITIES[tc.tool_name]
            kwargs = json.loads(tc.tool_args) if tc.tool_args else {}
            result = await temporalio.workflow.execute_activity(
                activity_fn,
                args=list(kwargs.values()),
                start_to_close_timeout=datetime.timedelta(minutes=2),
            )
            return ai.tool_result(
                tc.tool_call_id, tool_name=tc.tool_name, result=result
            )

        tasks = [asyncio.ensure_future(run_tool(tc)) for tc in msg.tool_calls]
        parts = await asyncio.gather(*tasks)
        yield ai.tool_message(*parts)


# ── Workflow ─────────────────────────────────────────────────────


@temporalio.workflow.defn
class WeatherWorkflow:
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


# ── Entry point ──────────────────────────────────────────────────

TASK_QUEUE = "temporal-direct"


async def main(user_query: str) -> None:
    client = await temporalio.client.Client.connect("localhost:7233")

    async with temporalio.worker.Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WeatherWorkflow],
        activities=[
            llm_call_activity,
            get_weather_activity,
            get_population_activity,
        ],
    ):
        workflow_id = f"direct-{uuid.uuid4().hex[:8]}"
        print(f"Workflow: {workflow_id}")
        print(f"Query:    {user_query}\n")

        result = await client.execute_workflow(
            WeatherWorkflow.run,
            user_query,
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
        print(result)


if __name__ == "__main__":
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What's the weather and population of New York and Los Angeles?"
    )
    asyncio.run(main(query))
