"""Durable agent via Middleware — default loop, I/O routed to Temporal activities.

The agent uses the **default loop** unchanged. A ``TemporalMiddleware`` is
passed to ``agent.run(middleware=[...])`` and intercepts ``wrap_model`` and
``wrap_tool`` to route all I/O through Temporal activities. Temporal's event
history provides durability — on replay, activities return cached results
without re-executing.

This is the recommended pattern when you don't need to customize the loop:
the middleware transparently replaces I/O with durable activities.

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
import json
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
# Declared with @ai.tool for schema extraction. The default loop will
# try to call them, but the middleware intercepts and routes execution
# to Temporal activities instead.


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
# A generic tool dispatch activity routes tool calls by name.
# The LLM call is also an activity.


TOOL_REGISTRY: dict[str, Any] = {
    "get_weather": lambda city: f"Sunny, 72F in {city}",
    "get_population": lambda city: {
        "new york": 8_336_817,
        "los angeles": 3_979_576,
    }.get(city.lower(), 1_000_000),
}


@dataclasses.dataclass
class ToolDispatchParams:
    tool_name: str
    tool_args: str  # JSON string


@dataclasses.dataclass
class ToolDispatchResult:
    result: Any
    is_error: bool = False


@temporalio.activity.defn
async def tool_dispatch_activity(params: ToolDispatchParams) -> ToolDispatchResult:
    """Execute a tool by name."""
    fn = TOOL_REGISTRY.get(params.tool_name)
    if fn is None:
        return ToolDispatchResult(
            result=f"Unknown tool: {params.tool_name}", is_error=True
        )

    kwargs = json.loads(params.tool_args) if params.tool_args else {}
    result = fn(**kwargs)
    return ToolDispatchResult(result=result)


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


# ── Middleware ───────────────────────────────────────────────────
#
# Intercepts wrap_model and wrap_tool to replace real I/O with
# Temporal activities. The default agent loop runs unchanged —
# it just sees a StreamResult from wrap_model and a Message from
# wrap_tool, same as without middleware.


class _BufferedStreamResult:
    """Wraps a single buffered Message to look like a StreamResult."""

    def __init__(self, message: ai.Message) -> None:
        self._message = message

    def __aiter__(self) -> AsyncGenerator[ai.Message]:
        return self._generate()

    async def _generate(self) -> AsyncGenerator[ai.Message]:
        yield self._message

    @property
    def tool_calls(self) -> list[ai.ToolCallPart]:
        return self._message.tool_calls

    @property
    def text(self) -> str:
        return self._message.text

    @property
    def usage(self) -> ai.Usage | None:
        return self._message.usage

    @property
    def output(self) -> Any:
        return self._message.output


class TemporalMiddleware(ai.Middleware):
    """Routes LLM calls and tool executions through Temporal activities.

    The middleware tracks messages itself so it can serialize the full
    conversation to the LLM activity at each step.
    """

    def __init__(
        self,
        initial_messages: list[ai.Message],
        tool_schemas: list[dict[str, Any]],
    ) -> None:
        self._messages = list(initial_messages)
        self._tool_schemas = tool_schemas

    async def wrap_model(self, call: ai.middleware.ModelContext, next: Any) -> Any:
        """LLM call → Temporal activity."""
        result = await temporalio.workflow.execute_activity(
            llm_call_activity,
            LLMParams(
                messages=[m.model_dump() for m in self._messages],
                tool_schemas=self._tool_schemas,
            ),
            start_to_close_timeout=datetime.timedelta(minutes=5),
            retry_policy=temporalio.common.RetryPolicy(maximum_attempts=3),
        )
        msg = ai.Message.model_validate(result.message)
        self._messages.append(msg)
        return _BufferedStreamResult(msg)

    async def wrap_tool(self, call: ai.middleware.ToolContext, next: Any) -> Any:
        """Tool execution → Temporal activity."""
        result = await temporalio.workflow.execute_activity(
            tool_dispatch_activity,
            ToolDispatchParams(
                tool_name=call.tool_name,
                tool_args=json.dumps(call.kwargs),
            ),
            start_to_close_timeout=datetime.timedelta(minutes=2),
        )
        tool_msg = ai.tool_message(
            ai.ToolResultPart(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                result=result.result,
                is_error=result.is_error,
            )
        )
        self._messages.append(tool_msg)
        return tool_msg


# ── Agent (default loop — no customization) ──────────────────────

weather_agent = ai.agent(tools=[get_weather, get_population])


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

        tool_schemas = [
            {
                "name": t.name,
                "description": t.description,
                "param_schema": t.param_schema,
            }
            for t in weather_agent._tools
        ]
        mw = TemporalMiddleware(messages, tool_schemas)

        final_text = ""
        async for msg in weather_agent.run(model, messages, middleware=[mw]):
            if msg.text:
                final_text = msg.text
        return final_text


# ── Entry point ──────────────────────────────────────────────────

TASK_QUEUE = "temporal-middleware"


async def main(user_query: str) -> None:
    client = await temporalio.client.Client.connect("localhost:7233")

    async with temporalio.worker.Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WeatherWorkflow],
        activities=[llm_call_activity, tool_dispatch_activity],
    ):
        workflow_id = f"middleware-{uuid.uuid4().hex[:8]}"
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
