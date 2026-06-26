"""Durable agent with a custom loop — every I/O call is a Temporal activity.

No middleware is involved. A custom ``Agent`` subclass overrides ``loop``
to call Temporal activities for every LLM call and tool execution.
Temporal's event history provides durability — on replay, activities
return cached results without re-executing.

This is the "lego bricks" approach: the framework gives you ``Agent``,
``Context``, ``@tool``, the message types, and ``ToolRunner``. You compose
them yourself.

The loop mirrors :py:meth:`ai.Agent.loop`: a streaming model call
feeds a ``ToolRunner``, tool calls are scheduled as they're emitted, and
results are folded back into the context. The only difference is that
the model "stream" is synthesized from the result of an LLM activity and
each tool call is dispatched as a Temporal activity.

Prerequisites:
    1. Temporal dev server:  temporal server start-dev
    2. AI_GATEWAY_API_KEY environment variable set

Usage:
    uv run python main.py
    uv run python main.py "What is the weather in Tokyo?"
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import sys
import uuid
from typing import TYPE_CHECKING, Any, ClassVar

import temporalio.activity
import temporalio.client
import temporalio.common
import temporalio.worker
import temporalio.workflow

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

with temporalio.workflow.unsafe.imports_passed_through():
    import ai


MODEL_ID = "gateway:anthropic/claude-sonnet-4.6"


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
    model_id: str
    messages: list[dict[str, Any]]
    tool_schemas: list[dict[str, Any]]


@dataclasses.dataclass
class LLMResult:
    message: dict[str, Any]


@temporalio.activity.defn
async def llm_call_activity(params: LLMParams) -> LLMResult:
    """Call the LLM, drain the stream, return the final message."""
    model = ai.get_model(params.model_id)
    messages = [ai.messages.Message.model_validate(m) for m in params.messages]
    tools = [ai.Tool.model_validate(t) for t in params.tool_schemas]

    async with ai.stream(model, messages, tools=tools) as s:
        async for _event in s:
            pass
        if s.message is None:
            raise RuntimeError("LLM stream ended without a final message")
        return LLMResult(message=s.message.model_dump())


# ── Agent with custom loop ───────────────────────────────────────
#
# The loop mirrors ai.Agent.loop: stream → schedule tools as they
# arrive → fold results back into the context. The only twist is
# that the "stream" comes from a Temporal activity (not the LLM
# directly) and each scheduled tool dispatches via another activity.


class WeatherAgent(ai.Agent):
    TOOLS: ClassVar[list[ai.AgentTool]] = [get_weather, get_population]

    async def loop(
        self, context: ai.Context
    ) -> AsyncGenerator[ai.events.AgentEvent]:
        tool_schemas = [t.model_dump(mode="json") for t in context.tools]

        while context.keep_running():
            # 1. LLM call via activity → complete message
            result = await temporalio.workflow.execute_activity(
                llm_call_activity,
                LLMParams(
                    model_id=context.model.id,
                    messages=[m.model_dump() for m in context.messages],
                    tool_schemas=tool_schemas,
                ),
                start_to_close_timeout=datetime.timedelta(minutes=5),
                retry_policy=temporalio.common.RetryPolicy(maximum_attempts=3),
            )
            llm_msg = ai.messages.Message.model_validate(result.message)

            # 2. Wrap the complete message in a synthetic stream so we can
            #    drive the rest of the loop with ToolRunner — same shape as
            #    the default loop. ``Stream.replay_message`` replays events
            #    from llm_msg.
            async with (
                ai.Stream.replay_message(llm_msg) as stream,
                ai.ToolRunner() as tr,
            ):
                async for event in ai.util.merge(stream, tr.events()):
                    yield event

                    if isinstance(event, ai.events.ToolEnd):
                        tr.schedule(_activity_tool_call(event.tool_call))

                context.add(stream.message)
                context.add(tr.get_tool_message())


weather_agent = WeatherAgent()


def _activity_tool_call(
    tc: ai.messages.ToolCallPart,
) -> ai.agents.ToolCallCallable:
    """Build a ``ToolCallCallable`` that runs the tool as a Temporal activity.

    ``ToolRunner.schedule`` accepts any zero-arg callable that returns
    a coroutine resolving to a ``ToolCallResult``. This lets us route
    tool execution through a Temporal activity (durable!) while keeping
    the rest of the loop identical to ``Agent.loop``.
    """

    async def _call() -> ai.events.ToolCallResult:
        activity_fn = TOOL_ACTIVITIES[tc.tool_name]
        kwargs = json.loads(tc.tool_args) if tc.tool_args else {}
        result = await temporalio.workflow.execute_activity(
            activity_fn,
            args=list(kwargs.values()),
            start_to_close_timeout=datetime.timedelta(minutes=2),
        )
        return ai.tool_result(
            tool_call_id=tc.tool_call_id,
            tool_name=tc.tool_name,
            result=result,
        )

    return _call


# ── Workflow ─────────────────────────────────────────────────────


@temporalio.workflow.defn
class WeatherWorkflow:
    @temporalio.workflow.run
    async def run(self, user_query: str) -> str:
        model = ai.get_model(MODEL_ID)
        messages: list[ai.messages.Message] = [
            ai.system_message(
                "Answer questions using the weather and population tools."
            ),
            ai.user_message(user_query),
        ]

        final_text = ""
        async with weather_agent.run(model, messages) as stream:
            async for event in stream:
                if isinstance(event, ai.events.StreamEnd):
                    final_text = event.message.text
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
    import asyncio

    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What's the weather and population of New York and Los Angeles?"
    )
    asyncio.run(main(query))
