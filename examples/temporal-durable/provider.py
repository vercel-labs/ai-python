"""Provider-based durability: a DurabilityProvider backed by Temporal activities.

The agent uses the **default loop** unchanged. A TemporalDurabilityProvider
is passed to ``agent.run(durability=...)``, and the framework auto-routes
``models.stream()`` and ``ToolCall.__call__()`` through the provider via
context var.

The provider ignores the factory closures it receives (they can't be
serialized to a Temporal activity) and calls its own Temporal activities
instead. It tracks the conversation internally so it can serialize the
correct messages to the LLM activity at each turn.
"""

from __future__ import annotations

import datetime
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

import temporalio.common
import temporalio.workflow

with temporalio.workflow.unsafe.imports_passed_through():
    import activities

    import ai


# ── Tools ────────────────────────────────────────────────────────
#
# Defined with @tool for schema extraction.  The default loop will
# try to call them via ToolCall.__call__(), but the provider intercepts
# and routes to Temporal activities instead.


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    raise RuntimeError("should not be called inside workflow")


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    raise RuntimeError("should not be called inside workflow")


# ── Temporal durability provider ─────────────────────────────────


class _ActivityStreamResult:
    """StreamResult-like wrapper around a buffered message from an activity."""

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


class TemporalDurabilityProvider:
    """DurabilityProvider that routes LLM and tool I/O through Temporal activities.

    Temporal's event history provides durability — on replay, activities
    return cached results automatically.  No checkpoint needed on our side.

    The provider maintains its own copy of the conversation so it can
    serialize the correct messages to the LLM activity at each step.
    """

    def __init__(
        self,
        initial_messages: list[ai.Message],
        tool_schemas: list[dict[str, Any]],
    ) -> None:
        self._messages: list[ai.Message] = list(initial_messages)
        self._tool_schemas = tool_schemas

    async def execute_stream(
        self,
        fn: Callable[[], Awaitable[Any]],
    ) -> _ActivityStreamResult:
        """Call the LLM via a Temporal activity instead of fn()."""
        result = await temporalio.workflow.execute_activity(
            activities.llm_call_activity,
            activities.LLMCallParams(
                messages=[m.model_dump() for m in self._messages],
                tool_schemas=self._tool_schemas,
            ),
            start_to_close_timeout=datetime.timedelta(minutes=5),
            retry_policy=temporalio.common.RetryPolicy(maximum_attempts=3),
        )
        msg = ai.Message.model_validate(result.message)
        # Track the assistant response.
        self._messages.append(msg)
        return _ActivityStreamResult(msg)

    async def execute_tool(
        self,
        fn: Callable[[], Awaitable[ai.ToolResultPart]],
        *,
        tool_call_id: str,
        tool_name: str,
    ) -> ai.ToolResultPart:
        """Execute a tool via a Temporal activity instead of fn()."""
        # Find tool_args from the last assistant message we recorded.
        tool_args = ""
        for msg in reversed(self._messages):
            for tc in msg.tool_calls:
                if tc.tool_call_id == tool_call_id:
                    tool_args = tc.tool_args
                    break

        result = await temporalio.workflow.execute_activity(
            activities.tool_dispatch_activity,
            activities.ToolDispatchParams(
                tool_name=tool_name,
                tool_args=tool_args,
            ),
            start_to_close_timeout=datetime.timedelta(minutes=2),
        )

        tool_result = ai.ToolResultPart(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=result.result,
            is_error=result.is_error,
        )
        # Track the tool result. The default loop yields a tool_message
        # after gathering all results, and _collect_messages appends it
        # to context.messages. We mirror that here so our next LLM call
        # has the full conversation.
        self._messages.append(ai.Message(role="tool", parts=[tool_result]))
        return tool_result

    def checkpoint(self) -> ai.Checkpoint:
        """Temporal is the event store — no checkpoint needed."""
        return ai.Checkpoint()


# ── Agent (uses default loop) ────────────────────────────────────

weather_agent = ai.agent(tools=[get_weather, get_population])


# ── Workflow ─────────────────────────────────────────────────────


@temporalio.workflow.defn
class ProviderWorkflow:
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

        provider = TemporalDurabilityProvider(messages, tool_schemas)

        final_text = ""
        async for msg in weather_agent.run(model, messages, durability=provider):
            if msg.text:
                final_text = msg.text
        return final_text
