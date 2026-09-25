# durable_agent.py
import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import field
from typing import Any

import pydantic
from rotor import (
    ChildDone,
    ChildFailed,
    DurableProcess,
    Start,
    on,
    stream,
)
from rotor.patterns import task
from rotor.testing import LocalRuntime

import ai

###########


class Value(pydantic.BaseModel):
    result: ai.events.ToolCallResult | None = None
    error_message: str | None = None


def wrap_tool_call(
    tc: ai.agents.BoundToolCall,
    inner: Callable[..., Awaitable[ai.events.ToolCallResult]],
) -> Callable[..., Awaitable[ai.events.ToolCallResult]]:
    async def deferred(**kwargs: Any) -> ai.events.ToolCallResult:
        # We don't call the actual tool callable at all, we just wait
        # on a hook.  The loop processing the hook needs to figure out
        # how to actually launch the tool somehow.
        value = await ai.hook(
            f"eval_{tc.id}",
            payload=Value,
            metadata={"part": tc._part.model_dump(mode="json")},
            tool_call_id=tc.id,
        )
        if isinstance(value.result, ai.events.ToolCallResult):
            return value.result
        else:
            raise RuntimeError(value.error_message)

    return deferred


class UltraServerlessAgent(ai.Agent):
    async def loop(
        self, context: ai.Context
    ) -> AsyncGenerator[ai.events.AgentEvent]:
        """All this does is it interposes wrap_tool_call in front of tools."""
        while context.keep_running():
            async with (
                ai.experimental_telemetry.span(
                    ai.experimental_telemetry.LoopTurnSpanData()
                ),
                ai.stream(context=context) as stream,
                ai.ToolRunner() as tr,
            ):
                async for event in ai.util.merge(stream, tr.events()):
                    yield event

                    if isinstance(event, ai.events.ToolEnd):
                        tool = self.resolve(event.tool_call)
                        tr.schedule(tool.wrap(wrap_tool_call))

                context.add(stream.message)
                # This adds the tool message to the history, which
                # also has the effect of causing another turn through
                # the loop.
                context.add(tr.get_tool_message())


###########


model = ai.get_model("openai/gpt-5.6-luna")


async def read_weather(city: str) -> str:
    readings = {
        "lisbon": "24 C and sunny",
        "london": "16 C and raining",
    }
    return readings[city.lower()]


@ai.tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return await read_weather(city)


TOOLS = [get_weather]

RESOLVER = ai.ToolResolver(TOOLS)


class TransientToolError(Exception):
    """An example error the tool policy may retry."""


def retry_tool(error: Exception, attempt: int) -> str | None:
    if isinstance(error, TransientToolError) and attempt <= 3:
        return "2s"
    return None


@task(retry=retry_tool)
async def run_tool(call: dict) -> dict:
    """One tool call, one durable child with its own retry clock."""
    part = ai.messages.ToolCallPart.model_validate(call)
    tool = RESOLVER.resolve(part)
    return (await tool()).model_dump(mode="json")


class AgentState:
    messages: list[dict] = field(default_factory=list)


class WeatherAgent(DurableProcess[AgentState]):
    @on
    async def start(self, msg: Start):
        self.state.messages.extend(
            [
                ai.system_message(
                    "Call get_weather once per city before answering."
                ).model_dump(mode="json"),
                ai.user_message(msg.input).model_dump(mode="json"),
            ]
        )
        await self._generate()

    async def _generate(self):
        if len(self.state.messages) >= 15:
            self.stop(output="Stopped after 15 messages")

        history = [
            ai.messages.Message.model_validate(message)
            for message in self.state.messages
        ]

        agent = UltraServerlessAgent(tools=TOOLS)
        async with agent.run(model, history) as response:
            async for event in response:
                if (
                    isinstance(event, ai.events.HookEvent)
                    and event.hook.status == "pending"
                ):
                    # Got a hook. Spawn a task to run it and defer it.
                    call = event.hook.metadata["part"]
                    self.spawn(
                        run_tool, input={"call": call}, key=event.hook.hook_id
                    )

                    ai.defer_hook(event.hook)

                await stream(event.model_dump(mode="json"))

        self.state.messages = [
            message.model_dump(mode="json") for message in response.messages
        ]
        if response.messages[-1].role == "assistant":
            self.stop(output=response.messages[-1].text)

    @on(run_tool.Done)
    @on(run_tool.Failed)
    async def tool_done(self, msg: ChildDone | ChildFailed):
        result = (
            Value(result=ai.events.ToolCallResult.model_validate(msg.output))
            if isinstance(msg, ChildDone)
            else Value(error_message=msg.reason)
        )
        async with ai.agents.hooks.use_hook_registry(ai.HookRegistry()):
            ai.resolve_hook(msg.key, result)
            await self._generate()


async def main():
    async with LocalRuntime(WeatherAgent, run_tool) as rt:
        handle = await rt.client.start(
            WeatherAgent,
            input="Compare the weather in Lisbon and London.",
            scope="weather-demo",
        )
        await rt.drain()
        print((await handle.snapshot()).output)


asyncio.run(main())
