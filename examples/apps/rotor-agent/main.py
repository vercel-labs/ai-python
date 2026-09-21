# durable_agent.py
import asyncio
from dataclasses import field

from rotor import (
    ChildDone,
    ChildFailed,
    DurableProcess,
    Start,
    on,
    record,
    stream,
)
from rotor.patterns import Fanout, task
from rotor.testing import LocalRuntime

import ai

model = ai.get_model("openai/gpt-5.6-luna")


async def read_weather(city: str) -> str:
    readings = {
        "lisbon": "24 C and sunny",
        "london": "16 C and raining",
    }
    return readings.get(city.lower(), "No reading is available")


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
    tools: Fanout = field(default_factory=Fanout)
    results: list[dict] = field(default_factory=list)
    turns: int = 0


def error_result(call_data: dict, reason: object) -> dict:
    call = ai.messages.ToolCallPart.model_validate(call_data)
    return ai.tool_result(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        result=str(reason),
        is_error=True,
    ).model_dump(mode="json")


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
        if self.state.turns >= 8:
            self.stop(output="Stopped after 8 model turns")

        if self.state.results:
            messages = [
                ai.events.ToolCallResult.model_validate(result).message
                for result in self.state.results
            ]
            self.state.messages.append(
                ai.tool_message(*messages).model_dump(mode="json")
            )
            self.state.results.clear()

        history = [
            ai.messages.Message.model_validate(message)
            for message in self.state.messages
        ]
        async with ai.stream(
            model, history, tools=[t.tool for t in TOOLS]
        ) as response:
            async for event in response:
                await stream(event.model_dump(mode="json"))
        reply = response.message

        self.state.messages.append(reply.model_dump(mode="json"))
        self.state.turns += 1
        record("model_turn", {"turn": self.state.turns})
        if not reply.tool_calls:
            self.stop(output=reply.text)

        for call in reply.tool_calls:
            call_data = call.model_dump(mode="json")
            key = self.state.tools.expect(data=call_data)
            self.spawn(run_tool, input={"call": call_data}, key=key)

    @on(run_tool.Done)
    @on(run_tool.Failed)
    async def tool_done(self, msg: ChildDone | ChildFailed):
        call = self.state.tools.settle(key=msg.key)
        if call is None:
            return
        result = (
            msg.output
            if isinstance(msg, ChildDone)
            else error_result(call, msg.reason)
        )
        self.state.results.append(result)
        if self.state.tools.settled:
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
