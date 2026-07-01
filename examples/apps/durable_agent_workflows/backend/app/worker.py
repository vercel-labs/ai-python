from __future__ import annotations

import asyncio
import traceback
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

import vercel._internal.workflow.py_sandbox

# The workflow sandbox re-imports non-passthrough modules under determinism
# restrictions, which the ai package (via httpx -> tempfile -> shutil) does
# not survive. Share the host's ai module instead. This must be registered
# here: the worker service is the process that runs the sandbox.
vercel._internal.workflow.py_sandbox._PASSTHROUGHS.update({"ai"})

import ai  # noqa: E402
import pydantic  # noqa: E402
import vercel.workflow  # noqa: E402

# The app uses one registry for all workflow decorators so queue messages
# are dispatched by the same Workflows instance.
workflow = vercel.workflow.Workflows()

MODEL_ID = "gateway:anthropic/claude-sonnet-4.6"
SYSTEM_PROMPT = """\
You are a coding assistant running inside a durable workflow. Help the user
inspect and modify the current project. Use bash when you need to read files,
run commands, or verify behavior. Keep answers concise.
"""

AGENT_EVENT_ADAPTER: pydantic.TypeAdapter[ai.events.AgentEvent] = pydantic.TypeAdapter(
    ai.events.AgentEvent
)


class TurnInput(pydantic.BaseModel):
    messages: list[ai.messages.Message]


class TurnOutput(pydantic.BaseModel):
    messages: list[ai.messages.Message]
    events: list[dict[str, Any]] = pydantic.Field(default_factory=list)
    error: str | None = None


@workflow.step
async def llm_step(
    model_data: dict[str, object],
    messages_data: list[dict[str, object]],
    tools_data: list[dict[str, object]],
) -> dict[str, object]:
    model = ai.Model.model_validate(model_data)
    messages = [
        ai.messages.Message.model_validate(message) for message in messages_data
    ]
    tools = [ai.Tool.model_validate(tool) for tool in tools_data]

    message: ai.messages.Message | None = None
    async with ai.stream(model, messages, tools=tools) as model_stream:
        async for event in model_stream:
            if isinstance(event, ai.events.StreamEnd):
                message = event.message

        if message is None:
            message = model_stream.message

    return message.model_dump(mode="json")


llm_step.max_retries = 0


@workflow.step
async def _bash(command: str, timeout: int | None = None) -> str:
    proc = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        return f"Command timed out after {timeout}s."

    output = stdout.decode() if stdout else ""
    if proc.returncode != 0:
        return f"[exit code {proc.returncode}]\n{output}"
    return output


_bash.max_retries = 0


@ai.tool
async def bash(command: str, timeout: int | None = None) -> str:
    """Execute a bash command. Use timeout in seconds to limit long commands."""
    return await _bash(command, timeout)


class DurableAgent(ai.Agent):
    TOOLS: ClassVar[list[ai.AgentTool]] = [bash]

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        tools_data = [tool.model_dump(mode="json") for tool in context.tools]

        while context.keep_running():
            result = await llm_step(
                context.model.model_dump(mode="json"),
                [message.model_dump(mode="json") for message in context.messages],
                tools_data,
            )
            assistant_message = ai.messages.Message.model_validate(result)
            context.add(assistant_message)

            async for event in ai.events.replay_message_events(assistant_message):
                yield event

            async with ai.ToolRunner() as runner:
                for tool_call in assistant_message.tool_calls:
                    runner.schedule(context.resolve(tool_call))

                async for tool_event in runner.events():
                    yield tool_event

                tool_message = runner.get_tool_message()
                if tool_message is not None:
                    context.add(tool_message)


durable_agent = DurableAgent()


async def _run_turn(turn_input: dict[str, Any]) -> TurnOutput:
    _turn_input = TurnInput.model_validate(turn_input)
    messages = _turn_input.messages
    events: list[dict[str, Any]] = []

    try:
        model = ai.get_model(MODEL_ID)
        async with durable_agent.run(model, messages) as run:
            async for event in run:
                events.append(event.model_dump(mode="json"))
            messages = run.messages
    except Exception as error:
        output = TurnOutput(
            messages=messages,
            events=events,
            error=f"{type(error).__name__}: {error}",
        )
        print(
            f"[durable_agent_workflows] error in run_turn:\n{traceback.format_exc()}",
            flush=True,
        )
    else:
        output = TurnOutput(messages=messages, events=events)

    return output


@workflow.workflow
async def run_turn(turn_input: dict[str, Any]) -> dict[str, Any]:
    try:
        output = await _run_turn(turn_input)
        return output.model_dump(mode="json")
    except Exception:
        print(
            f"[durable_agent_workflows] run_turn failed:\n{traceback.format_exc()}",
            flush=True,
        )
        raise
