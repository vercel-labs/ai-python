from __future__ import annotations

import asyncio
import datetime
import os
import traceback
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

import temporalio.activity
import temporalio.client
import temporalio.common
import temporalio.worker
import temporalio.workflow

# The workflow sandbox re-imports non-passthrough modules under determinism
# restrictions, which the ai package (via httpx -> tempfile -> shutil) does
# not survive. Share the host's modules instead. pydantic must come along:
# a sandbox copy would make the ai models fail isinstance checks.
with temporalio.workflow.unsafe.imports_passed_through():
    import ai
    import pydantic

MODEL_ID = "gateway:anthropic/claude-sonnet-4.6"
TASK_QUEUE = "durable-agent-temporal"
SYSTEM_PROMPT = """\
You are a coding assistant running inside a durable workflow. Help the user
inspect and modify the current project. Use bash when you need to read files,
run commands, or verify behavior. Keep answers concise.
"""

AGENT_EVENT_ADAPTER: pydantic.TypeAdapter[ai.events.AgentEvent] = pydantic.TypeAdapter(
    ai.events.AgentEvent
)

NO_RETRIES = temporalio.common.RetryPolicy(maximum_attempts=1)


class TurnInput(pydantic.BaseModel):
    messages: list[ai.messages.Message]


class TurnOutput(pydantic.BaseModel):
    messages: list[ai.messages.Message]
    events: list[dict[str, Any]] = pydantic.Field(default_factory=list)
    error: str | None = None


@temporalio.activity.defn
async def llm_activity(
    model_data: dict[str, Any],
    messages_data: list[dict[str, Any]],
    tools_data: list[dict[str, Any]],
) -> dict[str, Any]:
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


@temporalio.activity.defn
async def bash_activity(command: str, timeout: int | None = None) -> str:
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


@ai.tool
async def bash(command: str, timeout: int | None = None) -> str:
    """Execute a bash command. Use timeout in seconds to limit long commands."""
    return await temporalio.workflow.execute_activity(
        bash_activity,
        args=[command, timeout],
        start_to_close_timeout=datetime.timedelta(minutes=5),
        retry_policy=NO_RETRIES,
    )


class DurableAgent(ai.Agent):
    TOOLS: ClassVar[list[ai.AgentTool]] = [bash]

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        tools_data = [tool.model_dump(mode="json") for tool in context.tools]

        while context.keep_running():
            result = await temporalio.workflow.execute_activity(
                llm_activity,
                args=[
                    context.model.model_dump(mode="json"),
                    [message.model_dump(mode="json") for message in context.messages],
                    tools_data,
                ],
                start_to_close_timeout=datetime.timedelta(minutes=5),
                retry_policy=NO_RETRIES,
            )
            assistant_message = ai.messages.Message.model_validate(result)
            context.add(assistant_message)

            async with ai.Stream.replay_message(assistant_message) as replay:
                async for event in replay:
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
            f"[durable_agent_temporal] error in run_turn:\n{traceback.format_exc()}",
            flush=True,
        )
    else:
        output = TurnOutput(messages=messages, events=events)

    return output


@temporalio.workflow.defn
class RunTurn:
    # Draw message/part ids from the workflow's deterministic RNG so they
    # are stable across replay. ``workflow.random`` is passed as a factory
    # (it's only valid inside the workflow) and resolved on each call.
    @temporalio.workflow.run
    @ai.messages.use_random(temporalio.workflow.random)
    async def run(self, turn_input: dict[str, Any]) -> dict[str, Any]:
        try:
            output = await _run_turn(turn_input)
            return output.model_dump(mode="json")
        except Exception:
            print(
                f"[durable_agent_temporal] run_turn failed:\n{traceback.format_exc()}",
                flush=True,
            )
            raise


async def main() -> None:
    client = await temporalio.client.Client.connect(
        os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    )
    async with temporalio.worker.Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RunTurn],
        activities=[llm_activity, bash_activity],
    ):
        print(f"[durable_agent_temporal] worker running on {TASK_QUEUE!r}", flush=True)
        await asyncio.Event().wait()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
