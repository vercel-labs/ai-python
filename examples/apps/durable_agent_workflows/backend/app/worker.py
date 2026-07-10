from __future__ import annotations

import asyncio
import os
import random
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


def _install_telemetry() -> None:
    """Export spans over OTLP when a collector endpoint is configured.

    For local development: ``uv run python -m ai.telemetry.utils.viewer``
    and set ``OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318``.
    """
    if "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ:
        return
    from ai.telemetry import otel
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk import resources
    from opentelemetry.sdk import trace as sdk_trace
    from opentelemetry.sdk.trace import export

    provider = sdk_trace.TracerProvider(
        resource=resources.Resource.create({"service.name": "durable-agent-workflows"})
    )
    provider.add_span_processor(
        export.BatchSpanProcessor(trace_exporter.OTLPSpanExporter())
    )
    trace.set_tracer_provider(provider)
    otel.install()


# Host only: the sandbox re-imports this module for the workflow, and
# the otel SDK does not load under its restrictions. Workflow code
# still reaches the adapter through the passed-through ``ai`` module.
if not vercel._internal.workflow.py_sandbox.in_sandbox():
    _install_telemetry()

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
    # Time base for span timestamps inside the workflow, stamped by the
    # server at start. Workflow inputs are recorded, so it is identical
    # on every replay; workflow code itself cannot read the clock.
    submitted_at_ns: int


class TurnOutput(pydantic.BaseModel):
    messages: list[ai.messages.Message]
    events: list[dict[str, Any]] = pydantic.Field(default_factory=list)
    error: str | None = None


class TickingClock:
    """Replay-stable clock for span timestamps inside the workflow.

    Starts at a recorded instant and ticks forward a fixed step per
    reading — the same recipe the workflow runtime uses to derive ULIDs
    from the run's started_at. Replays read identical timestamps.
    """

    def __init__(self, now_ns: int, tick_ns: int = 1_000_000) -> None:
        self.now_ns = now_ns
        self.tick_ns = tick_ns

    def time_ns(self) -> int:
        self.now_ns += self.tick_ns
        return self.now_ns


@workflow.step
async def llm_step(
    model_data: dict[str, object],
    messages_data: list[dict[str, object]],
    tools_data: list[dict[str, object]],
    parent_ref: dict[str, object] | None,
) -> dict[str, object]:
    model = ai.Model.model_validate(model_data)
    messages = [
        ai.messages.Message.model_validate(message) for message in messages_data
    ]
    tools = [ai.Tool.model_validate(tool) for tool in tools_data]

    # The step runs in its own process, parenting under the ref
    # carried in the input continues the workflow's trace.
    parent = (
        ai.telemetry.SpanRef.model_validate(parent_ref)
        if parent_ref is not None
        else None
    )
    message: ai.messages.Message | None = None
    async with (
        ai.telemetry.span("llm_step", parent=parent),
        ai.stream(model, messages, tools=tools) as model_stream,
    ):
        async for event in model_stream:
            if isinstance(event, ai.events.StreamEnd):
                message = event.message

        if message is None:
            message = model_stream.message

    # The worker can be frozen once the step hands back control; push
    # buffered spans out while we still can.
    await ai.telemetry.flush()
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
        # The loop runs inside the run span; its ref lets spans opened in
        # the step process parent under it.
        ref = ai.telemetry.current_ref()
        ref_data = ref.model_dump(mode="json") if ref is not None else None

        while context.keep_running():
            result = await llm_step(
                context.model.model_dump(mode="json"),
                [message.model_dump(mode="json") for message in context.messages],
                tools_data,
                ref_data,
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

    # Ids and span timestamps are the two ambient nondeterministic inputs
    # of the ai package; replays must reproduce both. Ids draw from the
    # sandbox's random module, which is seeded per run; timestamps count
    # up from the recorded time base in the input.
    with (
        ai.messages.use_random(lambda: random.Random(random.getrandbits(64))),
        ai.telemetry.use_clock(TickingClock(_turn_input.submitted_at_ns)),
    ):
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
                "[durable_agent_workflows] error in run_turn:\n"
                f"{traceback.format_exc()}",
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
