"""Multi-agent hooks demo — FastAPI server.

Runs a multi-agent graph over a WebSocket connection.  Two sub-agents
fan out in parallel, each gated by an approval hook.  Once both
complete, a third agent summarises their results.  Hook resolutions
arrive back on the same WebSocket from the Textual TUI client.

    uv run fastapi dev server.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import warnings
from collections.abc import AsyncGenerator
from typing import Any

import fastapi
import pydantic

import ai

# ToolResultPart.result is typed as dict but tools can return plain strings.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

app = fastapi.FastAPI(title="multiagent-textual")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@ai.tool
async def contact_mothership(question: str) -> str:
    """Contact the mothership for important decisions."""
    return "Soon."


@ai.tool
async def contact_data_centers(question: str) -> str:
    """Contact the data centers for status updates."""
    return "We are not sure yet."


# ---------------------------------------------------------------------------
# Hook payload
# ---------------------------------------------------------------------------


class Approval(pydantic.BaseModel):
    granted: bool
    reason: str


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL = ai.get_model("gateway:anthropic/claude-sonnet-4")


# ---------------------------------------------------------------------------
# Gated agent factory
#
# Creates an agent whose loop gates one tool behind an approval hook.
# Everything else (streaming, tool execution, history) uses the
# standard stream → tool → stream cycle.
# ---------------------------------------------------------------------------


class GatedToolCall:
    """ToolCall-shaped wrapper that awaits a per-branch approval hook.

    ``ToolRunner.schedule`` accepts any zero-arg callable returning a
    ``ToolCallResult``; this wrapper inserts the hook await + denial
    path before the underlying tool runs.
    """

    def __init__(self, tc: ai.ToolCall, label: str) -> None:
        self._tc = tc
        self._label = label

    async def __call__(self) -> ai.events.ToolCallResult:
        tc = self._tc
        approval = await ai.hook(
            f"{self._label}_{tc.id}",
            payload=Approval,
            metadata={"branch": self._label, "tool": tc.name},
        )
        if approval.granted:
            return await tc()
        return ai.tool_result(
            tool_call_id=tc.id,
            tool_name=tc.name,
            result=f"Denied: {approval.reason}",
            is_error=True,
        )


def _gated_agent(
    tools: list[ai.AgentTool],
    approval_tool: str,
    label: str,
) -> ai.Agent:
    class GatedAgent(ai.Agent):
        async def loop(
            self, context: ai.Context
        ) -> AsyncGenerator[ai.events.AgentEvent]:
            while context.keep_running():
                async with (
                    ai.stream(context=context) as s,
                    ai.ToolRunner() as tr,
                ):
                    async for event in ai.util.merge(s, tr.events()):
                        yield event
                        if isinstance(event, ai.events.ToolEnd):
                            tc = context.resolve(event.tool_call)
                            if tc.name == approval_tool:
                                tr.schedule(GatedToolCall(tc, label))
                            else:
                                tr.schedule(tc)

                    context.add(s.message)
                    context.add(tr.get_tool_message())

    return GatedAgent(tools=tools)


mothership_agent = _gated_agent(
    tools=[contact_mothership],
    approval_tool="contact_mothership",
    label="mothership",
)

data_centers_agent = _gated_agent(
    tools=[contact_data_centers],
    approval_tool="contact_data_centers",
    label="data_centers",
)


# ---------------------------------------------------------------------------
# Orchestrator — fan-out, hooks, fan-in
# ---------------------------------------------------------------------------


class Orchestrator(ai.Agent):
    """Run two gated agents in parallel, then summarise their results."""

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        query = context.messages[-1].text

        # Fan out: both branches stream concurrently via yield_from.
        # Each yield_from wraps every inner event in a PartialToolCallResult
        # carrying the branch label, so the TUI can route to the right panel.
        async with (
            mothership_agent.run(
                context.model,
                [
                    ai.system_message(
                        "You are assistant 1. Use contact_mothership "
                        "when asked about the future."
                    ),
                    ai.user_message(query),
                ],
            ) as mothership_stream,
            data_centers_agent.run(
                context.model,
                [
                    ai.system_message(
                        "You are assistant 2. Use contact_data_centers "
                        "when asked about the future."
                    ),
                    ai.user_message(query),
                ],
            ) as data_centers_stream,
        ):
            r1, r2 = await asyncio.gather(
                ai.yield_from(
                    mothership_stream,
                    label="mothership",
                    aggregator=ai.agents.MessageAggregator,
                ),
                ai.yield_from(
                    data_centers_stream,
                    label="data_centers",
                    aggregator=ai.agents.MessageAggregator,
                ),
            )

        combined = f"Mothership: {r1}\nData centers: {r2}"

        # Fan in: stream the summary directly.  Top-level events from the
        # orchestrator are unlabeled — the client routes them to the summary
        # panel as its default.
        summary_agent = ai.agent()
        async with summary_agent.run(
            context.model,
            [
                ai.system_message(
                    "You are assistant 3. Summarise the results from the "
                    "other assistants."
                ),
                ai.user_message(combined),
            ],
        ) as summary_stream:
            async for event in summary_stream:
                yield event


orchestrator = Orchestrator()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_message(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure ToolResultPart.result is always a dict for safe deserialisation."""
    for part in data.get("parts", []):
        if part.get("kind") == "tool_result" and isinstance(part.get("result"), str):
            part["result"] = {"value": part["result"]}
    return data


def _normalise_event(data: dict[str, Any]) -> dict[str, Any]:
    message = data.get("message")
    if isinstance(message, dict):
        data["message"] = _normalise_message(message)
    # Recurse into PartialToolCallResult wrappers so the inner event's
    # message gets normalised too.
    inner = data.get("value")
    if isinstance(inner, dict):
        data["value"] = _normalise_event(inner)
    return data


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def ws_endpoint(websocket: fastapi.WebSocket) -> None:
    await websocket.accept()
    print("Client connected")

    # Background task: read hook resolutions from the client.
    async def read_resolutions() -> None:
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                print(f"  Resolution received: {data['hook_id']}")
                ai.resolve_hook(
                    data["hook_id"],
                    Approval(granted=data["granted"], reason=data["reason"]),
                )
        except fastapi.WebSocketDisconnect:
            pass

    reader = asyncio.create_task(read_resolutions())

    try:
        async with orchestrator.run(
            MODEL, [ai.user_message("When will the robots take over?")]
        ) as result:
            async for event in result:
                data = _normalise_event(event.model_dump())
                await websocket.send_json(data)

                if isinstance(event, ai.events.HookEvent):
                    print(f"  Hook {event.hook.status}: {event.hook.hook_id}")
    finally:
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader

    # Signal completion
    with contextlib.suppress(Exception):
        await websocket.send_json({"type": "done"})

    print("Run complete")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
