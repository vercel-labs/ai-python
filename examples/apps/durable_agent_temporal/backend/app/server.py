from __future__ import annotations

import asyncio
import collections.abc
import os
import uuid

import ai
import ai.agents.ui.ai_sdk as ai_sdk
import fastapi
import fastapi.middleware.cors
import fastapi.responses
import pydantic
import temporalio.client

from . import worker

# Connected lazily on first use so the server can boot before the
# Temporal dev server is up.
_temporal: temporalio.client.Client | None = None
_temporal_lock = asyncio.Lock()


async def _temporal_client() -> temporalio.client.Client:
    global _temporal
    async with _temporal_lock:
        if _temporal is None:
            _temporal = await temporalio.client.Client.connect(
                os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
            )
    return _temporal


app = fastapi.FastAPI(title="durable-agent-temporal")
app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(pydantic.BaseModel):
    messages: list[ai_sdk.UIMessage]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def post_chat(request: ChatRequest) -> fastapi.responses.StreamingResponse:
    messages, _ = ai_sdk.to_messages(request.messages)
    if not messages:
        raise fastapi.HTTPException(status_code=400, detail="No messages to run")

    client = await _temporal_client()
    output_data = await client.execute_workflow(
        worker.RunTurn.run,
        worker.TurnInput(
            messages=_with_system_message(messages),
        ).model_dump(mode="json"),
        id=f"turn-{uuid.uuid4().hex[:8]}",
        task_queue=worker.TASK_QUEUE,
    )
    output = worker.TurnOutput.model_validate(output_data)
    if output.error is not None:
        raise fastapi.HTTPException(status_code=500, detail=output.error)

    return fastapi.responses.StreamingResponse(
        _to_sse(output.events),
        headers=ai_sdk.UI_MESSAGE_STREAM_HEADERS,
    )


def _with_system_message(
    messages: list[ai.messages.Message],
) -> list[ai.messages.Message]:
    if messages and messages[0].role == "system":
        return messages
    return [ai.system_message(worker.SYSTEM_PROMPT), *messages]


async def _to_sse(
    events_data: list[dict[str, object]],
) -> collections.abc.AsyncIterator[str]:
    async def events() -> collections.abc.AsyncIterator[ai.events.AgentEvent]:
        for event_data in events_data:
            yield worker.AGENT_EVENT_ADAPTER.validate_python(event_data)

    async for chunk in ai_sdk.to_sse(events()):
        yield chunk
