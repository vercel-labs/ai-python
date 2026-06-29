from __future__ import annotations

import collections.abc
import os

_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault(
    "WORKFLOW_LOCAL_DATA_DIR",
    os.path.join(_BACKEND_DIR, ".workflow-data"),
)

import vercel._internal.workflow.py_sandbox  # noqa: E402

vercel._internal.workflow.py_sandbox._PASSTHROUGHS.update({"ai"})

import ai  # noqa: E402
import ai.agents.ui.ai_sdk as ai_sdk  # noqa: E402
import fastapi  # noqa: E402
import fastapi.middleware.cors  # noqa: E402
import fastapi.responses  # noqa: E402
import pydantic  # noqa: E402
import vercel.workflow  # noqa: E402

from . import worker


app = fastapi.FastAPI(title="durable-agent-workflows")
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

    run = await vercel.workflow.start(
        worker.run_turn,
        worker.TurnInput(
            messages=_with_system_message(messages),
        ).model_dump(mode="json"),
    )
    output = worker.TurnOutput.model_validate(await run.return_value())
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
