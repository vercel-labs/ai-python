"""FastAPI application entry point."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import agent as agent_
import fastapi
import fastapi.exceptions
import fastapi.middleware.cors
import fastapi.responses
import pydantic

import ai

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

app = fastapi.FastAPI(
    title="py-ai-fastapi-chat",
    description="Chat demo using Python Vercel AI SDK",
)

app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(fastapi.exceptions.RequestValidationError)
async def log_validation_errors(
    request: fastapi.Request, exc: fastapi.exceptions.RequestValidationError
) -> fastapi.responses.JSONResponse:
    """Log pydantic validation failures so 422s aren't silent in dev."""
    print(
        f"[422] {request.method} {request.url.path}: {exc.errors()}",
        file=sys.stderr,
        flush=True,
    )
    return fastapi.responses.JSONResponse(
        {"detail": exc.errors()}, status_code=422
    )


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


class ChatRequest(pydantic.BaseModel):
    """Request body for the chat endpoint."""

    messages: list[ai.ui.ai_sdk.UIMessage]


@app.post("/api/chat")
async def chat(request: ChatRequest) -> fastapi.responses.StreamingResponse:
    """Handle chat requests and stream responses."""
    messages, approvals = ai.ui.ai_sdk.to_messages(request.messages)

    async def stream_response() -> AsyncGenerator[str]:
        async with agent_.chat_agent.run(agent_.MODEL, messages) as result:
            # Pre-register hook resolutions so the agent loop's hooks
            # find them immediately on the resume turn.
            ai.ui.ai_sdk.apply_approvals(approvals)

            # We need to monitor the stream for HookEvents to abort;
            # since ui.ai_sdk.to_sse consumes a stream, we have a wrapper
            # async generator that does this check and yields the events.
            async def process() -> AsyncGenerator[ai.events.AgentEvent]:
                async for event in result:
                    if (
                        isinstance(event, ai.events.HookEvent)
                        and event.hook.status == "pending"
                    ):
                        event.hook.defer()
                    yield event

            async for chunk in ai.ui.ai_sdk.to_sse(process()):
                yield chunk

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=ai.ui.ai_sdk.UI_MESSAGE_STREAM_HEADERS,
    )
