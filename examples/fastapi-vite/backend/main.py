"""FastAPI application entry point."""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator

import agent as agent_
import fastapi
import fastapi.exceptions
import fastapi.middleware.cors
import fastapi.responses
import pydantic

import ai

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
    return fastapi.responses.JSONResponse({"detail": exc.errors()}, status_code=422)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


class ChatRequest(pydantic.BaseModel):
    """Request body for the chat endpoint."""

    messages: list[ai.agents.ui.ai_sdk.UIMessage]
    session_id: str | None = None


@app.post("/chat")
async def chat(request: ChatRequest) -> fastapi.responses.StreamingResponse:
    """Handle chat requests and stream responses."""
    messages, approvals = ai.agents.ui.ai_sdk.to_messages(request.messages)
    # Pre-register hook resolutions so the agent loop's hooks find them
    # immediately on the resume turn.
    ai.agents.ui.ai_sdk.apply_approvals(approvals)

    async def stream_response() -> AsyncGenerator[str]:
        async with agent_.chat_agent.run(agent_.MODEL, messages) as result:
            async for chunk in ai.agents.ui.ai_sdk.to_sse(result):
                yield chunk

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=ai.agents.ui.ai_sdk.UI_MESSAGE_STREAM_HEADERS,
    )
