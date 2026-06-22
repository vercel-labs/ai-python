"""FastAPI application entry point."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Protocol, cast

import agent as agent_
import fastapi
import fastapi.exceptions
import fastapi.middleware.cors
import fastapi.responses
import pydantic

import ai

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import starlette.types


class _VercelHeaders(Protocol):
    def set_headers(self, headers: dict[str, str] | None) -> None: ...


class VercelOIDCHeadersMiddleware:
    def __init__(self, app: starlette.types.ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: starlette.types.Scope,
        receive: starlette.types.Receive,
        send: starlette.types.Send,
    ) -> None:
        headers = _vercel_headers()
        if scope.get("type") != "http" or headers is None:
            await self.app(scope, receive, send)
            return

        headers.set_headers(_scope_headers(scope))
        try:
            await self.app(scope, receive, send)
        finally:
            headers.set_headers(None)


def _vercel_headers() -> _VercelHeaders | None:
    try:
        return cast(
            "_VercelHeaders",
            importlib.import_module("vercel.headers"),
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"vercel", "vercel.headers"}:
            raise
        return None


def _scope_headers(scope: starlette.types.Scope) -> dict[str, str]:
    return {
        _header_text(key): _header_text(value)
        for key, value in scope.get("headers", [])
    }


def _header_text(value: object) -> str:
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode("latin1")
    return str(value)


app = fastapi.FastAPI(
    title="py-ai-fastapi-chat",
    description="Chat demo using Python Vercel AI SDK",
)

app.add_middleware(VercelOIDCHeadersMiddleware)

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
            # We need to monitor the stream for HookEvents to abort;
            # since ui.ai_sdk.to_sse consumes a stream, we have a wrapper
            # async generator that does this check and yields the events.
            async def process() -> AsyncGenerator[ai.events.AgentEvent]:
                async for event in result:
                    if (
                        isinstance(event, ai.events.HookEvent)
                        and event.hook.status == "pending"
                    ):
                        ai.abort_pending_hook(event.hook)
                    yield event

            async for chunk in ai.agents.ui.ai_sdk.to_sse(process()):
                yield chunk

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=ai.agents.ui.ai_sdk.UI_MESSAGE_STREAM_HEADERS,
    )
