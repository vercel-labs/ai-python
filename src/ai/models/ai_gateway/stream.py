"""AI Gateway v3 streaming adapter — language-model endpoint.

Handles text, tool-call, reasoning, and inline file streaming via SSE.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import httpx
import pydantic

from ...types import events as events_
from ...types import media
from ...types import messages as messages_
from ...types import proto as proto_
from ...types import usage as usage_
from ..core import client as client_
from ..core import model as model_
from ..core.helpers import files
from . import _common, errors

# ---------------------------------------------------------------------------
# Request building — Message list → v3 prompt
# ---------------------------------------------------------------------------


async def _file_part_to_v3(part: messages_.FilePart) -> dict[str, Any]:
    """Convert a :class:`FilePart` to a v3 ``file`` content part."""
    data = part.data
    if isinstance(data, str) and media.is_downloadable_url(data):
        downloaded, _ = await files.download(data)
        data = downloaded

    entry: dict[str, Any] = {
        "type": "file",
        "mediaType": part.media_type,
        "data": media.data_to_data_url(data, part.media_type),
    }
    if part.filename is not None:
        entry["filename"] = part.filename
    return entry


async def _messages_to_prompt(
    messages: list[messages_.Message],
) -> list[dict[str, Any]]:
    """Convert ``Message`` list to the v3 prompt wire format."""
    result: list[dict[str, Any]] = []

    for msg in messages:
        match msg.role:
            case "system":
                text = "".join(
                    p.text for p in msg.parts if isinstance(p, messages_.TextPart)
                )
                result.append({"role": "system", "content": text})

            case "user":
                content: list[dict[str, Any]] = []
                for p in msg.parts:
                    if isinstance(p, messages_.TextPart):
                        content.append({"type": "text", "text": p.text})
                    elif isinstance(p, messages_.FilePart):
                        content.append(await _file_part_to_v3(p))
                result.append({"role": "user", "content": content})

            case "assistant":
                assistant_content: list[dict[str, Any]] = []
                for part in msg.parts:
                    match part:
                        case messages_.ReasoningPart(text=text):
                            assistant_content.append(
                                {"type": "reasoning", "text": text}
                            )
                        case messages_.TextPart(text=text):
                            assistant_content.append({"type": "text", "text": text})
                        case messages_.ToolCallPart() as tp:
                            tool_input: Any = (
                                json.loads(tp.tool_args) if tp.tool_args else {}
                            )
                            assistant_content.append(
                                {
                                    "type": "tool-call",
                                    "toolCallId": tp.tool_call_id,
                                    "toolName": tp.tool_name,
                                    "input": tool_input,
                                }
                            )
                result.append({"role": "assistant", "content": assistant_content})

            case "tool":
                tool_results: list[dict[str, Any]] = []
                for part in msg.parts:
                    if isinstance(part, messages_.ToolResultPart):
                        output = (
                            {
                                "type": "error-text",
                                "value": (
                                    str(part.result) if part.result is not None else ""
                                ),
                            }
                            if part.is_error
                            else {
                                "type": "json",
                                "value": part.result,
                            }
                        )
                        tool_results.append(
                            {
                                "type": "tool-result",
                                "toolCallId": part.tool_call_id,
                                "toolName": part.tool_name,
                                "output": output,
                            }
                        )
                if tool_results:
                    result.append({"role": "tool", "content": tool_results})

    return result


async def _build_request_body(
    messages: list[messages_.Message],
    tools: Sequence[proto_.ToolLike] | None = None,
    output_type: type[Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the ``LanguageModelV3CallOptions`` request body."""
    body: dict[str, Any] = {
        "prompt": await _messages_to_prompt(messages),
    }
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.param_schema,
            }
            for tool in tools
        ]
    if output_type is not None and issubclass(output_type, pydantic.BaseModel):
        body["responseFormat"] = {
            "type": "json",
            "schema": output_type.model_json_schema(),
            "name": output_type.__name__,
        }
    if kwargs.get("provider_options"):
        body["providerOptions"] = kwargs["provider_options"]
    return body


# ---------------------------------------------------------------------------
# SSE response parsing — v3 stream parts → public Event
# ---------------------------------------------------------------------------


def _expand_tool_call(data: dict[str, Any]) -> list[events_.Event]:
    """Expand a complete ``tool-call`` part into Start + Delta + End."""
    tc_id = data.get("toolCallId", "")
    tool_name = data.get("toolName", "")
    tool_input = data.get("input", "")
    args_str = tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
    return [
        events_.ToolStart(tool_call_id=tc_id, tool_name=tool_name),
        events_.ToolDelta(tool_call_id=tc_id, chunk=args_str),
        events_.ToolEnd(tool_call_id=tc_id),
    ]


def _parse_usage(data: Any) -> usage_.Usage:
    """Parse v3 usage data into an internal ``Usage``."""
    if not isinstance(data, dict):
        return usage_.Usage()

    input_tokens_obj = data.get("inputTokens")
    output_tokens_obj = data.get("outputTokens")

    if isinstance(input_tokens_obj, dict) or isinstance(output_tokens_obj, dict):
        inp = input_tokens_obj if isinstance(input_tokens_obj, dict) else {}
        out = output_tokens_obj if isinstance(output_tokens_obj, dict) else {}
        return usage_.Usage(
            input_tokens=inp.get("total") or 0,
            output_tokens=out.get("total") or 0,
            reasoning_tokens=out.get("reasoning"),
            cache_read_tokens=inp.get("cacheRead"),
            cache_write_tokens=inp.get("cacheWrite"),
            raw=data,
        )

    return usage_.Usage(
        input_tokens=data.get("prompt_tokens") or data.get("inputTokens") or 0,
        output_tokens=(data.get("completion_tokens") or data.get("outputTokens") or 0),
        raw=data,
    )


def _parse_stream_part(data: dict[str, Any]) -> list[events_.Event]:
    """Convert a ``LanguageModelV3StreamPart`` to public events."""
    match data.get("type", ""):
        case "text-start":
            return [events_.TextStart(block_id=data.get("id", "text"))]

        case "text-delta":
            return [
                events_.TextDelta(
                    block_id=data.get("id", "text"),
                    chunk=data.get("textDelta", data.get("delta", "")),
                )
            ]

        case "text-end":
            return [events_.TextEnd(block_id=data.get("id", "text"))]

        case "reasoning-start":
            return [events_.ReasoningStart(block_id=data.get("id", "reasoning"))]

        case "reasoning-delta":
            return [
                events_.ReasoningDelta(
                    block_id=data.get("id", "reasoning"),
                    chunk=data.get("delta", ""),
                )
            ]

        case "reasoning-end":
            return [events_.ReasoningEnd(block_id=data.get("id", "reasoning"))]

        case "tool-input-start":
            return [
                events_.ToolStart(
                    tool_call_id=data.get("id", ""),
                    tool_name=data.get("toolName", ""),
                )
            ]

        case "tool-input-delta":
            return [
                events_.ToolDelta(
                    tool_call_id=data.get("id", ""),
                    chunk=data.get("delta", ""),
                )
            ]

        case "tool-input-end":
            return [events_.ToolEnd(tool_call_id=data.get("id", ""))]

        case "tool-call":
            return _expand_tool_call(data)

        case "file":
            return [
                events_.FileEvent(
                    block_id=data.get("id", ""),
                    media_type=data.get("mediaType", "application/octet-stream"),
                    data=data.get("data", ""),
                )
            ]

        case "finish":
            usage_data = data.get("usage")
            usage = _parse_usage(usage_data) if usage_data else None
            return [events_.StreamEnd(usage=usage)]

        case _:
            return []


# ---------------------------------------------------------------------------
# Public adapter function
# ---------------------------------------------------------------------------


async def stream(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    *,
    tools: Sequence[proto_.ToolLike] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    **kwargs: Any,
) -> AsyncGenerator[events_.Event]:
    """Stream an LLM response through the AI Gateway v3 protocol.

    Yields :class:`~ai.types.events.Event` objects as the response streams in.
    Pure delta emitter — the :class:`~ai.models.Stream` wrapper aggregates
    parts into the final :class:`~ai.types.Message`.
    """
    body = await _build_request_body(
        messages, tools=tools, output_type=output_type, **kwargs
    )
    headers = _common.request_headers(
        client, model, model_type="language", streaming=True
    )
    url = f"{client.base_url.rstrip('/')}/language-model"

    try:
        async with client.http.stream(
            "POST",
            url,
            json=body,
            headers=headers,
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise errors.create_gateway_error(
                    response_body=response.text,
                    status_code=response.status_code,
                    api_key_provided=bool(client.api_key),
                )

            yield events_.StreamStart()
            async for data in _common.parse_sse_lines(response):
                for event in _parse_stream_part(data):
                    yield event
    except errors.GatewayError:
        raise
    except httpx.TimeoutException as exc:
        raise errors.GatewayTimeoutError(cause=exc) from exc
    except Exception as exc:
        raise errors.GatewayResponseError(
            message=f"Unexpected error during streaming: {exc}",
            cause=exc,
        ) from exc
