"""OpenAI protocol — chat completions API.

Message/tool conversion and streaming via the official ``openai`` SDK.
OpenAI-compatible providers own the SDK client used by this protocol.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pydantic

from ... import types
from ...models import core
from . import _sdk, errors

if TYPE_CHECKING:
    import openai

# ---------------------------------------------------------------------------
# Message / tool conversion — internal types → OpenAI wire format
# ---------------------------------------------------------------------------


def _tools_to_openai(
    tools: Sequence[types.tools.Tool],
) -> list[dict[str, Any]]:
    """Convert internal Tool objects to OpenAI tool schema format.

    Built-in tools are rejected upstream by ``stream(...)``; this helper
    only processes custom (host-executed) tools.
    """
    result: list[dict[str, Any]] = []
    for tool in tools:
        if tool.kind == "provider":
            continue
        args = tool.args
        if not isinstance(args, types.tools.FunctionToolArgs):
            raise TypeError(f"function tool {tool.name!r} has invalid args")
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": args.description or "",
                    "parameters": args.params,
                },
            }
        )
    return result


async def _file_part_to_openai(
    part: types.messages.FilePart,
) -> dict[str, Any]:
    """Convert a :class:`FilePart` to an OpenAI content-array element.

    * ``image/*`` -> ``image_url`` (URL or ``data:`` URL)
    * ``audio/*`` -> ``input_audio`` (base-64 only; URLs auto-downloaded)
    * ``application/pdf`` -> ``file`` (base-64 only; URLs auto-downloaded)
    * ``text/*`` -> ``text`` (decoded to string)
    * anything else -> ``ValueError``
    """
    mt = part.media_type
    data = part.data

    if mt.startswith("image/"):
        media_type = "image/jpeg" if mt == "image/*" else mt
        url = types.media.data_to_data_url(data, media_type)
        return {"type": "image_url", "image_url": {"url": url}}

    if mt.startswith("audio/"):
        if isinstance(data, str) and types.media.is_downloadable_url(data):
            downloaded, _ = await core.helpers.files.download(data)
            data = downloaded
        fmt = mt.split("/", 1)[1] if "/" in mt else mt
        b64 = types.media.data_to_base64(data)
        return {
            "type": "input_audio",
            "input_audio": {"data": b64, "format": fmt},
        }

    if mt == "application/pdf":
        if isinstance(data, str) and types.media.is_downloadable_url(data):
            downloaded, _ = await core.helpers.files.download(data)
            data = downloaded
        data_url = types.media.data_to_data_url(data, mt)
        filename = part.filename or "document.pdf"
        return {
            "type": "file",
            "file": {"filename": filename, "file_data": data_url},
        }

    if mt.startswith("text/"):
        if isinstance(data, bytes):
            text_content = data.decode("utf-8")
        elif types.media.is_url(data):
            text_content = data
        else:
            text_content = base64.b64decode(data).decode("utf-8")
        return {"type": "text", "text": text_content}

    raise ValueError(f"Unsupported media type for OpenAI: {mt}")


async def _messages_to_openai(
    messages: list[types.messages.Message],
) -> list[dict[str, Any]]:
    """Convert internal messages to OpenAI API format.

    * ``tool_calls`` on assistant messages
    * tool results as separate ``role: "tool"`` messages
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        match msg.role:
            case "assistant":
                content = ""
                reasoning = ""
                tool_calls: list[dict[str, Any]] = []

                for part in msg.parts:
                    match part:
                        case types.messages.ReasoningPart(text=text):
                            reasoning += text
                        case types.messages.TextPart(text=text):
                            content += text
                        case types.messages.ToolCallPart():
                            tool_calls.append(
                                {
                                    "id": part.tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": part.tool_name,
                                        "arguments": part.tool_args,
                                    },
                                }
                            )
                        case (
                            types.messages.BuiltinToolCallPart()
                            | types.messages.BuiltinToolReturnPart()
                        ):
                            raise NotImplementedError(
                                "OpenAI chat-completions protocol does not "
                                "support BuiltinToolCallPart or "
                                "BuiltinToolReturnPart in the message history. "
                                "Route via the AI Gateway provider until a "
                                "native Responses protocol ships."
                            )

                entry: dict[str, Any] = {"role": "assistant"}
                if content:
                    entry["content"] = content
                if reasoning:
                    entry["reasoning"] = reasoning
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                result.append(entry)

            case "tool":
                for part in msg.parts:
                    if isinstance(part, types.messages.ToolResultPart):
                        model_input = part.get_model_input()
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": part.tool_call_id,
                                "content": str(model_input)
                                if model_input is not None
                                else "",
                            }
                        )

            case "system":
                content_text = "".join(
                    p.text for p in msg.parts if isinstance(p, types.messages.TextPart)
                )
                result.append({"role": "system", "content": content_text})

            case "user":
                has_files = any(
                    isinstance(p, types.messages.FilePart) for p in msg.parts
                )
                if not has_files:
                    text = "".join(
                        p.text
                        for p in msg.parts
                        if isinstance(p, types.messages.TextPart)
                    )
                    result.append({"role": "user", "content": text})
                else:
                    parts: list[dict[str, Any]] = []
                    for p in msg.parts:
                        match p:
                            case types.messages.TextPart(text=text):
                                parts.append({"type": "text", "text": text})
                            case types.messages.FilePart():
                                parts.append(await _file_part_to_openai(p))
                    result.append({"role": "user", "content": parts})
    return result


# ---------------------------------------------------------------------------
def _coerce_params(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("openai stream params must be a dict")


async def stream(
    sdk_client: openai.AsyncOpenAI,
    model: core.model.Model,
    messages: list[types.messages.Message],
    *,
    tools: Sequence[types.tools.Tool] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    params: Any = None,
    provider: str,
) -> AsyncGenerator[types.events.Event]:
    """Stream through the OpenAI chat completions protocol using *sdk_client*."""
    openai_sdk = _sdk.import_sdk(provider=provider)
    if tools and any(t.kind == "provider" for t in tools):
        raise NotImplementedError(
            "OpenAI built-in tools require the Responses API. "
            "The chat-completions protocol does not support them. "
            "Route via the AI Gateway provider until a native Responses "
            "protocol ships."
        )

    stream_params = _coerce_params(params)
    openai_messages = await _messages_to_openai(messages)
    openai_tools = _tools_to_openai(tools) if tools else None

    stream_options = {
        "include_usage": True,
        **dict(stream_params.pop("stream_options", {}) or {}),
    }
    api_kwargs: dict[str, Any] = dict(stream_params)
    api_kwargs.update(
        {
            "model": model.id,
            "messages": openai_messages,
            "stream": True,
            "stream_options": stream_options,
        }
    )
    if openai_tools:
        api_kwargs["tools"] = openai_tools

    if output_type is not None:
        openai_pydantic = _sdk.import_pydantic(provider=provider)

        api_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": output_type.__name__,
                "schema": openai_pydantic.to_strict_json_schema(output_type),
                "strict": True,
            },
        }

    try:
        sdk_stream = await sdk_client.chat.completions.create(**api_kwargs)

        text_started = False
        reasoning_started = False
        tc_state: dict[int, dict[str, Any]] = {}
        usage: types.usage.Usage | None = None

        yield types.events.StreamStart()

        async for chunk in sdk_stream:
            if chunk.usage is not None:
                raw = chunk.usage.model_dump(exclude_none=True)
                reasoning_tokens: int | None = None
                cache_read: int | None = None
                cd = getattr(chunk.usage, "completion_tokens_details", None)
                if cd:
                    reasoning_tokens = getattr(cd, "reasoning_tokens", None)
                pd = getattr(chunk.usage, "prompt_tokens_details", None)
                if pd:
                    cache_read = getattr(pd, "cached_tokens", None)
                usage = types.usage.Usage(
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                    reasoning_tokens=reasoning_tokens,
                    cache_read_tokens=cache_read,
                    raw=raw,
                )

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            reasoning_value = None
            if hasattr(delta, "reasoning") and delta.reasoning:
                reasoning_value = delta.reasoning
            elif hasattr(delta, "model_extra") and delta.model_extra:
                reasoning_value = delta.model_extra.get("reasoning")

            if reasoning_value:
                if not reasoning_started:
                    reasoning_started = True
                    yield types.events.ReasoningStart(block_id="reasoning")
                yield types.events.ReasoningDelta(
                    chunk=reasoning_value,
                    block_id="reasoning",
                )

            if delta.content:
                if reasoning_started:
                    yield types.events.ReasoningEnd(block_id="reasoning")
                    reasoning_started = False

                if not text_started:
                    text_started = True
                    yield types.events.TextStart(block_id="text")
                yield types.events.TextDelta(chunk=delta.content, block_id="text")

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tc_state:
                        tc_state[idx] = {
                            "id": tc.id,
                            "name": None,
                            "started": False,
                        }
                    if tc.id:
                        tc_state[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tc_state[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tid = tc_state[idx]["id"]
                            tname = tc_state[idx]["name"] or ""

                            if not tc_state[idx]["started"] and tid:
                                tc_state[idx]["started"] = True
                                yield types.events.ToolStart(
                                    tool_call_id=tid,
                                    tool_name=tname,
                                )

                            if tid:
                                yield types.events.ToolDelta(
                                    chunk=tc.function.arguments,
                                    tool_call_id=tid,
                                )

            if choice.finish_reason is not None:
                if reasoning_started:
                    yield types.events.ReasoningEnd(block_id="reasoning")
                    reasoning_started = False
                if text_started:
                    yield types.events.TextEnd(block_id="text")
                    text_started = False
                for tc in tc_state.values():
                    if tc["started"] and tc["id"]:
                        yield types.events.ToolEnd(
                            tool_call_id=tc["id"],
                            tool_call=types.messages.DUMMY_TOOL_CALL,
                        )
                        tc["started"] = False

        yield types.events.StreamEnd(usage=usage)
    except openai_sdk.OpenAIError as exc:
        raise errors.map_error(
            exc,
            provider=provider,
            model_id=model.id,
        ) from exc
