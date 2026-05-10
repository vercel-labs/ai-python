"""OpenAI adapter — chat completions API.

Message/tool conversion and streaming via the official ``openai`` SDK.
The SDK client is constructed from :class:`Client` params on each call.
"""

from collections.abc import AsyncGenerator, Sequence
from typing import Any

import openai
import pydantic

from ... import types
from .. import core
from .params import OpenAIChatParams

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
            import base64 as _b64

            text_content = _b64.b64decode(data).decode("utf-8")
        return {"type": "text", "text": text_content}

    raise ValueError(f"Unsupported media type for OpenAI: {mt}")


async def _messages_to_openai(
    messages: list[types.messages.Message],
    *,
    system_message_mode: str = "system",
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
                                "OpenAI chat-completions adapter does not "
                                "support BuiltinToolCallPart or "
                                "BuiltinToolReturnPart in the message history. "
                                "Route via the AI Gateway provider until a "
                                "native Responses adapter ships."
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
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": part.tool_call_id,
                                "content": str(part.result)
                                if part.result is not None
                                else "",
                            }
                        )

            case "system":
                if system_message_mode == "remove":
                    continue
                content_text = "".join(
                    p.text for p in msg.parts if isinstance(p, types.messages.TextPart)
                )
                role = "developer" if system_message_mode == "developer" else "system"
                result.append({"role": role, "content": content_text})

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
# SDK client factory
# ---------------------------------------------------------------------------


def _make_client(client: core.client.Client) -> openai.AsyncOpenAI:
    """Construct an ``AsyncOpenAI`` from our generic ``Client``."""
    return openai.AsyncOpenAI(
        base_url=client.base_url,
        api_key=client.api_key or "",
    )


def _coerce_openai_params(value: Any) -> OpenAIChatParams:
    if value is None:
        return OpenAIChatParams()
    if isinstance(value, OpenAIChatParams):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        items = list(value)
        if len(items) == 1 and isinstance(items[0], OpenAIChatParams):
            return items[0]
    raise TypeError(f"openai stream params must be {OpenAIChatParams.__name__}")


def _merge_extra_body(
    api_kwargs: dict[str, Any],
    extra_body: dict[str, Any] | None,
) -> None:
    """Apply raw body fields after typed params so new provider options can pass."""
    if not extra_body:
        return
    merged = dict(api_kwargs.get("extra_body") or {})
    merged.update(extra_body)
    api_kwargs["extra_body"] = merged


# ---------------------------------------------------------------------------
# Public adapter function
# ---------------------------------------------------------------------------


async def stream(
    client: core.client.Client,
    model: core.model.Model,
    messages: list[types.messages.Message],
    *,
    tools: Sequence[types.tools.Tool] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    thinking: bool = False,
    budget_tokens: int | None = None,
    reasoning_effort: str | None = None,
    **kwargs: Any,
) -> AsyncGenerator[types.events.Event]:
    """Stream an LLM response via the OpenAI chat completions API.

    Yields :class:`~ai.types.events.Event` objects as the response streams in.
    Pure delta emitter — the :class:`~ai.models.Stream` wrapper aggregates
    parts into the final :class:`~ai.types.messages.Message`.

    Extra keyword arguments beyond the ``StreamFn`` protocol:

    * ``thinking`` — enable reasoning/thinking output.
    * ``budget_tokens`` — max tokens for reasoning (mutually exclusive
      with ``reasoning_effort``).
    * ``reasoning_effort`` — effort level: ``"none"``, ``"minimal"``,
      ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``
      (mutually exclusive with ``budget_tokens``).
    """
    if tools and any(t.kind == "provider" for t in tools):
        raise NotImplementedError(
            "OpenAI built-in tools require the Responses API. "
            "The chat-completions adapter does not support them. "
            "Route via the AI Gateway provider until a native Responses "
            "adapter ships."
        )

    sdk_client = _make_client(client)
    openai_params = _coerce_openai_params(kwargs.get("params"))
    system_message_mode = openai_params.system_message_mode
    if system_message_mode is None and openai_params.force_reasoning:
        system_message_mode = "developer"
    openai_messages = await _messages_to_openai(
        messages,
        system_message_mode=system_message_mode or "system",
    )
    openai_tools = _tools_to_openai(tools) if tools else None

    api_kwargs: dict[str, Any] = {
        "model": model.id,
        "messages": openai_messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if openai_tools:
        api_kwargs["tools"] = openai_tools

    if output_type is not None:
        from openai.lib._pydantic import to_strict_json_schema

        strict_json_schema = (
            True
            if openai_params.strict_json_schema is None
            else openai_params.strict_json_schema
        )
        api_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": output_type.__name__,
                "schema": to_strict_json_schema(output_type),
                "strict": strict_json_schema,
            },
        }

    if openai_params.logit_bias is not None:
        api_kwargs["logit_bias"] = openai_params.logit_bias
    if openai_params.logprobs is not None:
        if isinstance(openai_params.logprobs, bool):
            api_kwargs["logprobs"] = openai_params.logprobs
        else:
            api_kwargs["logprobs"] = True
            api_kwargs["top_logprobs"] = openai_params.logprobs
    if openai_params.top_logprobs is not None:
        api_kwargs["top_logprobs"] = openai_params.top_logprobs
    if openai_params.store is not None:
        api_kwargs["store"] = openai_params.store
    if openai_params.prediction is not None:
        api_kwargs["prediction"] = openai_params.prediction
    if openai_params.max_completion_tokens is not None:
        api_kwargs["max_completion_tokens"] = openai_params.max_completion_tokens
    if openai_params.service_tier is not None:
        api_kwargs["service_tier"] = openai_params.service_tier
    if openai_params.metadata is not None:
        api_kwargs["metadata"] = openai_params.metadata
    if openai_params.user is not None:
        api_kwargs["user"] = openai_params.user
    if openai_params.prompt_cache_key is not None:
        api_kwargs["prompt_cache_key"] = openai_params.prompt_cache_key
    if openai_params.prompt_cache_retention is not None:
        api_kwargs["prompt_cache_retention"] = openai_params.prompt_cache_retention
    if openai_params.parallel_tool_calls is not None:
        api_kwargs["parallel_tool_calls"] = openai_params.parallel_tool_calls
    if openai_params.reasoning_effort is not None:
        api_kwargs["reasoning_effort"] = openai_params.reasoning_effort
    if openai_params.text_verbosity is not None:
        api_kwargs["verbosity"] = openai_params.text_verbosity
    if openai_params.safety_identifier is not None:
        api_kwargs["safety_identifier"] = openai_params.safety_identifier

    # Enable reasoning/thinking via Vercel AI Gateway's unified format
    if thinking:
        reasoning_config: dict[str, Any] = {"enabled": True}
        if budget_tokens is not None:
            reasoning_config["max_tokens"] = budget_tokens
        elif reasoning_effort is not None:
            reasoning_config["effort"] = reasoning_effort
        api_kwargs["extra_body"] = {"reasoning": reasoning_config}
    _merge_extra_body(api_kwargs, openai_params.extra_body)
    if openai_params.extra_headers:
        api_kwargs["extra_headers"] = dict(openai_params.extra_headers)

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

            # Reasoning / thinking content
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
                    chunk=reasoning_value, block_id="reasoning"
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
                                    tool_call_id=tid, tool_name=tname
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
    finally:
        await sdk_client.close()
