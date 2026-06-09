"""OpenAI-compatible wire protocols.

Message/tool conversion and streaming via the official ``openai`` SDK.
OpenAI-compatible providers own the SDK client used by these protocols.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from ... import errors as ai_errors
from ... import types
from ...models import core
from ...models.core import params as params_
from .. import base
from . import _sdk, errors
from . import tools as openai_tools

if TYPE_CHECKING:
    import openai
    import pydantic

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


def _tool_result_to_openai(value: Any) -> str | list[dict[str, Any]]:
    """Convert a tool result's model-facing value to OpenAI content (chat).

    A :class:`ContentOutput` expands into a content array with ``text``
    and ``image_url`` parts.  Everything else is sent as a string: ``str``
    raw, ``None`` as ``""``, anything else JSON-encoded.
    """
    if isinstance(value, types.messages.ContentOutput):
        parts: list[dict[str, Any]] = []
        for item in value.value:
            if isinstance(item, types.messages.FilePart):
                mt = item.media_type
                if mt.startswith("image/"):
                    data_url = types.media.data_to_data_url(item.data, mt)
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    )
                else:
                    parts.append({"type": "text", "text": f"[file: {mt}]"})
            else:
                parts.append({"type": "text", "text": item.text})
        return parts
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return _json_dumps(value)


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
                        tool_content = _tool_result_to_openai(
                            part.get_model_input()
                        )
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": part.tool_call_id,
                                "content": tool_content,
                            }
                        )

            case "system":
                content_text = "".join(
                    p.text
                    for p in msg.parts
                    if isinstance(p, types.messages.TextPart)
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
def _is_default(value: object) -> bool:
    return isinstance(value, params_.ModelProviderDefault)


def _not_default(value: object) -> bool:
    return not _is_default(value)


def _seed_value(seed: object) -> int | None:
    if seed is None or seed == -1 or isinstance(seed, params_.RandomSeed):
        return None
    if isinstance(seed, int):
        return seed
    if isinstance(seed, params_.ModelProviderDefault):
        return None
    raise TypeError("seed must be an int, RANDOM, DEFAULT, or None")


def _filter_extra_headers(
    headers: Mapping[str, str | params_.Unset] | None,
) -> dict[str, str] | None:
    if headers is None:
        return None
    return {
        key: value
        for key, value in headers.items()
        if not isinstance(value, params_.Unset)
    }


def _tool_ref_name(tool_ref: params_.ToolRef | str) -> str:
    return str(tool_ref)


def _tools_for_openai_allowed(
    tools: frozenset[params_.ToolRef],
    *,
    responses: bool,
) -> list[dict[str, Any]]:
    if responses:
        return [
            {"type": "function", "name": _tool_ref_name(tool)}
            for tool in sorted(tools)
        ]
    return [
        {"type": "function", "function": {"name": _tool_ref_name(tool)}}
        for tool in sorted(tools)
    ]


def _openai_tool_choice(
    tool_choice: params_.ToolChoiceMode
    | params_.ToolRef
    | params_.ToolSelection,
    *,
    responses: bool,
) -> Any:
    if isinstance(tool_choice, params_.ToolChoiceMode):
        return tool_choice.value
    if isinstance(tool_choice, params_.ToolRef):
        if responses:
            return {"type": "function", "name": _tool_ref_name(tool_choice)}
        return {
            "type": "function",
            "function": {"name": _tool_ref_name(tool_choice)},
        }
    if tool_choice.mode == params_.ToolChoiceMode.NONE:
        raise ValueError("OpenAI allowed tool selection does not support none")
    if responses:
        return {
            "type": "allowed_tools",
            "mode": tool_choice.mode.value,
            "tools": _tools_for_openai_allowed(
                tool_choice.tools,
                responses=True,
            ),
        }
    return {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": tool_choice.mode.value,
            "tools": _tools_for_openai_allowed(
                tool_choice.tools,
                responses=False,
            ),
        },
    }


def _apply_sampling(
    api_kwargs: dict[str, Any],
    request_params: params_.InferenceRequestParams,
    *,
    chat: bool,
) -> None:
    sampling = request_params.sampling
    if isinstance(sampling, params_.ModelProviderDefault):
        return
    for sampler in sampling.values():
        match sampler:
            case params_.TemperatureSamplerParams(temperature=temperature):
                if _not_default(temperature):
                    api_kwargs["temperature"] = temperature
            case params_.TopPSamplerParams(top_p=top_p):
                if _not_default(top_p):
                    api_kwargs["top_p"] = top_p
            case params_.SeedSamplerParams(seed=seed):
                if _not_default(seed):
                    value = _seed_value(seed)
                    if value is not None:
                        if chat:
                            api_kwargs["seed"] = value
                        else:
                            raise ValueError(
                                "OpenAI responses does not support seed"
                            )
            case params_.TopKSamplerParams(top_k=top_k):
                if _not_default(top_k) and top_k is not None:
                    raise ValueError("OpenAI does not support top_k")
            case params_.MinPSamplerParams(min_p=min_p):
                if _not_default(min_p) and min_p is not None:
                    raise ValueError("OpenAI does not support min_p")
            case params_.RepetitionPenaltyParams() as repetition:
                if _not_default(repetition.frequency_penalty):
                    api_kwargs["frequency_penalty"] = (
                        repetition.frequency_penalty
                    )
                if _not_default(repetition.presence_penalty):
                    api_kwargs["presence_penalty"] = repetition.presence_penalty
                if (
                    _not_default(repetition.repetition_penalty)
                    and repetition.repetition_penalty is not None
                ):
                    raise ValueError(
                        "OpenAI does not support repetition_penalty"
                    )
                if (
                    _not_default(repetition.consideration_window)
                    and repetition.consideration_window is not None
                ):
                    raise ValueError(
                        "OpenAI does not support consideration_window"
                    )
    _ = chat


def _apply_common_openai_params(
    api_kwargs: dict[str, Any],
    request_params: params_.InferenceRequestParams,
    *,
    provider: str,
    responses: bool,
) -> None:
    _ = provider
    _apply_sampling(api_kwargs, request_params, chat=not responses)

    reasoning = request_params.reasoning
    output = request_params.output
    effort: str | params_.ModelProviderDefault | None = params_.DEFAULT
    if not isinstance(reasoning, params_.ModelProviderDefault):
        effort = reasoning.effort
    summary = params_.DEFAULT if output is None else output.reasoning_summary
    if _not_default(effort) or _not_default(summary):
        if responses:
            reasoning_kwargs: dict[str, Any] = dict(
                api_kwargs.get("reasoning") or {}
            )
            if _not_default(effort):
                reasoning_kwargs["effort"] = (
                    "none" if effort is None else effort
                )
            if _not_default(summary):
                reasoning_kwargs["summary"] = summary
            api_kwargs["reasoning"] = reasoning_kwargs
        else:
            if _not_default(effort):
                api_kwargs["reasoning_effort"] = (
                    "none" if effort is None else effort
                )
            if _not_default(summary) and summary is not None:
                raise ValueError(
                    "OpenAI chat completions does not support reasoning summary"
                )

    if request_params.tool_calling is not None:
        tool_calling = request_params.tool_calling
        if _not_default(tool_calling.max_tool_calls):
            if responses:
                api_kwargs["max_tool_calls"] = tool_calling.max_tool_calls
            elif tool_calling.max_tool_calls is not None:
                raise ValueError(
                    "OpenAI chat completions does not support max_tool_calls"
                )
        if _not_default(tool_calling.parallel_tool_calls):
            api_kwargs["parallel_tool_calls"] = tool_calling.parallel_tool_calls
        api_kwargs["tool_choice"] = _openai_tool_choice(
            tool_calling.tool_choice,
            responses=responses,
        )

    if request_params.provider_service is not None:
        service = request_params.provider_service
        if _not_default(service.service_tier):
            api_kwargs["service_tier"] = service.service_tier

    if request_params.safety_identifier is not None:
        api_kwargs["safety_identifier"] = request_params.safety_identifier
    if request_params.metadata is not None:
        api_kwargs["metadata"] = dict(request_params.metadata)

    if request_params.context_management is not None:
        context_management = request_params.context_management
        if context_management.compaction is not None:
            if responses:
                api_kwargs["context_management"] = [
                    {
                        "type": "compaction",
                        "compact_threshold": (
                            context_management.compaction.value
                        ),
                    }
                ]
            else:
                raise ValueError(
                    "OpenAI chat completions does not support "
                    "context management"
                )

    if request_params.output is not None:
        output = request_params.output
        if output.max_tokens is not None:
            api_kwargs[
                "max_output_tokens" if responses else "max_completion_tokens"
            ] = output.max_tokens
        if output.include is not None:
            if responses:
                api_kwargs["include"] = sorted(output.include)
            else:
                raise ValueError(
                    "OpenAI chat completions does not support output include"
                )
        if _not_default(output.text_verbosity):
            if responses:
                text_config = dict(api_kwargs.get("text") or {})
                text_config["verbosity"] = output.text_verbosity
                api_kwargs["text"] = text_config
            elif output.text_verbosity is not None:
                raise ValueError(
                    "OpenAI chat completions does not support text verbosity"
                )

    if request_params.cache is not None:
        cache = request_params.cache
        if cache.key is not None:
            api_kwargs["prompt_cache_key"] = cache.key
        if _not_default(cache.retention):
            api_kwargs["prompt_cache_retention"] = cache.retention

    extra_headers = _filter_extra_headers(request_params.extra_headers)
    if extra_headers is not None:
        api_kwargs["extra_headers"] = extra_headers
    if request_params.extra_query is not None:
        api_kwargs["extra_query"] = dict(request_params.extra_query)
    if request_params.extra_body is not None:
        api_kwargs["extra_body"] = dict(request_params.extra_body)


def _coerce_params(
    value: params_.InferenceRequestParams | None,
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, params_.InferenceRequestParams):
        api_kwargs: dict[str, Any] = {}
        _apply_common_openai_params(
            api_kwargs,
            value,
            provider="openai",
            responses=False,
        )
        return api_kwargs
    raise TypeError("openai stream params must be InferenceRequestParams")


async def stream(
    sdk_client: openai.AsyncOpenAI,
    model: core.model.Model,
    messages: list[types.messages.Message],
    *,
    tools: Sequence[types.tools.Tool] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    params: params_.InferenceRequestParams | None = None,
    provider: str,
) -> AsyncGenerator[types.events.Event]:
    """Stream through the OpenAI chat completions protocol."""
    openai_sdk = _sdk.import_sdk(provider=provider)
    if tools and any(t.kind == "provider" for t in tools):
        raise NotImplementedError(
            "OpenAI built-in tools require the Responses API. "
            "The chat-completions protocol does not support them. "
            "Route via the AI Gateway provider until a native Responses "
            "protocol ships."
        )

    stream_params = _coerce_params(params)
    if params is not None:
        stream_params = {}
        _apply_common_openai_params(
            stream_params,
            params,
            provider=provider,
            responses=False,
        )
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

            # providers expose reasoning in reasoning_content (e.g. deepseek)
            # or reasoning (some gpt-oss providers)
            reasoning_value = None
            if hasattr(delta, "reasoning") and delta.reasoning:
                reasoning_value = delta.reasoning
            elif hasattr(delta, "model_extra") and delta.model_extra:
                reasoning_value = delta.model_extra.get("reasoning")
                if not reasoning_value:
                    reasoning_value = delta.model_extra.get("reasoning_content")

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
                yield types.events.TextDelta(
                    chunk=delta.content, block_id="text"
                )

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


class OpenAIChatCompletionsProtocol(base.ProviderProtocol[Any]):
    """OpenAI Chat Completions protocol."""

    def stream(
        self,
        client: openai.AsyncOpenAI,
        model: core.model.Model,
        messages: list[types.messages.Message],
        *,
        tools: Sequence[types.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
        provider: str,
    ) -> AsyncGenerator[types.events.Event]:
        return stream(
            client,
            model,
            messages,
            tools=tools,
            output_type=output_type,
            params=params,
            provider=provider,
        )


_OPENAI_METADATA_KEY = "openai"
_RESPONSES_PROTECTED_PARAMS = frozenset({"model", "input", "stream"})
_BUILTIN_OUTPUT_TYPES = frozenset(
    {
        "web_search_call",
        "file_search_call",
        "code_interpreter_call",
        "image_generation_call",
        "mcp_call",
        "mcp_approval_request",
        "local_shell_call",
        "shell_call",
        "shell_call_output",
        "apply_patch_call",
        "tool_search_call",
        "tool_search_output",
        "computer_call",
    }
)


def _coerce_responses_params(
    value: params_.InferenceRequestParams | None,
    *,
    provider: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, params_.InferenceRequestParams):
        api_kwargs: dict[str, Any] = {}
        _apply_common_openai_params(
            api_kwargs,
            value,
            provider=provider,
            responses=True,
        )
        return api_kwargs
    raise TypeError(
        "openai responses stream params must be InferenceRequestParams"
    )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


def _model_dump(value: pydantic.BaseModel) -> dict[str, Any]:
    return value.model_dump(exclude_none=True)


def _openai_metadata(part: Any) -> dict[str, Any]:
    metadata = getattr(part, "provider_metadata", None)
    if not isinstance(metadata, Mapping):
        return {}
    openai_metadata = metadata.get(_OPENAI_METADATA_KEY)
    if not isinstance(openai_metadata, Mapping):
        return {}
    return dict(openai_metadata)


def _metadata_item_id(metadata: Mapping[str, Any]) -> str | None:
    value = metadata.get("item_id") or metadata.get("itemId")
    return value if isinstance(value, str) else None


def _provider_metadata_for_item(
    item: Mapping[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    item_id = item.get("id")
    data = {
        "raw_item": dict(item),
        **({"item_id": item_id} if isinstance(item_id, str) else {}),
        **{k: v for k, v in extra.items() if v is not None},
    }
    return {_OPENAI_METADATA_KEY: data}


def _provider_metadata_for_response(
    response: Mapping[str, Any],
) -> dict[str, Any]:
    response_id = response.get("id")
    model = response.get("model")
    status = response.get("status")
    data = {
        **(
            {"response_id": response_id} if isinstance(response_id, str) else {}
        ),
        **({"model": model} if isinstance(model, str) else {}),
        **({"status": status} if isinstance(status, str) else {}),
    }
    return {_OPENAI_METADATA_KEY: data} if data else {}


def _maybe_item_reference(
    part: Any,
    *,
    use_item_references: bool,
) -> dict[str, Any] | None:
    if not use_item_references:
        return None
    item_id = _metadata_item_id(_openai_metadata(part))
    if item_id is None:
        return None
    return {"type": "item_reference", "id": item_id}


def _raw_item_from_metadata(part: Any) -> dict[str, Any] | None:
    raw_item = _openai_metadata(part).get("raw_item")
    if isinstance(raw_item, Mapping):
        return dict(raw_item)
    return None


def _tool_result_to_responses(value: Any) -> str | list[dict[str, Any]]:
    """Convert a tool result's model-facing value to a Responses ``output``.

    Returns a plain string for ordinary values (``str`` raw, ``None`` as
    ``""``, anything else JSON-encoded), or an array of ``input_text`` /
    ``input_image`` / ``input_file`` parts for a :class:`ContentOutput`
    (the Responses API accepts both shapes on ``function_call_output.output``).
    """
    if isinstance(value, types.messages.ContentOutput):
        parts: list[dict[str, Any]] = []
        for item in value.value:
            if isinstance(item, types.messages.FilePart):
                data_url = types.media.data_to_data_url(
                    item.data, item.media_type
                )
                if item.media_type.startswith("image/"):
                    parts.append({"type": "input_image", "image_url": data_url})
                else:
                    entry: dict[str, Any] = {
                        "type": "input_file",
                        "file_data": data_url,
                    }
                    if item.filename is not None:
                        entry["filename"] = item.filename
                    parts.append(entry)
            else:
                parts.append({"type": "input_text", "text": item.text})
        return parts
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return _json_dumps(value)


async def _file_part_to_responses(
    part: types.messages.FilePart,
) -> dict[str, Any]:
    media_type = (
        "image/jpeg" if part.media_type == "image/*" else part.media_type
    )
    data = part.data

    if media_type.startswith("image/"):
        return {
            "type": "input_image",
            "image_url": types.media.data_to_data_url(data, media_type),
        }

    if media_type == "application/pdf":
        if isinstance(data, str) and types.media.is_downloadable_url(data):
            return {"type": "input_file", "file_url": data}
        return {
            "type": "input_file",
            "filename": part.filename or "document.pdf",
            "file_data": types.media.data_to_data_url(data, media_type),
        }

    if media_type.startswith("text/"):
        if isinstance(data, bytes):
            text_content = data.decode("utf-8")
        elif types.media.is_url(data):
            text_content = data
        else:
            text_content = base64.b64decode(data).decode("utf-8")
        return {"type": "input_text", "text": text_content}

    raise ValueError(
        f"Unsupported media type for OpenAI Responses: {media_type}"
    )


async def _messages_to_responses(
    messages: list[types.messages.Message],
    *,
    use_item_references: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for msg in messages:
        match msg.role:
            case "system":
                text = "".join(
                    p.text
                    for p in msg.parts
                    if isinstance(p, types.messages.TextPart)
                )
                if text:
                    result.append({"role": "system", "content": text})

            case "user":
                content: list[dict[str, Any]] = []
                for part in msg.parts:
                    match part:
                        case types.messages.TextPart(text=text):
                            content.append({"type": "input_text", "text": text})
                        case types.messages.FilePart():
                            content.append(await _file_part_to_responses(part))
                result.append({"role": "user", "content": content})

            case "assistant":
                assistant_content: list[dict[str, Any]] = []

                for part in msg.parts:
                    if item_reference := _maybe_item_reference(
                        part,
                        use_item_references=use_item_references,
                    ):
                        _flush_assistant_content(result, assistant_content)
                        result.append(item_reference)
                        continue

                    if raw_item := _raw_item_from_metadata(part):
                        _flush_assistant_content(result, assistant_content)
                        result.append(raw_item)
                        continue

                    match part:
                        case types.messages.TextPart(text=text):
                            assistant_content.append(
                                {"type": "output_text", "text": text}
                            )
                        case types.messages.ReasoningPart(text=text):
                            _flush_assistant_content(result, assistant_content)
                            metadata = _openai_metadata(part)
                            encrypted_content = metadata.get(
                                "reasoning_encrypted_content"
                            ) or metadata.get("reasoningEncryptedContent")
                            if encrypted_content is not None:
                                result.append(
                                    {
                                        "type": "reasoning",
                                        "summary": [
                                            {
                                                "type": "summary_text",
                                                "text": text,
                                            }
                                        ],
                                        "encrypted_content": encrypted_content,
                                    }
                                )
                        case types.messages.ToolCallPart():
                            _flush_assistant_content(result, assistant_content)
                            result.append(
                                {
                                    "type": "function_call",
                                    "call_id": part.tool_call_id,
                                    "name": part.tool_name,
                                    "arguments": part.tool_args,
                                }
                            )
                        case (
                            types.messages.BuiltinToolCallPart()
                            | types.messages.BuiltinToolReturnPart()
                        ):
                            _flush_assistant_content(result, assistant_content)

                _flush_assistant_content(result, assistant_content)

            case "tool":
                for part in msg.parts:
                    if isinstance(part, types.messages.ToolResultPart):
                        result.append(
                            {
                                "type": "function_call_output",
                                "call_id": part.tool_call_id,
                                "output": _tool_result_to_responses(
                                    part.get_model_input()
                                ),
                            }
                        )

            case "internal":
                continue

    return result


def _flush_assistant_content(
    result: list[dict[str, Any]],
    assistant_content: list[dict[str, Any]],
) -> None:
    if not assistant_content:
        return
    result.append({"role": "assistant", "content": list(assistant_content)})
    assistant_content.clear()


def _tools_to_responses(
    tools: Sequence[types.tools.Tool],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for tool in tools:
        if tool.kind == "function":
            args = tool.args
            if not isinstance(args, types.tools.FunctionToolArgs):
                raise TypeError(f"function tool {tool.name!r} has invalid args")
            result.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": args.description or "",
                    "parameters": args.params,
                }
            )
            continue

        args = tool.args
        tool_id = getattr(type(args), "openai_id", None)
        if not isinstance(args, openai_tools.OpenAIProviderArgs):
            raise TypeError(
                f"provider tool {tool.name!r} is not an OpenAI tool"
            )

        match tool_id:
            case "openai.web_search":
                result.append({"type": "web_search", **_model_dump(args)})
            case "openai.web_search_preview":
                result.append(
                    {"type": "web_search_preview", **_model_dump(args)}
                )
            case "openai.file_search":
                data = _model_dump(args)
                ranking = data.pop("ranking", None)
                if ranking is not None:
                    data["ranking_options"] = ranking
                result.append({"type": "file_search", **data})
            case "openai.code_interpreter":
                data = _model_dump(args)
                if "container" not in data:
                    data["container"] = {"type": "auto"}
                result.append({"type": "code_interpreter", **data})
            case "openai.image_generation":
                result.append({"type": "image_generation", **_model_dump(args)})
            case "openai.local_shell":
                result.append({"type": "local_shell"})
            case "openai.shell":
                result.append({"type": "shell", **_model_dump(args)})
            case "openai.apply_patch":
                result.append({"type": "apply_patch"})
            case "openai.mcp":
                result.append({"type": "mcp", **_model_dump(args)})
            case "openai.tool_search":
                result.append({"type": "tool_search", **_model_dump(args)})
            case _:
                raise NotImplementedError(
                    f"unsupported OpenAI provider tool {tool_id}"
                )

    return result


def _event_to_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, Mapping):
        return dict(event)
    if hasattr(event, "model_dump"):
        dumped = event.model_dump(exclude_none=True, mode="json")
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    if hasattr(event, "to_dict"):
        dumped = event.to_dict()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {
        key: value
        for key in dir(event)
        if not key.startswith("_")
        and not callable(value := getattr(event, key, None))
    }


def _usage_from_response(
    response: Mapping[str, Any],
) -> types.usage.Usage | None:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None

    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    if not isinstance(input_details, Mapping):
        input_details = {}
    if not isinstance(output_details, Mapping):
        output_details = {}

    return types.usage.Usage(
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        reasoning_tokens=output_details.get("reasoning_tokens"),
        cache_read_tokens=input_details.get("cached_tokens"),
        raw=dict(usage),
    )


def _image_media_type(
    params: Mapping[str, Any],
    tools: Sequence[types.tools.Tool],
) -> str:
    for tool in tools:
        if isinstance(tool.args, openai_tools.ImageGenerationArgs):
            fmt = str(tool.args.output_format or "png")
            return "image/jpeg" if fmt == "jpeg" else f"image/{fmt}"
    text = params.get("text")
    if isinstance(text, Mapping):
        text_fmt = text.get("output_format")
        if isinstance(text_fmt, str):
            return "image/jpeg" if text_fmt == "jpeg" else f"image/{text_fmt}"
    return "image/png"


def _state_key(item: Mapping[str, Any], data: Mapping[str, Any]) -> str:
    value = item.get("id") or item.get("call_id") or data.get("item_id")
    if isinstance(value, str):
        return value
    output_index = data.get("output_index")
    return str(output_index) if output_index is not None else ""


def _builtin_tool_name(item: Mapping[str, Any]) -> str:
    item_type = item.get("type")
    match item_type:
        case "web_search_call":
            return "web_search"
        case "file_search_call":
            return "file_search"
        case "code_interpreter_call":
            return "code_interpreter"
        case "image_generation_call":
            return "image_generation"
        case "mcp_call" | "mcp_approval_request":
            name = item.get("name")
            return f"mcp.{name}" if isinstance(name, str) else "mcp"
        case "local_shell_call":
            return "local_shell"
        case "shell_call" | "shell_call_output":
            return "shell"
        case "apply_patch_call":
            return "apply_patch"
        case "tool_search_call" | "tool_search_output":
            return "tool_search"
        case "computer_call":
            return "computer_use"
        case _:
            return str(item_type or "")


def _builtin_tool_call_id(item: Mapping[str, Any]) -> str:
    value = item.get("call_id") or item.get("id")
    return value if isinstance(value, str) else ""


def _builtin_tool_args(item: Mapping[str, Any]) -> str:
    item_type = item.get("type")
    match item_type:
        case "code_interpreter_call":
            return _json_dumps(
                {
                    "code": item.get("code"),
                    "container_id": item.get("container_id"),
                }
            )
        case "mcp_call" | "mcp_approval_request":
            arguments = item.get("arguments")
            return (
                arguments
                if isinstance(arguments, str)
                else _json_dumps(arguments)
            )
        case "local_shell_call" | "shell_call":
            return _json_dumps({"action": item.get("action")})
        case "apply_patch_call":
            return _json_dumps(
                {
                    "call_id": item.get("call_id"),
                    "operation": item.get("operation"),
                }
            )
        case "tool_search_call":
            return _json_dumps(
                {
                    "arguments": item.get("arguments"),
                    "call_id": item.get("call_id"),
                }
            )
        case _:
            return "{}"


def _builtin_tool_result(item: Mapping[str, Any]) -> Any:
    item_type = item.get("type")
    match item_type:
        case "web_search_call":
            return {"action": item.get("action")}
        case "file_search_call":
            return {
                "queries": item.get("queries"),
                "results": item.get("results"),
            }
        case "code_interpreter_call":
            return {
                "container_id": item.get("container_id"),
                "outputs": item.get("outputs"),
            }
        case "image_generation_call":
            return {"result": item.get("result")}
        case "mcp_call":
            return {
                "server_label": item.get("server_label"),
                "name": item.get("name"),
                "arguments": item.get("arguments"),
                "output": item.get("output"),
                "error": item.get("error"),
            }
        case "mcp_approval_request":
            return {
                "server_label": item.get("server_label"),
                "name": item.get("name"),
                "arguments": item.get("arguments"),
                "approval_request_id": item.get("approval_request_id")
                or item.get("id"),
            }
        case "shell_call_output":
            return {"output": item.get("output")}
        case "tool_search_output":
            return {"tools": item.get("tools")}
        case "computer_call":
            return {"status": item.get("status") or "completed"}
        case _:
            return None


def _index_key(data: Mapping[str, Any]) -> str | None:
    output_index = data.get("output_index")
    return str(output_index) if output_index is not None else None


def _lookup_state(
    states_by_item: dict[str, dict[str, Any]],
    states_by_index: dict[str, dict[str, Any]],
    data: Mapping[str, Any],
) -> dict[str, Any] | None:
    item_id = data.get("item_id")
    if isinstance(item_id, str) and item_id in states_by_item:
        return states_by_item[item_id]
    index = _index_key(data)
    if index is not None:
        return states_by_index.get(index)
    return None


async def _stream_responses(
    sdk_client: openai.AsyncOpenAI,
    model: core.model.Model,
    messages: list[types.messages.Message],
    *,
    tools: Sequence[types.tools.Tool] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    params: params_.InferenceRequestParams | None = None,
    provider: str,
) -> AsyncGenerator[types.events.Event]:
    openai_sdk = _sdk.import_sdk(provider=provider)
    stream_params = _coerce_responses_params(params, provider=provider)
    protected = sorted(_RESPONSES_PROTECTED_PARAMS & stream_params.keys())
    if protected:
        raise ValueError(
            "openai responses params cannot override protocol-owned fields: "
            + ", ".join(protected)
        )

    request_tools = list(tools or ())
    use_item_references = stream_params.get("store") is not False
    response_input = await _messages_to_responses(
        messages,
        use_item_references=use_item_references,
    )
    response_tools = (
        _tools_to_responses(request_tools) if request_tools else None
    )

    api_kwargs: dict[str, Any] = dict(stream_params)
    api_kwargs.update(
        {"model": model.id, "input": response_input, "stream": True}
    )
    if response_tools:
        api_kwargs["tools"] = response_tools

    if output_type is not None:
        openai_pydantic = _sdk.import_pydantic(provider=provider)
        text_config = dict(api_kwargs.get("text") or {})
        text_config["format"] = {
            "type": "json_schema",
            "name": output_type.__name__,
            "schema": openai_pydantic.to_strict_json_schema(output_type),
            "strict": True,
        }
        api_kwargs["text"] = text_config

    image_media_type = _image_media_type(stream_params, request_tools)
    text_blocks: set[str] = set()
    reasoning_blocks: set[str] = set()
    reasoning_delta_blocks: set[str] = set()
    reasoning_ended_blocks: set[str] = set()
    function_states_by_item: dict[str, dict[str, Any]] = {}
    function_states_by_index: dict[str, dict[str, Any]] = {}
    builtin_states_by_item: dict[str, dict[str, Any]] = {}
    builtin_states_by_index: dict[str, dict[str, Any]] = {}
    usage: types.usage.Usage | None = None
    response_metadata: dict[str, Any] | None = None

    try:
        sdk_stream = await sdk_client.responses.create(**api_kwargs)
        yield types.events.StreamStart()

        async for sdk_event in sdk_stream:
            data = _event_to_dict(sdk_event)
            event_type = data.get("type")

            if event_type == "response.created":
                response = data.get("response")
                if isinstance(response, Mapping):
                    response_metadata = _provider_metadata_for_response(
                        response
                    )
                continue

            if event_type in {"response.completed", "response.incomplete"}:
                response = data.get("response")
                if isinstance(response, Mapping):
                    usage = _usage_from_response(response) or usage
                    response_metadata = _provider_metadata_for_response(
                        response
                    )
                continue

            if event_type == "response.failed":
                response = data.get("response")
                if isinstance(response, Mapping):
                    usage = _usage_from_response(response) or usage
                    response_metadata = _provider_metadata_for_response(
                        response
                    )
                continue

            if event_type == "error":
                error = data.get("error")
                if isinstance(error, Mapping):
                    message = error.get("message") or error.get("code") or error
                else:
                    message = error or data
                raise ai_errors.ProviderResponseError(
                    str(message), provider=provider
                )

            if event_type == "response.output_item.added":
                item = data.get("item")
                if not isinstance(item, Mapping):
                    continue
                item = dict(item)
                item_type = item.get("type")
                state_key = _state_key(item, data)
                index = _index_key(data)

                if item_type == "message":
                    block_id = str(item.get("id") or state_key or "text")
                    text_blocks.add(block_id)
                    yield types.events.TextStart(
                        block_id=block_id,
                        provider_metadata=_provider_metadata_for_item(item),
                    )
                    continue

                if item_type == "reasoning":
                    block_id = f"{item.get('id') or state_key}:0"
                    reasoning_blocks.add(block_id)
                    yield types.events.ReasoningStart(
                        block_id=block_id,
                        provider_metadata=_provider_metadata_for_item(
                            item,
                            reasoning_encrypted_content=item.get(
                                "encrypted_content"
                            ),
                        ),
                    )
                    continue

                if item_type in {"function_call", "custom_tool_call"}:
                    tool_call_id = str(item.get("call_id") or state_key)
                    tool_name = str(item.get("name") or "")
                    new_state: dict[str, Any] = {
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "arguments": "",
                        "delta_emitted": False,
                    }
                    if state_key:
                        function_states_by_item[state_key] = new_state
                    if index is not None:
                        function_states_by_index[index] = new_state
                    yield types.events.ToolStart(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        provider_metadata=_provider_metadata_for_item(item),
                    )
                    arguments = item.get("arguments") or item.get("input")
                    if isinstance(arguments, str) and arguments:
                        new_state["arguments"] = arguments
                        new_state["delta_emitted"] = True
                        yield types.events.ToolDelta(
                            tool_call_id=tool_call_id,
                            chunk=arguments,
                        )
                    continue

                if item_type in _BUILTIN_OUTPUT_TYPES:
                    tool_call_id = _builtin_tool_call_id(item)
                    tool_name = _builtin_tool_name(item)
                    new_state = {
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "arguments": "",
                        "delta_emitted": False,
                    }
                    if state_key:
                        builtin_states_by_item[state_key] = new_state
                    if index is not None:
                        builtin_states_by_index[index] = new_state
                    yield types.events.BuiltinToolStart(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        provider_metadata=_provider_metadata_for_item(item),
                    )
                    continue

            if event_type == "response.output_text.delta":
                block_id = str(data.get("item_id") or "text")
                if block_id not in text_blocks:
                    text_blocks.add(block_id)
                    yield types.events.TextStart(block_id=block_id)
                delta = data.get("delta")
                if isinstance(delta, str) and delta:
                    yield types.events.TextDelta(block_id=block_id, chunk=delta)
                continue

            if event_type == "response.output_text.done":
                continue

            if event_type in {
                "response.function_call_arguments.delta",
                "response.custom_tool_call_input.delta",
            }:
                function_state = _lookup_state(
                    function_states_by_item,
                    function_states_by_index,
                    data,
                )
                delta = data.get("delta")
                if (
                    function_state is not None
                    and isinstance(delta, str)
                    and delta
                ):
                    function_state["arguments"] += delta
                    function_state["delta_emitted"] = True
                    yield types.events.ToolDelta(
                        tool_call_id=function_state["tool_call_id"],
                        chunk=delta,
                    )
                continue

            if event_type in {
                "response.function_call_arguments.done",
                "response.custom_tool_call_input.done",
            }:
                function_state = _lookup_state(
                    function_states_by_item,
                    function_states_by_index,
                    data,
                )
                arguments = data.get("arguments") or data.get("input")
                if (
                    function_state is not None
                    and isinstance(arguments, str)
                    and arguments
                    and not function_state["delta_emitted"]
                ):
                    function_state["arguments"] = arguments
                    function_state["delta_emitted"] = True
                    yield types.events.ToolDelta(
                        tool_call_id=function_state["tool_call_id"],
                        chunk=arguments,
                    )
                continue

            if event_type == "response.reasoning_summary_part.added":
                block_id = (
                    f"{data.get('item_id')}:{data.get('summary_index', 0)}"
                )
                if block_id not in reasoning_blocks:
                    reasoning_blocks.add(block_id)
                    yield types.events.ReasoningStart(block_id=block_id)
                continue

            if event_type in {
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            }:
                block_id = (
                    f"{data.get('item_id')}:{data.get('summary_index', 0)}"
                )
                if block_id not in reasoning_blocks:
                    reasoning_blocks.add(block_id)
                    yield types.events.ReasoningStart(block_id=block_id)
                delta = data.get("delta")
                if isinstance(delta, str) and delta:
                    reasoning_delta_blocks.add(block_id)
                    yield types.events.ReasoningDelta(
                        block_id=block_id, chunk=delta
                    )
                continue

            if event_type == "response.reasoning_summary_part.done":
                block_id = (
                    f"{data.get('item_id')}:{data.get('summary_index', 0)}"
                )
                reasoning_blocks.discard(block_id)
                reasoning_ended_blocks.add(block_id)
                yield types.events.ReasoningEnd(block_id=block_id)
                continue

            if event_type == "response.image_generation_call.partial_image":
                item_id = str(data.get("item_id") or "")
                partial = data.get("partial_image_b64")
                if isinstance(partial, str) and partial:
                    yield types.events.FileEvent(
                        block_id=f"{item_id}:partial",
                        media_type=image_media_type,
                        data=partial,
                    )
                continue

            if event_type == "response.code_interpreter_call_code.delta":
                builtin_state = _lookup_state(
                    builtin_states_by_item,
                    builtin_states_by_index,
                    data,
                )
                delta = data.get("delta")
                if (
                    builtin_state is not None
                    and isinstance(delta, str)
                    and delta
                ):
                    builtin_state["arguments"] += delta
                continue

            if event_type == "response.output_item.done":
                item = data.get("item")
                if not isinstance(item, Mapping):
                    continue
                item = dict(item)
                item_type = item.get("type")
                state_key = _state_key(item, data)
                index = _index_key(data)

                if item_type == "message":
                    block_id = str(item.get("id") or state_key or "text")
                    if block_id in text_blocks:
                        text_blocks.remove(block_id)
                        yield types.events.TextEnd(
                            block_id=block_id,
                            provider_metadata=_provider_metadata_for_item(item),
                        )
                    continue

                if item_type == "reasoning":
                    item_id = str(item.get("id") or state_key)
                    summaries = item.get("summary")
                    if isinstance(summaries, list) and summaries:
                        for idx, summary in enumerate(summaries):
                            block_id = f"{item_id}:{idx}"
                            if block_id in reasoning_ended_blocks:
                                continue
                            if block_id not in reasoning_blocks:
                                reasoning_blocks.add(block_id)
                                yield types.events.ReasoningStart(
                                    block_id=block_id,
                                    provider_metadata=_provider_metadata_for_item(
                                        item,
                                        reasoning_encrypted_content=item.get(
                                            "encrypted_content"
                                        ),
                                    ),
                                )
                            if (
                                block_id not in reasoning_delta_blocks
                                and isinstance(summary, Mapping)
                            ):
                                summary_mapping = cast(
                                    "Mapping[str, Any]", summary
                                )
                                text = summary_mapping.get("text")
                                if isinstance(text, str) and text:
                                    yield types.events.ReasoningDelta(
                                        block_id=block_id,
                                        chunk=text,
                                    )
                            reasoning_blocks.discard(block_id)
                            reasoning_ended_blocks.add(block_id)
                            yield types.events.ReasoningEnd(
                                block_id=block_id,
                                provider_metadata=_provider_metadata_for_item(
                                    item,
                                    reasoning_encrypted_content=item.get(
                                        "encrypted_content"
                                    ),
                                ),
                            )
                    else:
                        for block_id in list(reasoning_blocks):
                            if block_id.startswith(f"{item_id}:"):
                                reasoning_blocks.remove(block_id)
                                yield types.events.ReasoningEnd(
                                    block_id=block_id,
                                    provider_metadata=_provider_metadata_for_item(
                                        item,
                                        reasoning_encrypted_content=item.get(
                                            "encrypted_content"
                                        ),
                                    ),
                                )
                    continue

                if item_type in {"function_call", "custom_tool_call"}:
                    function_state = (
                        function_states_by_item.pop(state_key, None)
                        if state_key
                        else None
                    )
                    if function_state is None and index is not None:
                        function_state = function_states_by_index.get(index)
                    if index is not None:
                        function_states_by_index.pop(index, None)
                    tool_call_id = str(item.get("call_id") or state_key)
                    tool_name = str(item.get("name") or "")
                    arguments = item.get("arguments") or item.get("input")
                    if function_state is None:
                        yield types.events.ToolStart(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            provider_metadata=_provider_metadata_for_item(item),
                        )
                        if isinstance(arguments, str) and arguments:
                            yield types.events.ToolDelta(
                                tool_call_id=tool_call_id,
                                chunk=arguments,
                            )
                    elif (
                        isinstance(arguments, str)
                        and arguments
                        and not function_state["delta_emitted"]
                    ):
                        yield types.events.ToolDelta(
                            tool_call_id=function_state["tool_call_id"],
                            chunk=arguments,
                        )
                        tool_call_id = function_state["tool_call_id"]
                    else:
                        tool_call_id = function_state["tool_call_id"]
                    yield types.events.ToolEnd(
                        tool_call_id=tool_call_id,
                        tool_call=types.messages.DUMMY_TOOL_CALL,
                        provider_metadata=_provider_metadata_for_item(item),
                    )
                    continue

                if item_type in _BUILTIN_OUTPUT_TYPES:
                    builtin_state = (
                        builtin_states_by_item.pop(state_key, None)
                        if state_key
                        else None
                    )
                    if builtin_state is None and index is not None:
                        builtin_state = builtin_states_by_index.get(index)
                    if index is not None:
                        builtin_states_by_index.pop(index, None)
                    tool_call_id = _builtin_tool_call_id(item)
                    tool_name = _builtin_tool_name(item)
                    arguments = _builtin_tool_args(item)
                    if builtin_state is None:
                        yield types.events.BuiltinToolStart(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            provider_metadata=_provider_metadata_for_item(item),
                        )
                    else:
                        tool_call_id = builtin_state["tool_call_id"]
                        tool_name = builtin_state["tool_name"]
                    if arguments and (
                        builtin_state is None
                        or not builtin_state["delta_emitted"]
                    ):
                        yield types.events.BuiltinToolDelta(
                            tool_call_id=tool_call_id,
                            chunk=arguments,
                        )
                    yield types.events.BuiltinToolEnd(
                        tool_call_id=tool_call_id,
                        tool_call=types.messages.BuiltinToolCallPart(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            tool_args=arguments,
                            provider_metadata=_provider_metadata_for_item(item),
                        ),
                        provider_metadata=_provider_metadata_for_item(item),
                    )

                    if item_type == "image_generation_call":
                        result = item.get("result")
                        if isinstance(result, str) and result:
                            yield types.events.FileEvent(
                                block_id=str(item.get("id") or tool_call_id),
                                media_type=image_media_type,
                                data=result,
                                provider_metadata=_provider_metadata_for_item(
                                    item
                                ),
                            )

                    result_payload = _builtin_tool_result(item)
                    if result_payload is not None:
                        yield types.events.BuiltinToolResult(
                            tool_call_id=tool_call_id,
                            result=types.messages.BuiltinToolReturnPart(
                                tool_call_id=tool_call_id,
                                tool_name=tool_name,
                                result=result_payload,
                                provider_metadata=_provider_metadata_for_item(
                                    item
                                ),
                            ),
                        )
                    continue

        for block_id in list(text_blocks):
            yield types.events.TextEnd(block_id=block_id)
        for block_id in list(reasoning_blocks):
            yield types.events.ReasoningEnd(block_id=block_id)

        yield types.events.StreamEnd(
            usage=usage,
            provider_metadata=response_metadata,
        )
    except openai_sdk.OpenAIError as exc:
        raise errors.map_error(
            exc, provider=provider, model_id=model.id
        ) from exc


class OpenAIResponsesProtocol(base.ProviderProtocol[Any]):
    """OpenAI Responses API protocol."""

    def stream(
        self,
        client: openai.AsyncOpenAI,
        model: core.model.Model,
        messages: list[types.messages.Message],
        *,
        tools: Sequence[types.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
        provider: str,
    ) -> AsyncGenerator[types.events.Event]:
        return _stream_responses(
            client,
            model,
            messages,
            tools=tools,
            output_type=output_type,
            params=params,
            provider=provider,
        )


def default_protocol(provider: str) -> base.ProviderProtocol[Any]:
    """Return the best default OpenAI-compatible protocol for *provider*."""
    if provider == "openai":
        return OpenAIResponsesProtocol()
    return OpenAIChatCompletionsProtocol()
