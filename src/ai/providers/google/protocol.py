"""Google protocol — generateContent API.

Message/tool conversion and streaming via the official ``google-genai``
SDK. The Google provider owns the SDK client used by this protocol.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx

from ... import errors as ai_errors
from ... import types
from ...models.core import params as params_
from ...types import events
from .. import base, history_utils
from . import _sdk, errors

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    import google.genai as genai
    import pydantic

    from ...models import core

PROVIDER_NAME = "google"

_CODE_EXECUTION_TOOL = "code_execution"

# Models before Gemini 3 reject `thinking_level` and take a token budget
# instead; reasoning effort maps to these budgets there (-1 = automatic).
_THINKING_BUDGETS = {
    "minimal": 128,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}


def _provider_metadata(**values: Any) -> dict[str, Any]:
    """Namespace metadata as ``{"google": {...}}``."""
    return {PROVIDER_NAME: {**values}}


def _google_metadata(pm: dict[str, Any] | None) -> dict[str, Any]:
    """Read back the metadata written by :func:`_provider_metadata`."""
    meta = (pm or {}).get(PROVIDER_NAME)
    return meta if isinstance(meta, dict) else {}


def _signature_from_metadata(pm: dict[str, Any] | None) -> bytes | None:
    signature = _google_metadata(pm).get("thoughtSignature")
    if isinstance(signature, str) and signature:
        return base64.b64decode(signature)
    return None


# ---------------------------------------------------------------------------
# Message / tool conversion — internal types → Google wire format
# ---------------------------------------------------------------------------


def _split_tools(
    tools: Sequence[types.tools.Tool],
) -> tuple[list[types.tools.Tool], list[types.tools.Tool]]:
    """Split ``tools`` into host-executed and provider-executed declarations."""
    custom: list[types.tools.Tool] = []
    builtin: list[types.tools.Tool] = []
    for t in tools:
        if t.kind == "provider":
            builtin.append(t)
        else:
            custom.append(t)
    return custom, builtin


def _tools_to_google(
    custom: Sequence[types.tools.Tool],
    builtin: Sequence[types.tools.Tool],
) -> list[dict[str, Any]]:
    """Convert tools to Google wire format.

    Host-executed tools become one ``function_declarations`` entry;
    each provider-executed tool becomes its own entry keyed by the
    tool id without the ``google.`` prefix (``google_search`` etc.).
    """
    wire: list[dict[str, Any]] = []
    declarations: list[dict[str, Any]] = []
    for tool in custom:
        spec = tool.spec
        if spec is None:
            raise TypeError(f"function tool {tool.name!r} has no spec")
        declarations.append(
            {
                "name": tool.name,
                "description": spec.description or "",
                "parameters_json_schema": spec.params,
            }
        )
    if declarations:
        wire.append({"function_declarations": declarations})
    for tool in builtin:
        cfg = tool.tool_config
        tool_id = cfg.id if cfg is not None else None
        if cfg is None or tool_id is None or not tool_id.startswith("google."):
            raise ValueError(
                "GoogleModel does not support provider tool "
                f"{tool_id or tool.name!r}"
            )
        wire.append({tool_id.removeprefix("google."): {**cfg.args}})
    return wire


def _file_part_to_google(part: types.messages.FilePart) -> dict[str, Any]:
    """Convert a :class:`FilePart` to a Google content part.

    Downloadable URLs map to ``file_data``; everything else is sent
    inline as raw bytes.
    """
    media_type = (
        "image/jpeg" if part.media_type == "image/*" else part.media_type
    )
    if isinstance(part.data, str) and types.media.is_downloadable_url(
        part.data
    ):
        return {"file_data": {"file_uri": part.data, "mime_type": media_type}}
    return {
        "inline_data": {
            "data": base64.b64decode(types.media.data_to_base64(part.data)),
            "mime_type": media_type,
        }
    }


def _tool_result_to_google(
    part: types.messages.ToolResultPart,
) -> dict[str, Any]:
    """Convert a tool result to a ``function_response`` response object.

    Google requires a JSON object: dicts pass through, everything else
    is wrapped under ``"output"`` (or ``"error"`` for error results).
    """
    value = part.get_model_input()
    if isinstance(value, types.messages.ContentOutput):
        texts: list[str] = []
        for item in value.value:
            if isinstance(item, types.messages.FilePart):
                raise ValueError(
                    "Google does not support file parts in tool results"
                )
            texts.append(item.text)
        value = "".join(texts)
    if part.is_error:
        return {"error": value if value is not None else ""}
    if isinstance(value, dict):
        return value
    return {"output": value if value is not None else ""}


def _messages_to_google(
    messages: list[types.messages.Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert internal messages to Google API format.

    Returns ``(system_instruction, contents)``.  The system prompt is
    extracted separately because the Google API takes it as a config
    parameter.
    """
    system_instruction: str | None = None
    result: list[dict[str, Any]] = []

    for msg in history_utils.repair(messages):
        match msg.role:
            case "system":
                system_instruction = "".join(
                    p.text
                    for p in msg.parts
                    if isinstance(p, types.messages.TextPart)
                )
            case "assistant":
                parts: list[dict[str, Any]] = []
                for part in msg.parts:
                    match part:
                        case types.messages.ReasoningPart(
                            text=text,
                            provider_metadata=provider_metadata,
                        ):
                            # Thought summaries are not sent back; only
                            # signed thoughts round-trip.
                            signature = _signature_from_metadata(
                                provider_metadata
                            )
                            if signature:
                                parts.append(
                                    {
                                        "text": text,
                                        "thought": True,
                                        "thought_signature": signature,
                                    }
                                )
                        case types.messages.TextPart(
                            text=text,
                            provider_metadata=provider_metadata,
                        ):
                            entry: dict[str, Any] = {"text": text}
                            signature = _signature_from_metadata(
                                provider_metadata
                            )
                            if signature:
                                entry["thought_signature"] = signature
                            parts.append(entry)
                        case types.messages.ToolCallPart():
                            call_entry: dict[str, Any] = {
                                "function_call": {
                                    "id": part.tool_call_id,
                                    "name": part.tool_name,
                                    "args": json.loads(part.tool_args)
                                    if part.tool_args
                                    else {},
                                }
                            }
                            signature = _signature_from_metadata(
                                part.provider_metadata
                            )
                            if signature:
                                call_entry["thought_signature"] = signature
                            parts.append(call_entry)
                        case types.messages.BuiltinToolCallPart() if (
                            part.tool_name == _CODE_EXECUTION_TOOL
                        ):
                            exec_entry: dict[str, Any] = {
                                "executable_code": json.loads(part.tool_args)
                                if part.tool_args
                                else {},
                            }
                            signature = _signature_from_metadata(
                                part.provider_metadata
                            )
                            if signature:
                                exec_entry["thought_signature"] = signature
                            parts.append(exec_entry)
                        case types.messages.BuiltinToolReturnPart() if (
                            part.tool_name == _CODE_EXECUTION_TOOL
                        ):
                            result_entry: dict[str, Any] = {
                                "code_execution_result": part.result or {},
                            }
                            signature = _signature_from_metadata(
                                part.provider_metadata
                            )
                            if signature:
                                result_entry["thought_signature"] = signature
                            parts.append(result_entry)
                if parts:
                    result.append({"role": "model", "parts": parts})

            case "tool":
                tool_parts: list[dict[str, Any]] = []
                for part in msg.parts:
                    if isinstance(part, types.messages.ToolResultPart):
                        tool_parts.append(
                            {
                                "function_response": {
                                    "id": part.tool_call_id,
                                    "name": part.tool_name,
                                    "response": _tool_result_to_google(part),
                                }
                            }
                        )
                if tool_parts:
                    result.append({"role": "user", "parts": tool_parts})

            case "user":
                user_parts: list[dict[str, Any]] = []
                for p in msg.parts:
                    match p:
                        case types.messages.TextPart(text=text):
                            user_parts.append({"text": text})
                        case types.messages.FilePart():
                            user_parts.append(_file_part_to_google(p))
                if user_parts:
                    result.append({"role": "user", "parts": user_parts})

    return system_instruction, _merge_consecutive_roles(result)


def _merge_consecutive_roles(
    contents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge consecutive contents that share the same role."""
    if not contents:
        return contents

    merged: list[dict[str, Any]] = [contents[0]]
    for content in contents[1:]:
        if content["role"] == merged[-1]["role"]:
            merged[-1]["parts"] = merged[-1]["parts"] + content["parts"]
        else:
            merged.append(content)
    return merged


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


def _http_options(config: dict[str, Any]) -> dict[str, Any]:
    http_options = config.get("http_options")
    if not isinstance(http_options, dict):
        http_options = {}
        config["http_options"] = http_options
    return http_options


def _google_tool_choice(
    tool_choice: params_.ToolChoiceMode
    | params_.ToolRef
    | params_.ToolSelection,
) -> dict[str, Any]:
    if isinstance(tool_choice, params_.ToolChoiceMode):
        match tool_choice:
            case params_.ToolChoiceMode.AUTO:
                return {"mode": "AUTO"}
            case params_.ToolChoiceMode.REQUIRED:
                return {"mode": "ANY"}
            case params_.ToolChoiceMode.NONE:
                return {"mode": "NONE"}
    if isinstance(tool_choice, params_.ToolRef):
        return {"mode": "ANY", "allowed_function_names": [str(tool_choice)]}
    if tool_choice.mode in {
        params_.ToolChoiceMode.AUTO,
        params_.ToolChoiceMode.REQUIRED,
    }:
        return {
            # VALIDATED lets the model choose between the allowed tools
            # and plain text; ANY forces a tool call.
            "mode": "ANY"
            if tool_choice.mode == params_.ToolChoiceMode.REQUIRED
            else "VALIDATED",
            "allowed_function_names": sorted(str(t) for t in tool_choice.tools),
        }
    raise ValueError("Google does not support excluded tool subsets")


def _apply_sampling(
    config: dict[str, Any],
    request_params: params_.InferenceRequestParams,
) -> None:
    sampling = request_params.sampling
    if isinstance(sampling, params_.ModelProviderDefault):
        return
    for sampler in sampling.values():
        match sampler:
            case params_.TemperatureSamplerParams(temperature=temperature):
                if _not_default(temperature):
                    config["temperature"] = temperature
            case params_.TopPSamplerParams(top_p=top_p):
                if _not_default(top_p):
                    config["top_p"] = top_p
            case params_.TopKSamplerParams(top_k=top_k):
                if _not_default(top_k):
                    config["top_k"] = top_k
            case params_.SeedSamplerParams(seed=seed):
                if _not_default(seed):
                    seed_value = _seed_value(seed)
                    if seed_value is not None:
                        config["seed"] = seed_value
            case params_.MinPSamplerParams(min_p=min_p):
                if _not_default(min_p) and min_p is not None:
                    raise ValueError("Google does not support min_p")
            case params_.RepetitionPenaltyParams() as repetition:
                if (
                    _not_default(repetition.frequency_penalty)
                    and repetition.frequency_penalty is not None
                ):
                    config["frequency_penalty"] = repetition.frequency_penalty
                if (
                    _not_default(repetition.presence_penalty)
                    and repetition.presence_penalty is not None
                ):
                    config["presence_penalty"] = repetition.presence_penalty
                unsupported = {
                    "repetition_penalty": repetition.repetition_penalty,
                    "consideration_window": repetition.consideration_window,
                }
                for key, value in unsupported.items():
                    if _not_default(value) and value is not None:
                        raise ValueError(f"Google does not support {key}")


def _apply_google_params(
    config: dict[str, Any],
    request_params: params_.InferenceRequestParams,
    *,
    model_id: str,
    provider: str,
) -> None:
    _ = provider
    _apply_sampling(config, request_params)

    reasoning = request_params.reasoning
    output = request_params.output
    summary = params_.DEFAULT if output is None else output.reasoning_summary
    thinking: dict[str, Any] = {}
    if not isinstance(reasoning, params_.ModelProviderDefault) and _not_default(
        reasoning.effort
    ):
        if reasoning.effort is None:
            thinking["thinking_budget"] = 0
        elif model_id.startswith("gemini-3"):
            thinking["thinking_level"] = reasoning.effort
        else:
            thinking["thinking_budget"] = _THINKING_BUDGETS.get(
                cast("str", reasoning.effort), -1
            )
    if _not_default(summary):
        # `reasoning_summary` controls whether thought summaries are
        # surfaced in the response; it never turns thinking off (use
        # `reasoning.effort=None` for that).
        thinking["include_thoughts"] = summary is not None
    if thinking:
        config["thinking_config"] = thinking

    if request_params.tool_calling is not None:
        tool_calling = request_params.tool_calling
        if (
            _not_default(tool_calling.max_tool_calls)
            and tool_calling.max_tool_calls is not None
        ):
            raise ValueError("Google does not support max_tool_calls")
        if _not_default(tool_calling.parallel_tool_calls):
            raise ValueError(
                "Google does not support configuring parallel tool calls"
            )
        config["tool_config"] = {
            "function_calling_config": _google_tool_choice(
                tool_calling.tool_choice
            )
        }

    if request_params.provider_service is not None and _not_default(
        request_params.provider_service.service_tier
    ):
        config["service_tier"] = request_params.provider_service.service_tier

    if request_params.metadata is not None:
        raise ValueError("Google does not support request metadata")
    if request_params.safety_identifier is not None:
        raise ValueError("Google does not support safety identifiers")
    if request_params.context_management is not None:
        raise ValueError("Google does not support context management")

    if output is not None:
        if output.max_tokens is not None:
            config["max_output_tokens"] = output.max_tokens
        if output.include is not None:
            raise ValueError("Google does not support output include")
        if (
            _not_default(output.text_verbosity)
            and output.text_verbosity is not None
        ):
            raise ValueError("Google does not support text verbosity")

    if request_params.cache is not None:
        cache = request_params.cache
        if _not_default(cache.retention):
            raise ValueError("Google does not support cache retention")
        if cache.key is not None:
            config["cached_content"] = cache.key

    if request_params.extra_headers is not None:
        headers = {
            key: value
            for key, value in request_params.extra_headers.items()
            if not isinstance(value, params_.Unset)
        }
        if headers:
            _http_options(config)["headers"] = headers
    if request_params.extra_query is not None:
        raise ValueError("Google does not support extra query arguments")
    if request_params.extra_body is not None:
        _http_options(config)["extra_body"] = dict(request_params.extra_body)


# ---------------------------------------------------------------------------
# Public protocol function
# ---------------------------------------------------------------------------


async def stream(
    sdk_client: genai.Client,
    model: core.model.Model,
    messages: list[types.messages.Message],
    *,
    tools: Sequence[types.tools.Tool] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    params: params_.InferenceRequestParams | None = None,
    provider: str,
) -> AsyncGenerator[events.Event]:
    """Stream through the Google generateContent protocol using *sdk_client*.

    Yields :class:`~ai.types.events.Event` objects as the response streams in.
    Pure delta emitter — the :class:`~ai.models.Stream` wrapper aggregates
    parts into the final :class:`~ai.types.messages.Message`.
    """
    genai_errors = _sdk.import_errors(provider=provider)
    config: dict[str, Any] = {}
    if params is not None:
        if not isinstance(params, params_.InferenceRequestParams):
            raise TypeError(
                "google stream params must be InferenceRequestParams"
            )
        _apply_google_params(
            config, params, model_id=model.id, provider=provider
        )
    system_instruction, contents = _messages_to_google(messages)
    if system_instruction:
        config["system_instruction"] = system_instruction

    custom_tools, builtin_tools = _split_tools(tools or ())
    wire_tools = _tools_to_google(custom_tools, builtin_tools)
    if wire_tools:
        config["tools"] = wire_tools

    if output_type is not None:
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = output_type.model_json_schema()

    # Google streams complete parts per chunk without explicit block
    # boundaries; like the OpenAI chat-completions protocol we track one
    # text and one reasoning block with fixed ids.
    text_started = False
    reasoning_started = False
    text_signature: str | None = None
    reasoning_signature: str | None = None
    file_index = 0
    last_exec_tool_id = ""

    try:
        sdk_stream = await sdk_client.aio.models.generate_content_stream(
            model=model.id,
            contents=contents,
            config=cast("Any", config or None),
        )
        yield events.StreamStart()

        usage_metadata: Any = None
        finish_reason: str | None = None
        async for chunk in sdk_stream:
            feedback = chunk.prompt_feedback
            if feedback is not None and feedback.block_reason:
                raise ai_errors.ProviderResponseError(
                    "Google blocked the prompt: "
                    f"{feedback.block_reason_message or feedback.block_reason}",
                    provider=provider,
                )
            if chunk.usage_metadata is not None:
                usage_metadata = chunk.usage_metadata
            candidate = chunk.candidates[0] if chunk.candidates else None
            if candidate is not None and candidate.finish_reason is not None:
                finish_reason = str(candidate.finish_reason.value)
            content = candidate.content if candidate is not None else None
            for part in (content.parts if content is not None else None) or []:
                signature = (
                    base64.b64encode(part.thought_signature).decode("ascii")
                    if part.thought_signature
                    else None
                )
                if part.text is not None:
                    if part.thought:
                        if not reasoning_started:
                            reasoning_started = True
                            yield events.ReasoningStart(block_id="reasoning")
                        yield events.ReasoningDelta(
                            chunk=part.text, block_id="reasoning"
                        )
                        if signature is not None:
                            reasoning_signature = signature
                    else:
                        if reasoning_started:
                            reasoning_started = False
                            yield events.ReasoningEnd(
                                block_id="reasoning",
                                provider_metadata=(
                                    _provider_metadata(
                                        thoughtSignature=reasoning_signature
                                    )
                                    if reasoning_signature is not None
                                    else None
                                ),
                            )
                        if not text_started:
                            text_started = True
                            yield events.TextStart(block_id="text")
                        yield events.TextDelta(chunk=part.text, block_id="text")
                        if signature is not None:
                            text_signature = signature
                elif part.function_call is not None:
                    fc = part.function_call
                    tool_id = fc.id or types.messages.generate_id("call")
                    yield events.ToolStart(
                        tool_call_id=tool_id, tool_name=fc.name or ""
                    )
                    yield events.ToolDelta(
                        chunk=json.dumps(fc.args or {}, separators=(",", ":")),
                        tool_call_id=tool_id,
                    )
                    yield events.ToolEnd(
                        tool_call_id=tool_id,
                        tool_call=types.messages.DUMMY_TOOL_CALL,
                        provider_metadata=(
                            _provider_metadata(thoughtSignature=signature)
                            if signature is not None
                            else None
                        ),
                    )
                elif part.executable_code is not None:
                    tool_id = types.messages.generate_id("call")
                    last_exec_tool_id = tool_id
                    exec_args = json.dumps(
                        part.executable_code.model_dump(
                            mode="json", exclude_none=True
                        ),
                        separators=(",", ":"),
                    )
                    exec_metadata = (
                        _provider_metadata(thoughtSignature=signature)
                        if signature is not None
                        else _provider_metadata()
                    )
                    yield events.BuiltinToolStart(
                        tool_call_id=tool_id,
                        tool_name=_CODE_EXECUTION_TOOL,
                        provider_metadata=exec_metadata,
                    )
                    yield events.BuiltinToolDelta(
                        chunk=exec_args, tool_call_id=tool_id
                    )
                    yield events.BuiltinToolEnd(
                        tool_call_id=tool_id,
                        tool_call=types.messages.BuiltinToolCallPart(
                            tool_call_id=tool_id,
                            tool_name=_CODE_EXECUTION_TOOL,
                            provider_metadata=exec_metadata,
                        ),
                        provider_metadata=exec_metadata,
                    )
                elif part.code_execution_result is not None:
                    result_payload = part.code_execution_result.model_dump(
                        mode="json", exclude_none=True
                    )
                    yield events.BuiltinToolResult(
                        tool_call_id=last_exec_tool_id,
                        result=types.messages.BuiltinToolReturnPart(
                            tool_call_id=last_exec_tool_id,
                            tool_name=_CODE_EXECUTION_TOOL,
                            result=result_payload,
                            is_error=result_payload.get("outcome")
                            not in (None, "OUTCOME_OK"),
                            provider_metadata=(
                                _provider_metadata(thoughtSignature=signature)
                                if signature is not None
                                else _provider_metadata()
                            ),
                        ),
                    )
                elif part.inline_data is not None:
                    yield events.FileEvent(
                        block_id=f"file_{file_index}",
                        media_type=part.inline_data.mime_type or "",
                        data=part.inline_data.data or b"",
                    )
                    file_index += 1

        if reasoning_started:
            yield events.ReasoningEnd(
                block_id="reasoning",
                provider_metadata=(
                    _provider_metadata(thoughtSignature=reasoning_signature)
                    if reasoning_signature is not None
                    else None
                ),
            )
        if text_started:
            yield events.TextEnd(
                block_id="text",
                provider_metadata=(
                    _provider_metadata(thoughtSignature=text_signature)
                    if text_signature is not None
                    else None
                ),
            )

        if usage_metadata is not None:
            thoughts = usage_metadata.thoughts_token_count
            usage = types.usage.Usage(
                input_tokens=usage_metadata.prompt_token_count or 0,
                output_tokens=(
                    (usage_metadata.candidates_token_count or 0)
                    + (thoughts or 0)
                ),
                reasoning_tokens=thoughts,
                cache_read_tokens=usage_metadata.cached_content_token_count,
                raw=usage_metadata.model_dump(mode="json", exclude_none=True)
                or None,
            )
        else:
            usage = types.usage.Usage()
        yield events.StreamEnd(
            usage=usage,
            provider_metadata=(
                _provider_metadata(finishReason=finish_reason)
                if finish_reason is not None
                else None
            ),
        )
    except genai_errors.APIError as exc:
        raise errors.map_error(
            exc,
            provider=provider,
            model_id=model.id,
        ) from exc
    except httpx.HTTPError as exc:
        raise errors.map_httpx_error(exc, provider=provider) from exc


class GoogleGenerateContentProtocol(base.ProviderProtocol[Any]):
    """Google generateContent API protocol."""

    protocol_class_id: Literal["google_generate_content"] = (
        "google_generate_content"
    )

    def stream(
        self,
        client: genai.Client,
        model: core.model.Model,
        messages: list[types.messages.Message],
        *,
        tools: Sequence[types.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
        provider: str,
    ) -> AsyncGenerator[events.Event]:
        return stream(
            client,
            model,
            messages,
            tools=tools,
            output_type=output_type,
            params=params,
            provider=provider,
        )
