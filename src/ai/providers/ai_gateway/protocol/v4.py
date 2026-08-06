"""AI Gateway v4 wire protocol.

Owns everything the v4 spec defines: the prompt encoding (tagged file-data
unions with URL passthrough, the collapsed ``file`` tool-result content),
the standardized ``reasoning`` effort field, and the stream-part
vocabulary.  Version-stable pieces come from :mod:`._shared`.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal

import httpx

from .... import models, ops, types
from ... import base, history_utils
from .. import client as gateway_client
from .. import errors
from . import _shared

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    import pydantic

SPEC_VERSION = "4"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Message list → v4 prompt
# ---------------------------------------------------------------------------


def _file_data_to_wire(data: str | bytes) -> dict[str, Any]:
    """Convert :class:`FilePart` data to the v4 tagged file-data union.

    Downloadable URLs pass through as ``url`` — the gateway accepts all
    URLs natively, so nothing is downloaded client-side.  Everything else
    (bytes, base64 text, data URLs) becomes inline ``data``.
    """
    if isinstance(data, str) and types.media.is_downloadable_url(data):
        return {"type": "url", "url": data}
    return {"type": "data", "data": types.media.data_to_base64(data)}


def _file_part_to_wire(part: types.messages.FilePart) -> dict[str, Any]:
    """Convert a :class:`FilePart` to a v4 ``file`` content part.

    The same shape serves prompt file parts and inline tool-result
    content: v4 collapsed the v3 ``image-data``/``file-data``/... variants
    into one ``file`` type with tagged data.
    """
    entry: dict[str, Any] = {
        "type": "file",
        "mediaType": part.media_type,
        "data": _file_data_to_wire(part.data),
    }
    if part.filename is not None:
        entry["filename"] = part.filename
    return entry


def _tool_result_output(
    part: types.messages.ToolResultPart,
) -> dict[str, Any]:
    """Convert a tool result to its v4 ``output`` wire form.

    The v4 protocol carries a tagged output union.  A :class:`ContentOutput`
    becomes ``content``; an error result becomes ``error-text`` (for a
    ``str``) or ``error-json``; otherwise ``text`` (for a ``str``) or
    ``json``.  The text-vs-json call is made here, at the wire boundary.
    """
    value = part.get_model_input()
    if isinstance(value, types.messages.ContentOutput):
        parts: list[dict[str, Any]] = []
        for item in value.value:
            if isinstance(item, types.messages.FilePart):
                parts.append(_file_part_to_wire(item))
            else:
                parts.append({"type": "text", "text": item.text})
        return {"type": "content", "value": parts}
    if part.is_error:
        if value is None or isinstance(value, str):
            return {"type": "error-text", "value": value or ""}
        return {"type": "error-json", "value": value}
    if value is None or isinstance(value, str):
        return {"type": "text", "value": value or ""}
    return {"type": "json", "value": value}


async def _messages_to_prompt(
    messages: list[types.messages.Message],
) -> list[dict[str, Any]]:
    """Convert ``Message`` list to the v4 prompt wire format."""
    result: list[dict[str, Any]] = []

    for msg in history_utils.repair(messages):
        match msg.role:
            case "system":
                text = "".join(
                    p.text
                    for p in msg.parts
                    if isinstance(p, types.messages.TextPart)
                )
                result.append({"role": "system", "content": text})

            case "user":
                content: list[dict[str, Any]] = []
                for p in msg.parts:
                    if isinstance(p, types.messages.TextPart):
                        content.append({"type": "text", "text": p.text})
                    elif isinstance(p, types.messages.FilePart):
                        content.append(_file_part_to_wire(p))
                result.append({"role": "user", "content": content})

            case "assistant":
                assistant_content: list[dict[str, Any]] = []
                for part in msg.parts:
                    match part:
                        case types.messages.ReasoningPart(
                            text=text, provider_metadata=pm
                        ):
                            reasoning_entry: dict[str, Any] = {
                                "type": "reasoning",
                                "text": text,
                            }
                            # Replay the provider's reasoning metadata (e.g.
                            # the thinking-block signature) verbatim. Without
                            # it the provider drops the block and the model
                            # loses access to its prior reasoning. v4 mirrors
                            # inbound ``providerMetadata`` to outbound
                            # ``providerOptions``.
                            if pm:
                                reasoning_entry["providerOptions"] = pm
                            assistant_content.append(reasoning_entry)
                        case types.messages.TextPart(text=text):
                            assistant_content.append(
                                {"type": "text", "text": text}
                            )
                        case types.messages.ToolCallPart() as tp:
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
                        case types.messages.BuiltinToolCallPart() as btp:
                            btp_input: Any = (
                                json.loads(btp.tool_args)
                                if btp.tool_args
                                else {}
                            )
                            assistant_content.append(
                                {
                                    "type": "tool-call",
                                    "toolCallId": btp.tool_call_id,
                                    "toolName": btp.tool_name,
                                    "input": btp_input,
                                    "providerExecuted": True,
                                }
                            )
                        case types.messages.BuiltinToolReturnPart() as brp:
                            assistant_content.append(
                                {
                                    "type": "tool-result",
                                    "toolCallId": brp.tool_call_id,
                                    "toolName": brp.tool_name,
                                    "output": {
                                        "type": "json",
                                        "value": brp.result,
                                    },
                                    "providerExecuted": True,
                                }
                            )
                result.append(
                    {"role": "assistant", "content": assistant_content}
                )

            case "tool":
                tool_results: list[dict[str, Any]] = []
                for part in msg.parts:
                    if isinstance(part, types.messages.ToolResultPart):
                        output = _tool_result_output(part)
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


# ---------------------------------------------------------------------------
# Reasoning request options
# ---------------------------------------------------------------------------


def _apply_reasoning(
    body: dict[str, Any],
    request_params: models.InferenceRequestParams,
    *,
    provider: str | None,
) -> None:
    reasoning = request_params.reasoning
    output = request_params.output
    effort: str | models.ModelProviderDefault | None = models.DEFAULT
    if not isinstance(reasoning, models.ModelProviderDefault):
        effort = reasoning.effort
    summary = models.DEFAULT if output is None else output.reasoning_summary
    if isinstance(effort, models.ModelProviderDefault) and isinstance(
        summary, models.ModelProviderDefault
    ):
        return
    if provider == "openai":
        options = _shared.body_provider_options(body, "openai")
        if not isinstance(effort, models.ModelProviderDefault):
            options["reasoningEffort"] = effort
        if not isinstance(summary, models.ModelProviderDefault):
            options["reasoningSummary"] = summary
        return
    if provider == "anthropic":
        options = _shared.body_provider_options(body, "anthropic")
        if not isinstance(effort, models.ModelProviderDefault):
            if effort is None:
                options["thinking"] = {"type": "disabled"}
            else:
                options["effort"] = effort
                # The gateway only turns thinking on when a `thinking`
                # block is present; `effort` alone is a no-op upstream.
                thinking = dict(options.get("thinking") or {})
                thinking.setdefault("type", "adaptive")
                options["thinking"] = thinking
        if not isinstance(summary, models.ModelProviderDefault):
            # `reasoning_summary` only controls whether the reasoning summary
            # is surfaced; it never turns thinking off (use
            # `reasoning.effort=None` for that). `None` maps to
            # `display="omitted"` -- think, but don't emit a summary -- and is
            # ignored when thinking is already disabled.
            thinking = dict(options.get("thinking") or {})
            if thinking.get("type") != "disabled":
                thinking.setdefault("type", "adaptive")
                thinking["display"] = "omitted" if summary is None else summary
                options["thinking"] = thinking
        return
    # v4 standardizes `reasoning` as a plain effort level; there is no
    # summary field for providers without a dedicated branch above.
    if not isinstance(summary, models.ModelProviderDefault):
        raise ValueError(
            "AI Gateway v4 reasoning summary requires an OpenAI or "
            "Anthropic model"
        )
    if not isinstance(effort, models.ModelProviderDefault):
        body["reasoning"] = "none" if effort is None else effort


# ---------------------------------------------------------------------------
# v4 stream parts → public Event
# ---------------------------------------------------------------------------


def _parse_stream_part(
    data: dict[str, Any],
    streamed_tool_ids: set[str],
    provider_executed_ids: set[str] | None = None,
    response_metadata: dict[str, Any] | None = None,
) -> list[types.events.Event]:
    """Convert a ``LanguageModelV4StreamPart`` to public events.

    ``response_metadata`` accumulates the ``response-metadata`` part so the
    final ``finish`` part can surface it on :class:`StreamEnd`.
    """
    if provider_executed_ids is None:
        provider_executed_ids = set()
    if response_metadata is None:
        response_metadata = {}
    match data.get("type", ""):
        case "text-start":
            return [types.events.TextStart(block_id=data.get("id", "text"))]

        case "text-delta":
            return [
                types.events.TextDelta(
                    block_id=data.get("id", "text"),
                    chunk=data.get("delta", ""),
                )
            ]

        case "text-end":
            return [types.events.TextEnd(block_id=data.get("id", "text"))]

        case "reasoning-start":
            # Metadata on -start is gateway routing info (generationId),
            # not the provider's reasoning metadata; don't replay it.
            return [
                types.events.ReasoningStart(
                    block_id=data.get("id", "reasoning")
                )
            ]

        case "reasoning-delta":
            return [
                types.events.ReasoningDelta(
                    block_id=data.get("id", "reasoning"),
                    chunk=data.get("delta", ""),
                    provider_metadata=data.get("providerMetadata"),
                )
            ]

        case "reasoning-end":
            return [
                types.events.ReasoningEnd(
                    block_id=data.get("id", "reasoning"),
                    provider_metadata=data.get("providerMetadata"),
                )
            ]

        case "tool-input-start":
            tcid = data.get("id", "")
            streamed_tool_ids.add(tcid)
            if _shared.is_provider_executed(data):
                provider_executed_ids.add(tcid)
                return [
                    types.events.BuiltinToolStart(
                        tool_call_id=tcid,
                        tool_name=data.get("toolName", ""),
                    )
                ]
            return [
                types.events.ToolStart(
                    tool_call_id=tcid,
                    tool_name=data.get("toolName", ""),
                )
            ]

        case "tool-input-delta":
            tcid = data.get("id", "")
            if tcid in provider_executed_ids:
                return [
                    types.events.BuiltinToolDelta(
                        tool_call_id=tcid,
                        chunk=data.get("delta", ""),
                    )
                ]
            return [
                types.events.ToolDelta(
                    tool_call_id=tcid,
                    chunk=data.get("delta", ""),
                )
            ]

        case "tool-input-end":
            tcid = data.get("id", "")
            if tcid in provider_executed_ids:
                return [
                    types.events.BuiltinToolEnd(
                        tool_call_id=tcid,
                        tool_call=types.messages.BuiltinToolCallPart(
                            tool_call_id=tcid,
                            tool_name="",
                        ),
                    )
                ]
            return [
                types.events.ToolEnd(
                    tool_call_id=tcid,
                    tool_call=types.messages.DUMMY_TOOL_CALL,
                )
            ]

        case "tool-call":
            return _shared.expand_tool_call(
                data, streamed_tool_ids, provider_executed_ids
            )

        case "tool-result":
            tcid = data.get("toolCallId", "")
            tool_name = data.get("toolName", "")
            output = data.get("output") or data.get("result")
            is_error = bool(data.get("isError"))
            if _shared.is_provider_executed(data) or (
                tcid in provider_executed_ids
            ):
                provider_executed_ids.add(tcid)
                return [
                    types.events.BuiltinToolResult(
                        tool_call_id=tcid,
                        result=types.messages.BuiltinToolReturnPart(
                            tool_call_id=tcid,
                            tool_name=tool_name,
                            result=output,
                            is_error=is_error,
                        ),
                    )
                ]
            return []

        case "file":
            # v4 wraps generated file data in a tagged union.
            file_data = data.get("data", "")
            if isinstance(file_data, dict):
                if file_data.get("type") == "url":
                    file_data = file_data.get("url", "")
                else:
                    file_data = file_data.get("data", "")
            return [
                types.events.FileEvent(
                    block_id=data.get("id", ""),
                    media_type=data.get(
                        "mediaType", "application/octet-stream"
                    ),
                    data=file_data,
                )
            ]

        case "stream-start":
            for warning in data.get("warnings") or []:
                logger.warning("AI Gateway warning: %s", warning)
            return []

        case "response-metadata":
            response_metadata.update(data)
            return []

        case "error":
            raise gateway_client.errors.GatewayResponseError(
                message=f"Gateway stream error: {data.get('error')}",
            )

        case "finish":
            usage_data = data.get("usage")
            usage = _shared.parse_usage(usage_data) if usage_data else None
            finish_reason, finish_metadata = _shared.parse_finish_reason(
                data.get("finishReason")
            )
            return [
                types.events.StreamEnd(
                    usage=usage,
                    finish_reason=finish_reason,
                    provider_metadata=finish_metadata,
                    response_id=response_metadata.get("id"),
                    response_model=response_metadata.get("modelId"),
                )
            ]

        case _:
            return []


# ---------------------------------------------------------------------------
# Streaming orchestration
# ---------------------------------------------------------------------------


async def stream(
    gateway: gateway_client.GatewayClient,
    model: models.Model,
    messages: list[types.messages.Message],
    *,
    tools: Sequence[types.tools.Tool] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    params: models.InferenceRequestParams | None = None,
) -> AsyncGenerator[types.events.Event]:
    """Stream an LLM response through the AI Gateway v4 protocol."""
    body, extra_headers, extra_query = _shared.request_options(
        params,
        model_id=model.id,
    )
    if params is not None:
        _apply_reasoning(
            body,
            params,
            provider=_shared.provider_from_model_id(model.id),
        )
        if params.extra_body is not None:
            _shared.merge_extra_body(body, params.extra_body)
    body["prompt"] = await _messages_to_prompt(messages)
    if tools:
        body["tools"] = [_shared.tool_to_wire(tool) for tool in tools]
    response_format = _shared.response_format(output_type)
    if response_format is not None:
        body["responseFormat"] = response_format

    try:
        async with gateway.stream(
            "language-model",
            body,
            model=model,
            streaming=True,
            headers=extra_headers,
            query=extra_query,
            spec_version=SPEC_VERSION,
        ) as response:
            yield types.events.StreamStart()
            streamed_tool_ids: set[str] = set()
            provider_executed_ids: set[str] = set()
            response_metadata: dict[str, Any] = {}
            async for data in gateway.iter_sse(response):
                for event in _parse_stream_part(
                    data,
                    streamed_tool_ids,
                    provider_executed_ids,
                    response_metadata,
                ):
                    yield event
    except gateway_client.errors.GatewayError as exc:
        raise errors.map_error(exc) from exc
    except httpx.TimeoutException as exc:
        timeout_error = gateway_client.errors.GatewayTimeoutError()
        raise errors.map_error(timeout_error) from exc
    except Exception as exc:
        response_error = gateway_client.errors.GatewayResponseError(
            message=f"Unexpected error during streaming: {exc}",
        )
        raise errors.map_error(response_error) from exc


class GatewayV4Protocol(base.ProviderProtocol[gateway_client.GatewayClient]):
    """AI Gateway v4 wire protocol."""

    protocol_class_id: Literal["gateway_v4"] = "gateway_v4"

    def stream(
        self,
        client: gateway_client.GatewayClient,
        model: models.Model,
        messages: list[types.messages.Message],
        *,
        tools: Sequence[types.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: models.InferenceRequestParams | None = None,
        provider: str,
    ) -> AsyncGenerator[types.events.Event]:
        _ = provider
        return stream(
            client,
            model,
            messages,
            tools=tools,
            output_type=output_type,
            params=params,
        )

    async def generate_image(
        self,
        client: gateway_client.GatewayClient,
        model: models.Model,
        messages: list[types.messages.Message],
        *,
        params: ops.images.ImageParams,
        provider: str,
    ) -> ops.items.Item[list[types.messages.FilePart]]:
        _ = provider
        return await _shared.generate_image(
            client, model, messages, params=params, spec_version=SPEC_VERSION
        )

    async def generate_video(
        self,
        client: gateway_client.GatewayClient,
        model: models.Model,
        messages: list[types.messages.Message],
        *,
        params: ops.videos.VideoParams,
        provider: str,
    ) -> ops.items.Item[list[types.messages.FilePart]]:
        _ = provider
        return await _shared.generate_video(
            client, model, messages, params=params, spec_version=SPEC_VERSION
        )

    async def generate_audio(
        self,
        client: gateway_client.GatewayClient,
        model: models.Model,
        messages: list[types.messages.Message],
        *,
        params: ops.audio.AudioParams,
        provider: str,
    ) -> ops.items.Item[list[types.messages.FilePart]]:
        _ = provider
        return await _shared.generate_audio(
            client, model, messages, params=params, spec_version=SPEC_VERSION
        )
