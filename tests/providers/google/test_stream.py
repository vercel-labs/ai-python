"""Tests for Google stream parsing.

The adapter consumes ``GenerateContentResponse`` chunks and emits
framework events. Drained through :class:`models.Stream` to also
exercise event aggregation in ``core.api``.
"""

from __future__ import annotations

import base64
from typing import Any, cast

import pytest
from google.genai import types as genai_types

import ai
from ai import models
from ai.providers.google import protocol
from ai.types import events, messages

from .conftest import FakeGoogleClient, chunk

_MODEL = ai.Model(id="gemini-2.5-flash", provider=ai.get_provider("google"))


async def _drain(chunks: list[Any]) -> models.Stream:
    fake = FakeGoogleClient(chunks=chunks)
    s = models.Stream(
        protocol.stream(
            cast("Any", fake),
            _MODEL,
            [ai.user_message("Hi")],
            provider="google",
        )
    )
    async for _ in s:
        pass
    return s


async def test_text_deltas_aggregate_into_one_block() -> None:
    s = await _drain(
        [
            chunk([genai_types.Part(text="Hello")]),
            chunk([genai_types.Part(text=" world")]),
        ]
    )

    (part,) = s.message.parts
    assert isinstance(part, messages.TextPart)
    assert part.text == "Hello world"


async def test_thought_parts_emit_reasoning_with_signature() -> None:
    s = await _drain(
        [
            chunk(
                [
                    genai_types.Part(
                        text="thinking...",
                        thought=True,
                        thought_signature=b"sig-bytes",
                    ),
                    genai_types.Part(text="Answer"),
                ]
            ),
        ]
    )

    reasoning, text = s.message.parts
    assert isinstance(reasoning, messages.ReasoningPart)
    assert reasoning.text == "thinking..."
    assert reasoning.provider_metadata == {
        "google": {"thoughtSignature": base64.b64encode(b"sig-bytes").decode()}
    }
    assert isinstance(text, messages.TextPart)
    assert text.text == "Answer"


async def test_function_call_emits_tool_events() -> None:
    fake = FakeGoogleClient(
        chunks=[
            chunk(
                [
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            id="fc_1",
                            name="get_weather",
                            args={"city": "Tokyo"},
                        )
                    )
                ]
            )
        ]
    )

    seen: list[type] = []
    s = models.Stream(
        protocol.stream(
            cast("Any", fake),
            _MODEL,
            [ai.user_message("Hi")],
            provider="google",
        )
    )
    async for event in s:
        seen.append(type(event))

    assert seen == [
        events.StreamStart,
        events.ToolStart,
        events.ToolDelta,
        events.ToolEnd,
        events.StreamEnd,
    ]
    (call,) = s.message.tool_calls
    assert call.tool_call_id == "fc_1"
    assert call.tool_name == "get_weather"
    assert call.tool_args == '{"city":"Tokyo"}'


async def test_function_call_without_id_gets_generated_id() -> None:
    s = await _drain(
        [
            chunk(
                [
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            name="get_weather", args={}
                        )
                    )
                ]
            )
        ]
    )

    (call,) = s.message.tool_calls
    assert call.tool_call_id


async def test_code_execution_parts_emit_builtin_events() -> None:
    s = await _drain(
        [
            chunk(
                [
                    genai_types.Part(
                        executable_code=genai_types.ExecutableCode(
                            code="print(1)",
                            language=genai_types.Language.PYTHON,
                        )
                    ),
                    genai_types.Part(
                        code_execution_result=genai_types.CodeExecutionResult(
                            outcome=genai_types.Outcome.OUTCOME_OK,
                            output="1\n",
                        )
                    ),
                ]
            )
        ]
    )

    (call,) = s.message.builtin_tool_calls
    assert call.tool_name == "code_execution"
    assert call.tool_args == '{"code":"print(1)","language":"PYTHON"}'

    (ret,) = s.message.builtin_tool_returns
    assert ret.tool_call_id == call.tool_call_id
    assert ret.tool_name == "code_execution"
    assert ret.result == {"outcome": "OUTCOME_OK", "output": "1\n"}
    assert ret.is_error is False


async def test_code_execution_results_pair_by_id() -> None:
    s = await _drain(
        [
            chunk(
                [
                    genai_types.Part(
                        executable_code=genai_types.ExecutableCode(
                            code="print(1)",
                            language=genai_types.Language.PYTHON,
                            id="exec_1",
                        )
                    ),
                    genai_types.Part(
                        executable_code=genai_types.ExecutableCode(
                            code="print(2)",
                            language=genai_types.Language.PYTHON,
                            id="exec_2",
                        )
                    ),
                    genai_types.Part(
                        code_execution_result=genai_types.CodeExecutionResult(
                            outcome=genai_types.Outcome.OUTCOME_OK,
                            output="1\n",
                            id="exec_1",
                        )
                    ),
                    genai_types.Part(
                        code_execution_result=genai_types.CodeExecutionResult(
                            outcome=genai_types.Outcome.OUTCOME_OK,
                            output="2\n",
                            id="exec_2",
                        )
                    ),
                ]
            )
        ]
    )

    calls = s.message.builtin_tool_calls
    assert [c.tool_call_id for c in calls] == ["exec_1", "exec_2"]

    returns = s.message.builtin_tool_returns
    assert {r.tool_call_id: r.result["output"] for r in returns} == {
        "exec_1": "1\n",
        "exec_2": "2\n",
    }


async def test_inline_data_emits_file_event() -> None:
    fake = FakeGoogleClient(
        chunks=[
            chunk(
                [
                    genai_types.Part(
                        inline_data=genai_types.Blob(
                            data=b"\x89PNG", mime_type="image/png"
                        )
                    )
                ]
            )
        ]
    )

    file_events = []
    s = models.Stream(
        protocol.stream(
            cast("Any", fake),
            _MODEL,
            [ai.user_message("Draw a cat")],
            provider="google",
        )
    )
    async for event in s:
        if isinstance(event, events.FileEvent):
            file_events.append(event)

    (file_event,) = file_events
    assert file_event.media_type == "image/png"
    assert file_event.data == b"\x89PNG"


async def test_blocked_prompt_raises_response_error() -> None:
    with pytest.raises(ai.ProviderResponseError, match="blocked the prompt"):
        await _drain([chunk(block_reason=genai_types.BlockedReason.SAFETY)])


async def test_finish_reason_lands_in_provider_metadata() -> None:
    s = await _drain(
        [
            chunk(
                [genai_types.Part(text="partial")],
                finish_reason=genai_types.FinishReason.SAFETY,
            ),
        ]
    )

    assert s.message.provider_metadata == {"google": {"finishReason": "SAFETY"}}


async def test_usage_metadata_maps_to_usage() -> None:
    s = await _drain(
        [
            chunk([genai_types.Part(text="Hi")]),
            chunk(
                None,
                usage={
                    "prompt_token_count": 10,
                    "candidates_token_count": 5,
                    "thoughts_token_count": 3,
                    "cached_content_token_count": 2,
                },
            ),
        ]
    )

    usage = s.message.usage
    assert usage is not None
    assert usage.input_tokens == 10
    assert usage.output_tokens == 8
    assert usage.reasoning_tokens == 3
    assert usage.cache_read_tokens == 2
