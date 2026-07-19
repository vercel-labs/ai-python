"""Tests for the Google adapter's request shaping.

Focused on ``params`` translation, message/tool conversion, and SDK
error mapping.
"""

from __future__ import annotations

import base64
from typing import Any, cast

import httpx
import pydantic
import pytest
from google.genai import errors as genai_errors

import ai
from ai.providers.google import protocol
from ai.providers.google import tools as google_tools
from ai.types import messages

from .conftest import FakeGoogleClient

_MODEL = ai.Model(id="gemini-2.5-flash", provider=ai.get_provider("google"))


class _RaisingAsyncModels:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def generate_content_stream(self, **kwargs: Any) -> Any:
        raise self._exc


class _RaisingGoogleClient:
    def __init__(self, exc: Exception) -> None:
        self.aio = type("Aio", (), {"models": _RaisingAsyncModels(exc)})()


async def _drain(stream: Any) -> None:
    async for _ in stream:
        pass


async def test_params_translate_to_config() -> None:
    fake = FakeGoogleClient()

    await _drain(
        protocol.stream(
            cast("Any", fake),
            _MODEL,
            [ai.user_message("Hi")],
            params=ai.InferenceRequestParams(
                sampling={
                    ai.TemperatureSamplerParams: ai.TemperatureSamplerParams(
                        temperature=0.5
                    ),
                    ai.TopPSamplerParams: ai.TopPSamplerParams(top_p=0.9),
                    ai.TopKSamplerParams: ai.TopKSamplerParams(top_k=40),
                    ai.SeedSamplerParams: ai.SeedSamplerParams(seed=123),
                    ai.RepetitionPenaltyParams: ai.RepetitionPenaltyParams(
                        frequency_penalty=0.1, presence_penalty=0.2
                    ),
                },
                reasoning=ai.ReasoningParams(effort="high"),
                output=ai.OutputParams(
                    max_tokens=123, reasoning_summary="auto"
                ),
                tool_calling=ai.ToolCallingParams(
                    tool_choice=ai.ToolChoiceMode.REQUIRED,
                ),
                extra_headers={"x-goog-feature": "enabled"},
                extra_body={"future_option": {"enabled": True}},
            ),
            provider="google",
        )
    )

    config = fake.captured["config"]
    assert config["temperature"] == 0.5
    assert config["top_p"] == 0.9
    assert config["top_k"] == 40
    assert config["seed"] == 123
    assert config["frequency_penalty"] == 0.1
    assert config["presence_penalty"] == 0.2
    assert config["max_output_tokens"] == 123
    # Pre-Gemini-3 models take a token budget instead of a level.
    assert config["thinking_config"] == {
        "thinking_budget": 24576,
        "include_thoughts": True,
    }
    assert config["tool_config"] == {"function_calling_config": {"mode": "ANY"}}
    assert config["http_options"] == {
        "headers": {"x-goog-feature": "enabled"},
        "extra_body": {"future_option": {"enabled": True}},
    }


async def test_reasoning_effort_maps_to_thinking_level_on_gemini_3() -> None:
    fake = FakeGoogleClient()

    await _drain(
        protocol.stream(
            cast("Any", fake),
            ai.Model(id="gemini-3-flash-preview", provider=_MODEL.provider),
            [ai.user_message("Hi")],
            params=ai.InferenceRequestParams(
                reasoning=ai.ReasoningParams(effort="high")
            ),
            provider="google",
        )
    )

    assert fake.captured["config"]["thinking_config"] == {
        "thinking_level": "high"
    }


async def test_reasoning_disabled_maps_to_zero_budget() -> None:
    fake = FakeGoogleClient()

    await _drain(
        protocol.stream(
            cast("Any", fake),
            _MODEL,
            [ai.user_message("Hi")],
            params=ai.InferenceRequestParams(
                reasoning=ai.ReasoningParams(effort=None)
            ),
            provider="google",
        )
    )

    assert fake.captured["config"]["thinking_config"] == {"thinking_budget": 0}


async def test_random_seed_omitted_by_adapter() -> None:
    fake = FakeGoogleClient()

    await _drain(
        protocol.stream(
            cast("Any", fake),
            _MODEL,
            [ai.user_message("Hi")],
            params=ai.InferenceRequestParams(
                sampling={ai.SeedSamplerParams: ai.SeedSamplerParams(seed=-1)}
            ),
            provider="google",
        )
    )

    assert fake.captured["config"] is None


async def test_unsupported_params_rejected_by_adapter() -> None:
    unsupported = [
        (
            ai.InferenceRequestParams(
                sampling={ai.MinPSamplerParams: ai.MinPSamplerParams(min_p=0.1)}
            ),
            "min_p",
        ),
        (ai.InferenceRequestParams(metadata={"k": "v"}), "metadata"),
        (ai.InferenceRequestParams(extra_query={"k": "v"}), "extra query"),
        (
            ai.InferenceRequestParams(
                tool_calling=ai.ToolCallingParams(
                    tool_choice=ai.ToolChoiceMode.AUTO,
                    parallel_tool_calls=False,
                )
            ),
            "parallel tool calls",
        ),
    ]
    for params, match in unsupported:
        with pytest.raises(ValueError, match=match):
            await _drain(
                protocol.stream(
                    cast("Any", FakeGoogleClient()),
                    _MODEL,
                    [ai.user_message("Hi")],
                    params=params,
                    provider="google",
                )
            )


async def test_system_prompt_becomes_system_instruction() -> None:
    fake = FakeGoogleClient()

    await _drain(
        protocol.stream(
            cast("Any", fake),
            _MODEL,
            [ai.system_message("Be brief."), ai.user_message("Hi")],
            provider="google",
        )
    )

    assert fake.captured["config"]["system_instruction"] == "Be brief."
    assert fake.captured["contents"] == [
        {"role": "user", "parts": [{"text": "Hi"}]}
    ]


async def test_tool_round_trip_and_result_wrapping() -> None:
    fake = FakeGoogleClient()

    convo = [
        ai.user_message("What's the weather?"),
        messages.Message(
            role="assistant",
            parts=[
                messages.ToolCallPart(
                    tool_call_id="fc_1",
                    tool_name="get_weather",
                    tool_args='{"city":"Tokyo"}',
                )
            ],
        ),
        messages.Message(
            role="tool",
            parts=[
                messages.ToolResultPart(
                    tool_call_id="fc_1",
                    tool_name="get_weather",
                    result="sunny",
                )
            ],
        ),
    ]

    await _drain(
        protocol.stream(cast("Any", fake), _MODEL, convo, provider="google")
    )

    assert fake.captured["contents"] == [
        {"role": "user", "parts": [{"text": "What's the weather?"}]},
        {
            "role": "model",
            "parts": [
                {
                    "function_call": {
                        "id": "fc_1",
                        "name": "get_weather",
                        "args": {"city": "Tokyo"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": "fc_1",
                        "name": "get_weather",
                        "response": {"output": "sunny"},
                    }
                }
            ],
        },
    ]


async def test_dict_tool_result_passes_through_unwrapped() -> None:
    fake = FakeGoogleClient()

    convo = [
        ai.user_message("Weather?"),
        messages.Message(
            role="assistant",
            parts=[
                messages.ToolCallPart(
                    tool_call_id="fc_1", tool_name="get_weather", tool_args="{}"
                )
            ],
        ),
        messages.Message(
            role="tool",
            parts=[
                messages.ToolResultPart(
                    tool_call_id="fc_1",
                    tool_name="get_weather",
                    result={"condition": "sunny", "temp_c": 30},
                )
            ],
        ),
        ai.user_message("Thanks"),
    ]

    await _drain(
        protocol.stream(cast("Any", fake), _MODEL, convo, provider="google")
    )

    # The dict result passes through unwrapped, and the trailing user
    # message merges into the same user content as the tool response.
    tool_content = fake.captured["contents"][-1]
    assert tool_content["role"] == "user"
    assert tool_content["parts"] == [
        {
            "function_response": {
                "id": "fc_1",
                "name": "get_weather",
                "response": {"condition": "sunny", "temp_c": 30},
            }
        },
        {"text": "Thanks"},
    ]


async def test_thought_signature_round_trips_from_provider_metadata() -> None:
    fake = FakeGoogleClient()
    signature = base64.b64encode(b"sig-bytes").decode()

    await _drain(
        protocol.stream(
            cast("Any", fake),
            _MODEL,
            [
                ai.assistant_message(
                    ai.thinking(
                        "hidden",
                        provider_metadata={
                            "google": {"thoughtSignature": signature}
                        },
                    )
                ),
                ai.user_message("Hi"),
            ],
            provider="google",
        )
    )

    assert fake.captured["contents"][0] == {
        "role": "model",
        "parts": [
            {
                "text": "hidden",
                "thought": True,
                "thought_signature": b"sig-bytes",
            }
        ],
    }


async def test_text_part_signature_round_trips() -> None:
    fake = FakeGoogleClient()
    signature = base64.b64encode(b"sig-bytes").decode()

    await _drain(
        protocol.stream(
            cast("Any", fake),
            _MODEL,
            [
                messages.Message(
                    role="assistant",
                    parts=[
                        messages.TextPart(
                            text="answer",
                            provider_metadata={
                                "google": {"thoughtSignature": signature}
                            },
                        )
                    ],
                ),
                ai.user_message("Hi"),
            ],
            provider="google",
        )
    )

    assert fake.captured["contents"][0] == {
        "role": "model",
        "parts": [{"text": "answer", "thought_signature": b"sig-bytes"}],
    }


async def test_unsigned_reasoning_parts_are_dropped() -> None:
    fake = FakeGoogleClient()

    await _drain(
        protocol.stream(
            cast("Any", fake),
            _MODEL,
            [
                ai.assistant_message(ai.thinking("hidden"), "visible"),
                ai.user_message("Hi"),
            ],
            provider="google",
        )
    )

    assert fake.captured["contents"][0] == {
        "role": "model",
        "parts": [{"text": "visible"}],
    }


async def test_tools_translate_to_wire_format() -> None:
    fake = FakeGoogleClient()

    tool = ai.types.tools.Tool(
        kind="function",
        name="get_weather",
        spec=ai.types.tools.ToolSpec(
            description="Get the weather",
            params={"type": "object", "properties": {}},
        ),
    )

    await _drain(
        protocol.stream(
            cast("Any", fake),
            _MODEL,
            [ai.user_message("Hi")],
            tools=[tool, google_tools.google_search()],
            provider="google",
        )
    )

    assert fake.captured["config"]["tools"] == [
        {
            "function_declarations": [
                {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters_json_schema": {
                        "type": "object",
                        "properties": {},
                    },
                }
            ]
        },
        {"google_search": {}},
    ]


async def test_foreign_provider_tool_rejected() -> None:
    tool = ai.types.tools.Tool(
        kind="provider",
        name="web_search",
        tool_config=ai.types.tools.ToolConfig(
            id="anthropic.web_search_20260209"
        ),
    )

    with pytest.raises(ValueError, match="provider tool"):
        await _drain(
            protocol.stream(
                cast("Any", FakeGoogleClient()),
                _MODEL,
                [ai.user_message("Hi")],
                tools=[tool],
                provider="google",
            )
        )


async def test_output_type_sets_response_schema() -> None:
    fake = FakeGoogleClient()

    class Weather(pydantic.BaseModel):
        city: str

    await _drain(
        protocol.stream(
            cast("Any", fake),
            _MODEL,
            [ai.user_message("Hi")],
            output_type=Weather,
            provider="google",
        )
    )

    config = fake.captured["config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == Weather.model_json_schema()


async def test_sdk_errors_are_mapped_to_provider_hierarchy() -> None:
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://google.test/v1beta/models"),
    )
    sdk_error = genai_errors.APIError(
        429,
        {
            "error": {
                "code": 429,
                "message": "quota exceeded",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
        response,
    )

    with pytest.raises(ai.ProviderRateLimitError) as exc_info:
        await _drain(
            protocol.stream(
                cast("Any", _RaisingGoogleClient(sdk_error)),
                _MODEL,
                [ai.user_message("Hi")],
                provider="google",
            )
        )

    exc = exc_info.value
    assert exc.provider == "google"
    assert exc.http_context is not None
    assert exc.http_context.status_code == 429
    assert exc.http_context.request is response.request
    assert exc.http_context.response is response
    assert exc.type == "RESOURCE_EXHAUSTED"
    assert exc.__cause__ is sdk_error


async def test_transport_errors_are_mapped_to_provider_hierarchy() -> None:
    for sdk_error, expected in [
        (httpx.ConnectError("connection refused"), ai.ProviderConnectionError),
        (httpx.ConnectTimeout("timed out"), ai.ProviderTimeoutError),
    ]:
        with pytest.raises(expected) as exc_info:
            await _drain(
                protocol.stream(
                    cast("Any", _RaisingGoogleClient(sdk_error)),
                    _MODEL,
                    [ai.user_message("Hi")],
                    provider="google",
                )
            )

        exc = exc_info.value
        assert exc.provider == "google"
        assert exc.is_retryable is True
        assert exc.__cause__ is sdk_error


async def test_model_404_is_mapped_to_model_not_found() -> None:
    sdk_error = genai_errors.APIError(
        404,
        {"error": {"code": 404, "message": "model not found"}},
    )

    with pytest.raises(ai.ProviderModelNotFoundError) as exc_info:
        await _drain(
            protocol.stream(
                cast("Any", _RaisingGoogleClient(sdk_error)),
                _MODEL,
                [ai.user_message("Hi")],
                provider="google",
            )
        )

    assert exc_info.value.model_id == _MODEL.id


async def test_tool_choice_modes_map_to_function_calling_config() -> None:
    cases: list[tuple[ai.ToolChoiceMode | ai.ToolRef, dict[str, Any]]] = [
        (ai.ToolChoiceMode.AUTO, {"mode": "AUTO"}),
        (ai.ToolChoiceMode.NONE, {"mode": "NONE"}),
        (
            ai.ToolRef("get_weather"),
            {"mode": "ANY", "allowed_function_names": ["get_weather"]},
        ),
    ]
    for tool_choice, expected in cases:
        fake = FakeGoogleClient()
        await _drain(
            protocol.stream(
                cast("Any", fake),
                _MODEL,
                [ai.user_message("Hi")],
                params=ai.InferenceRequestParams(
                    tool_calling=ai.ToolCallingParams(tool_choice=tool_choice)
                ),
                provider="google",
            )
        )

        assert fake.captured["config"]["tool_config"] == {
            "function_calling_config": expected
        }


async def test_tool_selection_maps_auto_to_validated() -> None:
    for mode, expected in [
        (ai.ToolChoiceMode.AUTO, "VALIDATED"),
        (ai.ToolChoiceMode.REQUIRED, "ANY"),
    ]:
        fake = FakeGoogleClient()
        await _drain(
            protocol.stream(
                cast("Any", fake),
                _MODEL,
                [ai.user_message("Hi")],
                params=ai.InferenceRequestParams(
                    tool_calling=ai.ToolCallingParams(
                        tool_choice=ai.ToolSelection(
                            tools=frozenset({ai.ToolRef("get_weather")}),
                            mode=mode,
                        )
                    )
                ),
                provider="google",
            )
        )

        assert fake.captured["config"]["tool_config"] == {
            "function_calling_config": {
                "mode": expected,
                "allowed_function_names": ["get_weather"],
            }
        }


async def test_service_tier_maps_to_config() -> None:
    fake = FakeGoogleClient()

    await _drain(
        protocol.stream(
            cast("Any", fake),
            _MODEL,
            [ai.user_message("Hi")],
            params=ai.InferenceRequestParams(
                provider_service=ai.ProviderServiceParams(service_tier="flex")
            ),
            provider="google",
        )
    )

    assert fake.captured["config"]["service_tier"] == "flex"


async def test_builtin_code_execution_parts_round_trip() -> None:
    """Built-in tool parts serialize back to wire with their signatures."""
    fake = FakeGoogleClient()
    signature = base64.b64encode(b"sig-bytes").decode()

    call = messages.BuiltinToolCallPart(
        tool_call_id="call_1",
        tool_name="code_execution",
        tool_args='{"code":"print(1)","language":"PYTHON"}',
        provider_metadata={"google": {"thoughtSignature": signature}},
    )
    result = messages.BuiltinToolReturnPart(
        tool_call_id="call_1",
        tool_name="code_execution",
        result={"outcome": "OUTCOME_OK", "output": "1\n"},
        provider_metadata={"google": {}},
    )
    convo = [
        ai.user_message("Compute 1"),
        messages.Message(role="assistant", parts=[call, result]),
        ai.user_message("Thanks"),
    ]

    await _drain(
        protocol.stream(cast("Any", fake), _MODEL, convo, provider="google")
    )

    model_content = next(
        c for c in fake.captured["contents"] if c["role"] == "model"
    )
    assert model_content["parts"] == [
        {
            "executable_code": {"code": "print(1)", "language": "PYTHON"},
            "thought_signature": b"sig-bytes",
        },
        {
            "code_execution_result": {
                "outcome": "OUTCOME_OK",
                "output": "1\n",
            }
        },
    ]


async def test_tool_call_signature_round_trips() -> None:
    fake = FakeGoogleClient()
    signature = base64.b64encode(b"sig-bytes").decode()

    convo = [
        ai.user_message("Weather?"),
        messages.Message(
            role="assistant",
            parts=[
                messages.ToolCallPart(
                    tool_call_id="fc_1",
                    tool_name="get_weather",
                    tool_args="{}",
                    provider_metadata={
                        "google": {"thoughtSignature": signature}
                    },
                )
            ],
        ),
        messages.Message(
            role="tool",
            parts=[
                messages.ToolResultPart(
                    tool_call_id="fc_1",
                    tool_name="get_weather",
                    result="sunny",
                )
            ],
        ),
    ]

    await _drain(
        protocol.stream(cast("Any", fake), _MODEL, convo, provider="google")
    )

    model_content = next(
        c for c in fake.captured["contents"] if c["role"] == "model"
    )
    assert model_content["parts"] == [
        {
            "function_call": {
                "id": "fc_1",
                "name": "get_weather",
                "args": {},
            },
            "thought_signature": b"sig-bytes",
        }
    ]


async def test_file_parts_convert_to_inline_and_file_data() -> None:
    fake = FakeGoogleClient()

    await _drain(
        protocol.stream(
            cast("Any", fake),
            _MODEL,
            [
                messages.Message(
                    role="user",
                    parts=[
                        messages.TextPart(text="What are these?"),
                        messages.FilePart(
                            data=b"\x89PNG", media_type="image/png"
                        ),
                        messages.FilePart(
                            data="https://example.com/doc.pdf",
                            media_type="application/pdf",
                        ),
                    ],
                )
            ],
            provider="google",
        )
    )

    (content,) = fake.captured["contents"]
    assert content["parts"] == [
        {"text": "What are these?"},
        {"inline_data": {"data": b"\x89PNG", "mime_type": "image/png"}},
        {
            "file_data": {
                "file_uri": "https://example.com/doc.pdf",
                "mime_type": "application/pdf",
            }
        },
    ]


async def test_multipart_tool_result_flattens_text_and_rejects_files() -> None:
    def _convo(result: Any) -> list[messages.Message]:
        return [
            ai.user_message("Go"),
            messages.Message(
                role="assistant",
                parts=[
                    messages.ToolCallPart(
                        tool_call_id="fc_1", tool_name="tool", tool_args="{}"
                    )
                ],
            ),
            messages.Message(
                role="tool",
                parts=[
                    messages.ToolResultPart(
                        tool_call_id="fc_1",
                        tool_name="tool",
                        result=result,
                        result_kind="special",
                    )
                ],
            ),
        ]

    fake = FakeGoogleClient()
    await _drain(
        protocol.stream(
            cast("Any", fake),
            _MODEL,
            _convo(
                messages.ContentOutput(
                    value=[
                        messages.TextPart(text="part one, "),
                        messages.TextPart(text="part two"),
                    ]
                )
            ),
            provider="google",
        )
    )
    tool_content = fake.captured["contents"][-1]
    assert tool_content["parts"][0]["function_response"]["response"] == {
        "output": "part one, part two"
    }

    with pytest.raises(ValueError, match="file parts in tool results"):
        await _drain(
            protocol.stream(
                cast("Any", FakeGoogleClient()),
                _MODEL,
                _convo(
                    messages.ContentOutput(
                        value=[
                            messages.FilePart(
                                data=b"\x89PNG", media_type="image/png"
                            )
                        ]
                    )
                ),
                provider="google",
            )
        )


async def test_messages_to_google_repairs_history() -> None:
    """Conversion runs history_utils.repair: internal messages are dropped
    and orphaned tool calls get a synthetic error result."""
    msgs = [
        messages.Message(
            role="internal",
            parts=[messages.TextPart(text="app-only")],
        ),
        messages.Message(
            role="assistant",
            parts=[
                messages.ToolCallPart(
                    tool_call_id="tc-1", tool_name="search", tool_args="{}"
                )
            ],
        ),
    ]
    _, wire = protocol._messages_to_google(msgs)
    assert [m["role"] for m in wire] == ["model", "user"]
    (tool_response,) = wire[1]["parts"]
    assert "error" in tool_response["function_response"]["response"]
