from collections.abc import Sequence

import pytest

import ai
from ai import testing


async def _run_model_command(
    command: str,
    *,
    tools: Sequence[ai.tools.Tool] | None = None,
    model: ai.Model | None = None,
) -> str:
    selected_model = model if model is not None else ai.test_model()
    async with ai.stream(
        selected_model,
        [ai.user_message(command)],
        tools=tools,
    ) as stream:
        async for _ in stream:
            pass
    return stream.text


def test_model_creates_local_provider() -> None:
    model = ai.test_model()

    assert model.id == "test"
    assert isinstance(model.provider, testing.TestProvider)
    assert isinstance(model.provider.protocol, testing.TestProtocol)
    assert model.provider.api_key is None
    assert model.provider.is_configured()


def test_model_round_trips_through_serialization() -> None:
    model = ai.test_model()

    restored = ai.Model.model_validate_json(model.model_dump_json())

    assert restored == model
    assert isinstance(restored.provider, testing.TestProvider)


async def test_provider_lists_and_probes_local_model() -> None:
    model = ai.test_model()

    assert await model.provider.list_models() == ["test"]
    await model.provider.probe(model)


async def test_ping_streams_standard_text_events() -> None:
    async with ai.stream(
        ai.test_model(), [ai.user_message("ping: hello")]
    ) as stream:
        events = [event async for event in stream]

    assert stream.text == "pong: hello"
    assert [event.kind for event in events] == [
        "stream_start",
        "text_start",
        "text_delta",
        "text_end",
        "stream_end",
    ]
    assert stream.message.parts[0].id == "test-text-0"


async def test_agent_executes_tool_and_returns_canonical_json() -> None:
    calls: list[tuple[int, int]] = []

    @ai.tool
    async def add(left: int, right: int) -> int:
        calls.append((left, right))
        return left + right

    agent = ai.Agent(tools=[add])
    command = 'call: add {"right":3,"left":2}'
    async with agent.run(ai.test_model(), [ai.user_message(command)]) as stream:
        events = [event async for event in stream]

    assert calls == [(2, 3)]
    assert stream.output == (
        'return: {"is_error":false,"result":5,'
        '"tool_call_id":"test-tool-call-0","tool_name":"add"}'
    )
    tool_end = next(
        event for event in events if isinstance(event, ai.events.ToolEnd)
    )
    assert tool_end.tool_call.tool_args == '{"left":2,"right":3}'


async def test_tool_call_identifiers_are_deterministic() -> None:
    @ai.tool
    async def ready() -> bool:
        return True

    tool_call_ids: list[str] = []
    for _ in range(2):
        async with ai.stream(
            ai.test_model(),
            [ai.user_message("call: ready")],
            tools=[ready.tool],
        ) as stream:
            async for _ in stream:
                pass
        tool_call_ids.append(stream.tool_calls[0].tool_call_id)
        assert stream.tool_calls[0].tool_args == "{}"

    assert tool_call_ids == ["test-tool-call-0", "test-tool-call-0"]


async def test_tool_call_identifiers_are_unique_within_history() -> None:
    @ai.tool
    async def ready() -> bool:
        return True

    agent = ai.Agent(tools=[ready])
    model = ai.test_model()
    messages = [ai.user_message("call: ready")]

    for _ in range(2):
        async with agent.run(model, messages) as stream:
            async for _ in stream:
                pass
        messages = [*stream.messages, ai.user_message("call: ready")]

    tool_call_ids = [
        tool_call.tool_call_id
        for message in stream.messages
        for tool_call in message.tool_calls
    ]
    assert tool_call_ids == ["test-tool-call-0", "test-tool-call-1"]


async def test_unknown_tool_fails_with_available_tools() -> None:
    @ai.tool
    async def available() -> None:
        return None

    with pytest.raises(
        ValueError,
        match=(
            "unknown test tool 'missing'; "
            "available function tools: available"
        ),
    ):
        await _run_model_command("call: missing", tools=[available.tool])


@pytest.mark.parametrize(
    ("command", "error"),
    [
        ("call:", "call: requires a tool name"),
        ("call: available {invalid}", "arguments must be valid JSON"),
        ('call: available {"value":NaN}', "arguments must be valid JSON"),
        (
            'call: available {"value":1,"value":2}',
            "must not contain duplicate JSON keys",
        ),
        ("call: available []", "arguments must be a JSON object"),
        ("hello", "unsupported test model command"),
    ],
)
async def test_invalid_command_fails_loudly(
    command: str,
    error: str,
) -> None:
    @ai.tool
    async def available() -> None:
        return None

    with pytest.raises(ValueError, match=error):
        await _run_model_command(command, tools=[available.tool])


async def test_deeply_nested_json_fails_clearly() -> None:
    @ai.tool
    async def available() -> None:
        return None

    nested_value = "[" * 2000 + "0" + "]" * 2000
    command = f'call: available {{"value":{nested_value}}}'
    with pytest.raises(ValueError, match="argument nesting is too deep"):
        await _run_model_command(command, tools=[available.tool])


async def test_command_size_is_bounded() -> None:
    protocol = testing.TestProtocol(max_command_chars=8)
    model = ai.test_model().with_protocol(protocol)

    with pytest.raises(ValueError, match="exceeds max_command_chars"):
        await _run_model_command("ping: 123", model=model)


async def test_tool_result_size_is_bounded() -> None:
    protocol = testing.TestProtocol(max_result_chars=1)
    model = ai.test_model().with_protocol(protocol)
    tool_call_message = ai.assistant_message(
        ai.messages.ToolCallPart(
            tool_call_id="test-tool-call-0",
            tool_name="large_result",
            tool_args="{}",
        )
    )
    tool_message = ai.tool_message(
        tool_call_id="test-tool-call-0",
        tool_name="large_result",
        result="too large",
    )
    messages = [
        ai.user_message("call: large_result"),
        tool_call_message,
        tool_message,
    ]

    with pytest.raises(ValueError, match="exceeds max_result_chars"):
        async with ai.stream(model, messages) as stream:
            async for _ in stream:
                pass
