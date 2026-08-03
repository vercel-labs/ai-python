"""Example tests for ai.testing.FakeModel.

Temporary, deliberately non-trivial examples exercising the scripted
fake model: parallel tool calls, parallel subagents that themselves
call tools, plain ai.stream replay, multi-turn scripts, and the
failure modes.
"""

from __future__ import annotations

import json

import pytest

import ai
from ai.types import events as events_
from ai.types import messages as messages_


@ai.tool
async def double(word: str) -> str:
    """Double a word."""
    return word * 2


@ai.tool
async def moons(planet: str) -> str:
    """Count a planet's moons."""
    return {"mars": "2", "venus": "0"}[planet]


async def test_four_parallel_tool_calls() -> None:
    calls = [ai.testing.tool_call(double, word=w) for w in "abcd"]
    model = ai.testing.FakeModel(
        [
            ai.user_message("double these: a b c d"),
            ai.assistant_message("doubling", *calls),
            ai.assistant_message("done: aa bb cc dd"),
        ]
    )

    agent = ai.Agent(tools=[double])
    async with agent.run(
        model, [ai.user_message("double these: a b c d")]
    ) as stream:
        events = [event async for event in stream]

    messages = stream.messages
    assert messages[-1].text == "done: aa bb cc dd"

    # every call answered, correlated by the ids tool_call() minted
    results = {
        p.tool_call_id: p.result for m in messages for p in m.tool_results
    }
    expected = {
        c.tool_call_id: json.loads(c.tool_args)["word"] * 2 for c in calls
    }
    assert results == expected

    # the second model call saw exactly: user, assistant, tool message
    assert len(model.calls) == 2
    assert [m.role for m in model.calls[1]] == ["user", "assistant", "tool"]
    assert not model.unused

    tool_events = [e for e in events if isinstance(e, events_.ToolCallResult)]
    assert len(tool_events) == 4


async def test_parallel_subagents_with_tool_calls() -> None:
    model_ref: list[ai.Model] = []

    @ai.tool
    async def research(topic: str) -> ai.SubAgentTool:
        """Research a topic with a subagent."""
        inner = ai.Agent(tools=[moons])
        prompt = ai.user_message(f"research: {topic}")
        async with inner.run(model_ref[0], [prompt]) as stream:
            async for event in stream:
                yield event

    research_calls = [
        ai.testing.tool_call(research, topic="mars"),
        ai.testing.tool_call(research, topic="venus"),
    ]
    model = ai.testing.FakeModel(
        [
            ai.user_message("compare mars and venus"),
            ai.assistant_message("researching both", *research_calls),
            ai.assistant_message("mars: 2 moons; venus: 0 moons"),
        ],
        [
            ai.user_message("research: mars"),
            ai.assistant_message(
                "checking", ai.testing.tool_call(moons, planet="mars")
            ),
            ai.assistant_message("mars has 2 moons"),
        ],
        [
            ai.user_message("research: venus"),
            ai.assistant_message(
                "checking", ai.testing.tool_call(moons, planet="venus")
            ),
            ai.assistant_message("venus has 0 moons"),
        ],
    )
    model_ref.append(model)

    agent = ai.Agent(tools=[research])
    async with agent.run(
        model, [ai.user_message("compare mars and venus")]
    ) as stream:
        [event async for event in stream]

    messages = stream.messages
    assert messages[-1].text == "mars: 2 moons; venus: 0 moons"

    # each child transcript is attached to the right parent tool call
    by_id = {p.tool_call_id: p for m in messages for p in m.tool_results}
    mars, venus = (by_id[c.tool_call_id] for c in research_calls)
    assert mars.get_model_input() == "mars has 2 moons"
    assert venus.get_model_input() == "venus has 0 moons"

    # parent: 2 calls; each child: 2 calls
    assert len(model.calls) == 6
    assert not model.unused


async def test_multi_turn_with_repeated_user_message() -> None:
    model = ai.testing.FakeModel(
        [
            ai.user_message("hi"),
            ai.assistant_message("hello"),
            ai.user_message("hi"),
            ai.assistant_message("hello again"),
        ]
    )

    agent = ai.Agent()
    history: list[messages_.Message] = [ai.user_message("hi")]
    async with agent.run(model, history) as stream:
        async for _ in stream:
            pass
    assert stream.messages[-1].text == "hello"

    history = [*stream.messages, ai.user_message("hi")]
    async with agent.run(model, history) as stream:
        async for _ in stream:
            pass
    assert stream.messages[-1].text == "hello again"
    assert not model.unused


async def test_scripted_tool_message_asserts_results() -> None:
    # A tool message in the script is an assertion on the actual results.
    call = ai.testing.tool_call(double, word="hi")
    script = [
        ai.user_message("double hi"),
        ai.assistant_message("on it", call),
        ai.tool_message(
            tool_call_id=call.tool_call_id, tool_name="double", result="WRONG"
        ),
        ai.assistant_message("done"),
    ]
    model = ai.testing.FakeModel(script)

    agent = ai.Agent(tools=[double])
    # agent.run wraps errors from its task groups in ExceptionGroups
    expected_error = pytest.RaisesGroup(
        pytest.RaisesExc(AssertionError, match="WRONG"),
        flatten_subgroups=True,
    )
    with expected_error:
        async with agent.run(model, [ai.user_message("double hi")]) as stream:
            async for _ in stream:
                pass

    assert model.unused  # "done" never played


async def test_plain_stream_replays_tool_calls() -> None:
    # ai.stream does not execute tools: a single response, one model call
    call = ai.testing.tool_call(double, word="hi")
    model = ai.testing.FakeModel(
        [
            ai.user_message("double hi"),
            ai.assistant_message("sure", call),
        ]
    )

    async with ai.stream(model, [ai.user_message("double hi")]) as stream:
        [event async for event in stream]

    assert stream.message.text == "sure"
    assert [c.tool_call_id for c in stream.message.tool_calls] == [
        call.tool_call_id
    ]
    assert not model.unused


async def test_unmatched_input_fails_with_dump() -> None:
    model = ai.testing.FakeModel(
        [
            ai.user_message("expected input"),
            ai.assistant_message("hi"),
        ]
    )

    with pytest.raises(AssertionError) as err:
        async with ai.stream(model, [ai.user_message("surprise!")]) as stream:
            async for _ in stream:
                pass

    assert "surprise!" in str(err.value)  # what arrived, as a dump
    assert "expected input" in str(err.value)  # where the script diverges
    assert model.unused  # nothing was consumed


async def test_missing_tool_results_fail_with_ids() -> None:
    # Script makes two calls; drive the follow-up by hand answering
    # only one of them.
    calls = [
        ai.testing.tool_call(double, word="a"),
        ai.testing.tool_call(double, word="b"),
    ]
    model = ai.testing.FakeModel(
        [
            ai.user_message("go"),
            ai.assistant_message("on it", *calls),
            ai.assistant_message("done"),
        ]
    )

    async with ai.stream(model, [ai.user_message("go")]) as stream:
        async for _ in stream:
            pass

    partial_history = [
        ai.user_message("go"),
        stream.message,
        ai.tool_message(
            tool_call_id=calls[0].tool_call_id,
            tool_name="double",
            result="aa",
        ),
    ]
    with pytest.raises(AssertionError) as err:
        async with ai.stream(model, partial_history) as stream:
            async for _ in stream:
                pass

    # the unanswered call is named
    assert calls[1].tool_call_id in str(err.value)


async def test_foreign_tool_results_fail() -> None:
    # Results answering a call the script never made are rejected.
    call = ai.testing.tool_call(double, word="a")
    model = ai.testing.FakeModel(
        [
            ai.user_message("go"),
            ai.assistant_message("on it", call),
            ai.assistant_message("done"),
        ]
    )

    async with ai.stream(model, [ai.user_message("go")]) as stream:
        async for _ in stream:
            pass

    history = [
        ai.user_message("go"),
        stream.message,
        ai.tool_message(
            tool_call_id="call_bogus", tool_name="double", result="zz"
        ),
    ]
    with pytest.raises(AssertionError) as err:
        async with ai.stream(model, history) as stream:
            async for _ in stream:
                pass

    assert "call_bogus" in str(err.value)
