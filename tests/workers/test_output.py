"""Structured model output is a serializable view of the native transcript."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pydantic
import pytest

import ai
from ai import workers
from ai.types import events
from ai.types import messages as m

from . import fakes


class Count(pydantic.BaseModel):
    n: int


@pytest.mark.parametrize("retries", [0, 1, 2])
@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("native_prefix", [False, True])
async def test_output_continuation(
    retries: int, restore: bool, native_prefix: bool
) -> None:
    worker = fakes.FakeWorker()
    fields = set(worker.model_dump())
    attempts = 0

    def script(prompt: str) -> list[fakes.Step]:
        nonlocal attempts
        attempts += 1
        return fakes.text(
            "not JSON" if attempts <= retries else '```json\n{"n": 2}\n```'
        )

    worker.script(script)
    first = ai.user_message("count")
    async with worker:
        async with ai.stream(worker, [first], output_type=Count) as stream:
            seen = [event async for event in stream]
        assert stream.output == Count(n=2)
        assert len(worker.fake.turns) == retries + 1
        native = worker.messages[1]
        assert native.id == stream.message.id
        assert native is not stream.message
        assert native.text == (
            "not JSON" if retries else '```json\n{"n": 2}\n```'
        )
        assert native.parts[0].id == "t"
        assert stream.message.parts[0].id == "output"
        assert {event.message.id for event in seen} == {native.id}
        assert len(worker.messages) == 2 + 2 * retries
        assert stream.usage is not None
        assert stream.usage.input_tokens == 3 * (retries + 1)
        assert native.usage is not None and native.usage.input_tokens == 3
        for hidden in worker.messages[2:]:
            assert hidden.provider_metadata is not None
            assert hidden.provider_metadata["worker"]["model_hidden"] is True
        before = worker.model_dump_json()

    if restore:
        data = json.loads(before)
        assert set(data) == fields
        restored = workers.Worker.model_validate(data)
        assert isinstance(restored, fakes.FakeWorker)
        worker = restored
    worker.script(lambda prompt: fakes.text('{"n": 3}'))
    # Round-trip the caller's transcript independently, too.
    history = (
        list(worker.messages)
        if native_prefix
        else [
            first,
            m.Message.model_validate_json(stream.message.model_dump_json()),
        ]
    )
    second = ai.user_message("count again")
    plan = worker._plan([*history, second])
    assert plan.transcript == [second]
    assert plan.prompt == second.parts
    async with worker:
        async with ai.stream(
            worker, [*history, second], output_type=Count
        ) as continuation:
            _ = [event async for event in continuation]
        assert continuation.output == Count(n=3)
        assert len(worker.fake.turns) == 1
        assert worker.fake.turns[0].startswith("count again")
    assert (
        worker.messages[:-2]
        == fakes.FakeWorker.model_validate_json(before).messages
    )


async def test_projection_keeps_builtin_parts_and_suppresses_narration() -> (
    None
):
    worker = fakes.FakeWorker()
    worker.script(
        lambda prompt: [
            *fakes.text("Let me look."),
            *fakes.tool("read", "Read", '{"path":"a.py"}', "contents"),
            *fakes.text('{"n": 2}', block="answer"),
        ]
    )
    first = ai.user_message("count")
    async with worker:
        seen = [
            event async for event in worker.stream([first], output_type=Count)
        ]
        native = worker.messages[1]
        projected = seen[-1].message
        assert projected is not native
        assert {event.message.id for event in seen} == {native.id}
        assert all(event.message is projected for event in seen)
        assert projected.text == '{"n":2}'
        assert native.text == 'Let me look.{"n": 2}'
        assert [part.kind for part in projected.parts] == [
            "builtin_tool_call",
            "builtin_tool_return",
            "text",
        ]
        assert projected.builtin_tool_calls == native.builtin_tool_calls
        assert projected.builtin_tool_returns == native.builtin_tool_returns
        metadata = native.provider_metadata
        assert metadata is not None
        assert (
            metadata["worker"]["model_parts"]
            == projected.model_dump(mode="json")["parts"]
        )
        next_message = ai.user_message("again")
        assert worker._plan([first, projected, next_message]).transcript == [
            next_message
        ]


@pytest.mark.parametrize("native_prefix", [False, True])
@pytest.mark.parametrize("change", ["id", "role", "text", "part_id", "omit"])
async def test_changed_output_history_is_rejected(
    native_prefix: bool, change: str
) -> None:
    worker = fakes.FakeWorker()
    worker.script(lambda prompt: fakes.text('{"n": 2}'))
    first = ai.user_message("count")
    async with worker:
        async with ai.stream(worker, [first], output_type=Count) as stream:
            _ = [event async for event in stream]
        history = [
            message.model_copy(deep=True)
            for message in (
                worker.messages if native_prefix else [first, stream.message]
            )
        ]
        answer = history[-1]
        if change == "id":
            answer.id = "changed"
        elif change == "role":
            answer.role = "user"
        elif change == "text":
            assert isinstance(answer.parts[0], m.TextPart)
            answer.parts[0].text = '{"n":999}'
        elif change == "part_id":
            assert isinstance(answer.parts[0], m.TextPart)
            answer.parts[0].id = "changed"
        else:
            history.pop()
        with pytest.raises(workers.TranscriptMismatchError):
            async with ai.stream(
                worker, [*history, ai.user_message("again")]
            ) as continuation:
                _ = [event async for event in continuation]
        assert len(worker.fake.turns) == 1


async def test_agent_output_tool_loop_with_narration() -> None:
    worker = fakes.FakeWorker()
    ran: list[int] = []
    answers: list[tuple[str, bool]] = []

    @ai.tool
    async def count(n: int) -> int:
        """Return the count."""
        ran.append(n)
        return n

    async def call() -> None:
        answers.append(await worker._on_call("count", {"n": 2}))

    def script(prompt: str) -> Iterator[fakes.Step]:
        yield from fakes.text("I will count with a tool.")
        yield from fakes.tool("read", "Read")
        yield fakes.Run(call)
        yield from fakes.text('{"n": 2}')

    worker.script(script)
    async with worker:
        async with ai.Agent(tools=[count]).run(
            worker, [ai.user_message("count")], output_type=Count
        ) as run:
            seen = [event async for event in run]
        assert run.output == Count(n=2)
        assert ran == [2]
        assert answers == [("2", False)]
        assert len(worker.fake.turns) == 1
        assert [message.role for message in run.messages] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        tool_message = run.messages[1]
        assert tool_message.text == ""
        assert [part.kind for part in tool_message.parts] == [
            "builtin_tool_call",
            "builtin_tool_return",
            "tool_call",
        ]
        assert worker.messages[1].text == "I will count with a tool."
        assert worker.messages[1].id == tool_message.id
        assert not any(
            isinstance(event, events.TextDelta) and "I will" in event.chunk
            for event in seen
        )
        worker.script(lambda prompt: fakes.text('{"n": 3}'))
        async with ai.stream(
            worker, [*run.messages, ai.user_message("again")], output_type=Count
        ) as continuation:
            _ = [event async for event in continuation]
        assert continuation.output == Count(n=3)
