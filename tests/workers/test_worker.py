"""The worker core (L1 and L2) against a scripted wire, offline."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pydantic
import pytest

import ai
from ai import workers
from ai.types import events
from ai.types import messages as m
from ai.workers import _wire, _worker

from . import fakes


async def drain(feed: Any, until: type = events.StreamEnd) -> list[Any]:
    """Read events up to and including the first `until`."""
    seen: list[Any] = []
    async for event in feed:
        seen.append(event)
        if isinstance(event, until):
            break
    return seen


# -- the model protocol (L2) ---------------------------------------------------


async def test_a_worker_streams_like_a_model() -> None:
    worker = fakes.fake(fakes.echo)
    user = ai.user_message("hello")
    async with worker:
        async with ai.stream(worker, [user]) as s:
            kinds = [type(e).__name__ async for e in s]
        assert s.text == "hello"
        assert kinds[0] == "StreamStart" and kinds[-1] == "StreamEnd"
        # The worker's state and the stream agree, ids too.
        assert [msg.id for msg in worker.messages] == [user.id, s.message.id]
        assert worker.messages[1].text == "hello"
        assert worker.session_id == "s1"


async def test_the_next_call_sends_only_the_new_messages() -> None:
    worker = fakes.fake(fakes.echo)
    first = ai.user_message("one")
    async with worker:
        async with ai.stream(worker, [first]) as s:
            async for _ in s:
                pass
        second = ai.user_message("two")
        async with ai.stream(worker, [first, s.message, second]) as s2:
            async for _ in s2:
                pass
        assert [msg.text for msg in worker.wire.sent] == ["one", "two"]
        assert s2.text == "two"
        assert len(worker.messages) == 4


async def test_a_rewritten_history_is_refused() -> None:
    worker = fakes.fake(fakes.echo)
    async with worker:
        async with ai.stream(worker, [ai.user_message("one")]) as s:
            async for _ in s:
                pass
        with pytest.raises(workers.TranscriptMismatchError):
            async with ai.stream(
                worker, [ai.user_message("other"), ai.user_message("two")]
            ) as s:
                async for _ in s:
                    pass


async def test_a_fresh_worker_seeds_the_session_with_the_past() -> None:
    worker = fakes.fake(fakes.echo, instructions="Be brief.")
    history = [
        ai.system_message("You fix tests."),
        ai.user_message("one"),
        ai.assistant_message("done"),
        ai.user_message("two"),
    ]
    async with ai.stream(worker, history) as s:
        async for _ in s:
            pass
    started = worker.wire.started
    assert started is not None
    assert started["instructions"] == "Be brief.\n\nYou fix tests."
    assert [msg.text for msg in started["seed"]] == ["one", "done"]
    assert [msg.text for msg in worker.wire.sent] == ["two"]
    assert [msg.id for msg in worker.messages[:4]] == [x.id for x in history]
    await worker.close()


async def test_a_resumed_session_does_not_resend_the_past() -> None:
    worker = fakes.fake(fakes.echo, session_id="old")
    history = [
        ai.user_message("one"),
        ai.assistant_message("done"),
        ai.user_message("two"),
    ]
    async with ai.stream(worker, history) as s:
        async for _ in s:
            pass
    assert worker.wire.started is not None
    assert worker.wire.started["seed"] == []  # the CLI has it
    assert worker.wire.started["session_id"] == "old"
    assert [msg.text for msg in worker.wire.sent] == ["two"]
    await worker.close()


async def test_a_started_empty_session_is_replaced_by_a_seeded_one() -> None:
    worker = fakes.fake(fakes.echo)
    history = [ai.system_message("s"), ai.user_message("x")]
    async with worker:
        async with ai.stream(worker, history) as st:
            async for _ in st:
                pass
        first, second = worker._wires
        assert first.closed and first.sent == []
        assert second.started is not None
        assert second.started["session_id"] is None  # a new session
        assert second.started["instructions"] == "s"
        assert [msg.text for msg in second.sent] == ["x"]


async def test_new_tools_restart_an_idle_cli_on_the_same_session() -> None:
    worker = fakes.fake(fakes.echo)
    async with worker:
        first = ai.user_message("one")
        async with ai.stream(worker, [first], tools=[_tool("a")]) as st:
            async for _ in st:
                pass
        assert len(worker._wires) == 2  # the first CLI saw no tools
        assert worker.wire.started is not None
        assert worker.wire.started["session_id"] is None  # it was unused
        again = [first, st.message, ai.user_message("two")]
        async with ai.stream(worker, again, tools=[_tool("a")]) as st:
            async for _ in st:
                pass
        assert len(worker._wires) == 2  # same tools: no restart
        more = [*worker.messages, ai.user_message("three")]
        async with ai.stream(worker, more, tools=[_tool("b")]) as st:
            async for _ in st:
                pass
        assert len(worker._wires) == 3
        assert worker.wire.started is not None
        assert worker.wire.started["session_id"] == "s1"  # a used session


async def test_a_started_resumed_session_takes_its_known_past() -> None:
    worker = fakes.fake(fakes.echo, session_id="old")
    history = [
        ai.user_message("one"),
        ai.assistant_message("done"),
        ai.user_message("two"),
    ]
    async with worker:
        async with ai.stream(worker, history) as s:
            async for _ in s:
                pass
    assert [msg.text for msg in worker.wire.sent] == ["two"]


async def test_a_worker_cannot_continue_tool_calls_it_did_not_make() -> None:
    worker = fakes.fake(fakes.echo)
    call = m.ToolCallPart(tool_call_id="c", tool_name="t", tool_args="{}")
    history = [
        ai.user_message("go"),
        m.Message(role="assistant", parts=[call]),
        ai.tool_message(tool_call_id="c", result="r"),
    ]
    with pytest.raises(workers.TranscriptMismatchError):
        async with ai.stream(worker, history) as s:
            async for _ in s:
                pass


async def test_output_type_is_not_supported_yet() -> None:
    class Out(pydantic.BaseModel):
        answer: str

    worker = fakes.fake(fakes.echo)
    with pytest.raises(workers.UnsupportedError):
        async with ai.stream(
            worker, [ai.user_message("x")], output_type=Out
        ) as s:
            async for _ in s:
                pass


async def test_cli_tools_are_builtin_parts() -> None:
    def reply(msg: m.Message) -> list[Any]:
        return [
            _wire.Accepted(msg),
            *fakes.builtin(
                "b1", "commandExecution", '{"command": "ls"}', "a\nb"
            ),
            *fakes.text("two files"),
            _wire.TurnEnd(),
        ]

    worker = fakes.fake(reply)
    async with worker:
        async with ai.stream(worker, [ai.user_message("ls")]) as s:
            async for _ in s:
                pass
    (call,) = s.message.builtin_tool_calls
    (result,) = s.message.builtin_tool_returns
    assert call.tool_name == "commandExecution"
    assert call.tool_args == '{"command": "ls"}'
    assert result.result == "a\nb"
    assert s.message.text == "two files"


async def test_a_failed_turn_raises_from_stream() -> None:
    error = workers.NotAuthenticatedError("fake", "401", "log in")

    def reply(msg: m.Message) -> list[Any]:
        return [_wire.Accepted(msg), _wire.TurnEnd(reason="error", error=error)]

    worker = fakes.fake(reply)
    async with worker:
        with pytest.raises(workers.NotAuthenticatedError):
            async with ai.stream(worker, [ai.user_message("x")]) as s:
                async for _ in s:
                    pass
        # The state keeps the failed message, marked.
        assert worker.messages[-1].provider_metadata == {
            "worker": {"error": str(error)}
        }


async def test_one_stream_at_a_time() -> None:
    worker = fakes.fake()  # accepts, never answers
    async with worker:
        first = worker.stream([ai.user_message("one")])
        task = asyncio.create_task(anext(first))
        await asyncio.sleep(0.01)
        with pytest.raises(workers.WorkerError, match="one stream"):
            await anext(worker.stream([ai.user_message("two")]))
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await first.aclose()


# -- ai tools through the bridge ----------------------------------------------


async def test_an_agent_runs_ai_tools_for_the_cli() -> None:
    @ai.tool
    async def lookup(word: str) -> str:
        """Look a word up."""
        return word.upper()

    def reply(msg: m.Message) -> list[Any]:
        return [
            _wire.Accepted(msg),
            fakes.CallTool("lookup", {"word": "heron"}),
        ]

    worker = fakes.fake(reply)
    async with worker:
        async with ai.Agent(tools=[lookup]).run(
            worker, [ai.user_message("go")]
        ) as run:
            async for _ in run:
                pass
        roles = [msg.role for msg in worker.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        (call,) = worker.messages[1].tool_calls
        assert call.tool_name == "lookup"
        assert worker.messages[1].provider_metadata is None
        assert worker.messages[3].text == "got HERON"
        # The CLI got the result through MCP.
        assert await worker.wire.calls[0] == ("HERON", False)
        assert [msg.id for msg in run.messages] == [
            msg.id for msg in worker.messages
        ]
        assert worker.tools[0].name == "lookup"


async def test_parallel_calls_are_one_message() -> None:
    def reply(msg: m.Message) -> list[Any]:
        return [
            _wire.Accepted(msg),
            fakes.CallTool("a", then=lambda t, e: []),
            fakes.CallTool(
                "b", then=lambda t, e: [*fakes.text("both"), _wire.TurnEnd()]
            ),
        ]

    worker = fakes.fake(reply, tools=[_tool("a"), _tool("b")])
    async with worker:
        async with ai.stream(worker, [ai.user_message("go")]) as s:
            async for _ in s:
                pass
        assert s.message.tool_calls and len(s.message.tool_calls) == 2
        assert s.message.provider_metadata is None
        results = ai.tool_message(
            *[
                m.ToolResultPart(
                    tool_call_id=c.tool_call_id,
                    tool_name=c.tool_name,
                    result="ok",
                )
                for c in s.message.tool_calls
            ]
        )
        async with ai.stream(worker, [*worker.messages, results]) as s2:
            async for _ in s2:
                pass
        assert s2.text == "both"


async def test_cli_work_during_a_tool_call_comes_after_the_result() -> None:
    worker = fakes.fake(
        lambda msg: [
            _wire.Accepted(msg),
            fakes.CallTool("a", then=lambda t, e: [_wire.TurnEnd()]),
        ],
        tools=[_tool("a")],
    )
    async with worker:
        async with ai.stream(worker, [ai.user_message("go")]) as s:
            async for _ in s:
                pass
        # The CLI keeps working on its own tools while it waits.
        worker.wire.push(*fakes.builtin("b1", "read"))
        await asyncio.sleep(0.05)
        assert [msg.role for msg in worker.messages] == ["user", "assistant"]
        (call,) = s.message.tool_calls
        await worker.send(
            ai.tool_message(
                tool_call_id=call.tool_call_id, tool_name="a", result="r"
            )
        )
        await asyncio.sleep(0.05)
        roles = [msg.role for msg in worker.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert worker.messages[3].builtin_tool_calls[0].tool_name == "read"


async def test_a_new_message_fails_waiting_calls_with_a_tool_message() -> None:
    worker = fakes.fake(
        lambda msg: [
            _wire.Accepted(msg),
            fakes.CallTool("a", then=lambda t, e: []),
        ]
        if msg.text == "go"
        else fakes.echo(msg),
        tools=[_tool("a")],
    )
    async with worker:
        async with ai.stream(worker, [ai.user_message("go")]) as s:
            async for _ in s:
                pass
        # An agent was cancelled while its tool ran; the next run comes.
        async with ai.stream(
            worker, [*worker.messages, ai.user_message("next")]
        ) as s2:
            async for _ in s2:
                pass
        roles = [msg.role for msg in worker.messages]
        assert roles == ["user", "assistant", "tool", "user", "assistant"]
        synthetic = worker.messages[2]
        assert workers.is_interrupted(synthetic)
        assert synthetic.tool_results[0].result_kind == "error"
        assert await worker.wire.calls[0] == (_worker.NEW_MESSAGE, True)


async def test_a_result_for_no_call_is_refused() -> None:
    worker = fakes.fake()
    async with worker:
        with pytest.raises(ValueError, match="no tool call waits"):
            await worker.send(ai.tool_message(tool_call_id="x", result="r"))


# -- direct control (L1) -------------------------------------------------------


async def test_events_start_when_called_and_go_to_every_reader() -> None:
    worker = fakes.fake(fakes.echo)
    async with worker:
        a, b = worker.events(), worker.events()
        await worker.send("hi")  # before anyone iterates
        seen_a, seen_b = await drain(a), await drain(b)
        assert [type(e) for e in seen_a] == [type(e) for e in seen_b]
        assert isinstance(seen_a[0], workers.MessageEvent)
        assert seen_a[0].message.text == "hi"
        assert isinstance(seen_a[1], events.StreamStart)
        await a.aclose()
        await b.aclose()


async def test_a_steer_splits_the_reply_at_a_clean_point() -> None:
    worker = fakes.fake()
    async with worker:
        feed = worker.events()
        await worker.send("fix the tests")
        worker.wire.push(
            events.TextStart(block_id="t1"),
            events.TextDelta(chunk="Look", block_id="t1"),
        )
        steer = ai.user_message("skip the slow ones")
        await worker.send(steer)
        # The text block is open: the split waits for its end.
        worker.wire.push(
            events.TextDelta(chunk="ing.", block_id="t1"),
            events.TextEnd(block_id="t1"),
        )
        worker.wire.push(*fakes.text("Skipped.", block="t2"), _wire.TurnEnd())
        await drain(feed)  # the first part ends
        await drain(feed)  # the second part ends
        texts = [(msg.role, msg.text) for msg in worker.messages]
        assert texts == [
            ("user", "fix the tests"),
            ("assistant", "Looking."),
            ("user", "skip the slow ones"),
            ("assistant", "Skipped."),
        ]
        await feed.aclose()


async def test_interrupt_stops_the_turn_and_marks_the_reply() -> None:
    worker = fakes.fake()
    async with worker:
        await worker.send("long job")
        worker.wire.push(*fakes.text("Starting"))
        await asyncio.sleep(0.01)
        await worker.interrupt()
        assert worker.wire.interrupts == 1
        reply = worker.messages[-1]
        assert reply.text == "Starting"
        assert workers.is_interrupted(reply)
        await worker.interrupt()  # no turn: nothing happens
        assert worker.wire.interrupts == 1


async def test_an_ignored_interrupt_closes_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_worker, "INTERRUPT_TIMEOUT", 0.05)
    worker = fakes.fake()
    await worker.start()
    worker.wire.on_interrupt = None
    await worker.send("long job")
    worker.wire.push(*fakes.text("Starting"))
    await asyncio.sleep(0.01)
    await worker.interrupt()
    assert worker._session is None
    assert worker.wire.closed
    assert workers.is_interrupted(worker.messages[-1])
    await worker.start()  # continues the same session
    assert worker.wire.started is not None
    assert worker.wire.started["session_id"] == "s1"
    await worker.close()


async def test_a_cancelled_stream_interrupts_the_turn() -> None:
    worker = fakes.fake(lambda msg: [_wire.Accepted(msg), *fakes.text("work")])
    history = [ai.user_message("long job")]
    async with worker:
        async with ai.stream(worker, history) as s:
            async for event in s:
                if isinstance(event, events.TextEnd):
                    break  # stop reading: the turn must stop too
        assert worker.wire.interrupts == 1
        assert workers.is_interrupted(worker.messages[-1])
        # ai.Agent keeps no cancelled message; the next call still works.
        worker.wire.reply = fakes.echo
        async with ai.stream(worker, [*history, ai.user_message("again")]) as s:
            async for _ in s:
                pass
        assert s.text == "again"


async def test_a_cancelled_task_interrupts_the_turn() -> None:
    worker = fakes.fake(lambda msg: [_wire.Accepted(msg), *fakes.text("work")])

    async def consume() -> None:
        async with ai.stream(worker, [ai.user_message("long job")]) as s:
            async for _ in s:
                pass

    async with worker:
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.02)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert worker.wire.interrupts == 1
        assert workers.is_interrupted(worker.messages[-1])


async def test_a_crash_reaches_readers_and_start_recovers() -> None:
    worker = fakes.fake()
    await worker.start()
    feed = worker.events()
    await worker.send("go")
    worker.wire.push(*fakes.text("half"), RuntimeError("boom"))
    with pytest.raises(workers.WorkerError, match="boom"):
        await drain(feed, until=type(None))
    with pytest.raises(workers.WorkerError, match="stopped"):
        await worker.send("again")
    assert worker.messages[-1].provider_metadata is not None
    await worker.start()
    assert len(worker._wires) == 2
    await worker.close()


async def test_close_ends_readers_and_keeps_the_state() -> None:
    worker = fakes.fake(fakes.echo)
    await worker.start()
    feed = worker.events()
    await worker.send("hi")
    await drain(feed)
    await worker.close()
    # The reply's MessageEvent follows its StreamEnd; then the feed ends.
    rest = [e async for e in feed]
    assert [type(e) for e in rest] == [workers.MessageEvent]
    assert worker.wire.closed
    assert [msg.text for msg in worker.messages] == ["hi", "hi"]
    await worker.close()  # a second close does nothing
    with pytest.raises(workers.WorkerError, match="not started"):
        await worker.send("x")


# -- state ---------------------------------------------------------------------


async def test_the_state_is_plain_data() -> None:
    worker = workers.Codex(cwd="/w", model_name="gpt-5.5", sandbox="read-only")
    worker.messages = [ai.user_message("hi")]
    data = worker.model_dump(mode="json")
    again = workers.Worker.model_validate(data)
    assert type(again) is workers.Codex
    assert again.model_dump(mode="json") == data
    with pytest.raises(ValueError, match="unknown worker_class_id"):
        workers.Worker.model_validate({**data, "worker_class_id": "nope"})
    with pytest.raises(TypeError):
        workers.Worker(worker_class_id="codex")


async def test_a_copy_is_not_started_and_shares_no_list() -> None:
    worker = fakes.fake(fakes.echo)
    async with worker:
        await worker.send("hi")
        await asyncio.sleep(0.01)
        copy = worker.model_copy(update={"session_id": None})
        assert copy._session is None
        await worker.send("more")
        await asyncio.sleep(0.01)
        assert len(copy.messages) == 2 and len(worker.messages) == 4


async def test_an_agent_context_keeps_the_worker() -> None:
    worker = workers.Codex(cwd="/w")
    context = ai.Context(model=worker, messages=[], tools=[])
    data = context.model_dump(mode="json")
    again = ai.Context.model_validate(data)
    assert isinstance(again.model, workers.Codex)
    assert again.model.cwd == "/w"


# -- telemetry ----------------------------------------------------------------


async def test_sessions_messages_and_cli_tools_have_spans() -> None:
    def reply(msg: m.Message) -> list[Any]:
        return [
            _wire.Accepted(msg),
            *fakes.builtin("b1", "read", '{"path": "x"}', "data"),
            _wire.TurnEnd(),
        ]

    sink = ai.experimental_telemetry.DictSink()
    async with ai.experimental_telemetry.use_sink(sink):
        worker = fakes.fake(reply)
        async with worker:
            async with ai.stream(worker, [ai.user_message("go")]) as s:
                async for _ in s:
                    pass
    by_kind = {span.data.kind: span for span in sink.finished_spans}
    assert {
        "worker_session",
        "worker_message",
        "tool_execution",
        "ai_stream",
    } <= set(by_kind)
    session, message, tool = (
        by_kind["worker_session"],
        by_kind["worker_message"],
        by_kind["tool_execution"],
    )
    assert isinstance(session.data, workers.WorkerSessionSpanData)
    assert isinstance(message.data, workers.WorkerMessageSpanData)
    assert isinstance(
        tool.data, ai.experimental_telemetry.ToolExecutionSpanData
    )
    assert session.data.session_id == "s1"
    assert message.parent_id == session.id
    assert message.data.finish_reason == "stop"
    assert tool.parent_id == message.id
    assert tool.data.args == {"path": "x"}
    assert tool.data.result == "data"


def _tool(name: str) -> ai.tools.Tool:
    return ai.tools.Tool(
        kind="function",
        name=name,
        spec=ai.tools.ToolSpec(params={"type": "object"}),
    )
