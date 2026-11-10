"""The worker core against a scripted wire: model calls and sessions."""

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
from ai.workers import _session, _wire

from . import fakes


async def drain(feed: Any, until: type = events.StreamEnd) -> list[Any]:
    """Read events up to and including the first `until`."""
    seen: list[Any] = []
    async for event in feed:
        seen.append(event)
        if isinstance(event, until):
            break
    return seen


# -- the worker as a model ----------------------------------------------------


async def test_a_worker_streams_like_a_model() -> None:
    worker = fakes.fake(fakes.echo)
    user = ai.user_message("hello")
    async with worker.session() as session:
        async with ai.stream(worker, [user]) as s:
            kinds = [type(e).__name__ async for e in s]
        assert s.text == "hello"
        assert kinds[0] == "StreamStart" and kinds[-1] == "StreamEnd"
        # The session's state and the stream agree, ids too.
        assert [msg.id for msg in session.messages] == [user.id, s.message.id]
        assert session.messages[1].text == "hello"
        assert session.id == "s1"


async def test_the_next_call_sends_only_the_new_messages() -> None:
    worker = fakes.fake(fakes.echo)
    first = ai.user_message("one")
    async with worker.session() as session:
        async with ai.stream(worker, [first]) as s:
            async for _ in s:
                pass
        second = ai.user_message("two")
        async with ai.stream(worker, [first, s.message, second]) as s2:
            async for _ in s2:
                pass
        assert [msg.text for msg in worker.wire.sent] == ["one", "two"]
        assert s2.text == "two"
        assert len(session.messages) == 4


async def test_a_rewritten_history_is_refused() -> None:
    worker = fakes.fake(fakes.echo)
    async with worker.session():
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
    async with worker.session() as session:
        assert worker._wires == []  # the CLI starts with the first use
        async with ai.stream(worker, history, tools=[_tool("a")]) as s:
            async for _ in s:
                pass
        assert len(worker._wires) == 1  # it had the tools and the past
    started = worker.wire.started
    assert started is not None
    assert started["instructions"] == "Be brief.\n\nYou fix tests."
    assert [msg.text for msg in started["seed"]] == ["one", "done"]
    assert [msg.text for msg in worker.wire.sent] == ["two"]
    assert [msg.id for msg in session.messages[:4]] == [x.id for x in history]


async def test_a_resumed_session_does_not_resend_the_past() -> None:
    worker = fakes.fake(fakes.echo)
    history = [
        ai.user_message("one"),
        ai.assistant_message("done"),
        ai.user_message("two"),
    ]
    async with worker.session("old"):
        async with ai.stream(worker, history) as s:
            async for _ in s:
                pass
    assert worker.wire.started is not None
    assert worker.wire.started["seed"] == []  # the CLI has it
    assert worker.wire.started["session_id"] == "old"
    assert worker.session_id is None  # only a new session sets it
    assert [msg.text for msg in worker.wire.sent] == ["two"]


async def test_a_started_empty_session_is_replaced_by_a_seeded_one() -> None:
    worker = fakes.fake(fakes.echo)
    history = [ai.system_message("s"), ai.user_message("x")]
    async with worker.session() as session:
        await session.start()
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
    async with worker.session() as session:
        await session.start()
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
        more = [*session.messages, ai.user_message("three")]
        async with ai.stream(worker, more, tools=[_tool("b")]) as st:
            async for _ in st:
                pass
        assert len(worker._wires) == 3
        assert worker.wire.started is not None
        assert worker.wire.started["session_id"] == "s1"  # a used session


async def test_a_started_resumed_session_takes_its_known_past() -> None:
    worker = fakes.fake(fakes.echo)
    history = [
        ai.user_message("one"),
        ai.assistant_message("done"),
        ai.user_message("two"),
    ]
    async with worker.session("old") as session:
        await session.start()
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
    async with worker.session():
        with pytest.raises(workers.TranscriptMismatchError):
            async with ai.stream(worker, history) as s:
                async for _ in s:
                    pass
        assert worker._wires == []  # refused before the CLI starts


async def test_a_model_call_needs_no_session() -> None:
    worker = fakes.fake(fakes.echo)
    first = ai.user_message("one")
    async with ai.stream(worker, [first]) as s:
        async for _ in s:
            pass
    assert s.text == "one"
    assert worker.wire.closed  # the CLI ran for one turn
    # The next call continues the same CLI session, and sends the new part.
    second = ai.user_message("two")
    async with ai.stream(worker, [first, s.message, second]) as s2:
        async for _ in s2:
            pass
    assert s2.text == "two"
    assert len(worker._wires) == 2 and worker.wire.closed
    assert worker.wire.started is not None
    assert worker.wire.started["session_id"] == "s1"
    assert worker.wire.started["seed"] == []
    assert [msg.text for msg in worker.wire.sent] == ["two"]


async def test_a_worker_remembers_its_first_new_session() -> None:
    worker = fakes.fake(fakes.echo)
    assert worker.session_id is None
    first = ai.user_message("one")
    async with ai.stream(worker, [first]) as s:
        async for _ in s:
            pass
    assert worker.session_id == "s1"
    # The id is plain data: a worker in another process continues with it.
    again = workers.Worker.model_validate(worker.model_dump(mode="json"))
    assert isinstance(again, fakes.FakeWorker)
    again._reply = fakes.echo
    async with ai.stream(
        again, [first, s.message, ai.user_message("two")]
    ) as s:
        async for _ in s:
            pass
    assert s.text == "two"
    assert again.wire.started is not None
    assert again.wire.started["session_id"] == "s1"
    assert again.wire.started["seed"] == []
    assert [msg.text for msg in again.wire.sent] == ["two"]
    assert again.wire.closed
    # The worker stays with its session: it refuses another history.
    with pytest.raises(workers.TranscriptMismatchError):
        async with ai.stream(worker, [ai.user_message("other")]) as s:
            async for _ in s:
                pass
    # Without the id, the next call starts a new session and remembers it.
    worker.session_id = None
    async with ai.stream(worker, [ai.user_message("other")]) as s:
        async for _ in s:
            pass
    assert s.text == "other"
    assert worker.wire.started is not None
    assert worker.wire.started["session_id"] is None
    assert worker.session_id == "s1"  # the fake CLI names each session "s1"


async def test_in_a_block_the_model_calls_use_the_live_cli() -> None:
    worker = fakes.fake(fakes.echo, session_id="abc")
    first = ai.user_message("one")
    async with worker.session() as session:
        assert session.id == "abc"
        async with ai.stream(worker, [first]) as s:
            async for _ in s:
                pass
        # A copy has the same session_id: it is the same session.
        twin = worker.model_copy()
        more = [first, s.message, ai.user_message("two")]
        async with ai.stream(twin, more) as s:
            async for _ in s:
                pass
        assert len(worker._wires) == 1 and not worker.wire.closed
        assert worker.wire.started is not None
        assert worker.wire.started["session_id"] == "abc"
        assert [msg.text for msg in worker.wire.sent] == ["one", "two"]
    assert worker.wire.closed
    # After the block: a new process for the call, on the same session.
    more = [*session.messages, ai.user_message("three")]
    async with ai.stream(worker, more) as s:
        async for _ in s:
            pass
    assert len(worker._wires) == 2 and worker.wire.closed
    assert worker.wire.started is not None
    assert worker.wire.started["session_id"] == "abc"
    assert [msg.text for msg in worker.wire.sent] == ["three"]


async def test_two_tasks_open_sessions_of_one_worker() -> None:
    worker = fakes.fake(fakes.echo)

    async def chat(text: str) -> tuple[str, workers.Session]:
        async with worker.session() as session:
            async with ai.stream(worker, [ai.user_message(text)]) as s:
                async for _ in s:
                    pass
            return s.text, session

    (one, a), (two, b) = await asyncio.gather(chat("one"), chat("two"))
    assert (one, two) == ("one", "two")
    assert len(worker._wires) == 2  # each task had its own live CLI
    assert [msg.text for msg in a.messages] == ["one", "one"]
    assert [msg.text for msg in b.messages] == ["two", "two"]


async def test_without_a_session_the_cli_stays_for_a_tool_loop() -> None:
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
    async with ai.Agent(tools=[lookup]).run(
        worker, [ai.user_message("go")]
    ) as run:
        async for _ in run:
            pass
    assert [msg.role for msg in run.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert run.messages[3].text == "got HERON"
    assert len(worker._wires) == 1  # one CLI for the whole turn
    assert worker.wire.closed  # and it stopped when the turn ended


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
    async with worker.session():
        async with ai.stream(worker, [ai.user_message("ls")]) as s:
            async for _ in s:
                pass
    (call,) = s.message.builtin_tool_calls
    (result,) = s.message.builtin_tool_returns
    assert call.tool_name == "commandExecution"
    assert call.tool_args == '{"command": "ls"}'
    assert result.result == "a\nb"
    assert s.message.text == "two files"


async def test_turn_metadata_goes_on_the_last_reply() -> None:
    def reply(msg: m.Message) -> list[Any]:
        return [
            _wire.Accepted(msg),
            *fakes.text("hi"),
            _wire.TurnEnd(metadata={"fake": {"model": "m1"}}),
        ]

    worker = fakes.fake(reply)
    async with worker.session() as session:
        async with ai.stream(worker, [ai.user_message("x")]) as s:
            async for _ in s:
                pass
    assert s.message.provider_metadata == {"fake": {"model": "m1"}}
    assert session.messages[-1].provider_metadata == {"fake": {"model": "m1"}}


async def test_a_failed_turn_raises_from_stream() -> None:
    error = workers.NotAuthenticatedError("fake", "401", "log in")

    def reply(msg: m.Message) -> list[Any]:
        return [_wire.Accepted(msg), _wire.TurnEnd(reason="error", error=error)]

    worker = fakes.fake(reply)
    history = [ai.user_message("x")]
    async with worker.session() as session:
        with pytest.raises(workers.NotAuthenticatedError):
            async with ai.stream(worker, history) as s:
                async for _ in s:
                    pass
        # The state keeps the failed message, marked.
        assert session.messages[-1].provider_metadata == {
            "worker": {"error": str(error)}
        }
        # The caller does not have it; the next call still works.
        worker.wire.reply = fakes.echo
        async with ai.stream(worker, [*history, ai.user_message("again")]) as s:
            async for _ in s:
                pass
        assert s.text == "again"
        # Nothing new is refused in an open session.
        with pytest.raises(
            workers.TranscriptMismatchError, match="nothing new"
        ):
            async with ai.stream(worker, list(session.messages)) as s:
                async for _ in s:
                    pass


async def test_one_stream_at_a_time() -> None:
    worker = fakes.fake()  # accepts, never answers
    async with worker.session():
        first = worker.provider.stream(worker, [ai.user_message("one")])
        task = asyncio.create_task(anext(first))
        await asyncio.sleep(0.01)
        with pytest.raises(workers.WorkerError, match="one stream"):
            await anext(
                worker.provider.stream(worker, [ai.user_message("two")])
            )
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
    async with worker.session() as session:
        async with ai.Agent(tools=[lookup]).run(
            worker, [ai.user_message("go")]
        ) as run:
            async for _ in run:
                pass
        roles = [msg.role for msg in session.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        (call,) = session.messages[1].tool_calls
        assert call.tool_name == "lookup"
        assert session.messages[1].provider_metadata is None
        assert session.messages[3].text == "got HERON"
        # The CLI got the result through MCP.
        assert await worker.wire.calls[0] == ("HERON", False)
        assert [msg.id for msg in run.messages] == [
            msg.id for msg in session.messages
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
    async with worker.session() as session:
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
        async with ai.stream(worker, [*session.messages, results]) as s2:
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
    async with worker.session() as session:
        async with ai.stream(worker, [ai.user_message("go")]) as s:
            async for _ in s:
                pass
        # The CLI keeps working on its own tools while it waits.
        worker.wire.push(*fakes.builtin("b1", "read"))
        await asyncio.sleep(0.05)
        assert [msg.role for msg in session.messages] == ["user", "assistant"]
        (call,) = s.message.tool_calls
        await session.send(
            ai.tool_message(
                tool_call_id=call.tool_call_id, tool_name="a", result="r"
            )
        )
        await asyncio.sleep(0.05)
        roles = [msg.role for msg in session.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert session.messages[3].builtin_tool_calls[0].tool_name == "read"


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
    async with worker.session() as session:
        async with ai.stream(worker, [ai.user_message("go")]) as s:
            async for _ in s:
                pass
        # An agent was cancelled while its tool ran; the next run comes.
        async with ai.stream(
            worker, [*session.messages, ai.user_message("next")]
        ) as s2:
            async for _ in s2:
                pass
        roles = [msg.role for msg in session.messages]
        assert roles == ["user", "assistant", "tool", "user", "assistant"]
        synthetic = session.messages[2]
        assert synthetic.provider_metadata == {"worker": {"synthetic": True}}
        assert synthetic.tool_results[0].result_kind == "error"
        assert await worker.wire.calls[0] == (_session.NEW_MESSAGE, True)


async def test_a_result_for_no_call_is_refused() -> None:
    worker = fakes.fake()
    async with worker.session() as session:
        with pytest.raises(ValueError, match="no tool call waits"):
            await session.send(ai.tool_message(tool_call_id="x", result="r"))


# -- the session: direct control ----------------------------------------------


async def test_events_after_send_lose_nothing() -> None:
    worker = fakes.fake(fakes.echo)
    async with worker.session() as session:
        await session.send(ai.user_message("hi"))
        await asyncio.sleep(0.01)  # the turn ends before anyone reads
        seen = await drain(session.events())
        assert isinstance(seen[0], workers.MessageEvent)
        assert seen[0].message.text == "hi"
        assert isinstance(seen[1], events.StreamStart)
        # A reader that stopped needs no cleanup: the next one continues.
        rest = await drain(session.events(), until=workers.MessageEvent)
        assert [type(e) for e in rest] == [workers.MessageEvent]
        assert rest[0].message.role == "assistant"


async def test_events_that_nobody_reads_do_not_pile_up() -> None:
    worker = fakes.fake(fakes.echo)
    async with worker.session() as session:
        history: list[m.Message] = []
        sizes = []
        for text in ("one", "two", "three"):
            history = [*session.messages, ai.user_message(text)]
            async with ai.stream(worker, history) as s:
                async for _ in s:
                    pass
            sizes.append(len(session._feed))
        assert sizes[0] == sizes[1] == sizes[2]  # the latest turn only
        # A reader that reads keeps everything.
        seen = await drain(session.events())
        await session.send(ai.user_message("four"))
        rest = await drain(session.events(), until=workers.MessageEvent)
        assert isinstance(seen[0], workers.MessageEvent)
        assert rest[0].message.text == "three"  # the reply before "four"


async def test_a_steer_splits_the_reply_at_a_clean_point() -> None:
    worker = fakes.fake()
    async with worker.session() as session:
        feed = session.events()
        await session.send(ai.user_message("fix the tests"))
        worker.wire.push(
            events.TextStart(block_id="t1"),
            events.TextDelta(chunk="Look", block_id="t1"),
        )
        steer = ai.user_message("skip the slow ones")
        await session.send(steer)
        # The text block is open: the split waits for its end.
        worker.wire.push(
            events.TextDelta(chunk="ing.", block_id="t1"),
            events.TextEnd(block_id="t1"),
        )
        worker.wire.push(*fakes.text("Skipped.", block="t2"), _wire.TurnEnd())
        await drain(feed)  # the first part ends
        await drain(feed)  # the second part ends
        texts = [(msg.role, msg.text) for msg in session.messages]
        assert texts == [
            ("user", "fix the tests"),
            ("assistant", "Looking."),
            ("user", "skip the slow ones"),
            ("assistant", "Skipped."),
        ]


async def test_interrupt_stops_the_turn_and_marks_the_reply() -> None:
    worker = fakes.fake()
    async with worker.session() as session:
        await session.send(ai.user_message("long job"))
        worker.wire.push(*fakes.text("Starting"))
        await asyncio.sleep(0.01)
        await session.interrupt()
        assert worker.wire.interrupts == 1
        reply = session.messages[-1]
        assert reply.text == "Starting"
        # Not a finish reason of the vocabulary: the raw value is here.
        assert reply.provider_metadata == {
            "worker": {"stop_reason": "interrupted"}
        }
        await session.interrupt()  # no turn: nothing happens
        assert worker.wire.interrupts == 1


async def test_an_ignored_interrupt_stops_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_session, "INTERRUPT_TIMEOUT", 0.05)
    worker = fakes.fake()
    async with worker.session() as session:
        await session.start()
        worker.wire.on_interrupt = None
        feed = session.events()
        await session.send(ai.user_message("long job"))
        worker.wire.push(*fakes.text("Starting"))
        await asyncio.sleep(0.01)
        await session.interrupt()
        assert session._run is None
        assert worker.wire.closed
        assert session.messages[-1].provider_metadata == {
            "worker": {"stop_reason": "interrupted"}
        }
        assert (await drain(feed))[-1].finish_reason == "other"
        # The session stays open: the next use continues the CLI session.
        worker._reply = fakes.echo
        await session.send(ai.user_message("again"))
        assert len(worker._wires) == 2
        assert worker.wire.started is not None
        assert worker.wire.started["session_id"] == "s1"


async def test_a_cancelled_stream_interrupts_the_turn() -> None:
    worker = fakes.fake(lambda msg: [_wire.Accepted(msg), *fakes.text("work")])
    history = [ai.user_message("long job")]
    async with worker.session() as session:
        async with ai.stream(worker, history) as s:
            async for event in s:
                if isinstance(event, events.TextEnd):
                    break  # stop reading: the turn must stop too
        assert worker.wire.interrupts == 1
        assert session.messages[-1].provider_metadata == {
            "worker": {"stop_reason": "interrupted"}
        }
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

    async with worker.session() as session:
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.02)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert worker.wire.interrupts == 1
        assert session.messages[-1].provider_metadata == {
            "worker": {"stop_reason": "interrupted"}
        }


async def test_a_crash_reaches_the_reader_and_start_recovers() -> None:
    worker = fakes.fake()
    async with worker.session() as session:
        feed = session.events()
        await session.send(ai.user_message("go"))
        worker.wire.push(*fakes.text("half"), RuntimeError("boom"))
        with pytest.raises(workers.WorkerError, match="boom"):
            await drain(feed, until=type(None))
        with pytest.raises(workers.WorkerError, match="stopped"):
            await session.send(ai.user_message("again"))
        assert session.messages[-1].provider_metadata is not None
        await session.start()
        assert len(worker._wires) == 2


async def test_close_ends_readers_and_keeps_the_state() -> None:
    worker = fakes.fake(fakes.echo)
    session = worker.session()
    async with session:
        feed = session.events()
        await session.send(ai.user_message("hi"))
        await drain(feed)
    # The reply's MessageEvent follows its StreamEnd; then the feed ends.
    rest = [e async for e in feed]
    assert [type(e) for e in rest] == [workers.MessageEvent]
    assert worker.wire.closed
    assert session.id == "s1"
    assert [msg.text for msg in session.messages] == ["hi", "hi"]
    await session.close()  # a second close does nothing
    with pytest.raises(workers.WorkerError, match="not open"):
        await session.send(ai.user_message("x"))
    with pytest.raises(workers.WorkerError, match="not open"):
        await anext(session.events())


async def test_start_and_close_are_a_pair() -> None:
    worker = fakes.fake(fakes.echo)
    session = worker.session()
    await session.start()  # no block
    assert len(worker._wires) == 1
    await session.send(ai.user_message("hi"))
    await drain(session.events())
    await session.close()
    assert worker.wire.closed
    await session.start()  # open again: the same CLI session
    assert worker.wire.started is not None
    assert worker.wire.started["session_id"] == "s1"
    await session.close()


async def test_a_closed_session_continues_without_a_block() -> None:
    worker = fakes.fake(fakes.echo)
    async with worker.session() as session:
        async with ai.stream(worker, [ai.user_message("one")]) as s:
            async for _ in s:
                pass
    assert worker.session_id == session.id == "s1"
    more = [*session.messages, ai.user_message("two")]
    async with ai.stream(worker, more) as s:
        async for _ in s:
            pass
    assert worker.wire.started is not None
    assert worker.wire.started["session_id"] == "s1"  # not a new session
    assert [msg.text for msg in worker.wire.sent] == ["two"]


async def test_a_session_is_open_once_in_a_context() -> None:
    worker = fakes.fake(fakes.echo, session_id="abc")
    async with worker.session():
        with pytest.raises(workers.WorkerError, match="open already"):
            async with worker.session():
                pass
    async with worker.session() as session:  # the first one closed
        await session.send(ai.user_message("hi"))


async def test_the_feed_stays_when_new_tools_restart_the_cli() -> None:
    worker = fakes.fake(fakes.echo)
    async with worker.session() as session:
        await session.start()
        feed = session.events()
        async with ai.stream(
            worker, [ai.user_message("one")], tools=[_tool("a")]
        ) as st:
            async for _ in st:
                pass
        assert len(worker._wires) == 2
        seen = await drain(feed)
        assert isinstance(seen[0], workers.MessageEvent)
        assert seen[0].message.text == "one"


# -- state ---------------------------------------------------------------------


async def test_a_worker_is_plain_data() -> None:
    worker = workers.Codex(
        cwd="/w",
        model_name="gpt-5.5",
        sandbox="read-only",
        provider=workers.CodexProvider(executable="/bin/codex", env={"K": "v"}),
        session_id="abc",
    )
    data = worker.model_dump(mode="json")
    assert data["session_id"] == "abc" and "messages" not in data
    again = workers.Worker.model_validate(data)
    assert type(again) is workers.Codex
    assert again.model_dump(mode="json") == data
    assert again.provider == worker.provider
    assert again.session().id == "abc"
    assert again.session("other").id == "other"
    with pytest.raises(ValueError, match="unknown worker_class_id"):
        workers.Worker.model_validate({**data, "worker_class_id": "nope"})
    with pytest.raises(TypeError):
        workers.Worker(worker_class_id="codex")


async def test_a_seeded_session_starts_with_its_past() -> None:
    worker = fakes.fake(fakes.echo)
    history = [ai.user_message("one"), ai.assistant_message("done")]
    async with worker.session(messages=history) as session:
        await session.send(ai.user_message("two"))
        await drain(session.events())
        assert worker.wire.started is not None
        assert worker.wire.started["session_id"] is None
        assert [msg.text for msg in worker.wire.started["seed"]] == [
            "one",
            "done",
        ]
        assert len(history) == 2  # the caller's list stays as it was
        assert worker.session_id == session.id == "s1"
        assert [msg.id for msg in session.messages[:2]] == [
            msg.id for msg in history
        ]
        # A copy of the worker has no session.
        assert worker.model_copy()._session is None


async def test_an_agent_context_keeps_the_worker() -> None:
    worker = workers.Codex(cwd="/w")
    context = ai.Context(model=worker, messages=[], tools=[])
    assert context.model is worker
    # The data of a context loads its model as an ai.Model: save a worker
    # with `worker.model_dump()`, not in a context.
    with pytest.raises(pydantic.ValidationError):
        ai.Context.model_validate(context.model_dump(mode="json"))


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
        async with worker.session():
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
