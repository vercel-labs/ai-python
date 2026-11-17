"""The external harness core against a scripted wire: calls and sessions."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pydantic
import pytest

import ai
from ai import external
from ai.external import _session, _wire
from ai.types import events
from ai.types import messages as m

from . import fakes


async def drain(feed: Any, until: type = events.StreamEnd) -> list[Any]:
    """Read events up to and including the first `until`."""
    seen: list[Any] = []
    async for event in feed:
        seen.append(event)
        if isinstance(event, until):
            break
    return seen


# -- the external harness as a model ------------------------------------------


async def test_a_harness_streams_like_a_model() -> None:
    harness = fakes.fake(fakes.echo)
    user = ai.user_message("hello")
    async with harness.session() as session:
        async with ai.stream(harness, [user]) as s:
            kinds = [type(e).__name__ async for e in s]
        assert s.text == "hello"
        assert kinds[0] == "StreamStart" and kinds[-1] == "StreamEnd"
        # The session's state and the stream agree, ids too.
        assert [msg.id for msg in session.messages] == [user.id, s.message.id]
        assert session.messages[1].text == "hello"
        assert session.id == "s1"


async def test_the_next_call_sends_only_the_new_messages() -> None:
    harness = fakes.fake(fakes.echo)
    first = ai.user_message("one")
    async with harness.session() as session:
        async with ai.stream(harness, [first]) as s:
            async for _ in s:
                pass
        second = ai.user_message("two")
        async with ai.stream(harness, [first, s.message, second]) as s2:
            async for _ in s2:
                pass
        assert [msg.text for msg in harness.wire.sent] == ["one", "two"]
        assert s2.text == "two"
        assert len(session.messages) == 4


async def test_a_rewritten_history_is_refused() -> None:
    harness = fakes.fake(fakes.echo)
    async with harness.session():
        async with ai.stream(harness, [ai.user_message("one")]) as s:
            async for _ in s:
                pass
        with pytest.raises(external.TranscriptMismatchError):
            async with ai.stream(
                harness, [ai.user_message("other"), ai.user_message("two")]
            ) as s:
                async for _ in s:
                    pass


async def test_a_fresh_harness_seeds_the_session_with_the_past() -> None:
    harness = fakes.fake(fakes.echo, instructions="Be brief.")
    history = [
        ai.system_message("You fix tests."),
        ai.user_message("one"),
        ai.assistant_message("done"),
        ai.user_message("two"),
    ]
    async with harness.session() as session:
        assert harness._wires == []  # the CLI starts with the first use
        async with ai.stream(harness, history, tools=[_tool("a")]) as s:
            async for _ in s:
                pass
        assert len(harness._wires) == 1  # it had the tools and the past
    started = harness.wire.started
    assert started is not None
    assert started["instructions"] == "Be brief.\n\nYou fix tests."
    assert [msg.text for msg in started["seed"]] == ["one", "done"]
    assert [msg.text for msg in harness.wire.sent] == ["two"]
    assert [msg.id for msg in session.messages[:4]] == [x.id for x in history]


async def test_a_resumed_session_does_not_resend_the_past() -> None:
    harness = fakes.fake(fakes.echo)
    history = [
        ai.user_message("one"),
        ai.assistant_message("done"),
        ai.user_message("two"),
    ]
    async with harness.session("old"):
        async with ai.stream(harness, history) as s:
            async for _ in s:
                pass
    assert harness.wire.started is not None
    assert harness.wire.started["seed"] == []  # the CLI has it
    assert harness.wire.started["session_id"] == "old"
    assert [msg.text for msg in harness.wire.sent] == ["two"]


async def test_a_started_empty_session_is_replaced_by_a_seeded_one() -> None:
    harness = fakes.fake(fakes.echo)
    history = [ai.system_message("s"), ai.user_message("x")]
    async with harness.session() as session:
        await session.start()
        async with ai.stream(harness, history) as st:
            async for _ in st:
                pass
        first, second = harness._wires
        assert first.closed and first.sent == []
        assert second.started is not None
        assert second.started["session_id"] is None  # a new session
        assert second.started["instructions"] == "s"
        assert [msg.text for msg in second.sent] == ["x"]


async def test_new_tools_restart_an_idle_cli_on_the_same_session() -> None:
    harness = fakes.fake(fakes.echo)
    async with harness.session() as session:
        await session.start()
        first = ai.user_message("one")
        async with ai.stream(harness, [first], tools=[_tool("a")]) as st:
            async for _ in st:
                pass
        assert len(harness._wires) == 2  # the first CLI saw no tools
        assert harness.wire.started is not None
        assert harness.wire.started["session_id"] is None  # it was unused
        again = [first, st.message, ai.user_message("two")]
        async with ai.stream(harness, again, tools=[_tool("a")]) as st:
            async for _ in st:
                pass
        assert len(harness._wires) == 2  # same tools: no restart
        more = [*session.messages, ai.user_message("three")]
        async with ai.stream(harness, more, tools=[_tool("b")]) as st:
            async for _ in st:
                pass
        assert len(harness._wires) == 3
        assert harness.wire.started is not None
        assert harness.wire.started["session_id"] == "s1"  # a used session


async def test_a_started_resumed_session_takes_its_known_past() -> None:
    harness = fakes.fake(fakes.echo)
    history = [
        ai.user_message("one"),
        ai.assistant_message("done"),
        ai.user_message("two"),
    ]
    async with harness.session("old") as session:
        await session.start()
        async with ai.stream(harness, history) as s:
            async for _ in s:
                pass
    assert [msg.text for msg in harness.wire.sent] == ["two"]


async def test_a_harness_cannot_continue_tool_calls_it_did_not_make() -> None:
    harness = fakes.fake(fakes.echo)
    call = m.ToolCallPart(tool_call_id="c", tool_name="t", tool_args="{}")
    history = [
        ai.user_message("go"),
        m.Message(role="assistant", parts=[call]),
        ai.tool_message(tool_call_id="c", result="r"),
    ]
    async with harness.session():
        with pytest.raises(external.TranscriptMismatchError):
            async with ai.stream(harness, history) as s:
                async for _ in s:
                    pass
        assert harness._wires == []  # refused before the CLI starts


async def test_a_model_call_needs_an_open_session() -> None:
    harness = fakes.fake(fakes.echo)
    first = ai.user_message("one")
    with pytest.raises(external.ExternalHarnessError, match="open session"):
        async with ai.stream(harness, [first]) as s:
            async for _ in s:
                pass
    assert harness._wires == []  # refused before the CLI starts
    async with harness.session() as session:
        async with ai.stream(harness, [first]) as s:
            async for _ in s:
                pass
    assert harness.wire.closed
    # A closed session is not an open one.
    more = [*session.messages, ai.user_message("two")]
    with pytest.raises(external.ExternalHarnessError, match="open session"):
        async with ai.stream(harness, more) as s:
            async for _ in s:
                pass
    assert len(harness._wires) == 1


async def test_a_saved_session_id_continues_the_session() -> None:
    harness = fakes.fake(fakes.echo)
    first = ai.user_message("one")
    async with harness.session() as session:
        async with ai.stream(harness, [first]) as s:
            async for _ in s:
                pass
    assert session.id == "s1"
    # The external harness and the id are plain data: another process
    # continues with them.
    again = external.ExternalHarness.model_validate(
        harness.model_dump(mode="json")
    )
    assert isinstance(again, fakes.FakeHarness)
    again._reply = fakes.echo
    async with again.session(session.id):
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
    # The external harness remembers no session: the next one is new.
    async with harness.session() as other:
        assert other.id is None
        async with ai.stream(harness, [ai.user_message("other")]) as s:
            async for _ in s:
                pass
    assert s.text == "other"
    assert harness.wire.started is not None
    assert harness.wire.started["session_id"] is None
    assert [msg.text for msg in other.messages] == ["other", "other"]


async def test_in_a_block_the_model_calls_use_the_live_cli() -> None:
    harness = fakes.fake(fakes.echo)
    first = ai.user_message("one")
    async with harness.session("abc") as session:
        assert session.id == "abc"
        async with ai.stream(harness, [first]) as s:
            async for _ in s:
                pass
        more = [first, s.message, ai.user_message("two")]
        async with ai.stream(harness, more) as s:
            async for _ in s:
                pass
        assert len(harness._wires) == 1 and not harness.wire.closed
        assert harness.wire.started is not None
        assert harness.wire.started["session_id"] == "abc"
        assert [msg.text for msg in harness.wire.sent] == ["one", "two"]
        # A copy is another external harness: it has no open session.
        twin = harness.model_copy()
        with pytest.raises(external.ExternalHarnessError, match="open session"):
            async with ai.stream(twin, [ai.user_message("x")]) as s:
                async for _ in s:
                    pass
    assert harness.wire.closed


async def test_two_tasks_open_sessions_of_one_harness() -> None:
    harness = fakes.fake(fakes.echo)

    async def chat(text: str) -> tuple[str, external.Session]:
        async with harness.session() as session:
            async with ai.stream(harness, [ai.user_message(text)]) as s:
                async for _ in s:
                    pass
            return s.text, session

    (one, a), (two, b) = await asyncio.gather(chat("one"), chat("two"))
    assert (one, two) == ("one", "two")
    assert len(harness._wires) == 2  # each task had its own live CLI
    assert [msg.text for msg in a.messages] == ["one", "one"]
    assert [msg.text for msg in b.messages] == ["two", "two"]


async def test_the_cli_stays_for_a_tool_loop() -> None:
    @ai.tool
    async def lookup(word: str) -> str:
        """Look a word up."""
        return word.upper()

    def reply(msg: m.Message) -> list[Any]:
        return [
            _wire.Accepted(msg),
            fakes.CallTool("lookup", {"word": "heron"}),
        ]

    harness = fakes.fake(reply)
    async with harness.session():
        async with ai.Agent(tools=[lookup]).run(
            harness, [ai.user_message("go")]
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
        assert len(harness._wires) == 1  # one CLI for the whole turn
        assert not harness.wire.closed
    assert harness.wire.closed  # it stops when the block ends


async def test_output_type_is_not_supported_yet() -> None:
    class Out(pydantic.BaseModel):
        answer: str

    harness = fakes.fake(fakes.echo)
    with pytest.raises(external.UnsupportedError):
        async with ai.stream(
            harness, [ai.user_message("x")], output_type=Out
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

    harness = fakes.fake(reply)
    async with harness.session():
        async with ai.stream(harness, [ai.user_message("ls")]) as s:
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

    harness = fakes.fake(reply)
    async with harness.session() as session:
        async with ai.stream(harness, [ai.user_message("x")]) as s:
            async for _ in s:
                pass
    assert s.message.provider_metadata == {"fake": {"model": "m1"}}
    assert session.messages[-1].provider_metadata == {"fake": {"model": "m1"}}


async def test_a_failed_turn_raises_from_stream() -> None:
    error = external.NotAuthenticatedError("fake", "401", "log in")

    def reply(msg: m.Message) -> list[Any]:
        return [_wire.Accepted(msg), _wire.TurnEnd(reason="error", error=error)]

    harness = fakes.fake(reply)
    history = [ai.user_message("x")]
    async with harness.session() as session:
        with pytest.raises(external.NotAuthenticatedError):
            async with ai.stream(harness, history) as s:
                async for _ in s:
                    pass
        # The state keeps the failed message, marked.
        assert session.messages[-1].provider_metadata == {
            "external": {"error": str(error)}
        }
        # The caller does not have it; the next call still works.
        harness.wire.reply = fakes.echo
        async with ai.stream(
            harness, [*history, ai.user_message("again")]
        ) as s:
            async for _ in s:
                pass
        assert s.text == "again"
        # Nothing new is refused in an open session.
        with pytest.raises(
            external.TranscriptMismatchError, match="nothing new"
        ):
            async with ai.stream(harness, list(session.messages)) as s:
                async for _ in s:
                    pass


async def test_one_stream_at_a_time() -> None:
    harness = fakes.fake()  # accepts, never answers
    async with harness.session():
        first = harness.provider.stream(harness, [ai.user_message("one")])
        task = asyncio.create_task(anext(first))
        await asyncio.sleep(0.01)
        with pytest.raises(external.ExternalHarnessError, match="one stream"):
            await anext(
                harness.provider.stream(harness, [ai.user_message("two")])
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

    harness = fakes.fake(reply)
    async with harness.session() as session:
        async with ai.Agent(tools=[lookup]).run(
            harness, [ai.user_message("go")]
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
        assert await harness.wire.calls[0] == ("HERON", False)
        assert [msg.id for msg in run.messages] == [
            msg.id for msg in session.messages
        ]
        assert harness.tools[0].name == "lookup"


async def test_parallel_calls_are_one_message() -> None:
    def reply(msg: m.Message) -> list[Any]:
        return [
            _wire.Accepted(msg),
            fakes.CallTool("a", then=lambda t, e: []),
            fakes.CallTool(
                "b", then=lambda t, e: [*fakes.text("both"), _wire.TurnEnd()]
            ),
        ]

    harness = fakes.fake(reply, tools=[_tool("a"), _tool("b")])
    async with harness.session() as session:
        async with ai.stream(harness, [ai.user_message("go")]) as s:
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
        async with ai.stream(harness, [*session.messages, results]) as s2:
            async for _ in s2:
                pass
        assert s2.text == "both"


async def test_cli_work_during_a_tool_call_comes_after_the_result() -> None:
    harness = fakes.fake(
        lambda msg: [
            _wire.Accepted(msg),
            fakes.CallTool("a", then=lambda t, e: [_wire.TurnEnd()]),
        ],
        tools=[_tool("a")],
    )
    async with harness.session() as session:
        async with ai.stream(harness, [ai.user_message("go")]) as s:
            async for _ in s:
                pass
        # The CLI keeps working on its own tools while it waits.
        harness.wire.push(*fakes.builtin("b1", "read"))
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
    harness = fakes.fake(
        lambda msg: [
            _wire.Accepted(msg),
            fakes.CallTool("a", then=lambda t, e: []),
        ]
        if msg.text == "go"
        else fakes.echo(msg),
        tools=[_tool("a")],
    )
    async with harness.session() as session:
        async with ai.stream(harness, [ai.user_message("go")]) as s:
            async for _ in s:
                pass
        # An agent was cancelled while its tool ran; the next run comes.
        async with ai.stream(
            harness, [*session.messages, ai.user_message("next")]
        ) as s2:
            async for _ in s2:
                pass
        roles = [msg.role for msg in session.messages]
        assert roles == ["user", "assistant", "tool", "user", "assistant"]
        synthetic = session.messages[2]
        assert synthetic.provider_metadata == {"external": {"synthetic": True}}
        assert synthetic.tool_results[0].result_kind == "error"
        assert await harness.wire.calls[0] == (_session.NEW_MESSAGE, True)


async def test_a_result_for_no_call_is_refused() -> None:
    harness = fakes.fake()
    async with harness.session() as session:
        with pytest.raises(ValueError, match="no tool call waits"):
            await session.send(ai.tool_message(tool_call_id="x", result="r"))


# -- the session: direct control ----------------------------------------------


async def test_events_after_send_lose_nothing() -> None:
    harness = fakes.fake(fakes.echo)
    async with harness.session() as session:
        await session.send(ai.user_message("hi"))
        await asyncio.sleep(0.01)  # the turn ends before anyone reads
        seen = await drain(session.events())
        assert isinstance(seen[0], external.MessageEvent)
        assert seen[0].message.text == "hi"
        assert isinstance(seen[1], events.StreamStart)
        # A reader that stopped needs no cleanup: the next one continues.
        rest = await drain(session.events(), until=external.MessageEvent)
        assert [type(e) for e in rest] == [external.MessageEvent]
        assert rest[0].message.role == "assistant"


async def test_events_that_nobody_reads_do_not_pile_up() -> None:
    harness = fakes.fake(fakes.echo)
    async with harness.session() as session:
        history: list[m.Message] = []
        sizes = []
        for text in ("one", "two", "three"):
            history = [*session.messages, ai.user_message(text)]
            async with ai.stream(harness, history) as s:
                async for _ in s:
                    pass
            sizes.append(len(session._pending_events))
        assert sizes[0] == sizes[1] == sizes[2]  # the latest turn only
        # A reader that reads keeps everything.
        seen = await drain(session.events())
        await session.send(ai.user_message("four"))
        rest = await drain(session.events(), until=external.MessageEvent)
        assert isinstance(seen[0], external.MessageEvent)
        assert rest[0].message.text == "three"  # the reply before "four"


async def test_a_steer_splits_the_reply_at_a_clean_point() -> None:
    harness = fakes.fake()
    async with harness.session() as session:
        feed = session.events()
        await session.send(ai.user_message("fix the tests"))
        harness.wire.push(
            events.TextStart(block_id="t1"),
            events.TextDelta(chunk="Look", block_id="t1"),
        )
        steer = ai.user_message("skip the slow ones")
        await session.send(steer)
        # The text block is open: the split waits for its end.
        harness.wire.push(
            events.TextDelta(chunk="ing.", block_id="t1"),
            events.TextEnd(block_id="t1"),
        )
        harness.wire.push(*fakes.text("Skipped.", block="t2"), _wire.TurnEnd())
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
    harness = fakes.fake()
    async with harness.session() as session:
        await session.send(ai.user_message("long job"))
        harness.wire.push(*fakes.text("Starting"))
        await asyncio.sleep(0.01)
        await session.interrupt()
        assert harness.wire.interrupts == 1
        reply = session.messages[-1]
        assert reply.text == "Starting"
        # Not a finish reason of the vocabulary: the raw value is here.
        assert reply.provider_metadata == {
            "external": {"stop_reason": "interrupted"}
        }
        await session.interrupt()  # no turn: nothing happens
        assert harness.wire.interrupts == 1


async def test_an_ignored_interrupt_stops_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_session, "INTERRUPT_TIMEOUT", 0.05)
    harness = fakes.fake()
    async with harness.session() as session:
        await session.start()
        harness.wire.on_interrupt = None
        feed = session.events()
        await session.send(ai.user_message("long job"))
        harness.wire.push(*fakes.text("Starting"))
        await asyncio.sleep(0.01)
        await session.interrupt()
        assert session._run is None
        assert harness.wire.closed
        assert session.messages[-1].provider_metadata == {
            "external": {"stop_reason": "interrupted"}
        }
        assert (await drain(feed))[-1].finish_reason == "other"
        # The session stays open: the next use continues the CLI session.
        harness._reply = fakes.echo
        await session.send(ai.user_message("again"))
        assert len(harness._wires) == 2
        assert harness.wire.started is not None
        assert harness.wire.started["session_id"] == "s1"


async def test_a_cancelled_stream_interrupts_the_turn() -> None:
    harness = fakes.fake(lambda msg: [_wire.Accepted(msg), *fakes.text("work")])
    history = [ai.user_message("long job")]
    async with harness.session() as session:
        async with ai.stream(harness, history) as s:
            async for event in s:
                if isinstance(event, events.TextEnd):
                    break  # stop reading: the turn must stop too
        assert harness.wire.interrupts == 1
        assert session.messages[-1].provider_metadata == {
            "external": {"stop_reason": "interrupted"}
        }
        # ai.Agent keeps no cancelled message; the next call still works.
        harness.wire.reply = fakes.echo
        async with ai.stream(
            harness, [*history, ai.user_message("again")]
        ) as s:
            async for _ in s:
                pass
        assert s.text == "again"


async def test_a_cancelled_task_interrupts_the_turn() -> None:
    harness = fakes.fake(lambda msg: [_wire.Accepted(msg), *fakes.text("work")])

    async def consume() -> None:
        async with ai.stream(harness, [ai.user_message("long job")]) as s:
            async for _ in s:
                pass

    async with harness.session() as session:
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.02)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert harness.wire.interrupts == 1
        assert session.messages[-1].provider_metadata == {
            "external": {"stop_reason": "interrupted"}
        }


async def test_a_crash_reaches_the_reader_and_start_recovers() -> None:
    harness = fakes.fake()
    async with harness.session() as session:
        feed = session.events()
        await session.send(ai.user_message("go"))
        harness.wire.push(*fakes.text("half"), RuntimeError("boom"))
        with pytest.raises(external.ExternalHarnessError, match="boom"):
            await drain(feed, until=type(None))
        with pytest.raises(external.ExternalHarnessError, match="stopped"):
            await session.send(ai.user_message("again"))
        assert session.messages[-1].provider_metadata is not None
        await session.start()
        assert len(harness._wires) == 2


async def test_close_ends_readers_and_keeps_the_state() -> None:
    harness = fakes.fake(fakes.echo)
    session = harness.session()
    async with session:
        feed = session.events()
        await session.send(ai.user_message("hi"))
        await drain(feed)
    # The reply's MessageEvent follows its StreamEnd; then the feed ends.
    rest = [e async for e in feed]
    assert [type(e) for e in rest] == [external.MessageEvent]
    assert harness.wire.closed
    assert session.id == "s1"
    assert [msg.text for msg in session.messages] == ["hi", "hi"]
    await session.close()  # a second close does nothing
    with pytest.raises(external.ExternalHarnessError, match="not open"):
        await session.send(ai.user_message("x"))
    with pytest.raises(external.ExternalHarnessError, match="not open"):
        await anext(session.events())


async def test_start_needs_an_open_session() -> None:
    harness = fakes.fake(fakes.echo)
    session = harness.session()
    with pytest.raises(external.ExternalHarnessError, match="not open"):
        await session.start()  # the block is the only way to open
    assert harness._wires == []
    async with session:
        assert harness._wires == []
        await session.start()  # the CLI starts now, not with the first use
        assert len(harness._wires) == 1
        await session.send(ai.user_message("hi"))
        await drain(session.events())
    assert harness.wire.closed
    async with session:  # open again: the same CLI session
        await session.start()
        assert harness.wire.started is not None
        assert harness.wire.started["session_id"] == "s1"


async def test_a_session_is_open_once_in_a_context() -> None:
    one, two = fakes.fake(fakes.echo), fakes.fake(fakes.echo)
    async with one.session("abc"):
        # Two external harnesses, one CLI session: two writers.
        with pytest.raises(external.ExternalHarnessError, match="open already"):
            async with two.session("abc"):
                pass
    async with two.session("abc") as session:  # the first one closed
        await session.send(ai.user_message("hi"))


async def test_sessions_of_one_harness_do_not_nest() -> None:
    harness = fakes.fake(fakes.echo)
    async with harness.session() as session:
        with pytest.raises(external.ExternalHarnessError, match="open already"):
            async with session:
                pass
        with pytest.raises(external.ExternalHarnessError, match="do not nest"):
            async with harness.session():
                pass
        with pytest.raises(external.ExternalHarnessError, match="do not nest"):
            async with harness.session("other"):
                pass
        # The first one is still the open session.
        async with ai.stream(harness, [ai.user_message("hi")]) as s:
            async for _ in s:
                pass
        assert [msg.text for msg in session.messages] == ["hi", "hi"]
    async with harness.session():  # the first one closed
        pass


async def test_the_feed_stays_when_new_tools_restart_the_cli() -> None:
    harness = fakes.fake(fakes.echo)
    async with harness.session() as session:
        await session.start()
        feed = session.events()
        async with ai.stream(
            harness, [ai.user_message("one")], tools=[_tool("a")]
        ) as st:
            async for _ in st:
                pass
        assert len(harness._wires) == 2
        seen = await drain(feed)
        assert isinstance(seen[0], external.MessageEvent)
        assert seen[0].message.text == "one"


# -- state ---------------------------------------------------------------------


async def test_a_harness_is_plain_data() -> None:
    harness = external.Codex(
        cwd="/w",
        model_name="gpt-5.5",
        sandbox="read-only",
        provider=external.CodexProvider(
            executable="/bin/codex", env={"K": "v"}
        ),
    )
    data = harness.model_dump(mode="json")
    # Settings only: the session has the id and the conversation.
    assert "session_id" not in data and "messages" not in data
    again = external.ExternalHarness.model_validate(data)
    assert type(again) is external.Codex
    assert again.model_dump(mode="json") == data
    assert again.provider == harness.provider
    assert again.session().id is None
    assert again.session("abc").id == "abc"
    with pytest.raises(ValueError, match="unknown harness_class_id"):
        external.ExternalHarness.model_validate(
            {**data, "harness_class_id": "nope"}
        )
    with pytest.raises(TypeError):
        external.ExternalHarness(harness_class_id="codex")


async def test_a_seeded_session_starts_with_its_past() -> None:
    harness = fakes.fake(fakes.echo)
    history = [ai.user_message("one"), ai.assistant_message("done")]
    async with harness.session(messages=history) as session:
        await session.send(ai.user_message("two"))
        await drain(session.events())
        assert harness.wire.started is not None
        assert harness.wire.started["session_id"] is None
        assert [msg.text for msg in harness.wire.started["seed"]] == [
            "one",
            "done",
        ]
        assert len(history) == 2  # the caller's list stays as it was
        assert session.id == "s1"
        assert [msg.id for msg in session.messages[:2]] == [
            msg.id for msg in history
        ]


async def test_an_agent_context_keeps_the_harness() -> None:
    harness = external.Codex(cwd="/w")
    context = ai.Context(model=harness, messages=[], tools=[])
    assert context.model is harness
    # The data of a context loads its model as an ai.Model: save an external
    # harness with `harness.model_dump()`, not in a context.
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
        harness = fakes.fake(reply)
        async with harness.session():
            async with ai.stream(harness, [ai.user_message("go")]) as s:
                async for _ in s:
                    pass
    by_kind = {span.data.kind: span for span in sink.finished_spans}
    assert {
        "external_session",
        "external_message",
        "tool_execution",
        "ai_stream",
    } <= set(by_kind)
    session, message, tool = (
        by_kind["external_session"],
        by_kind["external_message"],
        by_kind["tool_execution"],
    )
    assert isinstance(session.data, external.ExternalHarnessSessionSpanData)
    assert isinstance(message.data, external.ExternalHarnessMessageSpanData)
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
