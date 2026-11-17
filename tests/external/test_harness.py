"""The external harness core against a scripted wire: calls and sessions."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import pickle
from collections.abc import Callable
from typing import Any

import pydantic
import pydantic_core
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
    async with harness:
        async with ai.stream(harness, [user]) as s:
            kinds = [type(e).__name__ async for e in s]
        assert s.text == "hello"
        assert kinds[0] == "StreamStart" and kinds[-1] == "StreamEnd"
        # The session's state and the stream agree, ids too.
        assert [msg.id for msg in harness.session.messages] == [
            user.id,
            s.message.id,
        ]
        assert harness.session.messages[1].text == "hello"
        assert harness.session.id == "s1"


async def test_the_next_call_sends_only_the_new_messages() -> None:
    harness = fakes.fake(fakes.echo)
    first = ai.user_message("one")
    async with harness:
        async with ai.stream(harness, [first]) as s:
            async for _ in s:
                pass
        second = ai.user_message("two")
        async with ai.stream(harness, [first, s.message, second]) as s2:
            async for _ in s2:
                pass
        assert [msg.text for msg in harness.wire.sent] == ["one", "two"]
        assert s2.text == "two"
        assert len(harness.session.messages) == 4


async def test_a_rewritten_history_is_refused() -> None:
    harness = fakes.fake(fakes.echo)
    async with harness:
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
    async with harness:
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
    assert [msg.id for msg in harness.session.messages[:4]] == [
        x.id for x in history
    ]


async def test_a_resumed_session_does_not_resend_the_past() -> None:
    harness = fakes.fake(fakes.echo, resume="old")
    history = [
        ai.user_message("one"),
        ai.assistant_message("done"),
        ai.user_message("two"),
    ]
    async with harness:
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
    async with harness:
        await harness.session.start()
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
    async with harness:
        await harness.session.start()
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
        more = [*harness.session.messages, ai.user_message("three")]
        async with ai.stream(harness, more, tools=[_tool("b")]) as st:
            async for _ in st:
                pass
        assert len(harness._wires) == 3
        assert harness.wire.started is not None
        assert harness.wire.started["session_id"] == "s1"  # a used session


async def test_a_started_resumed_session_takes_its_known_past() -> None:
    harness = fakes.fake(fakes.echo, resume="old")
    history = [
        ai.user_message("one"),
        ai.assistant_message("done"),
        ai.user_message("two"),
    ]
    async with harness:
        await harness.session.start()
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
    async with harness:
        with pytest.raises(external.TranscriptMismatchError):
            async with ai.stream(harness, history) as s:
                async for _ in s:
                    pass
        assert harness._wires == []  # refused before the CLI starts


async def test_a_model_call_needs_an_open_harness() -> None:
    harness = fakes.fake(fakes.echo)
    first = ai.user_message("one")
    with pytest.raises(
        external.ExternalHarnessError, match="open external harness"
    ):
        async with ai.stream(harness, [first]) as s:
            async for _ in s:
                pass
    assert harness._wires == []  # refused before the CLI starts
    async with harness:
        async with ai.stream(harness, [first]) as s:
            async for _ in s:
                pass
    assert harness.wire.closed
    # A closed external harness is not an open one.
    more = [*harness.session.messages, ai.user_message("two")]
    with pytest.raises(
        external.ExternalHarnessError, match="open external harness"
    ):
        async with ai.stream(harness, more) as s:
            async for _ in s:
                pass
    assert len(harness._wires) == 1


async def test_a_saved_session_id_continues_the_session() -> None:
    harness = fakes.fake(fakes.echo)
    first = ai.user_message("one")
    async with harness:
        async with ai.stream(harness, [first]) as s:
            async for _ in s:
                pass
    assert harness.session.id == "s1"
    # The id is plain data: a new external harness continues with it.
    again = fakes.fake(fakes.echo, resume=harness.session.id)
    async with again:
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
    # An external harness that opens again continues its session.
    async with harness:
        assert harness.session.id == "s1"
        more = [*harness.session.messages, ai.user_message("other")]
        async with ai.stream(harness, more) as s:
            async for _ in s:
                pass
    assert s.text == "other"
    assert harness.wire.started is not None
    assert harness.wire.started["session_id"] == "s1"
    # The same settings with a new session: a new conversation.
    fresh = dataclasses.replace(harness)
    assert fresh.session.id is None
    assert fresh.session.messages == []


async def test_in_a_block_the_model_calls_use_the_live_cli() -> None:
    harness = fakes.fake(fakes.echo, resume="abc")
    first = ai.user_message("one")
    async with harness:
        assert harness.session.id == "abc"
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
        # A copy is another external harness, and it is closed.
        twin = copy.copy(harness)
        with pytest.raises(
            external.ExternalHarnessError, match="open external harness"
        ):
            async with ai.stream(twin, [ai.user_message("x")]) as s:
                async for _ in s:
                    pass
    assert harness.wire.closed


async def test_two_conversations_are_two_harnesses() -> None:
    async def chat(text: str) -> tuple[str, fakes.FakeHarness]:
        async with fakes.fake(fakes.echo) as harness:
            async with ai.stream(harness, [ai.user_message(text)]) as s:
                async for _ in s:
                    pass
            return s.text, harness

    (one, a), (two, b) = await asyncio.gather(chat("one"), chat("two"))
    assert (one, two) == ("one", "two")
    assert len(a._wires) == len(b._wires) == 1  # each had its own live CLI
    assert [msg.text for msg in a.session.messages] == ["one", "one"]
    assert [msg.text for msg in b.session.messages] == ["two", "two"]


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
    async with harness:
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
    async with harness:
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
    async with harness:
        async with ai.stream(harness, [ai.user_message("x")]) as s:
            async for _ in s:
                pass
    assert s.message.provider_metadata == {"fake": {"model": "m1"}}
    assert harness.session.messages[-1].provider_metadata == {
        "fake": {"model": "m1"}
    }


async def test_a_failed_turn_raises_from_stream() -> None:
    error = external.NotAuthenticatedError("fake", "401", "log in")

    def reply(msg: m.Message) -> list[Any]:
        return [_wire.Accepted(msg), _wire.TurnEnd(reason="error", error=error)]

    harness = fakes.fake(reply)
    history = [ai.user_message("x")]
    async with harness:
        with pytest.raises(external.NotAuthenticatedError):
            async with ai.stream(harness, history) as s:
                async for _ in s:
                    pass
        # The state keeps the failed message, marked.
        assert harness.session.messages[-1].provider_metadata == {
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
        # Nothing new is refused in an open external harness.
        with pytest.raises(
            external.TranscriptMismatchError, match="nothing new"
        ):
            async with ai.stream(harness, list(harness.session.messages)) as s:
                async for _ in s:
                    pass


async def test_one_stream_at_a_time() -> None:
    harness = fakes.fake()  # accepts, never answers
    async with harness:
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
    async with harness:
        async with ai.Agent(tools=[lookup]).run(
            harness, [ai.user_message("go")]
        ) as run:
            async for _ in run:
                pass
        roles = [msg.role for msg in harness.session.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        (call,) = harness.session.messages[1].tool_calls
        assert call.tool_name == "lookup"
        assert harness.session.messages[1].provider_metadata is None
        assert harness.session.messages[3].text == "got HERON"
        # The CLI got the result through MCP.
        assert await harness.wire.calls[0] == ("HERON", False)
        assert [msg.id for msg in run.messages] == [
            msg.id for msg in harness.session.messages
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
    async with harness:
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
        async with ai.stream(
            harness, [*harness.session.messages, results]
        ) as s2:
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
    async with harness:
        async with ai.stream(harness, [ai.user_message("go")]) as s:
            async for _ in s:
                pass
        # The CLI keeps working on its own tools while it waits.
        harness.wire.push(*fakes.builtin("b1", "read"))
        await asyncio.sleep(0.05)
        assert [msg.role for msg in harness.session.messages] == [
            "user",
            "assistant",
        ]
        (call,) = s.message.tool_calls
        await harness.session.send(
            ai.tool_message(
                tool_call_id=call.tool_call_id, tool_name="a", result="r"
            )
        )
        await asyncio.sleep(0.05)
        roles = [msg.role for msg in harness.session.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert (
            harness.session.messages[3].builtin_tool_calls[0].tool_name
            == "read"
        )


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
    async with harness:
        async with ai.stream(harness, [ai.user_message("go")]) as s:
            async for _ in s:
                pass
        # An agent was cancelled while its tool ran; the next run comes.
        async with ai.stream(
            harness, [*harness.session.messages, ai.user_message("next")]
        ) as s2:
            async for _ in s2:
                pass
        roles = [msg.role for msg in harness.session.messages]
        assert roles == ["user", "assistant", "tool", "user", "assistant"]
        synthetic = harness.session.messages[2]
        assert synthetic.provider_metadata == {"external": {"synthetic": True}}
        assert synthetic.tool_results[0].result_kind == "error"
        assert await harness.wire.calls[0] == (_session.NEW_MESSAGE, True)


async def test_a_result_for_no_call_is_refused() -> None:
    harness = fakes.fake()
    async with harness:
        with pytest.raises(ValueError, match="no tool call waits"):
            await harness.session.send(
                ai.tool_message(tool_call_id="x", result="r")
            )


# -- the session: direct control ----------------------------------------------


async def test_events_after_send_lose_nothing() -> None:
    harness = fakes.fake(fakes.echo)
    async with harness:
        await harness.session.send(ai.user_message("hi"))
        await asyncio.sleep(0.01)  # the turn ends before anyone reads
        seen = await drain(harness.session.events())
        assert isinstance(seen[0], external.MessageEvent)
        assert seen[0].message.text == "hi"
        assert isinstance(seen[1], events.StreamStart)
        # A reader that stopped needs no cleanup: the next one continues.
        rest = await drain(
            harness.session.events(), until=external.MessageEvent
        )
        assert [type(e) for e in rest] == [external.MessageEvent]
        assert rest[0].message.role == "assistant"


async def test_events_that_nobody_reads_do_not_pile_up() -> None:
    harness = fakes.fake(fakes.echo)
    async with harness:
        history: list[m.Message] = []
        sizes = []
        for text in ("one", "two", "three"):
            history = [*harness.session.messages, ai.user_message(text)]
            async with ai.stream(harness, history) as s:
                async for _ in s:
                    pass
            sizes.append(len(harness.session._pending_events))
        assert sizes[0] == sizes[1] == sizes[2]  # the latest turn only
        # A reader that reads keeps everything.
        seen = await drain(harness.session.events())
        await harness.session.send(ai.user_message("four"))
        rest = await drain(
            harness.session.events(), until=external.MessageEvent
        )
        assert isinstance(seen[0], external.MessageEvent)
        assert rest[0].message.text == "three"  # the reply before "four"


async def test_a_steer_splits_the_reply_at_a_clean_point() -> None:
    harness = fakes.fake()
    async with harness:
        feed = harness.session.events()
        await harness.session.send(ai.user_message("fix the tests"))
        harness.wire.push(
            events.TextStart(block_id="t1"),
            events.TextDelta(chunk="Look", block_id="t1"),
        )
        steer = ai.user_message("skip the slow ones")
        await harness.session.send(steer)
        # The text block is open: the split waits for its end.
        harness.wire.push(
            events.TextDelta(chunk="ing.", block_id="t1"),
            events.TextEnd(block_id="t1"),
        )
        harness.wire.push(*fakes.text("Skipped.", block="t2"), _wire.TurnEnd())
        await drain(feed)  # the first part ends
        await drain(feed)  # the second part ends
        texts = [(msg.role, msg.text) for msg in harness.session.messages]
        assert texts == [
            ("user", "fix the tests"),
            ("assistant", "Looking."),
            ("user", "skip the slow ones"),
            ("assistant", "Skipped."),
        ]


async def test_interrupt_stops_the_turn_and_marks_the_reply() -> None:
    harness = fakes.fake()
    async with harness:
        await harness.session.send(ai.user_message("long job"))
        harness.wire.push(*fakes.text("Starting"))
        await asyncio.sleep(0.01)
        await harness.session.interrupt()
        assert harness.wire.interrupts == 1
        reply = harness.session.messages[-1]
        assert reply.text == "Starting"
        # Not a finish reason of the vocabulary: the raw value is here.
        assert reply.provider_metadata == {
            "external": {"stop_reason": "interrupted"}
        }
        await harness.session.interrupt()  # no turn: nothing happens
        assert harness.wire.interrupts == 1


async def test_an_ignored_interrupt_stops_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_session, "INTERRUPT_TIMEOUT", 0.05)
    harness = fakes.fake()
    async with harness:
        await harness.session.start()
        harness.wire.on_interrupt = None
        feed = harness.session.events()
        await harness.session.send(ai.user_message("long job"))
        harness.wire.push(*fakes.text("Starting"))
        await asyncio.sleep(0.01)
        await harness.session.interrupt()
        assert harness.session._run is None
        assert harness.wire.closed
        assert harness.session.messages[-1].provider_metadata == {
            "external": {"stop_reason": "interrupted"}
        }
        assert (await drain(feed))[-1].finish_reason == "other"
        # The session stays open: the next use continues the CLI session.
        harness._reply = fakes.echo
        await harness.session.send(ai.user_message("again"))
        assert len(harness._wires) == 2
        assert harness.wire.started is not None
        assert harness.wire.started["session_id"] == "s1"


async def test_a_cancelled_stream_interrupts_the_turn() -> None:
    harness = fakes.fake(lambda msg: [_wire.Accepted(msg), *fakes.text("work")])
    history = [ai.user_message("long job")]
    async with harness:
        async with ai.stream(harness, history) as s:
            async for event in s:
                if isinstance(event, events.TextEnd):
                    break  # stop reading: the turn must stop too
        assert harness.wire.interrupts == 1
        assert harness.session.messages[-1].provider_metadata == {
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

    async with harness:
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.02)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert harness.wire.interrupts == 1
        assert harness.session.messages[-1].provider_metadata == {
            "external": {"stop_reason": "interrupted"}
        }


async def test_a_crash_reaches_the_reader_and_start_recovers() -> None:
    harness = fakes.fake()
    async with harness:
        feed = harness.session.events()
        await harness.session.send(ai.user_message("go"))
        harness.wire.push(*fakes.text("half"), RuntimeError("boom"))
        with pytest.raises(external.ExternalHarnessError, match="boom"):
            await drain(feed, until=type(None))
        with pytest.raises(external.ExternalHarnessError, match="stopped"):
            await harness.session.send(ai.user_message("again"))
        assert harness.session.messages[-1].provider_metadata is not None
        await harness.session.start()
        assert len(harness._wires) == 2


async def test_close_ends_readers_and_keeps_the_state() -> None:
    harness = fakes.fake(fakes.echo)
    session = harness.session
    async with harness:
        feed = session.events()
        await session.send(ai.user_message("hi"))
        await drain(feed)
    # The reply's MessageEvent follows its StreamEnd; then the feed ends.
    rest = [e async for e in feed]
    assert [type(e) for e in rest] == [external.MessageEvent]
    assert harness.wire.closed
    assert session.id == "s1"
    assert [msg.text for msg in session.messages] == ["hi", "hi"]
    with pytest.raises(external.ExternalHarnessError, match="not open"):
        await session.send(ai.user_message("x"))
    with pytest.raises(external.ExternalHarnessError, match="not open"):
        await anext(session.events())


async def test_start_needs_an_open_harness() -> None:
    harness = fakes.fake(fakes.echo)
    session = harness.session
    with pytest.raises(external.ExternalHarnessError, match="not open"):
        await session.start()  # the block is the only way to open
    assert harness._wires == []
    async with harness:
        assert harness._wires == []
        await session.start()  # the CLI starts now, not with the first use
        assert len(harness._wires) == 1
        await session.send(ai.user_message("hi"))
        await drain(session.events())
    assert harness.wire.closed
    async with harness:  # open again: the same CLI session
        await session.start()
        assert harness.wire.started is not None
        assert harness.wire.started["session_id"] == "s1"


async def test_a_session_is_open_once() -> None:
    one = fakes.fake(fakes.echo, resume="abc")
    two = fakes.fake(fakes.echo, resume="abc")
    async with one:
        # Two external harnesses, one CLI session: two writers.
        with pytest.raises(external.ExternalHarnessError, match="open already"):
            async with two:
                pass
    async with two:  # the first one closed
        await two.session.send(ai.user_message("hi"))


async def test_a_harness_does_not_open_twice() -> None:
    harness = fakes.fake(fakes.echo)
    async with harness:
        with pytest.raises(external.ExternalHarnessError, match="open already"):
            async with harness:
                pass
        # It is still open.
        async with ai.stream(harness, [ai.user_message("hi")]) as s:
            async for _ in s:
                pass
        assert [msg.text for msg in harness.session.messages] == ["hi", "hi"]
    async with harness:  # it closed: it opens again
        pass


async def test_the_feed_stays_when_new_tools_restart_the_cli() -> None:
    harness = fakes.fake(fakes.echo)
    async with harness:
        await harness.session.start()
        feed = harness.session.events()
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


async def test_a_harness_is_settings_and_a_session() -> None:
    harness = external.Codex(
        cwd="/w",
        model_name="gpt-5.5",
        sandbox="read-only",
        provider=external.CodexProvider(
            executable="/bin/codex", env={"K": "v"}
        ),
    )
    # The fields are the settings: the session has the id and the
    # conversation, and no field keeps `resume` or `history`.
    fields = {field.name for field in dataclasses.fields(harness)}
    assert not {"resume", "history", "session_id", "messages"} & fields
    assert harness.id == "gpt-5.5"
    assert harness.session.harness is harness
    assert harness.session.id is None
    assert harness.session.messages == []
    history = [ai.user_message("one")]
    resumed = external.Codex(cwd="/w", resume="abc", history=history)
    assert resumed.session.id == "abc"
    assert [msg.id for msg in resumed.session.messages] == [history[0].id]
    # The same settings, a new session.
    fresh = dataclasses.replace(resumed, model_name="m")
    assert (fresh.cwd, fresh.model_name) == ("/w", "m")
    assert fresh.session.id is None
    assert fresh.session.messages == []
    with pytest.raises(TypeError):
        external.ExternalHarness(provider=external.CodexProvider())
    with pytest.raises(TypeError):
        external.Codex(cwdd="/w")  # type: ignore[call-arg]  # ty: ignore[unknown-argument]


@pytest.mark.parametrize(
    "harness_class",
    [external.ClaudeCode, external.Codex, external.Fx, external.Pi],
)
def test_a_copy_or_a_pickle_has_the_settings_and_the_conversation(
    harness_class: Callable[..., external.ExternalHarness],
) -> None:
    history = [ai.user_message("one")]
    harness = harness_class(
        cwd="/w", model_name="m", resume="abc", history=history
    )
    for again in (
        copy.copy(harness),
        copy.deepcopy(harness),
        pickle.loads(pickle.dumps(harness)),
    ):
        assert type(again) is harness_class
        assert (again.cwd, again.model_name) == ("/w", "m")
        assert again.provider == harness.provider
        assert again.session is not harness.session
        assert again.session.harness is again
        assert again.session.id == "abc"
        assert [msg.id for msg in again.session.messages] == [history[0].id]


async def test_an_open_harness_copies_and_pickles_as_a_closed_one() -> None:
    async with external.Codex(cwd="/w") as harness:  # no CLI starts
        for again in (copy.copy(harness), pickle.loads(pickle.dumps(harness))):
            with pytest.raises(external.ExternalHarnessError, match="not open"):
                await again.session.start()


async def test_a_seeded_session_starts_with_its_past() -> None:
    history = [ai.user_message("one"), ai.assistant_message("done")]
    harness = fakes.fake(fakes.echo, history=history)
    async with harness:
        await harness.session.send(ai.user_message("two"))
        await drain(harness.session.events())
        assert harness.wire.started is not None
        assert harness.wire.started["session_id"] is None
        assert [msg.text for msg in harness.wire.started["seed"]] == [
            "one",
            "done",
        ]
        assert len(history) == 2  # the caller's list stays as it was
        assert harness.session.id == "s1"
        assert [msg.id for msg in harness.session.messages[:2]] == [
            msg.id for msg in history
        ]


async def test_an_agent_context_keeps_the_harness() -> None:
    harness = external.Codex(cwd="/w")
    context = ai.Context(model=harness, messages=[], tools=[])
    assert context.model is harness
    # An external harness is not plain data: a context with one does not
    # dump.
    with pytest.raises(pydantic_core.PydanticSerializationError):
        context.model_dump(mode="json")


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
        async with harness:
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
