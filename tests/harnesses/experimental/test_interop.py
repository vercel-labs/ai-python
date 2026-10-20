"""A harness turn in the rest of `ai`: the UI adapter, aggregators, agents.

A harness is an agent, so its stream is a RUN, as coarse as its CLI
reports: one `StreamStart ... StreamEnd` around the whole turn, tool calls
and results inside. These pin that every `ai` consumer takes it as it is.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import ai
from ai import util
from ai.harnesses.experimental import _approval
from ai.types import events
from ai.types import messages as m
from ai.ui.ai_sdk import outbound_stream

from . import fakes


def _tool_turn(prompt: str) -> list[fakes.Step]:
    return [
        events.StreamStart(),
        *fakes.text("I'll write it.", block="a"),
        *fakes.tool("c1", "Write", '{"path": "x"}'),
        *fakes.text("Done.", block="b"),
    ]


def _ui_kinds(ui_events: list[Any]) -> list[str]:
    return [
        json.loads(outbound_stream.serialize_event(e))["type"]
        for e in ui_events
    ]


async def test_the_ui_adapter_renders_a_turn_as_one_step() -> None:
    harness, _ = await fakes.open_harness(fakes.FakeAdapter(_tool_turn))
    turn = harness.session().stream("go")
    kinds = _ui_kinds([e async for e in ai.ui.ai_sdk.to_stream(turn)])
    assert kinds.count("start-step") == kinds.count("finish-step") == 1
    assert "tool-input-start" in kinds
    assert "tool-output-available" in kinds
    assert kinds.count("text-start") == 2


async def test_the_ui_adapter_does_not_offer_to_answer_a_harness_approval() -> (
    None
):
    """The `approve=` hook answers it, not `ai.resolve_hook`."""

    async def approve(ctx: _approval.ApprovalContext) -> _approval.Decision:
        return _approval.Allow()

    adapter = fakes.AskingAdapter()
    adapter.script = lambda prompt: [
        events.StreamStart(),
        adapter.ask(),
        *fakes.text(prompt),
    ]
    harness, _ = await fakes.open_harness(adapter, approve=approve)
    turn = harness.session().stream("go")
    kinds = _ui_kinds([e async for e in ai.ui.ai_sdk.to_stream(turn)])
    assert not [k for k in kinds if "approval" in k]


async def test_a_message_aggregator_collects_a_turn() -> None:
    harness, _ = await fakes.open_harness(fakes.FakeAdapter(_tool_turn))
    aggregator = ai.agents.MessageAggregator()
    async for event in harness.session().stream("go"):
        aggregator.feed(event)
    bundle = aggregator.snapshot()
    assert [msg.role for msg in bundle.messages] == [
        "assistant",
        "tool",
        "assistant",
    ]
    assert aggregator.get_model_input() == "Done."


async def test_a_turn_merges_with_other_streams() -> None:
    harness, _ = await fakes.open_harness(fakes.FakeAdapter(_tool_turn))
    turn = harness.session().stream("go")

    async def ticks() -> Any:
        for i in range(3):
            yield i

    merged = [e async for e in util.merge(turn, ticks())]
    assert sum(isinstance(e, events.StreamEnd) for e in merged) == 1
    assert sum(isinstance(e, int) for e in merged) == 3
    assert turn.result.text == "Done."


# -- a harness as a tool of an `ai.Agent` -------------------------------------


async def test_a_harness_is_a_tool_of_an_agent() -> None:
    harness, _ = await fakes.open_harness(fakes.FakeAdapter(_tool_turn))
    session = harness.session()

    @ai.tool
    async def code(task: str) -> ai.SubAgentTool:
        """Hand a coding task to the coding agent."""
        async with session.stream(task) as turn:
            async for event in turn:
                yield event

    model = ai.testing.FakeModel(
        [
            ai.user_message("fix the bug"),
            ai.assistant_message(ai.testing.tool_call(code, task="fix it")),
            ai.assistant_message("Fixed."),
        ]
    )
    seen: list[str] = []
    async with ai.Agent(tools=[code]).run(
        model, [ai.user_message("fix the bug")]
    ) as stream:
        async for event in stream:
            if isinstance(event, events.PartialToolCallResult):
                seen.append(type(event.value).__name__)
    assert not model.unused
    assert "ToolCallResult" in seen, "the harness's own tool calls stream up"
    [result] = [p for msg in stream.messages for p in msg.tool_results]
    assert result.model_input == "Done.", "the parent model sees the answer"
    assert stream.output == "Fixed."
    assert session.messages[-1].text == "Done."


async def test_closing_the_agent_run_stops_the_harness_turn() -> None:
    pause = fakes.Pause()

    def script(prompt: str) -> list[fakes.Step]:
        if prompt != "go":
            return fakes.reply(prompt)
        return [events.StreamStart(), *fakes.text("working"), pause]

    harness, adapter = await fakes.open_harness(fakes.FakeAdapter(script))
    session = harness.session()

    @ai.tool
    async def code(task: str) -> ai.SubAgentTool:
        """Hand a coding task to the coding agent."""
        async with session.stream(task) as turn:
            async for event in turn:
                yield event

    model = ai.testing.FakeModel(
        [
            ai.user_message("go"),
            ai.assistant_message(ai.testing.tool_call(code, task="go")),
        ]
    )
    async with ai.Agent(tools=[code]).run(
        model, [ai.user_message("go")]
    ) as stream:
        async for event in stream:
            if isinstance(event, events.PartialToolCallResult) and isinstance(
                event.value, events.TextDelta
            ):
                break
    await asyncio.wait_for(pause.reached.wait(), 1)
    assert adapter.stops == [session.session_id], "the CLI's turn was ended"
    assert (await session.run("next")).text == "next"
    assert [msg.role for msg in session.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert isinstance(session.messages[1].parts[0], m.TextPart)


# -- combining turns and customizing the loop (docs: "Combine and customize")


async def test_turns_of_two_sessions_merge_into_one_stream() -> None:
    harness, _ = await fakes.open_harness(fakes.FakeAdapter(_tool_turn))

    async def tagged(label: str, turn: Any) -> Any:
        async with turn:
            async for event in turn:
                yield label, event

    a = harness.session().stream("a")
    b = harness.session().stream("b")
    seen = [
        label
        async for label, event in util.merge(tagged("a", a), tagged("b", b))
        if isinstance(event, events.StreamEnd)
    ]
    assert sorted(seen) == ["a", "b"]
    assert a.output == b.output == "Done."


async def test_turns_fan_out_with_yield_from_in_an_agent_loop() -> None:
    harness, _ = await fakes.open_harness(
        fakes.FakeAdapter(lambda prompt: fakes.reply(f"{prompt}-done"))
    )

    class FanOut(ai.Agent):
        async def loop(self, context: ai.Context) -> Any:
            left, right = harness.session(), harness.session()
            async with left.stream("left") as tl, right.stream("right") as tr:
                a, b = await asyncio.gather(
                    ai.yield_from(
                        tl, aggregator=ai.agents.MessageAggregator, label="l"
                    ),
                    ai.yield_from(
                        tr, aggregator=ai.agents.MessageAggregator, label="r"
                    ),
                )
            context.add(ai.assistant_message(f"{a} + {b}"))
            return
            yield

    # Never called: the loop makes no model call.
    model = ai.testing.FakeModel(
        [ai.user_message("go"), ai.assistant_message("unused")]
    )
    labels = set()
    async with FanOut().run(model, [ai.user_message("go")]) as stream:
        async for event in stream:
            if isinstance(event, events.PartialToolCallResult):
                labels.add(event.label)
    assert labels == {"l", "r"}
    assert stream.output == "left-done + right-done"


async def test_a_harness_turn_is_a_step_of_a_custom_agent_loop() -> None:
    harness, _ = await fakes.open_harness(fakes.FakeAdapter(_tool_turn))

    class CodeThenReview(ai.Agent):
        async def loop(self, context: ai.Context) -> Any:
            async with harness.session().stream(
                context.messages[-1].text
            ) as turn:
                async for event in turn:
                    if not isinstance(event, events.TextDelta):
                        yield event  # text dropped: the loop decides
            context.add(ai.assistant_message(turn.output))
            context.add(ai.user_message("review it"))
            async with ai.stream(context=context) as review:
                async for event in review:
                    yield event
            context.add(review.message)

    model = ai.testing.FakeModel(
        [
            ai.user_message("fix"),
            ai.assistant_message("Done."),
            ai.user_message("review it"),
            ai.assistant_message("YES"),
        ]
    )
    deltas = []
    async with CodeThenReview().run(model, [ai.user_message("fix")]) as stream:
        async for event in stream:
            if isinstance(event, events.TextDelta):
                deltas.append(event.chunk)
    assert deltas == ["YES"], "only the review's text got through"
    assert len(model.calls) == 1, "one model call, after the harness step"
    assert stream.output == "YES"


async def test_an_outer_loop_prompts_again_until_a_check_passes() -> None:
    harness, adapter = await fakes.open_harness()

    async def until(check: str) -> Any:
        session = harness.session()
        prompt = "first"
        for attempt in range(3):
            async with session.stream(prompt) as turn:
                async for event in turn:
                    yield event
            yield ("checked", attempt, turn.output == check)
            if turn.output == check:
                return
            prompt = check

    checks = [item async for item in until("fixed") if isinstance(item, tuple)]
    assert checks == [("checked", 0, False), ("checked", 1, True)]
    assert adapter.turns_started == ["first", "fixed"]
