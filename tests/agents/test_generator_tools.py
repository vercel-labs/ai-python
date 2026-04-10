"""Async-generator tools and nested streaming via yield_from."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pydantic

import ai
from ai import models
from ai.models.core.helpers import streaming as streaming_
from ai.types import messages as messages_

from ..conftest import MOCK_MODEL, mock_llm, text_msg, tool_call_msg

# ---------------------------------------------------------------------------
# Generator tool: yields intermediate messages, returns final text
# ---------------------------------------------------------------------------


@ai.tool  # type: ignore[arg-type]
async def progress_tool(query: str) -> AsyncGenerator[ai.Message]:
    """Tool that streams progress, then returns a final answer."""
    yield ai.Message(
        role="assistant",
        parts=[messages_.TextPart(text="Working...", state="done")],
        label="progress",
    )
    yield ai.Message(
        role="assistant",
        parts=[messages_.TextPart(text=f"Answer for {query}", state="done")],
    )


async def test_generator_tool_streams_and_returns_result() -> None:
    """Generator tool yields intermediate messages visible to consumer;
    final text becomes the tool result fed back to the LLM."""
    my_agent = ai.agent(tools=[progress_tool])

    # Turn 1: LLM calls progress_tool
    # Turn 2: LLM produces final text after seeing the tool result
    call = [tool_call_msg(tc_id="tc-1", name="progress_tool", args='{"query": "test"}')]
    reply = [text_msg("Done!", id="msg-2")]
    llm = mock_llm([call, reply])

    collected: list[ai.Message] = []
    async for msg in my_agent.run(MOCK_MODEL, [ai.user_message("Go")]):
        collected.append(msg)

    assert llm.call_count == 2

    # Intermediate progress message was forwarded to consumer.
    progress = [m for m in collected if m.label == "progress"]
    assert len(progress) == 1
    assert progress[0].text == "Working..."

    # Tool result was fed back to LLM.
    tool_results = [m for m in collected if m.role == "tool"]
    assert len(tool_results) >= 1
    assert tool_results[0].tool_results[0].result == "Answer for test"

    # Final response arrived.
    assert any(m.text == "Done!" for m in collected)


# ---------------------------------------------------------------------------
# yield_from: nested agent streams through outer agent
# ---------------------------------------------------------------------------


class _CapturingAdapter:
    """Like MockAdapter but records the messages arg on every call."""

    def __init__(self, responses: list[list[messages_.Message]]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.call_count = 0
        self.calls: list[list[messages_.Message]] = []

    async def stream(
        self,
        client: models.Client,
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[messages_.Message]:
        if self._idx >= len(self._responses):
            raise RuntimeError("_CapturingAdapter: no more responses")
        self.call_count += 1
        self.calls.append(list(messages))
        seq = self._responses[self._idx]
        self._idx += 1

        handler = streaming_.StreamHandler()
        for msg in seq:
            for i, part in enumerate(msg.parts):
                if isinstance(part, messages_.TextPart):
                    bid = f"text-{i}"
                    yield handler.handle_event(streaming_.TextStart(block_id=bid))
                    if part.text:
                        yield handler.handle_event(
                            streaming_.TextDelta(block_id=bid, delta=part.text)
                        )
                    yield handler.handle_event(streaming_.TextEnd(block_id=bid))
                elif isinstance(part, messages_.ToolCallPart):
                    yield handler.handle_event(
                        streaming_.ToolStart(
                            tool_call_id=part.tool_call_id,
                            tool_name=part.tool_name,
                        )
                    )
                    if part.tool_args:
                        yield handler.handle_event(
                            streaming_.ToolArgsDelta(
                                tool_call_id=part.tool_call_id,
                                delta=part.tool_args,
                            )
                        )
                    yield handler.handle_event(
                        streaming_.ToolEnd(tool_call_id=part.tool_call_id)
                    )
        yield handler.handle_event(streaming_.MessageDone())


@ai.tool
async def inner_fact(topic: str) -> str:
    """Return a fact about a topic."""
    return f"Fact about {topic}"


@ai.tool  # type: ignore[arg-type]
async def research_tool(topic: str) -> AsyncGenerator[ai.Message]:
    """Nested agent that researches a topic."""
    inner = ai.agent(tools=[inner_fact])

    msgs = [
        ai.system_message("Be concise."),
        ai.user_message(f"Research: {topic}"),
    ]
    async for msg in inner.run(MOCK_MODEL, msgs, label="inner"):
        yield msg


async def test_yield_from_nested_agent() -> None:
    """yield_from forwards inner messages to the consumer but does NOT
    add them to the outer agent's history (context.messages).

    The critical contract from agent.py:292: yield_from streams messages
    through the runtime queue without going through _collect_messages,
    so the parent agent's context.messages stays clean.
    """
    outer = ai.agent(tools=[research_tool])

    # Inner agent: text-only reply (no tools, to keep it simple)
    inner_reply = [text_msg("Mars has two moons.", id="inner-msg")]

    # Outer agent turn 1: calls research_tool
    outer_call = [
        tool_call_msg(tc_id="otc-1", name="research_tool", args='{"topic": "mars"}')
    ]
    # Outer agent turn 2: final answer (after seeing tool result)
    outer_reply = [text_msg("Summary: Mars has two moons.", id="outer-msg-2")]

    adapter = _CapturingAdapter([outer_call, inner_reply, outer_reply])
    models.register_stream("mock", adapter.stream)

    collected: list[ai.Message] = []
    async for msg in outer.run(MOCK_MODEL, [ai.user_message("Tell me about Mars")]):
        collected.append(msg)

    assert adapter.call_count == 3

    # Inner messages were forwarded to the consumer with label="inner".
    inner_msgs = [m for m in collected if m.label == "inner"]
    assert len(inner_msgs) > 0

    # The outer LLM's second call (index 2) must NOT contain any inner
    # agent messages.  It should only see: the original user message,
    # the outer assistant tool-call, and the outer tool-result.
    outer_turn2_msgs = adapter.calls[2]
    outer_turn2_roles = [m.role for m in outer_turn2_msgs]
    assert outer_turn2_roles == ["user", "assistant", "tool"]

    # Specifically: no inner assistant text or inner tool results leaked.
    for m in outer_turn2_msgs:
        assert m.label is None or m.label != "inner"
        if m.role == "assistant":
            # This must be the outer tool-call message, not inner text.
            assert len(m.tool_calls) > 0
