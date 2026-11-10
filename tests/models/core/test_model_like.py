"""`ModelLike`: an id and a provider go where a model goes."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pydantic

import ai
from ai.types import events
from ai.types import messages as m
from ai.types import tools as tools_


class EchoProvider:
    name = "echo"

    async def stream(
        self,
        model: Echo,
        messages: list[m.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: Any = None,
    ) -> AsyncGenerator[events.Event]:
        model.calls.append(list(messages))
        text = next(x.text for x in reversed(messages) if x.role == "user")
        yield events.StreamStart()
        yield events.TextStart(block_id="t")
        yield events.TextDelta(chunk=text, block_id="t")
        yield events.TextEnd(block_id="t")
        yield events.StreamEnd(finish_reason="stop")

    async def generate(
        self,
        model: Echo,
        messages: list[m.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: Any = None,
    ) -> m.Message:
        raise NotImplementedError


class Echo:
    """Not an `ai.Model`: it answers with the last user message's text."""

    id = "echo"

    def __init__(self) -> None:
        self.calls: list[list[m.Message]] = []
        self.provider = EchoProvider()


def test_a_model_and_a_stand_in_are_model_like() -> None:
    assert isinstance(Echo(), ai.models.ModelLike)
    assert isinstance(ai.get_model("openai/gpt-5"), ai.models.ModelLike)


async def test_stream_takes_a_model_like() -> None:
    async with ai.stream(Echo(), [ai.user_message("hello")]) as s:
        async for _ in s:
            pass
    assert s.text == "hello"


async def test_generate_takes_a_model_like() -> None:
    message = await ai.experimental_generate(Echo(), [ai.user_message("hi")])
    assert message.text == "hi"


async def test_an_agent_runs_on_a_model_like() -> None:
    model = Echo()
    async with ai.Agent().run(model, [ai.user_message("ping")]) as run:
        async for _ in run:
            pass
    assert [x.role for x in run.messages] == ["user", "assistant"]
    assert run.messages[-1].text == "ping"
    assert len(model.calls) == 1


def test_a_context_keeps_a_model_like_as_it_is() -> None:
    model = Echo()
    context = ai.Context(model=model, messages=[], tools=[])
    assert context.model is model
