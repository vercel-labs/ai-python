from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable, Sequence
from typing import Any, cast

import pydantic

import ai
from ai import models
from ai.types import builders
from ai.types import events as agent_events_
from ai.types import events as events_
from ai.types import messages as messages_
from ai.types import usage as usage_


class MockProvider(models.Provider):
    """Minimal provider for tests.

    Carries just enough state so that ``Model`` objects can be constructed.
    """

    def __init__(
        self,
        *,
        name: str = "mock",
        base_url: str = "http://mock.test",
        api_key_env: str | None = "MOCK_API_KEY",
    ) -> None:
        super().__init__(
            name=name,
            base_url=base_url,
            api_key_env=api_key_env,
        )
        self._stream_impl: Any | None = None
        self._generate_impl: Any | None = None

    async def list_models(self) -> list[str]:
        return []

    def stream(
        self,
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: models.InferenceRequestParams | None = None,
    ) -> AsyncGenerator[events_.Event]:
        if model.protocol is not None:
            return model.protocol.stream(
                None,
                model,
                messages,
                tools=tools,
                output_type=output_type,
                params=params,
                provider=self.name,
            )
        if self._stream_impl is None:
            raise RuntimeError(
                "MockProvider: no stream implementation configured"
            )
        return cast(
            "AsyncGenerator[events_.Event]",
            self._stream_impl(
                model,
                messages,
                tools=tools,
                output_type=output_type,
                params=params,
            ),
        )

    async def generate(
        self,
        model: models.Model,
        messages: list[messages_.Message],
        params: models.GenerateParams,
    ) -> messages_.Message:
        if model.protocol is not None:
            return await model.protocol.generate(
                None,
                model,
                messages,
                params,
                provider=self.name,
            )
        if self._generate_impl is None:
            raise RuntimeError(
                "MockProvider: no generate implementation configured"
            )
        return cast(
            "messages_.Message",
            await self._generate_impl(model, messages, params),
        )


MOCK_PROVIDER = MockProvider()


def mock_provider() -> MockProvider:
    """Provider factory returning the shared mock provider."""
    return MOCK_PROVIDER


# A fixed Model used in tests.
MOCK_MODEL: models.Model = models.Model(
    "mock-model",
    provider_factory=mock_provider,
)


async def emit_events_for_messages(
    seq: list[messages_.Message],
    *,
    usage: usage_.Usage | None = None,
) -> AsyncGenerator[events_.Event]:
    """Emit a stream of public ``events_.Event`` corresponding to ``seq``.

    Walks each message's parts and yields the appropriate
    ``Start`` / ``Delta`` / ``End`` events (and ``FileEvent``).  The output
    matches what a real adapter would produce.  Bookended by
    ``StreamStart`` / ``StreamEnd``.
    """
    yield events_.StreamStart()
    for msg in seq:
        for i, part in enumerate(msg.parts):
            if isinstance(part, messages_.TextPart):
                bid = f"text-{i}"
                yield events_.TextStart(block_id=bid)
                if part.text:
                    yield events_.TextDelta(block_id=bid, chunk=part.text)
                yield events_.TextEnd(block_id=bid)

            elif isinstance(part, messages_.ReasoningPart):
                bid = f"reasoning-{i}"
                yield events_.ReasoningStart(block_id=bid)
                if part.text:
                    yield events_.ReasoningDelta(block_id=bid, chunk=part.text)
                yield events_.ReasoningEnd(
                    block_id=bid,
                    provider_metadata=part.provider_metadata,
                )

            elif isinstance(part, messages_.ToolCallPart):
                yield events_.ToolStart(
                    tool_call_id=part.tool_call_id,
                    tool_name=part.tool_name,
                )
                if part.tool_args:
                    yield events_.ToolDelta(
                        tool_call_id=part.tool_call_id,
                        chunk=part.tool_args,
                    )
                yield events_.ToolEnd(
                    tool_call_id=part.tool_call_id, tool_call=part
                )

            elif isinstance(part, messages_.FilePart):
                yield events_.FileEvent(
                    block_id=part.id,
                    media_type=part.media_type,
                    data=part.data if isinstance(part.data, str) else "",
                )
    yield events_.StreamEnd(usage=usage)


class MockAdapter:
    """Mock stream adapter that yields pre-configured response sequences.

    Each call pops the next response list and emits events for it via
    :func:`emit_events_for_messages`.  Tracks ``call_count``.
    """

    def __init__(self, responses: list[list[messages_.Message]]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.call_count = 0

    async def stream(
        self,
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        if self._call_index >= len(self._responses):
            raise RuntimeError("MockAdapter: no more responses configured")
        self.call_count += 1
        seq = self._responses[self._call_index]
        self._call_index += 1

        async for event in emit_events_for_messages(seq):
            yield event


def mock_llm(responses: list[list[messages_.Message]]) -> MockAdapter:
    """Create a MockAdapter and attach it to the shared mock provider.

    Returns the adapter so tests can inspect ``call_count``.
    """
    adapter = MockAdapter(responses)
    MOCK_PROVIDER._stream_impl = adapter.stream
    return adapter


async def collect_messages(
    source: AsyncIterable[agent_events_.AgentEvent],
) -> list[messages_.Message]:
    """Collect terminal messages from an event stream."""
    result: list[messages_.Message] = []
    async for event in source:
        if isinstance(event, agent_events_.TerminalEvent):
            result.append(event.message)
    return result


class MockGenerateAdapter:
    """Mock generate adapter that returns pre-configured responses.

    Each call pops the next response.  Tracks ``call_count`` for assertions.
    """

    def __init__(self, responses: list[messages_.Message]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.call_count = 0

    async def generate(
        self,
        model: models.Model,
        messages: list[messages_.Message],
        params: models.GenerateParams,
    ) -> messages_.Message:
        if self._call_index >= len(self._responses):
            raise RuntimeError(
                "MockGenerateAdapter: no more responses configured"
            )
        self.call_count += 1
        msg = self._responses[self._call_index]
        self._call_index += 1
        return msg


def mock_generate(responses: list[messages_.Message]) -> MockGenerateAdapter:
    """Create a MockGenerateAdapter and attach it to the shared mock provider.

    Returns the adapter so tests can inspect ``call_count``.
    """
    adapter = MockGenerateAdapter(responses)
    MOCK_PROVIDER._generate_impl = adapter.generate
    return adapter


# ── Helpers ──────────────────────────────────────────────────────


def text_msg(
    text: str,
    *,
    id: str = "msg-1",
) -> messages_.Message:
    part: messages_.Part = messages_.TextPart(text=text)
    return messages_.Message(id=id, role="assistant", parts=[part])


def tool_call_msg(
    *,
    id: str = "msg-1",
    tc_id: str = "tc-1",
    name: str = "test_tool",
    args: str = "{}",
) -> messages_.Message:
    """Assistant message containing a tool call."""
    part: messages_.Part = messages_.ToolCallPart(
        tool_call_id=tc_id,
        tool_name=name,
        tool_args=args,
    )
    return messages_.Message(id=id, role="assistant", parts=[part])


def tool_result_msg(
    *,
    tc_id: str = "tc-1",
    name: str = "test_tool",
    result: Any = None,
    is_error: bool = False,
) -> messages_.Message:
    """Tool-result message."""
    return builders.tool_message(
        tool_call_id=tc_id,
        tool_name=name,
        result=result,
        is_error=is_error,
    )
