from collections.abc import AsyncGenerator
from typing import Any, Self

from . import events as events_
from . import messages
from . import usage as usage_


class StreamResult:
    """Wrapper around an event stream. Async-iterable; collects the final result.

    Yields :class:`~ai.types.events.Event` objects.  After iteration,
    convenience properties (``.text``, ``.tool_calls``, ``.usage``,
    ``.message``) are available — they delegate to the ``MessageEnd``
    event's ``message``.

    One ``StreamResult`` represents one turn: a single LLM request and
    its response.
    """

    def __init__(
        self,
        gen: AsyncGenerator[events_.Event],
        *,
        turn_id: str | None = None,
        input_messages: list[messages.Message] | None = None,
    ) -> None:
        self._gen = gen
        self._turn_id = turn_id
        self._input_messages = input_messages or []
        self._message: messages.Message | None = None
        self._usage: usage_.Usage | None = None

    @classmethod
    def from_generator(cls, gen: AsyncGenerator[events_.Event]) -> Self:
        """Create a :class:`StreamResult` from an async generator of events."""
        return cls(gen)

    def __aiter__(self) -> AsyncGenerator[events_.Event]:
        return self._iterate()

    async def _iterate(self) -> AsyncGenerator[events_.Event]:
        # Re-emit input messages as MessageStart + MessageEnd event pairs.
        for msg in self._input_messages:
            if msg.turn_id is None and self._turn_id is not None:
                msg = msg.model_copy(update={"turn_id": self._turn_id})
            yield events_.MessageStart(message=msg)
            yield events_.MessageEnd(message=msg)

        # Stream adapter events.
        async for event in self._gen:
            # Capture the final message from MessageEnd.
            if isinstance(event, events_.MessageEnd):
                self._message = event.message
                self._usage = event.usage
            yield event

    @property
    def turn_id(self) -> str | None:
        """The turn id stamped on this stream's response (if any)."""
        return self._turn_id

    @property
    def message(self) -> messages.Message | None:
        """The final assembled message, available after iteration."""
        return self._message

    @property
    def text(self) -> str:
        if self._message is None:
            return ""
        return "".join(
            p.text for p in self._message.parts if isinstance(p, messages.TextPart)
        )

    @property
    def tool_calls(self) -> list[messages.ToolCallPart]:
        if self._message is None:
            return []
        return [
            p for p in self._message.parts if isinstance(p, messages.ToolCallPart)
        ]

    @property
    def usage(self) -> usage_.Usage | None:
        return self._usage

    @property
    def output(self) -> Any:
        """Parsed structured output from the final message, if available."""
        if self._message is None:
            return None
        for p in self._message.parts:
            if isinstance(p, messages.StructuredOutputPart):
                return p.value
        return None
