"""Shared fakes for the Anthropic adapter tests.

The adapter consumes ``anthropic.AsyncAnthropic`` via
``messages.stream(**kwargs)``. To exercise the real adapter without
hitting the network we build a tiny stand-in that:

* captures the kwargs passed into ``messages.stream``;
* yields scripted SDK-shaped events from the async context;
* exposes a ``current_message_snapshot`` for the result-block lookup
  the adapter performs at ``content_block_stop`` for tool-result blocks.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any


class FakeUsage:
    """Mimics the snapshot's ``usage`` field."""

    def __init__(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class FakeSnapshot:
    """Mimics ``MessageStream.current_message_snapshot``."""

    def __init__(self, content: list[Any] | None = None) -> None:
        self.usage = FakeUsage()
        self.content = content or []


class FakeStream:
    """Async-iterable + async-context-manager stand-in for ``MessageStream``.

    Yields a fixed list of events, then exposes ``current_message_snapshot``
    so the adapter can resolve tool-result block names by ``tool_use_id``.
    """

    def __init__(
        self,
        events: Iterable[Any] = (),
        snapshot_content: list[Any] | None = None,
    ) -> None:
        self._events = list(events)
        self.current_message_snapshot = FakeSnapshot(snapshot_content)

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        return None

    def __aiter__(self) -> FakeStream:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


class FakeMessages:
    def __init__(
        self,
        captured: dict[str, Any],
        stream: FakeStream | None = None,
    ) -> None:
        self._captured = captured
        self._stream = stream or FakeStream()

    def stream(self, **kwargs: Any) -> FakeStream:
        self._captured.update(kwargs)
        return self._stream


class FakeAnthropicClient:
    """Stand-in for ``anthropic.AsyncAnthropic``."""

    def __init__(
        self,
        captured: dict[str, Any] | None = None,
        stream: FakeStream | None = None,
    ) -> None:
        self.messages = FakeMessages(
            captured if captured is not None else {}, stream
        )
        self.closed = False

    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# SDK-shaped event factories
# ---------------------------------------------------------------------------


def block_start(index: int, block_type: str, **fields: Any) -> SimpleNamespace:
    """Build an SDK ``content_block_start`` event."""
    block = SimpleNamespace(type=block_type, **fields)
    return SimpleNamespace(
        type="content_block_start", index=index, content_block=block
    )


def block_stop(index: int) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_stop", index=index)


def block_delta(index: int, delta_type: str, **fields: Any) -> SimpleNamespace:
    delta = SimpleNamespace(type=delta_type, **fields)
    return SimpleNamespace(type="content_block_delta", index=index, delta=delta)


def snapshot_block(block_type: str, **fields: Any) -> SimpleNamespace:
    """Build a snapshot content block (used for tool-result lookups)."""
    return SimpleNamespace(type=block_type, **fields)
