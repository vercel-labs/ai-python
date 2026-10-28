from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator, Collection, Sequence
from typing import Protocol

from . import agents
from .types import events

__all__ = ["DurableAgent", "Streamer"]


class Streamer(Protocol):
    def __call__(
        self, *, context: agents.Context
    ) -> contextlib.AbstractAsyncContextManager[
        AsyncGenerator[events.AgentEvent]
    ]: ...


class DurableAgent(agents.Agent):
    # We require the loop to run in lockstep with the client code, so that
    # tool streams are always sent before the next llm_step invocation.
    LOOP_BUFFER = 0

    def __init__(
        self,
        *,
        tools: Sequence[agents.AgentTool | agents.Tool] | None = None,
        streamer: Streamer,
        eager_tools: Collection[str] = (),
    ) -> None:
        super().__init__(tools=tools)
        self.streamer = streamer
        self.eager_tools = frozenset(eager_tools)

    async def loop(
        self, context: agents.Context
    ) -> AsyncGenerator[events.AgentEvent]:
        while context.keep_running():
            async with (
                self.streamer(context=context) as stream,
                agents.ToolRunner() as tr,
            ):
                final = False
                async for event in stream:
                    yield event

                    if isinstance(event, events.ToolEnd):
                        if (
                            final
                            or event.tool_call.tool_name in self.eager_tools
                        ):
                            tr.schedule(self.resolve(event.tool_call))
                    elif isinstance(event, events.StreamEnd):
                        tr.discard_all(
                            except_=self.resolve(event.message.tool_calls)
                        )
                        context.add(event.message)
                    elif isinstance(event, events.StreamStart):
                        final = True

                async for event in tr.events():
                    yield event

                context.add(tr.get_tool_message())
