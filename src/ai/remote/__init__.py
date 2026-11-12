"""Run tools, or an agent's whole loop, in a runner.

A runner runs functions somewhere else, such as in another process or
a sandbox; anything with ``run`` and ``iterate`` methods will do (see
:class:`Runner`).  Make one ambient with :func:`use_runner`::

    async with ai.remote.use_runner(runner):
        async with agent.run(model, messages) as stream:
            ...

Then a tool declared with ``execution_strategy="remote"`` or
``"remote-enabled"`` runs in the runner, and a :class:`RemoteAgent`
runs its whole loop there, tools included; events, messages added to
the context, and hooks come back, and hooks are resolved or cancelled
here as usual.  Code running in the runner runs every tool in place.
With no ambient runner, everything runs here, except that a
``"remote"`` tool fails.

The agent, its context and its tools are pickled to the runner, so
they must be picklable there.  Nothing the runner sends is unpickled,
since what runs there may not be trusted: events and tool results come
back as JSON and are validated here, tool results against the tool's
return type.  A tool without one gets its result as plain JSON data.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import uuid
from collections.abc import AsyncGenerator, Sequence
from typing import Any, cast

import pydantic

from .. import agents, models
from ..agents import agent as agent_
from ..agents import hooks as hooks_
from ..agents import runner as runner_
from ..agents import runtime as runtime_
from ..agents.runner import RemoteIterator, Runner, use_runner
from ..types import events
from ..types import messages as messages_

__all__ = [
    "RemoteAgent",
    "RemoteIterator",
    "Runner",
    "use_runner",
    "with_resolved_key",
]


def with_resolved_key(model: models.Model) -> models.Model:
    """Copy *model* with its API key read now, so it pickles along.

    Otherwise the provider reads its key's env var when it's called,
    and a runner may not have it.
    """
    key = model.provider.api_key
    if key is None:
        return model
    provider = model.provider.model_copy(update={"api_key_value": key})
    return model.model_copy(update={"provider": provider})


@dataclasses.dataclass
class _Added:
    """Messages the remote loop added to its copy of the context."""

    messages: list[messages_.Message]


class _SyncingContext(agents.Context):
    """The runner's copy of the context, reporting each ``add`` at once.

    The ``_Added`` goes on the runtime's queue, which is merged into the
    event stream ahead of the loop's own events, so it comes back while
    the loop is still waiting on the next model response.
    """

    def add(
        self,
        message: messages_.Message | Sequence[messages_.Message] | None,
    ) -> None:
        before = len(self.messages)
        super().add(message)
        if len(self.messages) > before:
            queue = runtime_.get_runtime()._event_queue
            added = _Added(self.messages[before:])
            queue.put_nowait(cast("events.AgentEvent", added))


# The hook registries of the loops running in this process, by key, so
# that hooks can be resolved and cancelled from the other side.
_registries: dict[str, hooks_.HookRegistry] = {}


async def _remote_loop(
    agent: RemoteAgent, context: agents.Context, key: str
) -> AsyncGenerator[events.AgentEvent | _Added | dict[str, Any]]:
    """Run in the runner: the agent's loop.

    Tools put partial results on the runtime, so this runs its own, to
    merge them in.
    """
    context.__class__ = _SyncingContext
    runner_._in_runner.set(True)
    registry = _registries[key] = hooks_.HookRegistry()
    # Left set: this generator's context is the runner's, not the run's.
    hooks_._hook_registry.set(registry)
    try:
        loop = agent.loop(context)
        async with contextlib.aclosing(runtime_.run(loop)) as stream:
            async for event in stream:
                if not isinstance(event, _Added) and event.replay:
                    # Not serialized with the event, but validated.
                    yield {**event.model_dump(mode="json"), "replay": True}
                else:
                    yield event
    finally:
        del _registries[key]


def _resolve_remote(
    key: str, label: str, resolution: dict[str, Any] | BaseException
) -> None:
    if registry := _registries.get(key):
        hooks_.resolve_hook(label, resolution, registry=registry)


async def _cancel_remote(key: str, label: str, reason: str | None) -> None:
    registry = _registries.get(key)
    if registry is not None and label in registry._live_hooks:
        await hooks_.cancel_hook(label, reason=reason, registry=registry)


class _HookBridge:
    """Stands in here for the remote loop's pending hooks.

    A stand-in is a live hook in this run's registry, so the caller
    resolves or cancels it as usual; the outcome is then passed on to
    the hook in the runner.
    """

    def __init__(self, runner: Runner, key: str) -> None:
        self._runner = runner
        self._key = key
        self._registry = hooks_.get_hook_registry()
        self._runtime = runtime_.get_runtime()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancelled: set[str] = set()

    def forward(self, event: events.HookEvent) -> bool:
        """Note a hook event from the runner; return whether to yield it.

        A cancellation made here was already reported here.
        """
        hook = event.hook
        if hook.status == "cancelled":
            return hook.hook_id not in self._cancelled
        if hook.status == "pending":
            future: asyncio.Future[dict[str, Any]] = asyncio.Future()
            self._registry._live_hooks[hook.hook_id] = (
                future,
                hook.metadata,
                self._runtime,
            )
            self._tasks[hook.hook_id] = asyncio.create_task(
                self._pass_on(hook.hook_id, future)
            )
        return True

    async def _pass_on(
        self, label: str, future: asyncio.Future[dict[str, Any]]
    ) -> None:
        try:
            try:
                resolution: dict[str, Any] | BaseException = await future
            except asyncio.CancelledError as exc:
                if not future.cancelled():
                    raise
                self._cancelled.add(label)
                reason = exc.args[0] if exc.args else None
                await self._runner.run(_cancel_remote, self._key, label, reason)
                return
            except Exception as exc:
                resolution = exc
            await self._runner.run(
                _resolve_remote, self._key, label, resolution
            )
        finally:
            self._registry._live_hooks.pop(label, None)
            self._tasks.pop(label, None)

    async def aclose(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class RemoteAgent(agents.Agent):
    """An agent that runs its loop in the ambient runner, if there is one.

    Override ``loop`` as usual; it's the loop that runs in the runner.
    """

    async def _loop(
        self, context: agents.Context
    ) -> AsyncGenerator[events.AgentEvent]:
        runner = runner_._runner.get()
        if runner is None or runner_._in_runner.get():
            async with contextlib.aclosing(self.loop(context)) as stream:
                async for event in stream:
                    yield event
            return
        key = uuid.uuid4().hex
        bridge = _HookBridge(runner, key)
        tools = messages_.tool_context(self._tools)
        items = pydantic.TypeAdapter[events.AgentEvent | _Added](
            events.AgentEvent | _Added
        )
        streaming = {
            t.name: (
                t.aggregator,
                pydantic.TypeAdapter(agent_._item_type(t.fn)),
            )
            for t in self._tools
            if t.aggregator is not None
        }
        try:
            async with runner.iterate(
                _remote_loop, self, context, key, free_running=True
            ) as s:
                async for data in s:
                    item = items.validate_python(data, context=tools)
                    if isinstance(item, _Added):
                        context.messages.extend(item.messages)
                        continue
                    if (
                        isinstance(item, events.PartialToolCallResult)
                        and item.label is None
                        and item.tool_name in streaming
                    ):
                        factory, adapter = streaming[item.tool_name]
                        item = item.model_copy(
                            update={
                                "aggregator_factory": factory,
                                "value": adapter.validate_python(item.value),
                            }
                        )
                    if not isinstance(item, events.HookEvent) or bridge.forward(
                        item
                    ):
                        yield item
        finally:
            await bridge.aclose()
