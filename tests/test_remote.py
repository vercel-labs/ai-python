from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import json
import pickle
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Callable,
    Iterable,
)
from typing import Any, Self, cast

import pydantic
import pydantic_core
import pytest

import ai
from ai import remote
from ai.types import events, messages

# Set here and never in the runner, which runs calls in an empty context.
where: contextvars.ContextVar[str] = contextvars.ContextVar(
    "where", default="runner"
)


def _copy[T](value: T) -> T:
    return pickle.loads(pickle.dumps(value))  # type: ignore[no-any-return]


def _reply(value: Any) -> Any:
    return json.loads(pydantic_core.to_json(value))


class InProcessRunner:
    """A runner that copies calls and runs them in a context of their own.

    That's what a runner in another process sees, short of the process;
    results come back as JSON data, as from one.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.free_running: list[bool] = []

    async def run(self, func: Callable[..., Any], *args: Any) -> Any:
        self.calls += 1
        func, args = _copy((func, args))

        async def call() -> Any:
            result = func(*args)
            return await result if inspect.isawaitable(result) else result

        task = asyncio.create_task(call(), context=contextvars.Context())
        return _reply(await task)

    def iterate(
        self,
        func: Callable[..., Iterable[Any] | AsyncIterable[Any]],
        *args: Any,
        free_running: bool = False,
    ) -> _Iterator:
        self.calls += 1
        self.free_running.append(free_running)
        return _Iterator(*_copy((func, args)))


def _runner() -> remote.Runner:
    return cast("remote.Runner", InProcessRunner())


class _Iterator:
    def __init__(
        self,
        func: Callable[..., Iterable[Any] | AsyncIterable[Any]],
        args: tuple[Any, ...],
    ) -> None:
        self._items: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self._task = asyncio.create_task(
            self._pump(func, args), context=contextvars.Context()
        )

    async def _pump(
        self,
        func: Callable[..., Iterable[Any] | AsyncIterable[Any]],
        args: tuple[Any, ...],
    ) -> None:
        try:
            it = func(*args)
            if isinstance(it, AsyncIterable):
                try:
                    async for item in it:
                        await self._items.put(("item", _reply(item)))
                finally:
                    if isinstance(it, AsyncGenerator):
                        await it.aclose()
            else:
                for item in it:
                    await self._items.put(("item", _reply(item)))
        except Exception as exc:
            await self._items.put(("err", exc))
        else:
            await self._items.put(("done", None))

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Any:
        kind, value = await self._items.get()
        if kind == "err":
            raise value
        if kind == "done":
            raise StopAsyncIteration
        return value

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)


@ai.tool
async def locate(x: int) -> str:
    """Say where this ran."""
    return f"{where.get()}:{x}"


@ai.tool(execution_strategy="remote")
async def remote_locate(x: int) -> str:
    """Say where this ran, in the runner."""
    return f"{where.get()}:{x}"


@ai.tool(execution_strategy="remote-enabled")
async def maybe_locate(x: int) -> str:
    """Say where this ran, in the runner if there is one."""
    return f"{where.get()}:{x}"


@ai.tool(require_approval=True)
async def gated(x: int) -> str:
    """Say where this ran, with approval."""
    return f"{where.get()}:{x}"


@ai.tool(require_approval=True, execution_strategy="remote")
async def remote_gated(x: int) -> str:
    """Say where this ran, in the runner, with approval."""
    return f"{where.get()}:{x}"


@ai.tool(execution_strategy="remote")
async def count(n: int) -> ai.StreamingTextTool:
    """Count up, saying where."""
    for i in range(n):
        yield f"{where.get()}:{i}"


@ai.tool
async def local_count(n: int) -> ai.StreamingTextTool:
    """Count up, saying where."""
    for i in range(n):
        yield f"{where.get()}:{i}"


class Place(pydantic.BaseModel):
    where: str


@ai.tool(execution_strategy="remote")
async def place() -> Place:
    """Say where this ran, as a model."""
    return Place(where=where.get())


@ai.tool(execution_strategy="remote")
async def places(n: int) -> ai.StreamingStatusTool[Place]:
    """Say where this ran, as models, over and over."""
    for _ in range(n):
        yield Place(where=where.get())


@ai.tool(execution_strategy="remote")
async def delegate(x: int) -> str:
    """Run a sub-agent with a remote tool."""
    model = _script(ai.testing.tool_call(remote_locate, x=x))
    agent = ai.Agent(tools=[remote_locate])
    async with agent.run(model, [ai.user_message("go")]) as stream:
        async for _ in stream:
            pass
    return str(_results(stream)["remote_locate"])


def _script(*calls: messages.ToolCallPart) -> ai.testing.FakeModel:
    return ai.testing.FakeModel(
        [
            ai.user_message("go"),
            ai.assistant_message("calling", *calls),
            ai.assistant_message("done"),
        ]
    )


def _results(stream: Any) -> dict[str, Any]:
    return {
        p.tool_name: p.result for m in stream.messages for p in m.tool_results
    }


async def _run(
    agent: ai.Agent, *calls: messages.ToolCallPart
) -> tuple[Any, list[messages.HookPart[Any]]]:
    hooks = []
    async with agent.run(_script(*calls), [ai.user_message("go")]) as stream:
        async for event in stream:
            if (
                isinstance(event, events.HookEvent)
                and event.hook.status == "pending"
            ):
                hooks.append(event.hook)
                ai.resolve_hook(event.hook, ai.tools.ToolApproval(granted=True))
    assert [m.role for m in stream.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert stream.messages[-1].text == "done"
    return stream, hooks


@pytest.fixture(autouse=True)
def _here() -> None:
    where.set("here")


_CALLS = [
    ai.testing.tool_call(locate, x=1),
    ai.testing.tool_call(maybe_locate, x=2),
    ai.testing.tool_call(remote_locate, x=3),
    ai.testing.tool_call(gated, x=4),
    ai.testing.tool_call(remote_gated, x=5),
]
_TOOLS = [locate, maybe_locate, remote_locate, gated, remote_gated]


@pytest.mark.parametrize(
    ("agent_cls", "expected"),
    [
        (
            ai.Agent,
            {
                "locate": "here:1",
                "maybe_locate": "runner:2",
                "remote_locate": "runner:3",
                "gated": "here:4",
                "remote_gated": "runner:5",
            },
        ),
        (
            remote.RemoteAgent,
            {
                "locate": "runner:1",
                "maybe_locate": "runner:2",
                "remote_locate": "runner:3",
                "gated": "runner:4",
                "remote_gated": "runner:5",
            },
        ),
    ],
)
async def test_where_tools_run(
    agent_cls: type[ai.Agent], expected: dict[str, str]
) -> None:
    runner = InProcessRunner()
    async with remote.use_runner(cast("remote.Runner", runner)):
        stream, hooks = await _run(agent_cls(tools=_TOOLS), *_CALLS)
    assert _results(stream) == expected
    assert [h.tool_call_id for h in hooks] == [
        c.tool_call_id for c in stream.messages[1].tool_calls[3:]
    ]
    assert runner.calls > 0
    assert all(runner.free_running)


@pytest.mark.parametrize("agent_cls", [ai.Agent, remote.RemoteAgent])
async def test_without_a_runner(agent_cls: type[ai.Agent]) -> None:
    stream, _ = await _run(agent_cls(tools=_TOOLS), *_CALLS)
    results = _results(stream)
    error = results.pop("remote_locate")
    assert "remote_locate" in error
    assert "use_runner" in error
    assert results.pop("remote_gated") == error.replace(
        "remote_locate", "remote_gated"
    )
    assert results == {
        "locate": "here:1",
        "maybe_locate": "here:2",
        "gated": "here:4",
    }
    assert [p.is_error for p in stream.messages[2].tool_results] == [
        False,
        False,
        True,
        False,
        True,
    ]


@pytest.mark.parametrize(
    ("agent_cls", "expected"),
    [
        (ai.Agent, {"count": "runner", "local_count": "here"}),
        (remote.RemoteAgent, {"count": "runner", "local_count": "runner"}),
    ],
)
async def test_streaming_tool(
    agent_cls: type[ai.Agent], expected: dict[str, str]
) -> None:
    agent = agent_cls(tools=[count, local_count])
    partials: dict[str, list[str]] = {}
    runner = InProcessRunner()
    async with remote.use_runner(cast("remote.Runner", runner)):
        async with agent.run(
            _script(
                ai.testing.tool_call(count, n=2),
                ai.testing.tool_call(local_count, n=2),
            ),
            [ai.user_message("go")],
        ) as stream:
            async for event in stream:
                if isinstance(event, events.PartialToolCallResult):
                    assert event.tool_name is not None
                    partials.setdefault(event.tool_name, []).append(event.value)
    assert partials == {
        name: [f"{w}:0", f"{w}:1"] for name, w in expected.items()
    }
    assert _results(stream) == {
        name: f"{w}:0{w}:1" for name, w in expected.items()
    }
    assert runner.free_running == [True]


@pytest.mark.parametrize("agent_cls", [ai.Agent, remote.RemoteAgent])
async def test_results_are_validated(agent_cls: type[ai.Agent]) -> None:
    model = _script(
        ai.testing.tool_call(place), ai.testing.tool_call(places, n=2)
    )
    agent = agent_cls(tools=[place, places])
    partials = []
    async with remote.use_runner(_runner()):
        async with agent.run(model, [ai.user_message("go")]) as stream:
            async for event in stream:
                if isinstance(event, events.PartialToolCallResult):
                    assert event.aggregator_factory is not None
                    partials.append(event.value)
    runner = Place(where="runner")
    assert partials == [runner, runner]
    assert _results(stream) == {"place": runner, "places": runner}


@pytest.mark.parametrize("agent_cls", [ai.Agent, remote.RemoteAgent])
async def test_remote_tool_runs_remote_tools_in_place(
    agent_cls: type[ai.Agent],
) -> None:
    async with remote.use_runner(_runner()):
        stream, _ = await _run(
            agent_cls(tools=[delegate]), ai.testing.tool_call(delegate, x=1)
        )
    assert _results(stream) == {"delegate": "runner:1"}


def test_execution_strategy_pickles() -> None:
    assert _copy(remote_locate).execution_strategy == "remote"
    assert _copy(locate).execution_strategy == "local"


async def _approval_run(
    agent: ai.Agent,
    call: messages.ToolCallPart,
    decide: Callable[[messages.HookPart[Any]], Any],
) -> tuple[Any, list[str]]:
    model = _script(call)
    statuses: list[str] = []
    async with agent.run(model, [ai.user_message("go")]) as stream:
        async for event in stream:
            if isinstance(event, events.HookEvent):
                statuses.append(event.hook.status)
                if event.hook.status == "pending":
                    result = decide(event.hook)
                    if inspect.isawaitable(result):
                        await result
    return stream, statuses


@pytest.mark.parametrize(
    "decide",
    [
        lambda hook: ai.resolve_hook(hook, ai.tools.ToolApproval(granted=True)),
        lambda hook: ai.resolve_hook(
            hook, ai.tools.ToolApproval(granted=False, reason="no")
        ),
        lambda hook: ai.cancel_hook(hook, reason="stop"),
    ],
    ids=["granted", "denied", "cancelled"],
)
async def test_hooks_match_local(
    decide: Callable[[messages.HookPart[Any]], Any],
) -> None:
    call = ai.testing.tool_call(gated, x=1)
    local, local_statuses = await _approval_run(
        ai.Agent(tools=[gated]), call, decide
    )
    async with remote.use_runner(_runner()):
        remote_, remote_statuses = await _approval_run(
            remote.RemoteAgent(tools=[gated]), call, decide
        )

    assert remote_statuses == local_statuses
    assert [ai.testing.fingerprint(m) for m in remote_.messages] == [
        ai.testing.fingerprint(m).replace('"here:', '"runner:')
        for m in local.messages
    ]
    assert not remote._registries


async def test_remote_context_additions_arrive_before_the_next_turn() -> None:
    model = _script(ai.testing.tool_call(locate, x=1))
    agent = remote.RemoteAgent(tools=[locate])
    seen = []
    async with remote.use_runner(_runner()):
        async with agent.run(model, [ai.user_message("go")]) as stream:
            async for event in stream:
                if isinstance(event, events.StreamStart):
                    seen.append([m.role for m in stream.messages])
    assert seen == [["user"], ["user", "assistant", "tool"]]


class _NotingAgent(remote.RemoteAgent):
    async def loop(
        self, context: ai.Context
    ) -> AsyncGenerator[events.AgentEvent]:
        async for event in super().loop(context):
            yield event
        context.add(ai.assistant_message(f"loop ran {where.get()}"))


@pytest.mark.parametrize(
    ("runner", "expected"), [(None, "here"), (_runner(), "runner")]
)
async def test_overridden_loop(runner: Any, expected: str) -> None:
    model = _script(ai.testing.tool_call(locate, x=1))
    agent = _NotingAgent(tools=[locate])
    async with (
        remote.use_runner(runner)
        if runner is not None
        else contextlib.nullcontext()
    ):
        async with agent.run(model, [ai.user_message("go")]) as stream:
            async for _ in stream:
                pass
    assert stream.messages[-1].text == f"loop ran {expected}"
    assert _results(stream) == {"locate": f"{expected}:1"}


def test_with_resolved_key(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ai.get_model("openai/gpt-5")
    monkeypatch.setenv(model.provider.api_key_env or "", "sekrit")
    copied = _copy(remote.with_resolved_key(model))
    monkeypatch.delenv(model.provider.api_key_env or "")
    assert copied.provider.api_key == "sekrit"
    assert model.provider.api_key is None
