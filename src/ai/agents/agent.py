"""Agent, Context, StreamResult, and the stream() function."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import inspect
import json
import typing
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Sequence,
)
from contextlib import AbstractAsyncContextManager
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Protocol,
    Self,
    cast,
    get_type_hints,
    overload,
)

import pydantic

# ``typing.TypeVar`` lacks the ``default=`` kwarg on Python <3.13.
# Use the typing_extensions backport so this works on 3.12 too.
from typing_extensions import TypeVar

from .. import models, telemetry, type_utils, types, util
from ..types import builders
from ..types import events as events_
from ..types.messages import MessageBundle
from . import _middleware as middleware_
from . import hooks as hooks_
from . import runtime


def _unwrap_singleton_group(exc: BaseException) -> BaseException:
    """Collapse nested singleton ``BaseExceptionGroup``s to the inner exception.

    A failing task inside an ``asyncio.TaskGroup`` is always re-raised
    inside an ``ExceptionGroup`` even when there's only one — which
    obscures the real type and message.  When the group has exactly
    one child we unwrap so the original exception, traceback, and
    ``__cause__`` survive.  Multi-error groups stay intact (they
    really do represent concurrent failures).
    """
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


def _error_tool_result(
    exc: BaseException,
    *,
    tool_call_id: str,
    tool_name: str,
) -> events_.ToolCallResult:
    """Build an error ``ToolCallResult`` from an exception.

    Unwraps singleton ``ExceptionGroup``s so the surfaced type and
    message reflect the actual failure.
    """
    unwrapped = _unwrap_singleton_group(exc)
    return tool_result(
        types.messages.ToolResultPart(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=f"{type(unwrapped).__name__}: {unwrapped}",
            result_kind="error",
        ),
        exception=unwrapped,
    )


def _process_interrupted_hooks(messages: list[types.messages.Message]) -> None:
    """Detect an interrupted hook tail and mangle ``messages`` in place.

    This prepares the next agent run to resume correctly.

    Two shapes are recognised:

    1. **Trailing assistant turn with tool calls** (single-tool gating
       or no-tool-results-yet bail-out): mark the assistant message
       ``replay=True`` so ``models.stream`` short-circuits and the
       loop's tool dispatcher re-runs the calls.

    2. **Trailing tool message containing ``is_hook_pending=True``
       results** (concurrent gating: some tools completed, others
       were suspended on a hook): fold the completed (non-pending)
       tool results onto the matching ``ToolCallPart.cached_result``
       of the preceding assistant turn, drop the tool message, and
       mark the assistant message ``replay=True``.  On replay, the
       completed calls short-circuit to the cached value; the
       suspended calls re-run (and pick up the pre-registered hook
       resolution).
    """
    if not messages:
        return

    last = messages[-1]

    # Case 1: trailing assistant turn with tool calls.
    if last.role == "assistant" and last.tool_calls:
        messages[-1] = last.model_copy(update={"replay": True})
        return

    # Case 2: trailing tool message with at least one pending-hook result.
    if (
        len(messages) >= 2
        and last.role == "tool"
        and last.tool_results
        and any(r.is_hook_pending for r in last.tool_results)
    ):
        prev = messages[-2]
        if prev.role != "assistant" or not prev.tool_calls:
            return

        completed_by_id = {
            r.tool_call_id: r
            for r in last.tool_results
            if not r.is_hook_pending
        }

        new_parts: list[types.messages.Part] = []
        for part in prev.parts:
            if (
                isinstance(part, types.messages.ToolCallPart)
                and part.tool_call_id in completed_by_id
            ):
                part = part.model_copy(
                    update={"cached_result": completed_by_id[part.tool_call_id]}
                )
            new_parts.append(part)

        messages[-2] = prev.model_copy(
            update={"parts": new_parts, "replay": True}
        )
        messages.pop()


def _aggregator_cls(
    factory: Any,
) -> type[events_.Aggregator[Any, Any, Any]] | None:
    """Resolve a tool's aggregator factory to the underlying class.

    Tools may declare the aggregator as a class directly (``LastAggregator``)
    or via an ``Aggregate`` marker that wraps it.  This normalizes both forms.
    """
    if factory is None:
        return None
    if isinstance(factory, type) and issubclass(factory, events_.Aggregator):
        return factory
    inner = getattr(factory, "_factory", None)
    if isinstance(inner, type) and issubclass(inner, events_.Aggregator):
        return inner
    return None


def _populate_model_inputs(
    messages: Sequence[types.messages.Message],
    tools_by_name: dict[str, AgentTool],
) -> None:
    """Set ``model_input`` on tool results that arrived without one.

    Tool execution sets ``model_input`` directly; this fills in the
    value for tool results that were reconstructed from a wire round-
    trip (e.g. the AI SDK UI inbound path) and never had it computed.
    """
    for msg in messages:
        if msg.role != "tool":
            continue
        for part in msg.tool_results:
            if part.has_model_input or part.is_error or part.is_hook_pending:
                continue
            tool = tools_by_name.get(part.tool_name)
            if tool is None:
                continue
            agg_cls = _aggregator_cls(tool.aggregator)
            if agg_cls is None:
                continue
            part.set_model_input(agg_cls.to_model_input(part.result))


class SimpleAggregator[Item, Result](events_.Aggregator[Item, Result, Result]):
    @classmethod
    def to_model_input(cls, snapshot: Result) -> Result:
        return snapshot


class ConcatAggregator(SimpleAggregator[str, str]):
    def __init__(self, *, delim: str = "") -> None:
        self._parts: list[str] = []
        self._delim = delim

    def feed(self, item: str) -> None:
        self._parts.append(item)

    def snapshot(self) -> str:
        return self._delim.join(self._parts)


class LastAggregator[T](SimpleAggregator[T, T | None]):
    def __init__(self) -> None:
        self._val: T | None = None

    def feed(self, item: T) -> None:
        self._val = item

    def snapshot(self) -> T | None:
        return self._val


class MessageAggregator(
    events_.Aggregator[events_.AgentEvent, MessageBundle, str]
):
    def __init__(self) -> None:
        self._messages: list[types.messages.Message] = []
        self._index_by_id: dict[str, int] = {}

    def feed(self, item: events_.AgentEvent) -> None:
        if isinstance(item, events_.PartialToolCallResult | events_.RunBlocked):
            return
        msg = item.message
        if msg is None:
            return
        # Later snapshots of a message replace earlier ones in place.
        # Snapshots of the same message are not always consecutive —
        # e.g. a tool-result message can land between two snapshots of
        # the assistant message that called it — so dedupe by id, not
        # by checking the tail.
        index = self._index_by_id.get(msg.id)
        if index is None:
            self._index_by_id[msg.id] = len(self._messages)
            self._messages.append(msg)
        else:
            self._messages[index] = msg

    def snapshot(self) -> MessageBundle:
        return MessageBundle(messages=tuple(self._messages))

    @classmethod
    def to_model_input(cls, snapshot: MessageBundle | Any) -> str:
        if not isinstance(snapshot, MessageBundle):
            snapshot = MessageBundle.model_validate(snapshot)
        for m in reversed(snapshot.messages):
            if m.role == "assistant" and m.text:
                return m.text
        return ""


class Aggregate:
    r"""Marker for declaring an aggregator on a tool's return type.

    Place inside ``Annotated`` metadata to attach an aggregator factory
    to an async-generator tool::

        type SubAgentTool = Annotated[
            AsyncGenerator[ai.events.AgentEvent], Aggregate(MessageAggregator)
        ]

        @ai.tool
        async def research(topic: str) -> SubAgentTool:
            ...

    Extra kwargs are passed to the factory each time it is invoked::

        type Joined = Annotated[
            AsyncGenerator[str], Aggregate(ConcatAggregator, delim="\\n")
        ]
    """

    def __init__(
        self,
        factory: Callable[..., events_.Aggregator[Any, Any, Any]],
        /,
        **kwargs: Any,
    ) -> None:
        self._factory = factory
        self._kwargs = kwargs

    def __call__(self) -> events_.Aggregator[Any, Any, Any]:
        return self._factory(**self._kwargs)

    def __repr__(self) -> str:
        kw = ", ".join(f"{k}={v!r}" for k, v in self._kwargs.items())
        sep = ", " if kw else ""
        name = getattr(
            self._factory, "__name__", self._factory.__class__.__name__
        )
        return f"Aggregate({name}{sep}{kw})"


type StreamingStatusTool[T] = Annotated[
    AsyncGenerator[T], Aggregate(LastAggregator)
]
"""Async-generator tool whose final yielded value becomes the tool result.

Intermediate yields stream to the consumer as ``PartialToolCallResult``
events; the last yield is what the model sees::

    @ai.tool
    async def fetch(url: str) -> StreamingStatusTool[str]:
        yield "connecting..."
        yield "downloading..."
        yield body  # this is the tool result
"""


type SubAgentTool = Annotated[
    AsyncGenerator[events_.AgentEvent], Aggregate(MessageAggregator)
]
"""Async-generator tool that streams a sub-agent's events.

The collected messages flow to the consumer; the final assistant text
becomes the tool result the parent model sees::

    @ai.tool
    async def research(topic: str) -> SubAgentTool:
        sub = ai.Agent(tools=[...])
        async with sub.run(model, messages) as stream:
            async for event in stream:
                yield event
"""


type StreamingTextTool = Annotated[
    AsyncGenerator[str], Aggregate(ConcatAggregator)
]
"""Async-generator tool whose yielded chunks concatenate into the result.

Each yield streams to the consumer as a ``PartialToolCallResult``;
the model sees the full concatenation as the tool result::

    @ai.tool
    async def render(prompt: str) -> StreamingTextTool:
        async for chunk in some_text_stream(prompt):
            yield chunk

For a custom delimiter, drop down to the marker form:
``Annotated[AsyncGenerator[str], Aggregate(ConcatAggregator, delim="\\n")]``.
"""


def _return_type_from_callable(fn: Callable[..., Any]) -> Any:
    try:
        return get_type_hints(fn, include_extras=True).get("return")
    except Exception:
        return None


def _stream_item_type(return_type: Any) -> Any:
    resolved = type_utils.resolve_type_alias(return_type)
    if typing.get_origin(resolved) is typing.Annotated:
        resolved = typing.get_args(resolved)[0]

    if typing.get_origin(resolved) is not AsyncGenerator:
        return None

    args = typing.get_args(resolved)
    return args[0] if args else Any


def _aggregator_result_type(
    aggregator: Callable[[], events_.Aggregator[Any, Any, Any]] | None,
    stream_item_type: Any,
) -> Any:
    agg_cls = _aggregator_cls(aggregator)
    if agg_cls is None:
        return None

    args = type_utils.generic_base_args(agg_cls, events_.Aggregator)
    if args is None:
        return None

    item_type, result_type, _model_input_type = args
    bindings = type_utils.bind_typevars(item_type, stream_item_type)
    return type_utils.replace_typevars(result_type, bindings)


def _tool_result_type_from_return_annotation(
    return_type: Any,
    aggregator: Callable[[], events_.Aggregator[Any, Any, Any]] | None,
) -> Any:
    """Return the type used to validate ``ToolResultPart.result``.

    Normal tools store the function's return value directly, so the function's
    return annotation is the result type.

    Async-generator tools store ``aggregator.snapshot()`` instead. For those,
    read the ``Result`` parameter from ``Aggregator[Item, Result, ModelInput]``
    on the configured aggregator class, binding generic aggregators from the
    stream item type when possible (for example ``StreamingStatusTool[str]`` +
    ``LastAggregator[T]`` becomes ``str | None``).
    """
    if return_type is None:
        return None

    item_type = _stream_item_type(return_type)
    if item_type is not None:
        return _aggregator_result_type(aggregator, item_type)

    return return_type


def _aggregate_from_return_type(fn: Callable[..., Any]) -> Aggregate | None:
    """Find an ``Aggregate`` marker in *fn*'s return-type metadata, if any.

    Handles three shapes:

    * ``Annotated[X, Aggregate(...)]`` directly,
    * a PEP 695 alias ``type Foo = Annotated[X, Aggregate(...)]``,
    * a parameterized alias ``type Foo[T] = Annotated[X[T], Aggregate(...)]``.
    """
    ret = _return_type_from_callable(fn)
    if ret is None:
        return None

    if isinstance(ret, typing.TypeAliasType):
        ret = ret.__value__
    else:
        origin = typing.get_origin(ret)
        if isinstance(origin, typing.TypeAliasType):
            ret = origin.__value__

    metadata = getattr(ret, "__metadata__", ())
    matches = [m for m in metadata if isinstance(m, Aggregate)]
    if len(matches) > 1:
        name = getattr(fn, "__name__", fn.__class__.__name__)
        raise TypeError(
            f"Tool {name!r} has multiple Aggregate markers in its "
            "return-type annotation; expected at most one"
        )
    return matches[0] if matches else None


Tool = types.tools.Tool


@dataclasses.dataclass(frozen=True)
class AgentTool:
    """Agent-owned executable tool paired with its model-facing declaration."""

    tool: Tool
    fn: Callable[..., Any]
    validator: type[pydantic.BaseModel] | None = None
    is_gen: bool = False
    aggregator: Callable[[], events_.Aggregator[Any, Any, Any]] | None = None
    return_type: Any = None

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def require_approval(self) -> bool:
        return self.tool.require_approval

    @property
    def _aggregator(
        self,
    ) -> Callable[[], events_.Aggregator[Any, Any, Any]] | None:
        return self.aggregator


def _validate_kwargs(
    tool: AgentTool,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Validate kwargs and return normalized Python values."""
    if tool.validator is not None:
        validated = tool.validator.model_validate(kwargs)
        return {f: getattr(validated, f) for f in type(validated).model_fields}
    return kwargs


@overload
def tool[**P, R](fn: Callable[P, Awaitable[R]], /) -> AgentTool: ...


@overload
def tool[**P, T](fn: Callable[P, AsyncGenerator[T]], /) -> AgentTool: ...


@overload
def tool[**P](
    *,
    require_approval: bool,
    return_type: Any = None,
) -> Callable[[Callable[P, Any]], AgentTool]: ...


@overload
def tool[**P](
    *,
    return_type: Any,
    require_approval: bool = False,
) -> Callable[[Callable[P, Any]], AgentTool]: ...


@overload
def tool[**P](
    *,
    aggregator: Callable[[], events_.Aggregator[Any, Any, Any]],
    require_approval: bool = False,
    return_type: Any = None,
) -> Callable[[Callable[P, AsyncGenerator[Any]]], AgentTool]: ...


def tool[**P, T, R](
    fn: Callable[P, Awaitable[R]]
    | Callable[P, AsyncGenerator[T]]
    | None = None,
    /,
    *,
    aggregator: Callable[[], events_.Aggregator[Any, Any, Any]] | None = None,
    require_approval: bool = False,
    return_type: Any = None,
) -> (
    Callable[[Callable[P, AsyncGenerator[Any]]], AgentTool]
    | Callable[[Callable[P, Awaitable[R]]], AgentTool]
    | AgentTool
):
    """Turn an async function into a :class:`Tool`.

    For async-generator tools, declare the aggregator either via the
    ``aggregator=`` keyword argument or by annotating the return type
    with an :class:`Aggregate` marker (e.g. via the :data:`SubAgentTool`
    or :data:`StreamingStatusTool` aliases).  Specifying both raises
    ``TypeError``. Pass ``return_type=`` to override the type used when
    validating round-tripped tool results.
    """

    def wrap(fn: Any) -> AgentTool:
        sig = inspect.signature(fn)
        hints = get_type_hints(fn) if hasattr(fn, "__annotations__") else {}

        fields: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            param_type = hints.get(param_name, str)
            if param.default is inspect.Parameter.empty:
                fields[param_name] = (param_type, ...)
            else:
                fields[param_name] = (param_type, param.default)

        validator = pydantic.create_model(f"{fn.__name__}_Args", **fields)

        annotated_return_type = _return_type_from_callable(fn)
        annotated_aggregate = _aggregate_from_return_type(fn)
        if annotated_aggregate is not None and aggregator is not None:
            raise TypeError(
                f"Tool {fn.__name__!r}: aggregator was declared both via "
                "the `aggregator=` argument and via an Aggregate marker "
                "in the return-type annotation; specify only one"
            )
        effective_aggregator = aggregator or annotated_aggregate

        tool_decl = Tool(
            kind="function",
            name=fn.__name__,
            spec=types.tools.ToolSpec(
                description=inspect.getdoc(fn) or "",
                params=validator.model_json_schema(),
            ),
            require_approval=require_approval,
        )

        return AgentTool(
            tool=tool_decl,
            fn=fn,
            validator=validator,
            is_gen=inspect.isasyncgenfunction(fn),
            aggregator=effective_aggregator,
            return_type=return_type
            or _tool_result_type_from_return_annotation(
                annotated_return_type, effective_aggregator
            ),
        )

    if fn is None:
        return wrap
    else:
        return wrap(fn)


class BoundToolCall:
    """Callable that binds a :class:`ToolCallPart` to its :class:`AgentTool`.

    Calling it executes the tool and returns a ``role="tool"`` message.
    """

    def __init__(
        self,
        part: types.messages.ToolCallPart,
        tool: AgentTool,
    ) -> None:
        self._part = part
        self._tool = tool
        self._kwargs: dict[str, Any] | None = None

    @property
    def id(self) -> str:
        return self._part.tool_call_id

    @property
    def name(self) -> str:
        return self._part.tool_name

    @property
    def fn(self) -> Callable[..., Awaitable[Any]]:
        return self._tool.fn

    @property
    def kwargs(self) -> dict[str, Any]:
        if self._kwargs is None:
            kwargs = (
                json.loads(self._part.tool_args) if self._part.tool_args else {}
            )
            self._kwargs = _validate_kwargs(self._tool, kwargs)
        return dict(self._kwargs)

    async def __call__(self, **overrides: Any) -> events_.ToolCallResult:
        """Execute the tool and return a :class:`ToolCallResult`."""
        data = telemetry.ToolExecutionSpanData(
            tool_name=self._part.tool_name,
            tool_call_id=self._part.tool_call_id,
        )

        # Replay-from-pending-hook short-circuit: if a prior run already
        # produced a result for this call (cached on the ToolCallPart
        # by ``_process_interrupted_hooks``), return it without
        # re-executing the tool.
        cached = self._part.cached_result
        if cached is not None:
            async with telemetry.span(data, replay=True):
                data.result = cached.result
                data.is_error = cached.is_error
                msg = builders.tool_message(cached)
                return events_.ToolCallResult(
                    message=msg, results=msg.tool_results
                )

        # Best-effort parse so middleware sees usable kwargs when possible.
        # If parsing fails, middleware still gets the raw tool_call_id /
        # tool_name and can replace kwargs before _real() executes.
        try:
            base_kwargs = self.kwargs
        except Exception:
            base_kwargs = {}

        if overrides:
            # Overrides come from user code, not the model — validate
            # eagerly so programming errors surface immediately.
            base_kwargs = _validate_kwargs(
                self._tool, {**base_kwargs, **overrides}
            )

        call = middleware_.ToolContext(
            tool_call_id=self._part.tool_call_id,
            tool_name=self._part.tool_name,
            kwargs=base_kwargs,
        )

        tool = self._tool

        async def _real(
            call: middleware_.ToolContext,
        ) -> events_.ToolCallResult:
            result: Any
            model_input: Any
            try:
                kwargs = _validate_kwargs(tool, call.kwargs)
                if tool.is_gen:
                    # Generator tool (e.g. agent-as-a-tool): drain the async
                    # generator, forward each yielded value to the runtime for
                    # real-time streaming, then capture both the aggregator
                    # snapshot (the rich shape that flows to the UI) and the
                    # model-facing value (what the LLM sees on its next turn).
                    assert tool.aggregator
                    agg = await _aggregate_from(
                        tool.fn(**kwargs),
                        tool_call_id=call.tool_call_id,
                        tool_name=call.tool_name,
                        aggregator=tool.aggregator,
                    )
                    result = agg.snapshot()
                    model_input = agg.get_model_input()
                else:
                    result = await tool.fn(**kwargs)
                    model_input = result
            except Exception as exc:
                return _error_tool_result(
                    exc,
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                )
            part = types.messages.ToolResultPart(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                result=result,
                result_kind=types.messages.ToolResultPart.kind_for(result),
            )
            part.set_model_input(model_input)
            return tool_result(part)

        data.args = base_kwargs
        chain = middleware_._build_tool_chain(_real)
        async with telemetry.span(data) as sp:
            res = await chain(call)
            if res.results:
                data.result = res.results[0].result
                data.is_error = any(p.is_error for p in res.results)
            # A tool exception is caught and converted to an error
            # result before it reaches this block, so it never hits the
            # span's own except path — thread it through explicitly.
            sp.error = res.exception
            return res


class GatedToolCall:
    """ToolCall-shaped wrapper that awaits an approval hook before executing.

    ``ToolRunner.schedule`` only consumes the ``__call__`` shape of
    ``ToolCall``; this wrapper supplies the same shape while inserting
    the hook await + denial path before the underlying tool runs.
    """

    def __init__(self, tc: ToolCall) -> None:
        self._tc = tc

    @property
    def id(self) -> str:
        return self._tc.id

    @property
    def name(self) -> str:
        return self._tc.name

    @property
    def fn(self) -> Callable[..., Awaitable[Any]]:
        return self._tc.fn

    @property
    def kwargs(self) -> dict[str, Any]:
        return self._tc.kwargs

    async def __call__(self) -> events_.ToolCallResult:
        tc = self._tc
        # If the model sent invalid arguments, skip the approval hook
        # and return the validation error directly as a tool result.
        try:
            hook_kwargs = tc.kwargs
        except Exception as exc:
            return _error_tool_result(
                exc,
                tool_call_id=tc.id,
                tool_name=tc.name,
            )
        try:
            approval = await hooks_.hook(
                f"approve_{tc.id}",
                payload=types.tools.ToolApproval,
                metadata={"tool": tc.name, "kwargs": hook_kwargs},
                tool_call_id=tc.id,
            )
        except hooks_.HookPendingError as e:
            return pending_tool_result(
                e.hook, tool_call_id=tc.id, tool_name=tc.name
            )
        if approval.granted:
            return await tc()
        return tool_result(
            tool_call_id=tc.id,
            tool_name=tc.name,
            result=f"Rejected: {approval.reason}",
            is_error=True,
        )


class ToolCallCallable(Protocol):
    """Anything ``ToolRunner.schedule`` can accept.

    Satisfied by :class:`ToolCall` and by any zero-arg callable returning
    a coroutine that resolves to a :class:`~ai.agents.events.ToolCallResult`
    — e.g. an inline closure that gates the tool behind an approval hook.
    """

    def __call__(self) -> Coroutine[Any, Any, events_.ToolCallResult]: ...


class ToolCall(ToolCallCallable, Protocol):
    """Something with all the key information for a tool call."""

    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def fn(self) -> Callable[..., Awaitable[Any]]: ...

    @property
    def kwargs(self) -> dict[str, Any]: ...


class _RestartableToolStream:
    def __init__(self, tr: ToolRunner) -> None:
        self._tr = tr

    def __aiter__(self) -> AsyncGenerator[events_.ToolCallResult]:
        return self._tr._iterate()


class ToolRunner:
    def __init__(self) -> None:
        self._new_results: list[events_.ToolCallResult] = []
        self._tool_results: list[events_.ToolCallResult] = []
        self._tg_base = util.TaskGroup()
        self._waiter: util.MultiWaiter[events_.ToolCallResult] = (
            util.MultiWaiter()
        )

    async def __aenter__(self) -> Self:
        self._tg = await self._tg_base.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        self._waiter.clear()
        return await self._tg_base.__aexit__(*args)

    def events(self) -> _RestartableToolStream:
        return _RestartableToolStream(self)

    def schedule(self, tc: ToolCallCallable) -> None:
        """Schedule a tool call (or any callable producing a ToolCallResult).

        See :class:`ToolCallCallable` — accepts both :class:`ToolCall` and
        any zero-arg callable returning a coroutine that resolves to a
        :class:`ToolCallResult`.  The latter lets you wrap a ``ToolCall``
        in custom logic (e.g. an approval hook await) and still ride the
        runner's merge-and-iterate flow.
        """
        self._waiter.add(self._tg.create_task(tc()))

    def add_result(self, res: events_.ToolCallResult) -> None:
        async def _feed() -> events_.ToolCallResult:
            return res

        self._waiter.add(self._tg.create_task(_feed()))

    def get_tool_message(self) -> types.messages.Message | None:
        if self._tool_results:
            return builders.tool_message(
                *[t.message for t in self._tool_results]
            )
        return None

    async def _iterate(self) -> AsyncGenerator[events_.ToolCallResult]:
        while self._waiter.tasks():
            t = await self._waiter
            try:
                res = t.result()
            except asyncio.CancelledError:
                # If a task got cancelled, that's fine.
                # Need to catch it or the whole runner gets zapped.
                continue

            assert res is not None
            self._tool_results.append(res)
            yield res


class Context(pydantic.BaseModel):
    """Everything that goes into the LLM."""

    model: models.Model
    messages: list[types.messages.Message]
    tools: list[Tool]
    output_type: type[pydantic.BaseModel] | None = pydantic.Field(
        default=None, exclude=True, repr=False
    )
    params: models.InferenceRequestParams | None = pydantic.Field(
        default=None, exclude=True, repr=False
    )

    _agent_tools_by_name: dict[str, AgentTool] = pydantic.PrivateAttr(
        default_factory=dict
    )

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self._agent_tools_by_name = {}

    def keep_running(self) -> bool:
        """Call at top of an agent loop to see whether to keep running."""
        if not self.messages:
            return False

        last_message = self.messages[-1]
        # Bail out if any tool result in the last message is a
        # pending-hook placeholder. There's nothing we can do until
        # those are resolved and we get called again.
        if any(r.is_hook_pending for r in last_message.tool_results):
            return False
        return last_message.replay or last_message.role not in (
            "assistant",
            "internal",
        )

    @overload
    def resolve(self, tool_part: types.messages.ToolCallPart) -> ToolCall: ...
    @overload
    def resolve(
        self, tool_part: Sequence[types.messages.ToolCallPart]
    ) -> list[ToolCall]: ...

    def resolve(
        self,
        tool_part: types.messages.ToolCallPart
        | Sequence[types.messages.ToolCallPart],
    ) -> ToolCall | list[ToolCall]:
        """Resolve ToolCallPart(s) into callable ToolCall object(s)."""
        if isinstance(tool_part, types.messages.ToolCallPart):
            tool = self._agent_tools_by_name.get(tool_part.tool_name)
            if tool is None:
                raise KeyError(
                    "No agent executor registered for tool "
                    f"{tool_part.tool_name!r}"
                )
            tc = BoundToolCall(part=tool_part, tool=tool)
            if tool.require_approval:
                return GatedToolCall(tc)
            return tc
        return [self.resolve(tp) for tp in tool_part]

    def add(
        self,
        message: types.messages.Message
        | Sequence[types.messages.Message]
        | None,
    ) -> None:
        """Append message(s) to the context, skipping any flagged ``replay``.

        Replay-flagged messages come from ``models.stream`` short-
        circuiting an existing assistant turn (resume-after-approval).
        The default loop calls ``context.add(stream.message)`` after
        every stream — the flag lets that be a no-op on replay rather
        than producing a duplicate turn.
        """
        if message is None:
            return
        msgs = (
            [message]
            if isinstance(message, types.messages.Message)
            else list(message)
        )
        for msg in msgs:
            if msg.replay:
                continue
            self.messages.append(msg)


# Agent run output type.  Defaults to ``str``: when ``Agent.run`` was
# called without an ``output_type``, ``AgentStream.output`` returns the
# final assistant message's concatenated text.
AgentOutputT = TypeVar("AgentOutputT", default=str)


class AgentStream(Generic[AgentOutputT]):
    """Async-iterable wrapper around an agent run's event stream.

    Exposes the run's :class:`Context` via :attr:`context` so callers can
    inspect (or use) the live messages/tools without threading them
    through their own bookkeeping::

        async with agent.run(model, messages) as stream:
            async for event in stream:
                ...
            print(stream.context.messages)

    Structurally satisfies the ``AsyncGenerator`` protocol by delegating
    to the underlying generator, so it can be passed directly to
    :func:`yield_from` and other APIs that expect an async generator.
    """

    def __init__(
        self,
        gen: AsyncGenerator[events_.AgentEvent],
        context: Context,
    ) -> None:
        self._gen = gen
        self._context = context
        self._blocked = False
        self._pending_hooks: dict[str, types.messages.HookPart[Any]] = {}

    def _track(self, event: events_.AgentEvent) -> events_.AgentEvent:
        # Fold delivered events into the exposed run state, so that the
        # properties below are always consistent with the events the
        # consumer has seen so far.
        if isinstance(event, events_.RunBlocked):
            self._blocked = True
        elif isinstance(event, events_.HookEvent):
            if event.hook.status == "pending":
                self._pending_hooks[event.hook.hook_id] = event.hook
            else:
                # A blocked run can only resume via a hook resolution
                # or cancellation (see RunBlocked), so this is also the
                # unblock signal.
                self._pending_hooks.pop(event.hook.hook_id, None)
                self._blocked = False
        return event

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> events_.AgentEvent:
        return self._track(await self._gen.__anext__())

    async def asend(self, value: Any) -> events_.AgentEvent:
        return self._track(await self._gen.asend(value))

    async def athrow(self, *args: Any, **kwargs: Any) -> events_.AgentEvent:
        return self._track(await self._gen.athrow(*args, **kwargs))

    async def aclose(self) -> None:
        await self._gen.aclose()

    @property
    def context(self) -> Context:
        return self._context

    @property
    def blocked(self) -> bool:
        """Whether the run is blocked awaiting external hook resolution.

        Set by a delivered :class:`~ai.types.events.RunBlocked` event,
        cleared by the hook resolution that follows.  True means
        nothing in the run can make progress until a pending hook (see
        :attr:`pending_hooks`) is resolved — e.g. a tool approval.
        After the run ends this keeps its final value, so serverless
        drivers can distinguish "turn complete" from "turn suspended
        awaiting input".
        """
        return self._blocked

    @property
    def pending_hooks(self) -> list[types.messages.HookPart[Any]]:
        """Hooks that are pending as of the last delivered event."""
        return list(self._pending_hooks.values())

    @property
    def messages(self) -> list[types.messages.Message]:
        return self._context.messages

    @property
    def output(self) -> AgentOutputT:
        """Return the run's output.

        Defaults to the final assistant message's concatenated text.
        When an ``output_type`` was passed, the assistant message's text
        is validated as JSON against that type and the parsed instance
        is returned.
        """
        last = self._context.messages[-1]
        return cast("AgentOutputT", last.get_output(self._context.output_type))


def tool_result(
    *items: types.messages.Message
    | types.messages.ToolResultPart
    | events_.ToolCallResult
    | list[types.messages.Message],
    tool_call_id: str | None = None,
    result: Any = None,
    tool_name: str = "",
    is_error: bool = False,
    exception: BaseException | None = None,
) -> events_.ToolCallResult:
    """Create a :class:`ToolCallResult` from tool messages, parts, or kwargs.

    Accepts ``ToolCallResult`` items (extracts their ``.message``),
    plain ``Message`` objects, ``ToolResultPart`` instances, or keyword
    arguments matching :func:`ai.tool_message`::

        yield ai.tool_result(*(t.result() for t in tasks))
        ai.tool_result(tool_call_id="tc-1", result="denied", is_error=True)

    Pass ``exception=`` to attach the raised :class:`BaseException` to
    the returned event for richer logging downstream — the wire-bound
    ``ToolResultPart`` only carries ``str(exc)``.
    """
    if tool_call_id is not None:
        msg = builders.tool_message(
            tool_call_id=tool_call_id,
            result=result,
            tool_name=tool_name,
            is_error=is_error,
        )
        return events_.ToolCallResult(
            message=msg, results=msg.tool_results, exception=exception
        )

    unwrapped: list[types.messages.Message | types.messages.ToolResultPart] = []
    for item in items:
        if isinstance(item, events_.ToolCallResult):
            unwrapped.append(item.message)
        elif isinstance(item, list):
            unwrapped.extend(item)
        else:
            unwrapped.append(item)
    msg = builders.tool_message(*unwrapped)
    return events_.ToolCallResult(
        message=msg, results=msg.tool_results, exception=exception
    )


def pending_tool_result(
    hook: types.messages.HookPart[Any],
    *,
    tool_call_id: str,
    tool_name: str = "",
) -> events_.ToolCallResult:
    """Build an error :class:`ToolCallResult` for a tool call pending on a hook.

    Use in approval-gated flows when a hook abort (e.g. ``HookPendingError``)
    leaves a tool call without a real result.  The placeholder is flagged
    ``is_error=True`` and keeps the assistant turn well-formed (every
    ``tool_call`` paired with a ``tool_result``) so the run can be replayed
    on the next invocation once the hook is resolved::

        try:
            approval = await ai.hook(...)
        except ai.HookPendingError as e:
            return ai.pending_tool_result(
                e.hook, tool_call_id=tc.id, tool_name=tc.name
            )

    The hook itself is surfaced separately via the ``HookPart`` already
    emitted by ``ai.hook()`` (status=``"pending"``) which downstream
    consumers (e.g. the ai-sdk UI bridge) use to render the actual
    approval state.
    """
    part = types.messages.ToolResultPart(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        result=f"Pending on hook {hook.hook_id!r}",
        result_kind="error",
        is_hook_pending=True,
    )
    msg = types.messages.Message(role="tool", parts=[part])
    return events_.ToolCallResult(message=msg, results=msg.tool_results)


async def yield_from[T, S, R](
    source: AsyncGenerator[T],
    *,
    aggregator: Callable[[], events_.Aggregator[T, S, R]],
    # TODO: is this what we really want for labelling?
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    label: object = None,
) -> R:
    """Drain *source*, forwarding each event to the current runtime.

    Use inside a custom loop to stream messages from a sub-agent to the
    consumer without adding them to the parent agent's message history::

        async with sub.run(model, msgs) as stream:
            result = await yield_from(stream, label="researcher")

    Works with :func:`asyncio.gather` for concurrent fan-out::

        async with a.run(model, m1) as sa, b.run(model, m2) as sb:
            r1, r2 = await asyncio.gather(
                yield_from(sa, label="a"),
                yield_from(sb, label="b"),
            )

    Each forwarded event is wrapped in a :class:`PartialToolCallResult`
    carrying ``label`` (and optionally ``tool_call_id`` / ``tool_name``)
    so the consumer can route by source.

    Returns the final message's text (empty string if no messages).
    """
    agg = await _aggregate_from(
        source,
        aggregator=aggregator,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        label=label,
    )
    return agg.get_model_input()


async def _aggregate_from[T, S, R](
    source: AsyncGenerator[T],
    *,
    aggregator: Callable[[], events_.Aggregator[T, S, R]],
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    label: object = None,
) -> events_.Aggregator[T, S, R]:
    """Drain *source* into a fresh aggregator, forwarding partial events.

    Returns the live aggregator so callers can consume both the snapshot
    (the rich shape stored on ``ToolResultPart.result``) and the
    model-facing value (set via ``ToolResultPart.set_model_input``)
    without re-aggregating.
    """
    agg = aggregator()

    rt = runtime.get_runtime()
    async with contextlib.aclosing(source) as src:
        async for item in src:
            agg.feed(item)
            await rt.put_event(
                events_.PartialToolCallResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    label=label,
                    value=item,
                    aggregator_factory=aggregator,
                )
            )
    return agg


class Agent:
    """Bag of configuration: model + tools + loop."""

    TOOLS: ClassVar[list[AgentTool]] = []
    """Class-level default tools.

    Subclasses can declare a fixed tool set here::

        class WeatherAgent(ai.Agent):
            TOOLS = [get_weather, get_population]

    Tools passed to ``__init__`` are appended to ``TOOLS``, so callers can
    extend the class-level set per instance.
    """

    def __init__(
        self,
        *,
        tools: Sequence[AgentTool | Tool] | None = None,
    ) -> None:
        self._tools: list[AgentTool] = []
        self._provider_tools: list[Tool] = []
        for t in [*self.TOOLS, *(tools or [])]:
            if isinstance(t, AgentTool):
                self._tools.append(t)
            else:
                self._provider_tools.append(t)

    @property
    def tools(self) -> list[AgentTool]:
        """The agent's registered tools (read-only copy)."""
        return list(self._tools)

    async def loop(
        self, context: Context
    ) -> AsyncGenerator[events_.AgentEvent]:
        """Stream, execute tools, repeat.

        Override in a subclass to customise the agent's control flow.
        (A custom loop keeps run/model/tool/hook spans — those live in
        the layers it calls — but loses per-step grouping unless it
        opens its own ``telemetry.span`` blocks.)
        """
        while context.keep_running():
            async with (
                telemetry.span(telemetry.LoopTurnSpanData()),
                models.stream(context=context) as stream,
                ToolRunner() as tr,
            ):
                async for event in util.merge(stream, tr.events()):
                    yield event

                    if isinstance(event, types.events.ToolEnd):
                        tool = context.resolve(event.tool_call)
                        tr.schedule(tool)

                context.add(stream.message)
                # This adds the tool message to the history, which
                # also has the effect of causing another turn through
                # the loop.
                context.add(tr.get_tool_message())

    @overload
    def run(
        self,
        model: models.Model,
        messages: list[types.messages.Message],
        *,
        params: models.InferenceRequestParams | None = None,
        _middleware: list[middleware_._Middleware] | None = None,
    ) -> AbstractAsyncContextManager[AgentStream[str]]: ...
    @overload
    def run[T: pydantic.BaseModel](
        self,
        model: models.Model,
        messages: list[types.messages.Message],
        *,
        output_type: type[T],
        params: models.InferenceRequestParams | None = None,
        _middleware: list[middleware_._Middleware] | None = None,
    ) -> AbstractAsyncContextManager[AgentStream[T]]: ...
    def run(
        self,
        model: models.Model,
        messages: list[types.messages.Message],
        *,
        output_type: type[pydantic.BaseModel] | None = None,
        params: models.InferenceRequestParams | None = None,
        _middleware: list[middleware_._Middleware] | None = None,
    ) -> AbstractAsyncContextManager[AgentStream[Any]]:
        """Run the agent loop, yielding events to the consumer.

        Used as an async context manager whose value the event stream,
        extended with the `context` and `messages` of the stream::

            async with agent.run(model, messages) as stream:
                async for event in stream:
                    ...
                print(stream.messages)

        Args:
            model: The model to use for LLM calls.
            messages: Initial conversation messages.
            output_type: Optional Pydantic model the model's output must
                conform to.  When set, ``stream.output`` validates the
                final assistant message's text against it.

        To attribute a sub-agent's events to a branch, wrap the run in
        ``yield_from(..., label=...)`` — the label flows via
        ``PartialToolCallResult`` rather than on individual messages.

        """
        return self._run(
            model,
            messages,
            output_type=output_type,
            params=params,
            _middleware=_middleware,
        )

    @contextlib.asynccontextmanager
    async def _run(
        self,
        model: models.Model,
        messages: list[types.messages.Message],
        *,
        output_type: type[pydantic.BaseModel] | None,
        params: models.InferenceRequestParams | None,
        _middleware: list[middleware_._Middleware] | None,
    ) -> AsyncIterator[AgentStream[Any]]:
        context = Context(
            model=model,
            messages=list(messages),
            tools=[
                *[t.tool for t in self._tools],
                *self._provider_tools,
            ],
            output_type=output_type,
            params=params,
        )
        context._agent_tools_by_name = {t.name: t for t in self._tools}
        _populate_model_inputs(context.messages, context._agent_tools_by_name)
        _process_interrupted_hooks(context.messages)

        async def _real(call: Context) -> AsyncGenerator[events_.AgentEvent]:
            tracker = events_.RunStateTracker()
            source = self.loop(call)
            async with contextlib.aclosing(runtime.run(source)) as events:
                async for event in events:
                    # Feed the tracker before the replay filter: replayed
                    # StreamEnds carry the tool calls the dispatcher is
                    # about to re-run, which the fold must count.
                    transition = tracker.feed(event)
                    # Drop replay-flagged events: they're a control-flow
                    # signal for the loop's tool dispatcher (which already
                    # ran by the time we see the event here), not user-
                    # facing output.
                    if not (
                        isinstance(event, events_.BaseEvent) and event.replay
                    ):
                        yield event
                    if transition is not None:
                        yield transition

        async def _stream() -> AsyncGenerator[events_.AgentEvent]:
            run_data = telemetry.RunSpanData(
                agent=type(self).__name__,
                model=model.id,
                messages=list(messages),
            )
            # Activate middleware for this run (and everything it calls).
            # When middleware is None (default), inherit the parent's
            # middleware from the context var — this lets nested agents
            # share middleware.  When middleware is explicitly provided,
            # *extend* the parent stack so that outer cross-cutting
            # concerns (tracing, durability) are preserved.  Pass
            # ``_middleware=[]`` to clear the stack entirely.
            async with telemetry.span(run_data):
                mw_token: middleware_.Token | None = None
                if _middleware is not None:
                    parent = middleware_.get()
                    mw_token = middleware_.activate(parent + _middleware)
                try:
                    chain = middleware_._build_agent_run_chain(_real)
                    async with contextlib.aclosing(chain(context)) as events:
                        async for event in events:
                            yield event
                finally:
                    if mw_token is not None:
                        middleware_.deactivate(mw_token)

        # close the event generator on exit from the ``run()`` block.
        astream = AgentStream(_stream(), context)
        try:
            yield astream
        finally:
            await astream.aclose()
