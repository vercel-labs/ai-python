"""Agent, Context, StreamResult, and the stream() function."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable, Sequence
from typing import Any, Protocol, Self, get_type_hints, overload

import pydantic

from .. import middleware as middleware_
from .. import models, types, util
from ..types import builders
from . import events as events_
from . import runtime


class Tool[**P, R]:
    """Wraps async function, introspects schema, attaches a validator"""

    def __init__(
        self,
        fn: Callable[P, Awaitable[R]],
        schema: types.ToolSchema,
        validator: type[pydantic.BaseModel] | None = None,
        *,
        is_gen: bool = False,
    ) -> None:
        self._fn = fn
        self._validator = validator
        self._is_gen = is_gen
        self.schema = schema

    def parse_args(self, json_args: str) -> dict[str, Any]:
        """Parse and validate JSON args into Python kwargs."""
        kwargs = json.loads(json_args) if json_args else {}
        return self.validate_kwargs(kwargs)

    def validate_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Validate kwargs and return normalized Python values."""
        if self._validator is not None:
            validated = self._validator.model_validate(kwargs)
            return dict(validated.model_dump())
        return kwargs

    async def execute_kwargs(
        self,
        kwargs: dict[str, Any],
        *,
        tool_name: str | None,
        tool_call_id: str | None,
    ) -> R:
        """Validate kwargs and call the underlying tool implementation."""
        kwargs = self.validate_kwargs(kwargs)
        if not self._is_gen:
            return await self._fn(**kwargs)  # type: ignore[call-arg]

        # Generator tool (e.g. agent-as-a-tool): drain the async
        # generator, forward each yielded message to the runtime for
        # real-time streaming, and return the final text as the result.
        return await yield_from(  # type: ignore[return-value]
            self._fn(**kwargs),  # type: ignore[arg-type,call-arg]
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )

    @property
    def name(self) -> str:
        return self.schema.name

    @property
    def description(self) -> str:
        return self.schema.description

    @property
    def param_schema(self) -> dict[str, Any]:
        return self.schema.param_schema

    @property
    def fn(self) -> Callable[P, Awaitable[R]]:
        return self._fn


def tool[**P, R](fn: Callable[P, Awaitable[R]]) -> Tool[P, R]:
    """Decorator: turn an async function into a :class:`Tool`."""
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

    schema = types.ToolSchema(
        name=fn.__name__,
        description=inspect.getdoc(fn) or "",
        param_schema=validator.model_json_schema(),
        return_type=hints.get("return", None),
    )

    return Tool(
        fn=fn,
        schema=schema,
        validator=validator,
        is_gen=inspect.isasyncgenfunction(fn),
    )


class ToolCall:
    """Callable that binds a :class:`ToolCallPart` to its :class:`Tool`.

    Calling it executes the tool and returns a ``role="tool"`` message.
    """

    def __init__(self, part: types.ToolCallPart, tool: Tool[..., Any]) -> None:
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
            self._kwargs = self._tool.parse_args(self._part.tool_args)
        return dict(self._kwargs)

    async def __call__(self, **overrides: Any) -> events_.ToolCallResult:
        """Execute the tool and return a :class:`ToolCallResult`."""
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
            base_kwargs = self._tool.validate_kwargs({**base_kwargs, **overrides})

        call = middleware_.ToolContext(
            tool_call_id=self._part.tool_call_id,
            tool_name=self._part.tool_name,
            kwargs=base_kwargs,
        )

        tool = self._tool

        async def _real(call: middleware_.ToolContext) -> events_.ToolCallResult:
            try:
                result = await tool.execute_kwargs(
                    call.kwargs,
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                )
            except Exception as exc:
                return tool_result(
                    types.ToolResultPart(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.tool_name,
                        result=str(exc),
                        is_error=True,
                    )
                )
            return tool_result(
                types.ToolResultPart(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    result=result,
                )
            )

        chain = middleware_._build_tool_chain(_real)
        return await chain(call)


class ToolRunner:
    def __init__(self, stream: models.Stream) -> None:
        self._stream = stream
        # finish_future gets set when the stream exhausts. We won't
        # exhaust until that happens, since the stream can cause more
        # tools to get triggered.
        self._finish_future = stream.finish_future
        # A future that gets signalled when we add a new tool, so that
        # asyncio.wait gets woken up and cycles around in the loop to
        # wait on the new thing as well.
        self._sched_waiter: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        self._active: set[
            asyncio.Future[events_.ToolCallResult] | asyncio.Future[None]
        ] = {self._finish_future}
        self._tool_results: list[events_.ToolCallResult] = []
        self._tg_base = asyncio.TaskGroup()

    async def __aenter__(self) -> Self:
        self._tg = await self._tg_base.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._tg_base.__aexit__(*args)

    def events(self) -> AsyncGenerator[events_.ToolCallResult]:
        return self._iterate()

    def schedule(self, tc: ToolCall) -> None:
        self._active.add(self._tg.create_task(tc()))
        self._sched_waiter.set_result(None)

    def get_tool_message(self) -> types.Message | None:
        if self._tool_results:
            return builders.tool_message(*[t.message for t in self._tool_results])
        return None

    async def _iterate(self) -> AsyncGenerator[events_.ToolCallResult]:
        while self._active:
            done, _ = await asyncio.wait(
                [*self._active, self._sched_waiter],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in done:
                self._active.discard(t)
                if t is self._finish_future:
                    t.result()
                elif t is self._sched_waiter:
                    t.result()
                    self._sched_waiter = asyncio.get_running_loop().create_future()
                else:
                    res = t.result()
                    assert res is not None
                    self._tool_results.append(res)
                    yield res


class Context(pydantic.BaseModel):
    """Everything that goes into the LLM."""

    model: models.Model[Any]
    messages: list[types.Message]
    tools: list[Tool[..., Any]]

    _tools_by_name: dict[str, Tool[..., Any]] = pydantic.PrivateAttr()

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self._tools_by_name = {t.name: t for t in self.tools}

    def keep_running(self) -> bool:
        """Call at top of an agent loop to see whether to keep running."""
        return bool(
            self.messages and self.messages[-1].role not in ("assistant", "internal")
        )

    @overload
    def resolve(self, tool_part: types.ToolCallPart) -> ToolCall: ...
    @overload
    def resolve(self, tool_part: Sequence[types.ToolCallPart]) -> list[ToolCall]: ...

    def resolve(
        self, tool_part: types.ToolCallPart | Sequence[types.ToolCallPart]
    ) -> ToolCall | list[ToolCall]:
        """Resolve ToolCallPart(s) into callable ToolCall object(s)."""
        if isinstance(tool_part, types.ToolCallPart):
            return ToolCall(
                part=tool_part, tool=self._tools_by_name[tool_part.tool_name]
            )
        return [self.resolve(tp) for tp in tool_part]

    def add(self, message: types.Message | Sequence[types.Message] | None) -> None:
        if message is None:
            return
        if isinstance(message, types.Message):
            self.messages.append(message)
        else:
            self.messages.extend(message)


class LoopFn(Protocol):
    def __call__(self, context: Context) -> AsyncGenerator[events_.AgentEvent]: ...


def tool_result(
    *items: types.Message
    | types.ToolResultPart
    | events_.ToolCallResult
    | list[types.Message],
    tool_call_id: str | None = None,
    result: Any = None,
    tool_name: str = "",
    is_error: bool = False,
) -> events_.ToolCallResult:
    """Create a :class:`ToolCallResult` from tool messages, parts, or kwargs.

    Accepts ``ToolCallResult`` items (extracts their ``.message``),
    plain ``Message`` objects, ``ToolResultPart`` instances, or keyword
    arguments matching :func:`ai.tool_message`::

        yield ai.tool_result(*(t.result() for t in tasks))
        ai.tool_result(tool_call_id="tc-1", result="denied", is_error=True)
    """
    if tool_call_id is not None:
        msg = builders.tool_message(
            tool_call_id=tool_call_id,
            result=result,
            tool_name=tool_name,
            is_error=is_error,
        )
        return events_.ToolCallResult(message=msg, results=msg.tool_results)

    unwrapped: list[types.Message | types.ToolResultPart] = []
    for item in items:
        if isinstance(item, events_.ToolCallResult):
            unwrapped.append(item.message)
        elif isinstance(item, list):
            unwrapped.extend(item)
        else:
            unwrapped.append(item)
    msg = builders.tool_message(*unwrapped)
    return events_.ToolCallResult(message=msg, results=msg.tool_results)


async def yield_from(
    source: AsyncIterable[events_.AgentEvent],
    *,
    # TODO: is this what we really want for labelling?
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    label: object = None,
) -> str:
    """Drain *source*, forwarding each event to the current runtime.

    Use inside a custom loop to stream messages from a sub-agent to the
    consumer without adding them to the parent agent's message history::

        result = await yield_from(sub.run(model, msgs), label="researcher")

    Works with :func:`asyncio.gather` for concurrent fan-out::

        r1, r2 = await asyncio.gather(
            yield_from(a.run(model, m1), label="a"),
            yield_from(b.run(model, m2), label="b"),
        )

    Each forwarded event is wrapped in a :class:`PartialToolCallResult`
    carrying ``label`` (and optionally ``tool_call_id`` / ``tool_name``)
    so the consumer can route by source.

    Returns the final message's text (empty string if no messages).
    """
    rt = runtime.get_runtime()
    last: types.Message | None = None
    async for item in source:
        if isinstance(item, types.Message):
            last = item
            continue
        await rt.put_event(
            events_.PartialToolCallResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                label=label,
                value=item,
            )
        )
        if isinstance(item, events_.TerminalEvent):
            last = item.message
    return last.text if last else ""


class Agent:
    """Bag of configuration: model + tools + loop."""

    def __init__(
        self,
        *,
        tools: list[Tool[..., Any]] | None = None,
    ) -> None:
        self._tools: list[Tool[..., Any]] = tools or []
        self._loop_fn: LoopFn | None = None

    @property
    def tools(self) -> list[Tool[..., Any]]:
        """The agent's registered tools (read-only copy)."""
        return list(self._tools)

    # TODO: remove?
    def loop(self, fn: LoopFn) -> LoopFn:
        """Decorator: override the default loop function."""
        self._loop_fn = fn
        return fn

    async def default_loop(
        self, context: Context
    ) -> AsyncGenerator[events_.AgentEvent]:
        """Stream, execute tools, repeat."""
        while context.keep_running():
            async with (
                models.stream(
                    context.model, context.messages, tools=context.tools
                ) as stream,
                ToolRunner(stream) as tr,
            ):
                async for event in util.merge(stream, tr.events()):
                    yield event

                    if isinstance(event, types.ToolEnd):
                        tool = context.resolve(event.tool_call)
                        tr.schedule(tool)

                context.add(stream.message)
                # This adds the tool message to the history, which
                # also has the effect of causing another turn through
                # the loop.
                context.add(tr.get_tool_message())

    async def run(
        self,
        model: models.Model[Any],
        messages: list[types.Message],
        *,
        middleware: list[middleware_.Middleware] | None = None,
    ) -> AsyncGenerator[events_.AgentEvent]:
        """Run the agent loop, yielding events to the consumer.

        Args:
            model: The model to use for LLM calls.
            messages: Initial conversation messages.
            middleware: Optional list of middleware to apply to this run.
                First in the list = outermost.  Middleware wraps model
                calls, tool calls, hooks, and the run itself.

        To attribute a sub-agent's events to a branch, wrap the run in
        ``yield_from(..., label=...)`` — the label flows via
        ``PartialToolCallResult`` rather than on individual messages.
        """
        call = middleware_.AgentRunContext(
            model=model,
            messages=messages,
            tools=self._tools,
        )

        loop_fn = self._loop_fn or self.default_loop

        async def _real(
            call: middleware_.AgentRunContext,
        ) -> AsyncGenerator[events_.AgentEvent]:
            context = Context(
                model=call.model,
                messages=list(call.messages),
                tools=call.tools,
            )
            source = loop_fn(context)
            async for event in runtime.run(source):
                yield event

        # Activate middleware for this run (and everything it calls).
        # When middleware is None (default), inherit the parent's middleware
        # from the context var — this lets nested agents share middleware.
        # When middleware is explicitly provided, *extend* the parent stack
        # so that outer cross-cutting concerns (tracing, durability) are
        # preserved.  Pass ``middleware=[]`` to clear the stack entirely.
        mw_token: middleware_.Token | None = None
        if middleware is not None:
            parent = middleware_.get()
            mw_token = middleware_.activate(parent + middleware)
        try:
            chain = middleware_._build_agent_run_chain(_real)
            async for event in chain(call):
                yield event
        finally:
            if mw_token is not None:
                middleware_.deactivate(mw_token)


def agent(
    *,
    tools: list[Tool[..., Any]] | None = None,
) -> Agent:
    """Create an Agent."""
    return Agent(tools=tools)
