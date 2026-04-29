"""Agent, Context, StreamResult, and the stream() function."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable
from typing import Any, Protocol, get_type_hints

import pydantic

from .. import middleware as middleware_
from .. import models, types
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

    async def execute_kwargs(self, kwargs: dict[str, Any]) -> R:
        """Validate kwargs and call the underlying tool implementation."""
        kwargs = self.validate_kwargs(kwargs)
        if not self._is_gen:
            return await self._fn(**kwargs)  # type: ignore[call-arg]

        # Generator tool (e.g. agent-as-a-tool): drain the async
        # generator, forward each yielded message to the runtime for
        # real-time streaming, and return the final text as the result.
        return await yield_from(self._fn(**kwargs))  # type: ignore[arg-type,call-arg,return-value]

    async def __call__(self, json_args: str) -> R:
        """Parse json_args into kwargs, validate, and call the function."""
        return await self.execute_kwargs(self.parse_args(json_args))

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

    async def __call__(self, **overrides: Any) -> types.Message:
        """Execute the tool and return a single tool-result message."""
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

        async def _real(call: middleware_.ToolContext) -> types.Message:
            try:
                result = await tool.execute_kwargs(call.kwargs)
            except Exception as exc:
                return builders.tool_message(
                    types.ToolResultPart(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.tool_name,
                        result=str(exc),
                        is_error=True,
                    )
                )
            return builders.tool_message(
                types.ToolResultPart(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    result=result,
                )
            )

        chain = middleware_._build_tool_chain(_real)
        return await chain(call)


class Context(pydantic.BaseModel):
    """Everything that goes into the LLM."""

    model: models.Model
    messages: list[types.Message]
    tools: list[Tool[..., Any]]

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def resolve(self, tool_parts: list[types.ToolCallPart]) -> list[ToolCall]:
        """Resolve ToolCallParts into callable ToolCall objects."""
        tools_by_name = {t.name: t for t in self.tools}
        return [
            ToolCall(part=tp, tool=tools_by_name[tp.tool_name]) for tp in tool_parts
        ]


StreamItem = events_.AgentEvent | types.Message


class LoopFn(Protocol):
    def __call__(self, context: Context) -> AsyncGenerator[StreamItem]: ...


async def _message_events(
    message: types.Message,
) -> AsyncGenerator[events_.AgentEvent]:
    yield events_.MessageStart(message=message)
    yield events_.MessageEnd(message=message)


async def _coerce_events(
    source: AsyncIterable[StreamItem],
) -> AsyncGenerator[events_.AgentEvent]:
    async for item in source:
        if isinstance(item, types.Message):
            async for event in _message_events(item):
                yield event
        else:
            yield item


async def _default_loop(context: Context) -> AsyncGenerator[events_.AgentEvent]:
    while True:
        stream = models.stream(
            context.model,
            context.messages,
            tools=context.tools,
        )
        async for stream_event in stream:
            yield stream_event

        # Bridge: emit MessageStart/MessageEnd around the assistant message
        # the model stream just produced, so _collect_messages and downstream
        # consumers (AI-SDK outbound, label stamping) see the same boundary
        # events they did under the previous adapter contract.
        if stream.message is not None and stream.message.parts:
            async for boundary in _message_events(stream.message):
                yield boundary

        tool_calls = context.resolve(stream.tool_calls)
        if not tool_calls:
            break

        # Execute tool calls in parallel.
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(tc()) for tc in tool_calls]

        # Yield one merged tool-result message — history auto-collects it.
        # Left un-stamped: the tool result is the input of the *next* turn,
        # so the next stream() call will stamp it with that turn's id.
        tool_msg = builders.tool_message(*(t.result() for t in tasks))
        async for boundary in _message_events(tool_msg):
            yield boundary


async def _collect_messages(
    source: AsyncIterable[StreamItem],
    messages: list[types.Message],
) -> AsyncGenerator[events_.AgentEvent]:
    """Intercept yielded events and collect MessageEnd messages into *messages*.

    This runs on the **producer** side (same coroutine as the loop function),
    so ``messages`` is always up-to-date by the time the loop reads it for
    the next model call — avoiding the race that would occur if collection
    happened on the consumer side of the runtime queue.
    """
    async for event in _coerce_events(source):
        if isinstance(event, events_.MessageEnd):
            message = event.message
            for i, existing in enumerate(messages):
                if existing.id == message.id:
                    messages[i] = message
                    break
            else:
                messages.append(message)
        yield event


async def yield_from(source: AsyncIterable[StreamItem]) -> str:
    """Drain *source*, forwarding each event to the current runtime.

    Use inside a custom loop to stream messages from a sub-agent to the
    consumer without adding them to the parent agent's message history::

        result = await yield_from(sub.run(model, msgs, label="researcher"))

    Works with :func:`asyncio.gather` for concurrent fan-out::

        r1, r2 = await asyncio.gather(
            yield_from(a.run(model, m1, label="a")),
            yield_from(b.run(model, m2, label="b")),
        )

    Returns the final message's text (empty string if no messages).
    """
    rt = runtime.get_runtime()
    last: types.Message | None = None
    async for item in _coerce_events(source):
        await rt.put_event(item)
        if isinstance(item, events_.MessageEnd):
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
        self._loop_fn: LoopFn = _default_loop

    @property
    def tools(self) -> list[Tool[..., Any]]:
        """The agent's registered tools (read-only copy)."""
        return list(self._tools)

    def loop(self, fn: LoopFn) -> LoopFn:
        """Decorator: override the default loop function."""
        self._loop_fn = fn
        return fn

    async def run(
        self,
        model: models.Model,
        messages: list[types.Message],
        *,
        label: str | None = None,
        middleware: list[middleware_.Middleware] | None = None,
    ) -> AsyncGenerator[events_.AgentEvent]:
        """Run the agent loop, yielding events to the consumer.

        Args:
            model: The model to use for LLM calls.
            messages: Initial conversation messages.
            label: Optional label applied to every yielded message.
                Useful for multi-agent graphs where the consumer needs
                to route messages by source.
            middleware: Optional list of middleware to apply to this run.
                First in the list = outermost.  Middleware wraps model
                calls, tool calls, hooks, and the run itself.
        """
        call = middleware_.AgentRunContext(
            model=model,
            messages=messages,
            tools=self._tools,
            label=label,
        )

        loop_fn = self._loop_fn

        async def _real(
            call: middleware_.AgentRunContext,
        ) -> AsyncGenerator[events_.AgentEvent]:
            context = Context(
                model=call.model,
                messages=list(call.messages),
                tools=call.tools,
            )
            source = _collect_messages(loop_fn(context), context.messages)
            async for event in runtime.run(source):
                if call.label is not None:
                    event_message: types.Message | None = None
                    if isinstance(event, events_.MessageEnd) or (
                        isinstance(event, events_.MessageStart)
                        and event.message is not None
                    ):
                        event_message = event.message

                    if event_message is not None:
                        event = event.model_copy(
                            update={
                                "message": event_message.model_copy(
                                    update={"source_label": call.label}
                                )
                            }
                        )
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
            async for message in chain(call):
                yield message
        finally:
            if mw_token is not None:
                middleware_.deactivate(mw_token)


def agent(
    *,
    tools: list[Tool[..., Any]] | None = None,
) -> Agent:
    """Create an Agent."""
    return Agent(tools=tools)
