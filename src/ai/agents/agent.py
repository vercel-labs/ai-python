"""Agent, Context, StreamResult, and the stream() function."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable
from typing import Any, Protocol, get_type_hints

import pydantic

from .. import _durability as _dctx
from .. import models, types
from ..telemetry import events as telemetry
from ..types import builders
from . import checkpoint as checkpoint_
from . import durability as durability_
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
        """Execute the tool and return a single tool-result message.

        If a durability provider is active, the call is routed through
        it for recording or replay.
        """
        provider = _dctx.get_provider()
        prep_error: Exception | None = None
        try:
            base_kwargs = self.kwargs
        except Exception as exc:
            prep_error = exc
            base_kwargs = {}

        try:
            final_kwargs = self._tool.validate_kwargs({**base_kwargs, **overrides})
            prep_error = None
        except Exception as exc:
            if prep_error is None or overrides:
                raise
            final_kwargs = None

        telemetry.handle(
            telemetry.ToolCallStartEvent(
                tool_name=self.name,
                tool_call_id=self.id,
                args=(
                    json.dumps(final_kwargs)
                    if final_kwargs is not None
                    else self._part.tool_args
                ),
            )
        )
        t0 = telemetry.time_ms()
        error_str: str | None = None

        async def _execute() -> types.ToolResultPart:
            nonlocal error_str
            try:
                if prep_error is not None or final_kwargs is None:
                    raise prep_error or ValueError("Failed to prepare tool kwargs")
                result = await self._tool.execute_kwargs(final_kwargs)
            except Exception as exc:
                error_str = str(exc)
                return types.ToolResultPart(
                    tool_call_id=self._part.tool_call_id,
                    tool_name=self._part.tool_name,
                    result=str(exc),
                    is_error=True,
                )
            return types.ToolResultPart(
                tool_call_id=self._part.tool_call_id,
                tool_name=self._part.tool_name,
                result=result,
            )

        if provider is not None:
            result_part = await provider.execute_tool(
                _execute,
                tool_call_id=self.id,
                tool_name=self.name,
            )
        else:
            result_part = await _execute()

        telemetry.handle(
            telemetry.ToolCallFinishEvent(
                tool_name=self.name,
                tool_call_id=self.id,
                result=result_part.result,
                error=error_str,
                duration_ms=telemetry.time_ms() - t0,
            )
        )
        return builders.tool_message(
            types.Message(role="tool", parts=[result_part])
        )


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


class LoopFn(Protocol):
    def __call__(self, context: Context) -> AsyncGenerator[types.Message]: ...


async def _default_loop(context: Context) -> AsyncGenerator[types.Message]:
    step_index = 0
    while True:
        telemetry.handle(telemetry.StepStartEvent(step_index=step_index))

        stream = await models.stream(
            context.model, context.messages, tools=context.tools
        )
        done_message: types.Message | None = None
        async for message in stream:
            done_message = message
            yield message

        if done_message is not None:
            telemetry.handle(
                telemetry.StepFinishEvent(step_index=step_index, message=done_message)
            )
        step_index += 1

        tool_calls = context.resolve(stream.tool_calls)
        if not tool_calls:
            break

        # Execute tool calls in parallel.
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(tc()) for tc in tool_calls]

        # Yield one merged tool-result message — history auto-collects it.
        yield builders.tool_message(*(t.result() for t in tasks))


async def _collect_messages(
    source: AsyncGenerator[types.Message],
    messages: list[types.Message],
) -> AsyncGenerator[types.Message]:
    """Intercept yielded messages and collect done ones into *messages*.

    This runs on the **producer** side (same coroutine as the loop function),
    so ``messages`` is always up-to-date by the time the loop reads it for
    the next model call — avoiding the race that would occur if collection
    happened on the consumer side of the runtime queue.
    """
    async for message in source:
        if message.is_done:
            for i, existing in enumerate(messages):
                if existing.id == message.id:
                    messages[i] = message
                    break
            else:
                messages.append(message)
        yield message


async def yield_from(source: AsyncIterable[types.Message]) -> str:
    """Drain *source*, forwarding each message to the current runtime.

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
    async for message in source:
        await rt.put_message(message)
        last = message
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
        durability: durability_.DurabilityProvider | None = None,
        checkpoint: checkpoint_.Checkpoint | None = None,
    ) -> AsyncGenerator[types.Message]:
        """Run the agent loop, yielding messages to the consumer.

        Args:
            model: The model to use for LLM calls.
            messages: Initial conversation messages.
            label: Optional label applied to every yielded message.
                Useful for multi-agent graphs where the consumer needs
                to route messages by source.
            durability: Explicit durability provider.  If ``None`` but
                *checkpoint* is given, an :class:`EventLogProvider` is
                created automatically.
            checkpoint: Checkpoint to resume from.  Implies eventlog
                durability when no explicit *durability* is provided.
        """
        # Convenience: checkpoint implies eventlog provider.
        if checkpoint is not None and durability is None:
            durability = durability_.EventLogProvider(checkpoint)

        context = Context(model=model, messages=list(messages), tools=self._tools)

        # Set the durability provider on the shared context var so that
        # models.stream() and ToolCall.__call__() auto-detect it.
        token = _dctx.set_provider(durability) if durability is not None else None
        try:
            source = _collect_messages(self._loop_fn(context), context.messages)
            async for message in runtime.run(source):
                if label is not None:
                    message = message.model_copy(update={"label": label})
                yield message
        finally:
            if token is not None:
                _dctx.reset_provider(token)


def agent(
    *,
    tools: list[Tool[..., Any]] | None = None,
) -> Agent:
    """Create an Agent."""
    return Agent(tools=tools)
