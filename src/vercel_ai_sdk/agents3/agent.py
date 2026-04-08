"""Agent, Context, StreamResult, and the stream() function."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Protocol, get_type_hints

import pydantic

from .. import _durability as _dctx
from .. import models, types
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

    async def __call__(self, json_args: str) -> R:
        """Parse json_args into kwargs, validate, and call the function."""
        kwargs = json.loads(json_args) if json_args else {}
        if self._validator is not None:
            self._validator.model_validate(kwargs)

        if not self._is_gen:
            return await self._fn(**kwargs)

        # Generator tool (e.g. agent-as-a-tool): drain the async
        # generator, pipe each yielded message to the runtime for
        # real-time streaming, and return the final text as the result.
        rt = runtime.get_runtime()
        last: types.Message | None = None
        async for message in self._fn(**kwargs):
            await rt.put_message(message)
            last = message
        return last.text if last else ""

    @property
    def name(self) -> str:
        return self.schema.name

    @property
    def description(self) -> str:
        return self.schema.description

    @property
    def param_schema(self) -> dict[str, Any]:
        return self.schema.param_schema


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

    Calling it executes the tool and returns a :class:`ToolResultPart`.
    """

    def __init__(self, part: types.ToolCallPart, tool: Tool[..., Any]) -> None:
        self._part = part
        self._tool = tool

    @property
    def id(self) -> str:
        return self._part.tool_call_id

    @property
    def name(self) -> str:
        return self._part.tool_name

    @property
    def args(self) -> str:
        return self._part.tool_args

    async def __call__(self) -> types.ToolResultPart:
        """Execute the tool and return a :class:`ToolResultPart`.

        If a durability provider is active, the call is routed through
        it for recording or replay.
        """
        provider = _dctx.get_provider()

        async def _execute() -> types.ToolResultPart:
            try:
                result = await self._tool(self._part.tool_args)
            except Exception as exc:
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
            return await provider.execute_tool(
                _execute,
                tool_call_id=self.id,
                tool_name=self.name,
            )
        return await _execute()


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
    while True:
        stream = await models.stream(
            context.model, context.messages, tools=context.tools
        )
        async for message in stream:
            yield message

        tool_calls = context.resolve(stream.tool_calls)
        if not tool_calls:
            break

        # Execute tool calls in parallel.
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(tc()) for tc in tool_calls]

        # Yield a tool-result message — history auto-collects it.
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
        durability: durability_.DurabilityProvider | None = None,
        checkpoint: checkpoint_.Checkpoint | None = None,
    ) -> AsyncGenerator[types.Message]:
        """Run the agent loop, yielding messages to the consumer.

        Args:
            model: The model to use for LLM calls.
            messages: Initial conversation messages.
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
                yield message
        finally:
            if token is not None:
                _dctx.reset_provider(token)


def agent(
    *,
    tools: list[Tool[..., Any]] | None = None,
    system: str | None = None,
) -> Agent:
    """Create an Agent."""
    return Agent(tools=tools)
