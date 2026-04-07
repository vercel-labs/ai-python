"""Agent, Context, StreamResult, and the stream() function."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Protocol, get_type_hints

import pydantic

from .. import models, types
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
        # real-time streaming, and collect all messages as the result.
        rt = runtime.get_runtime()
        messages: list[types.Message] = []
        async for message in self._fn(**kwargs):
            await rt.put_message(message)
            messages.append(message)
        return messages

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
    """
    Intermediate object that context resolves tool parts to.
    Binds together ToolPart and Tool, allows users to inspect it.
    """

    def __init__(self, part: types.ToolPart, tool: Tool[..., Any]) -> None:
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

    async def __call__(self) -> types.Message:
        """Execute the tool and return a message with the resolved part."""
        result = await self._tool(self._part.tool_args)
        updated = self._part.with_result(result)
        return types.Message(role="assistant", parts=[updated])


class Context(pydantic.BaseModel):
    """Everything that goes into the LLM."""

    messages: list[types.Message]
    tools: list[Tool[..., Any]]

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def resolve(self, tool_parts: list[types.ToolPart]) -> list[ToolCall]:
        """Resolve ToolParts into callable ToolCall objects."""
        tools_by_name = {t.name: t for t in self.tools}
        return [
            ToolCall(part=tp, tool=tools_by_name[tp.tool_name]) for tp in tool_parts
        ]


class LoopFn(Protocol):
    def __call__(self, context: Context) -> AsyncGenerator[types.Message]: ...


async def _default_loop(context: Context) -> AsyncGenerator[types.Message]:
    while True:
        stream = models.stream(context.model, context.messages)
        async for message in stream:
            yield message

        tool_calls = context.resolve(stream.message.tool_calls)
        if not tool_calls:
            break

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(tc()) for tc in tool_calls]

        for task in tasks:
            yield task.result


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
        self, model: models.Model, messages: list[types.Message]
    ) -> AsyncGenerator[types.Message]:
        """Run the agent loop, yielding messages to the consumer."""
        context = Context(messages=messages, tools=self._tools)

        async for message in runtime.run(self._loop_fn(context)):
            yield message


def agent(
    *,
    model: models.Model,
    tools: list[Tool[..., Any]] | None = None,
    system: str | None = None,
) -> Agent:
    """Create an Agent."""
    return Agent(tools=tools)
