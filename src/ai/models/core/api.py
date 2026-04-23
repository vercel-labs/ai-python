"""Top-level orchestration — stream(), generate(), check_connection().

These wire together adapters, middleware chains, and auto-client creation.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pydantic

from ... import middleware as middleware_
from ...types import events as events_
from ...types import integrity as integrity_
from ...types import messages as messages_
from ...types import stream as stream_
from ...types import tools as tools_
from . import adapters
from . import client as client_
from . import model as model_
from . import types as types_


def stream(
    model: model_.Model,
    messages: list[messages_.Message],
    *,
    tools: Sequence[tools_.ToolLike] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    turn_id: str | None = None,
    **kwargs: Any,
) -> stream_.StreamResultLike:
    """Stream an LLM response.

    Returns a :class:`StreamResultLike` that is async-iterable and
    collects the final ``Message``.  After iteration, access ``.text``,
    ``.tool_calls``, ``.usage``, etc.

    Call-site is a plain ``async for`` — no outer ``await`` needed::

        async for msg in ai.stream(model, messages):
            ...

    One call is one turn: a single request and its response.  The model
    response carries ``turn_id``; re-emitted input messages keep any
    existing ``turn_id`` from prior turns and only receive the current
    one when unstamped.  If *turn_id* is not provided, one is generated.

    The client is resolved from the model: ``model.client`` if set,
    otherwise auto-created from ``model.base_url`` / ``model.api_key_env``.

    Middleware dispatch and adapter setup are deferred to the first
    iteration; any async preflight work happens there.
    """
    messages = integrity_.prepare_messages(messages)

    if turn_id is None:
        turn_id = messages_.generate_id("turn")

    call = middleware_.ModelContext(
        model=model,
        messages=messages,
        tools=tools,
        output_type=output_type,
        kwargs=kwargs,
    )

    # Capture in closure for the inner function.
    _turn_id = turn_id

    async def _real(call: middleware_.ModelContext) -> stream_.StreamResultLike:
        c = client_.auto_client(call.model)
        adapter_fn = adapters.get_stream_adapter(call.model.adapter)
        return types_.StreamResult(
            adapter_fn(
                c,
                call.model,
                call.messages,
                tools=call.tools,
                output_type=call.output_type,
                **call.kwargs,
            ),
            turn_id=_turn_id,
            input_messages=call.messages,
        )

    async def _driver() -> AsyncGenerator[events_.Event]:
        chain = middleware_._build_model_chain(_real)
        inner = await chain(call)
        async for event in inner:
            yield event

    return stream_.StreamResult.from_generator(_driver())


async def generate(
    model: model_.Model,
    messages: list[messages_.Message],
    params: types_.GenerateParams,
    **kwargs: Any,
) -> messages_.Message:
    """Generate a response (images, video, etc.).

    Resolves the adapter function from ``model.adapter``, auto-creates a
    :class:`Client` from the model if no explicit client is set.

    ``params`` is required and controls the generation type:

    * :class:`ImageParams` — image generation (``/image-model``).
    * :class:`VideoParams` — video generation (``/video-model``).
    """
    messages = integrity_.prepare_messages(messages)

    call = middleware_.GenerateContext(
        model=model,
        messages=messages,
        params=params,
    )

    async def _real(call: middleware_.GenerateContext) -> messages_.Message:
        c = client_.auto_client(call.model)
        adapter_fn = adapters.get_generate_adapter(call.model.adapter)
        return await adapter_fn(c, call.model, call.messages, params=call.params)

    chain = middleware_._build_generate_chain(_real)
    return await chain(call)


async def check_connection(
    model: model_.Model,
) -> bool:
    """Check whether the model's provider is reachable and the model exists.

    Returns ``True`` when the credentials are valid **and** the model is
    available on the remote side — i.e. a subsequent :func:`stream` or
    :func:`generate` call should succeed (network conditions permitting).

    This only hits free metadata endpoints; no tokens or credits are
    consumed.

    The client is resolved from the model: ``model.client`` if set,
    otherwise created by the provider.

    Non-auth transport errors (network failures, 5xx) are raised rather
    than returning ``False`` so that callers can distinguish "bad
    credentials / unknown model" from "provider unreachable".
    """
    c = client_.auto_client(model)
    return await model.provider.check(c, model)
