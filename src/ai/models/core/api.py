"""Top-level orchestration — stream(), generate(), check_connection().

These wire together adapters, middleware chains, and auto-client creation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pydantic

from ... import middleware as middleware_
from ...types import messages as messages_
from ...types import stream as stream_
from ...types import tools as tools_
from ..ai_gateway import types as ai_gw_types
from . import adapters, stream_result
from . import client as client_
from . import model as model_


async def stream(
    model: model_.Model,
    messages: list[messages_.Message],
    *,
    tools: Sequence[tools_.ToolLike] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    client: client_.Client | None = None,
    **kwargs: Any,
) -> stream_.StreamResultLike:
    """Stream an LLM response.

    Returns a :class:`StreamResultLike` that is async-iterable and
    collects the final ``Message``.  After iteration, access ``.text``,
    ``.tool_calls``, ``.usage``, etc.

    Without middleware the concrete type is :class:`StreamResult`; with
    middleware it may be any :class:`~ai.StreamResultLike`.
    """
    call = middleware_.ModelContext(
        model=model,
        messages=messages,
        tools=tools,
        output_type=output_type,
        client=client,
        kwargs=kwargs,
    )

    async def _real(call: middleware_.ModelContext) -> stream_.StreamResultLike:
        c = call.client or client_.auto_client(call.model)
        adapter_fn = adapters.get_stream_adapter(call.model.adapter)
        return stream_result.StreamResult(
            adapter_fn(
                c,
                call.model,
                call.messages,
                tools=call.tools,
                output_type=call.output_type,
                **call.kwargs,
            )
        )

    chain = middleware_._build_model_chain(_real)
    return await chain(call)


async def generate(
    model: model_.Model,
    messages: list[messages_.Message],
    params: ai_gw_types.GenerateParams | None = None,
    *,
    client: client_.Client | None = None,
) -> messages_.Message:
    """Generate a response (images, video, etc.).

    Resolves the adapter function from ``model.adapter``, auto-creates a
    :class:`Client` from env vars if none is provided.

    ``params`` controls the generation type:

    * :class:`ImageParams` — image generation (``/image-model``).
    * :class:`VideoParams` — video generation (``/video-model``).
    * ``None`` — auto-detect from ``model.capabilities``.
    """
    call = middleware_.GenerateContext(
        model=model,
        messages=messages,
        params=params,
        client=client,
    )

    async def _real(call: middleware_.GenerateContext) -> messages_.Message:
        c = call.client or client_.auto_client(call.model)
        adapter_fn = adapters.get_generate_adapter(call.model.adapter)
        return await adapter_fn(c, call.model, call.messages, params=call.params)

    chain = middleware_._build_generate_chain(_real)
    return await chain(call)


async def check_connection(
    model: model_.Model,
    *,
    client: client_.Client | None = None,
) -> bool:
    """Check whether *client* can reach *model*'s provider and the model exists.

    Returns ``True`` when the credentials are valid **and** the model is
    available on the remote side — i.e. a subsequent :func:`stream` or
    :func:`generate` call should succeed (network conditions permitting).

    This only hits free metadata endpoints; no tokens or credits are
    consumed.

    If no *client* is given, one is auto-created from environment
    variables (same logic as :func:`stream`).

    Non-auth transport errors (network failures, 5xx) are raised rather
    than returning ``False`` so that callers can distinguish "bad
    credentials / unknown model" from "provider unreachable".
    """
    c = client or client_.auto_client(model)
    check_fn = adapters.get_check_fn(model.provider)
    return await check_fn(c, model)
