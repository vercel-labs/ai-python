"""Hooks: suspension points that require external input to continue.

Usage inside an agent loop::

    result = await hook(
        "approve_delete", payload=ToolApproval, metadata={"tool": "rm"}
    )
    if result.granted:
        ...

Resolution from outside the loop::

    resolve_hook("approve_delete", {"granted": True})

Cancellation::

    await cancel_hook("approve_delete", reason="denied")

"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pydantic

from .. import telemetry, types
from ..types import messages as messages_
from . import _middleware as middleware_
from . import runtime as runtime_

# ---------------------------------------------------------------------------
# Module-level hook registries
#
# _live_hooks:
#   Populated by hook() when it suspends inside a running agent.
#   Maps hook label -> (future, metadata dict, Runtime).
#   Consumed by resolve_hook() / cancel_hook() to unblock the awaiting
#   coroutine.  Entries are removed when the hook resolves, cancels, or
#   the run completes.
#
# _pending_resolutions:
#   Populated by resolve_hook() when no live hook exists yet (serverless
#   re-entry: the user calls resolve_hook() *before* agent.run() replays).
#   Maps hook label -> (payload type, validated resolution dict).
#   Consumed by hook() at the start of execution — if a pre-registered
#   resolution exists for the label, the hook returns immediately without
#   suspending.  Entries are removed on consumption.
# ---------------------------------------------------------------------------

_live_hooks: dict[
    str, tuple[asyncio.Future[dict[str, Any]], dict[str, Any], runtime_.Runtime]
] = {}

_pending_resolutions: dict[str, dict[str, Any] | BaseException] = {}


class HookPendingError(Exception):
    """Exception for aborting due to a hook."""

    type: str = "gateway_error"

    def __init__(
        self,
        hook: messages_.HookPart[Any],
    ) -> None:
        super().__init__(hook.hook_id)
        self.hook = hook


def _label(target: str | messages_.HookPart[Any]) -> str:
    """Normalize a hook label or a HookPart down to its label string."""
    return target.hook_id if isinstance(target, messages_.HookPart) else target


def cleanup_run(labels: set[str]) -> None:
    """Remove all registry entries associated with a finished run."""
    for label in labels:
        _live_hooks.pop(label, None)
        _pending_resolutions.pop(label, None)


async def hook[T: pydantic.BaseModel](
    hook: str | messages_.HookPart[Any],
    *,
    payload: type[T],
    metadata: dict[str, Any] | None = None,
    tool_call_id: str | None = None,
) -> T:
    """Create a hook suspension point and await its resolution.

    Args:
        hook: Unique identifier for this hook instance, or a HookPart
            whose ``hook_id`` supplies it.
        payload: Pydantic model class — the resolution data must validate
            against this type.  The return value is a validated instance.
        metadata: Arbitrary metadata surfaced in the pending signal message
            and checkpoint.  Useful for UI rendering (e.g. which tool needs
            approval, what arguments it received).
        tool_call_id: The tool call this hook suspends, if any.  Stamped
            onto the emitted :class:`~ai.messages.HookPart` so consumers
            (run-blocked tracking, UIs) can attribute the suspension to
            its tool call.  Approval gating sets this automatically;
            custom gating wrappers should pass it too.

    """
    call = middleware_.HookContext(
        label=_label(hook),
        payload=payload,
        metadata=metadata or {},
        tool_call_id=tool_call_id,
    )

    chain = middleware_._build_hook_chain(_hook_impl)
    result = await chain(call)
    return cast("T", result)


async def _hook_impl(call: middleware_.HookContext) -> pydantic.BaseModel:
    """Core hook logic — the innermost ``next`` in the middleware chain."""
    rt = runtime_.get_runtime()
    label = call.label
    payload = call.payload
    hook_metadata = call.metadata

    data = telemetry.HookSpanData(
        label=label, hook_type=payload.__name__, metadata=hook_metadata
    )

    # Pre-registered resolution (serverless re-entry).
    pre_registered = _pending_resolutions.pop(label, None)
    if pre_registered is not None:
        async with telemetry.span(data, replay=True):
            if isinstance(pre_registered, BaseException):
                raise pre_registered
            data.status = "resolved"
            return payload(**pre_registered)

    # No resolution available — suspend.  The span covers the whole
    # suspension: how long the run sat waiting on external input.
    async with telemetry.span(data):
        future: asyncio.Future[dict[str, Any]] = asyncio.Future()

        _live_hooks[label] = (future, hook_metadata, rt)
        rt.track_hook_label(label)

        # Emit pending signal.
        hook_part: messages_.HookPart[Any] = messages_.HookPart(
            hook_id=label,
            hook_type=payload.__name__,
            status="pending",
            metadata=hook_metadata,
            tool_call_id=call.tool_call_id,
        )

        await rt.put_hook(hook_part)

        # Await resolution — may be resolved externally or cancelled.
        try:
            resolution = await future
        except asyncio.CancelledError:
            data.status = "cancelled"
            raise

        # Clean up live registry.
        _live_hooks.pop(label, None)
        data.status = "resolved"

        # Emit resolved signal.
        await rt.put_hook(
            messages_.HookPart(
                hook_id=label,
                hook_type=payload.__name__,
                status="resolved",
                metadata=hook_metadata,
                resolution=resolution,
                tool_call_id=call.tool_call_id,
            )
        )

        return payload(**resolution)


def resolve_hook(
    hook: str | messages_.HookPart[Any],
    data: pydantic.BaseModel | dict[str, Any] | BaseException,
    *,
    payload: type[pydantic.BaseModel] | None = None,
) -> None:
    """Resolve a hook by label.

    Works in two modes:

    1. **Live hook exists** (long-running): validates data (if ``payload``
       type is provided), resolves the future immediately, unblocking the
       awaiting coroutine.

    2. **No live hook yet** (serverless re-entry): stashes the resolution
       in the pre-registration registry.  When ``hook()`` executes during
       replay, it finds the pre-registered value and returns without
       suspending.

    Passing an exception sends it to the awaiter (or stashes it for the
    next replay) so the awaiting ``ai.hook(...)`` call raises rather than
    returns.  See :func:`abort_pending_hook` for the common case of
    propagating a :class:`HookPendingError`.

    Args:
        hook: The hook label to resolve, or a HookPart whose ``hook_id``
            supplies it.
        data: Resolution data — a dict, pydantic model instance, or an
            exception to raise in the awaiter.
        payload: Optional pydantic model class for validation.  Ignored
            when *data* is an exception.

    """
    label = _label(hook)
    resolution: dict[str, Any] | BaseException
    if isinstance(data, BaseException):
        resolution = data
    elif isinstance(data, pydantic.BaseModel):
        resolution = data.model_dump()
    elif isinstance(data, dict):
        if payload is not None:
            # Validate against the payload type.
            validated = payload(**data)
            resolution = validated.model_dump()
        else:
            resolution = data
    else:
        raise TypeError(
            f"Expected dict or pydantic model, got {type(data).__name__}"
        )

    # Path 1: live hook — resolve the future directly.
    if label in _live_hooks:
        future, _, _rt = _live_hooks[label]
        if isinstance(resolution, BaseException):
            future.set_exception(resolution)
        else:
            future.set_result(resolution)
        return

    # Path 2: no live hook — pre-register for later consumption.
    _pending_resolutions[label] = resolution


def abort_pending_hook(hook_part: messages_.HookPart[Any]) -> None:
    """Abort the hook identified by ``hook_part.hook_id``.

    The abort carries a :class:`HookPendingError` wrapping *hook_part*.

    Convenience wrapper around :func:`resolve_hook` for the serverless
    pattern where a caller has a :class:`~ai.messages.HookPart` (e.g.
    from inbound conversion) and needs to surface it back through the
    awaiting ``ai.hook(...)`` site as a structured suspension.
    """
    resolve_hook(hook_part.hook_id, HookPendingError(hook_part))


async def cancel_hook(
    hook: str | messages_.HookPart[Any], *, reason: str | None = None
) -> None:
    """Cancel a pending hook.

    Only works for live hooks (long-running mode).  Raises ValueError
    if the hook is not currently pending.  ``hook`` may be a label
    string or a HookPart whose ``hook_id`` supplies it.
    """
    label = _label(hook)
    if label not in _live_hooks:
        raise ValueError(f"No pending hook with label: {label!r}")

    future, hook_metadata, rt = _live_hooks.pop(label)
    future.cancel(reason)

    # Emit cancelled signal.
    await rt.put_hook(
        messages_.HookPart(
            hook_id=label,
            hook_type="",  # not available at cancel site
            status="cancelled",
            metadata=hook_metadata,
        )
    )


# ── Built-in hook payloads ────────────────────────────────────────

ToolApproval = types.tools.ToolApproval

TOOL_APPROVAL_HOOK_TYPE = ToolApproval.__name__
