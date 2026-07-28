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
import contextvars
from typing import TYPE_CHECKING, Any, cast

import pydantic

from .. import experimental_telemetry as telemetry
from .. import types, util
from ..types import messages as messages_
from . import _middleware as middleware_
from . import runtime as runtime_

if TYPE_CHECKING:
    from collections.abc import Iterator


class HookRegistry:
    """Holds hook state: live suspensions and pre-registered resolutions.

    ``_live_hooks``:
        Populated by ``hook()`` when it suspends inside a running agent.
        Maps hook label -> (future, metadata dict, Runtime).
        Consumed by ``resolve_hook()`` / ``cancel_hook()`` to unblock the
        awaiting coroutine.  Each entry is removed by the suspended hook
        coroutine itself when it finishes -- resolved, cancelled, or torn
        down with its run.

    ``_pending_resolutions``:
        Populated by ``resolve_hook()`` when no live hook exists yet
        (serverless re-entry: the user calls ``resolve_hook()`` inside
        the ``agent.run()`` block, before iterating the stream).  Maps
        hook label -> resolution dict or exception.  Consumed by
        ``hook()`` at the start of execution -- if a pre-registered
        resolution exists for the label, the hook returns immediately
        without suspending.  Entries are removed on consumption.

    Every hook operation takes an optional ``registry`` argument; when
    omitted, the current registry is used.  ``agent.run()`` makes its
    registry current both for the agent loop and for the consumer's
    ``async with`` block, and a nested run reuses the enclosing run's
    registry, so hooks created anywhere in a run tree are resolvable
    from the outermost consumer.  Create a ``HookRegistry`` and pass
    it explicitly to resolve from outside the run's context (e.g. a
    UI callback on another task) or to isolate a run's hooks.

    Whoever owns a registry owns its contents: a pre-registered
    resolution that no hook ever consumes stays until the registry is
    dropped.  The default per-run registry dies with the run; if you
    keep a long-lived one, avoid pre-registering labels that may never
    run.
    """

    def __init__(self) -> None:
        self._live_hooks: dict[
            str,
            tuple[
                asyncio.Future[dict[str, Any]],
                dict[str, Any],
                runtime_.Runtime,
            ],
        ] = {}
        self._pending_resolutions: dict[
            str, dict[str, Any] | BaseException
        ] = {}


_hook_registry: contextvars.ContextVar[HookRegistry] = contextvars.ContextVar(
    "hook_registry"
)


def _registry(registry: HookRegistry | None) -> HookRegistry:
    """Return the registry a hook operation should use."""
    if registry is not None:
        return registry
    try:
        return _hook_registry.get()
    except LookupError:
        raise LookupError(
            "no current HookRegistry: hook operations work inside an "
            "agent.run() block, or pass an explicit registry"
        ) from None


def get_hook_registry() -> HookRegistry:
    """Return the current HookRegistry.

    Set inside an ``agent.run()`` block (and the run's loop) and within
    a ``use_hook_registry()`` context.  Raises LookupError when there
    is none.
    """
    return _registry(None)


@util.contextmanager_any_sync
def use_hook_registry(registry: HookRegistry) -> Iterator[None]:
    """Make *registry* the current HookRegistry within this context.

    Scoped to the calling task and tasks spawned from it, and restored
    on exit.  ``agent.run()`` uses this to make the run's registry
    current for the consumer's block.
    """
    token = _hook_registry.set(registry)
    try:
        yield
    finally:
        _hook_registry.reset(token)


class HookDeferredException(Exception):  # noqa: N818
    """Exception for deferring due to a hook."""

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


async def hook[T: pydantic.BaseModel](
    hook: str | messages_.HookPart[Any],
    *,
    payload: type[T],
    metadata: dict[str, Any] | None = None,
    tool_call_id: str | None = None,
    registry: HookRegistry | None = None,
) -> T:
    """Create a hook suspension point and await its resolution.

    Args:
        hook: Unique identifier for this hook instance, or a HookPart
            whose ``hook_id`` supplies it.
        payload: Pydantic model class — the resolution data must validate
            against this type.  The return value is a validated instance.
        metadata: Arbitrary metadata surfaced in the deferred signal message
            and checkpoint.  Useful for UI rendering (e.g. which tool needs
            approval, what arguments it received).
        tool_call_id: The tool call this hook suspends, if any.  Stamped
            onto the emitted :class:`~ai.messages.HookPart` so consumers
            (run-blocked tracking, UIs) can attribute the suspension to
            its tool call.  Approval gating sets this automatically;
            custom gating wrappers should pass it too.
        registry: The :class:`HookRegistry` to register the hook in.
            Defaults to the current registry.

    """
    call = middleware_.HookContext(
        label=_label(hook),
        payload=payload,
        metadata=metadata or {},
        tool_call_id=tool_call_id,
        registry=registry,
    )

    chain = middleware_._build_hook_chain(_hook_impl)
    result = await chain(call)
    return cast("T", result)


async def _hook_impl(call: middleware_.HookContext) -> pydantic.BaseModel:
    """Core hook logic — the innermost ``next`` in the middleware chain."""
    rt = runtime_.get_runtime()
    registry = _registry(call.registry)
    label = call.label
    payload = call.payload
    hook_metadata = call.metadata

    data = telemetry.HookSpanData(
        label=label,
        hook_type=payload.__name__,
        metadata=hook_metadata,
        tool_call_id=call.tool_call_id,
    )

    # Pre-registered resolution (serverless re-entry).
    pre_registered = registry._pending_resolutions.pop(label, None)
    if pre_registered is not None:
        async with telemetry.span(data, replay=True) as sp:
            if isinstance(pre_registered, BaseException):
                raise pre_registered
            sp.data.status = "resolved"
            sp.data.resolution = pre_registered
            return payload(**pre_registered)

    # No resolution available — suspend.  The span covers the whole
    # suspension: how long the run sat waiting on external input.
    async with telemetry.span(data) as sp:
        future: asyncio.Future[dict[str, Any]] = asyncio.Future()

        registry._live_hooks[label] = (future, hook_metadata, rt)

        # Emit pending signal.
        hook_part: messages_.HookPart[Any] = messages_.HookPart(
            hook_id=label,
            hook_type=payload.__name__,
            status="pending",
            metadata=hook_metadata,
            tool_call_id=call.tool_call_id,
        )

        await rt.put_hook(hook_part)
        sp.add_event(telemetry.HOOK_DEFERRED)
        await sp.push()

        # Await resolution — may be resolved externally or cancelled.
        try:
            resolution = await future
        except asyncio.CancelledError as exc:
            sp.data.status = "cancelled"
            # ``cancel_hook(reason=...)`` rides on the CancelledError.
            attrs: dict[str, Any] = {}
            if exc.args and exc.args[0] is not None:
                attrs["reason"] = exc.args[0]
            sp.add_event(telemetry.HOOK_CANCELLED, attrs)
            await sp.push()
            raise
        finally:
            # Clean up live registry.
            registry._live_hooks.pop(label, None)

        sp.data.status = "resolved"
        sp.data.resolution = resolution
        sp.add_event(telemetry.HOOK_RESOLVED)
        await sp.push()

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
    registry: HookRegistry | None = None,
) -> None:
    """Resolve a hook by label.

    Works in two modes:

    1. **Live hook exists** (long-running): validates data (if ``payload``
       type is provided), resolves the future immediately, unblocking the
       awaiting coroutine.

    2. **No live hook yet** (serverless re-entry): stashes the resolution
       in the pre-registration registry.  When ``hook()`` executes during
       replay, it finds the pre-registered value and returns without
       suspending.  Pre-registration must target the run's registry:
       either call this inside the ``async with agent.run(...)`` block
       (before iterating the stream), or pass the registry explicitly.

    Passing an exception sends it to the awaiter (or stashes it for the
    next replay) so the awaiting ``ai.hook(...)`` call raises rather than
    returns.  See :func:`defer_hook` for the common case of
    propagating a :class:`HookDeferredException`.

    Args:
        hook: The hook label to resolve, or a HookPart whose ``hook_id``
            supplies it.
        data: Resolution data — a dict, pydantic model instance, or an
            exception to raise in the awaiter.
        payload: Optional pydantic model class for validation.  Ignored
            when *data* is an exception.
        registry: The :class:`HookRegistry` to resolve in.  Defaults to
            the current registry.  Pass the run's registry explicitly
            when resolving from a task outside the ``agent.run()`` block
            (e.g. a UI callback).

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

    reg = _registry(registry)

    # Path 1: live hook — resolve the future directly.
    if label in reg._live_hooks:
        future, _, _rt = reg._live_hooks[label]
        if isinstance(resolution, BaseException):
            future.set_exception(resolution)
        else:
            future.set_result(resolution)
        return

    # Path 2: no live hook — pre-register for later consumption.
    reg._pending_resolutions[label] = resolution


def defer_hook(
    hook_part: messages_.HookPart[Any],
    *,
    registry: HookRegistry | None = None,
) -> None:
    """Defer the hook identified by ``hook_part.hook_id``.

    The deferred exception carries a :class:`HookDeferredException` wrapping
    *hook_part*.

    Convenience wrapper around :func:`resolve_hook` for the serverless
    pattern where a caller has a :class:`~ai.messages.HookPart` (e.g.
    from inbound conversion) and needs to surface it back through the
    awaiting ``ai.hook(...)`` site as a structured suspension.
    """
    resolve_hook(
        hook_part.hook_id, HookDeferredException(hook_part), registry=registry
    )


async def cancel_hook(
    hook: str | messages_.HookPart[Any],
    *,
    reason: str | None = None,
    registry: HookRegistry | None = None,
) -> None:
    """Cancel a deferred hook.

    Only works for live hooks (long-running mode).  Raises ValueError
    if the hook is not currently deferred.  ``hook`` may be a label
    string or a HookPart whose ``hook_id`` supplies it.  ``registry``
    selects the :class:`HookRegistry` to use, defaulting to the current
    one.
    """
    label = _label(hook)
    reg = _registry(registry)
    if label not in reg._live_hooks:
        raise ValueError(f"No deferred hook with label: {label!r}")

    future, hook_metadata, rt = reg._live_hooks.pop(label)
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
