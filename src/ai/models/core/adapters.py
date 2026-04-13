"""Adapter and check-function registries.

Maps adapter/provider strings to their handler functions.  Adapter modules
are imported lazily on first use to keep import-time lightweight.
"""

from __future__ import annotations

from . import proto

# ---------------------------------------------------------------------------
# Stream / generate adapter registry
# ---------------------------------------------------------------------------

_stream_adapters: dict[str, proto.StreamFn] = {}
_generate_adapters: dict[str, proto.GenerateFn] = {}
_adapters_loaded = False


def _ensure_adapters() -> None:
    """Lazily register built-in adapter functions on first call."""
    global _adapters_loaded  # noqa: PLW0603
    if _adapters_loaded:
        return
    _adapters_loaded = True

    from ..ai_gateway.generate import generate as ai_gw_generate
    from ..ai_gateway.stream import stream as ai_gw_stream
    from ..anthropic.adapter import stream as anthropic_stream
    from ..openai.adapter import stream as openai_stream

    _stream_adapters["ai-gateway-v3"] = ai_gw_stream
    _generate_adapters["ai-gateway-v3"] = ai_gw_generate
    _stream_adapters["openai"] = openai_stream
    _stream_adapters["anthropic"] = anthropic_stream


def register_stream(adapter: str, fn: proto.StreamFn) -> None:
    """Register a stream adapter function for the given adapter key.

    Use this to add custom adapters (or override built-in ones).
    """
    _stream_adapters[adapter] = fn


def register_generate(adapter: str, fn: proto.GenerateFn) -> None:
    """Register a generate adapter function for the given adapter key.

    Use this to add custom adapters (or override built-in ones).
    """
    _generate_adapters[adapter] = fn


def get_stream_adapter(adapter: str) -> proto.StreamFn:
    """Return the stream adapter for *adapter*, raising on miss."""
    _ensure_adapters()
    fn = _stream_adapters.get(adapter)
    if fn is None:
        registered = ", ".join(sorted(_stream_adapters)) or "(none)"
        raise KeyError(
            f"No stream adapter registered for adapter={adapter!r}. "
            f"Registered: {registered}"
        )
    return fn


def get_generate_adapter(adapter: str) -> proto.GenerateFn:
    """Return the generate adapter for *adapter*, raising on miss."""
    _ensure_adapters()
    fn = _generate_adapters.get(adapter)
    if fn is None:
        registered = ", ".join(sorted(_generate_adapters)) or "(none)"
        raise KeyError(
            f"No generate adapter registered for adapter={adapter!r}. "
            f"Registered: {registered}"
        )
    return fn


# ---------------------------------------------------------------------------
# Connection-check registry — keyed by *provider* (not adapter) because the
# check verifies "can this client reach this provider and does this model
# exist there".
# ---------------------------------------------------------------------------

_check_fns: dict[str, proto.CheckConnFn] = {}
_check_fns_loaded = False


def _ensure_check_fns() -> None:
    """Lazily register built-in check functions on first call."""
    global _check_fns_loaded  # noqa: PLW0603
    if _check_fns_loaded:
        return
    _check_fns_loaded = True

    from ..ai_gateway import check as ai_gw_check
    from ..anthropic import check as anthropic_check
    from ..openai import check as openai_check

    _check_fns["ai-gateway"] = ai_gw_check.check
    _check_fns["anthropic"] = anthropic_check.check
    _check_fns["openai"] = openai_check.check


def register_check(provider: str, fn: proto.CheckConnFn) -> None:
    """Register a connection-check function for a provider.

    Use this to add checks for custom providers.
    """
    _check_fns[provider] = fn


def get_check_fn(provider: str) -> proto.CheckConnFn:
    """Return the check function for *provider*, raising on miss."""
    _ensure_check_fns()
    fn = _check_fns.get(provider)
    if fn is None:
        registered = ", ".join(sorted(_check_fns)) or "(none)"
        raise KeyError(
            f"No check function registered for provider={provider!r}. "
            f"Registered: {registered}"
        )
    return fn
