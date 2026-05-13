"""Adapter registries.

Maps adapter strings to their handler functions.  Adapter modules
are imported lazily on first use to keep import-time lightweight.

.. note::

    Connection checks are no longer dispatched through a registry.
    :func:`~ai.models.core.api.check_connection` delegates through
    :meth:`ai.models.Model.check` to the model's provider.
"""

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

    from ...providers.ai_gateway.adapter import generate as ai_gw_generate
    from ...providers.ai_gateway.adapter import stream as ai_gw_stream
    from ...providers.anthropic.adapter import stream as anthropic_stream
    from ...providers.openai.adapter import stream as openai_stream

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
