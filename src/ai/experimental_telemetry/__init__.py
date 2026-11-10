"""Telemetry: spans, sinks, adapters, and the ambient current span.

Experimental: not part of the stable API, may change or be removed.

See :mod:`.span` for the full story.
"""

import warnings
from typing import TYPE_CHECKING, Any

from .span import (
    FIRST_TOKEN,
    HOOK_CANCELLED,
    HOOK_DEFERRED,
    HOOK_RESOLVED,
    RESPONSE_COMPLETE,
    AdapterCallable,
    AdapterMixin,
    AdapterProtocol,
    AiGenerateSpanData,
    AiStreamSpanData,
    CustomSpanData,
    DecideSpanData,
    DictSink,
    EmbedSpanData,
    GenerateAudioSpanData,
    GenerateImageSpanData,
    GenerateVideoSpanData,
    HookSpanData,
    LoopTurnSpanData,
    RerankSpanData,
    RunSpanData,
    Sink,
    Span,
    SpanData,
    SpanError,
    SpanEvent,
    ToolExecutionSpanData,
    TranscribeSpanData,
    adapter,
    create_span,
    current_span,
    is_enabled,
    now_ns,
    push_all,
    register,
    span,
    unregister,
    use_sink,
    use_span,
    use_time,
)

__all__ = [
    "FIRST_TOKEN",
    "HOOK_CANCELLED",
    "HOOK_DEFERRED",
    "HOOK_RESOLVED",
    "RESPONSE_COMPLETE",
    "AdapterCallable",
    "AdapterMixin",
    "AdapterProtocol",
    "AiGenerateSpanData",
    "AiStreamSpanData",
    "CustomSpanData",
    "DecideSpanData",
    "DictSink",
    "EmbedSpanData",
    "GenerateAudioSpanData",
    "GenerateImageSpanData",
    "GenerateVideoSpanData",
    "HookSpanData",
    "LoopTurnSpanData",
    "RerankSpanData",
    "RunSpanData",
    "Sink",
    "Span",
    "SpanData",
    "SpanError",
    "SpanEvent",
    "ToolExecutionSpanData",
    "TranscribeSpanData",
    "adapter",
    "create_span",
    "current_span",
    "is_enabled",
    "now_ns",
    "push_all",
    "register",
    "span",
    "unregister",
    "use_sink",
    "use_span",
    "use_time",
]

if TYPE_CHECKING:
    EvaluateSpanData = DecideSpanData
else:

    def __getattr__(name: str) -> Any:
        # Deprecated names, kept as aliases of their replacements.
        if name == "EvaluateSpanData":
            warnings.warn(
                "ai.experimental_telemetry.EvaluateSpanData is deprecated, "
                "use ai.experimental_telemetry.DecideSpanData",
                DeprecationWarning,
                stacklevel=2,
            )
            return DecideSpanData
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
