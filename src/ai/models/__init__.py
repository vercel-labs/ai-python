"""models — composable model layer.

A :class:`Model` holds JSON-safe provider settings instead of a live
provider object.  The provider and its client are built lazily on first
use.  Models produced by ``get_model`` round-trip through
``model.model_dump(mode="json")`` / ``Model.model_validate()``.

Usage::

    import ai
    model = ai.get_model("openai:gpt-5.4")
    model = ai.get_model("anthropic:claude-sonnet-4-6")
    model = ai.get_model("anthropic/claude-sonnet-4")  # defaults to Gateway

    # custom provider configuration — JSON-friendly args
    model = ai.Model(
        "llama3",
        provider=ai.ProviderRef(
            "openai",
            base_url="http://localhost:11434/v1",
        ),
    )

    # stream — auto-creates client from env vars
    msgs = [ai.user_message("hello")]
    async with ai.stream(model, msgs) as s:
        async for event in s:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)

    # models serialize and rebuild their provider on first use
    data = model.model_dump(mode="json")
    model = ai.Model.model_validate(data)

    # list available models
    ids = await ai.get_provider("openai").list_models()
"""

from ..providers.base import Provider, ProviderProtocol
from .core.api import (
    Executor,
    GenerateExecutor,
    GenerateRequest,
    Stream,
    StreamExecutor,
    StreamRequest,
    generate,
    probe,
    stream,
)
from .core.model import Model, ProtocolRef, ProviderRef, get_model
from .core.params import (
    DEFAULT,
    GLOBAL,
    RANDOM,
    UNSET,
    CacheParams,
    CloudRegion,
    ContextManagementParams,
    GenerateParams,
    GeoRegion,
    ImageParams,
    InferenceRequestParams,
    MinPSamplerParams,
    ModelProviderDefault,
    OutputParams,
    ProviderRankingStrategy,
    ProviderServiceParams,
    RandomSeed,
    ReasoningParams,
    RepetitionPenaltyParams,
    RoutingParams,
    RoutingTarget,
    RoutingTargetChain,
    SeedSamplerParams,
    TemperatureSamplerParams,
    TokenThreshold,
    ToolCallingParams,
    ToolChoiceMode,
    ToolRef,
    ToolSelection,
    TopKSamplerParams,
    TopPSamplerParams,
    Unset,
    VideoParams,
)

__all__ = [
    "DEFAULT",
    "GLOBAL",
    "RANDOM",
    "UNSET",
    "CacheParams",
    "CloudRegion",
    "ContextManagementParams",
    "Executor",
    "GenerateExecutor",
    "GenerateParams",
    "GenerateRequest",
    "GeoRegion",
    "ImageParams",
    "InferenceRequestParams",
    "MinPSamplerParams",
    "Model",
    "ModelProviderDefault",
    "OutputParams",
    "ProtocolRef",
    "Provider",
    "ProviderProtocol",
    "ProviderRankingStrategy",
    "ProviderRef",
    "ProviderServiceParams",
    "RandomSeed",
    "ReasoningParams",
    "RepetitionPenaltyParams",
    "RoutingParams",
    "RoutingTarget",
    "RoutingTargetChain",
    "SeedSamplerParams",
    "Stream",
    "StreamExecutor",
    "StreamRequest",
    "TemperatureSamplerParams",
    "TokenThreshold",
    "ToolCallingParams",
    "ToolChoiceMode",
    "ToolRef",
    "ToolSelection",
    "TopKSamplerParams",
    "TopPSamplerParams",
    "Unset",
    "VideoParams",
    "generate",
    "get_model",
    "probe",
    "stream",
]
