"""models — composable model layer.

A :class:`Model` holds a *recipe* for its provider — a factory callable
plus its arguments — instead of a live provider object.  The provider and
its client are built lazily on first use.  When the factory is a named,
module-level callable and the args are JSON-friendly (everything
``get_model`` produces), the model serializes: ``model.model_dump()`` /
``Model.model_validate()`` round-trip.  Any other callable (a lambda, a
closure over live objects) works normally in-process, but ``model_dump``
raises and ``model.serializable`` is ``False``.

Usage::

    import ai
    model = ai.get_model("openai:gpt-5.4")
    model = ai.get_model("anthropic:claude-sonnet-4-6")
    model = ai.get_model("anthropic/claude-sonnet-4")  # defaults to Gateway

    # custom provider configuration — JSON-friendly args
    model = ai.Model(
        "llama3",
        provider_factory=ai.get_provider,
        provider_args={"id": "openai", "base_url": "http://localhost:11434/v1"},
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

    # anything non-serializable (clients, custom auth) lives inside a
    # named module-level factory; its import path is what's serialized
    def my_provider() -> ai.Provider:
        return ai.get_provider("openai", client=shared_client)

    model = ai.Model("gpt-5.4", provider_factory=my_provider)

    # if the model never crosses a process boundary, any callable works —
    # the model just isn't serializable (model_dump() raises)
    model = ai.Model("gpt-5.4", provider_factory=lambda: provider)
    assert model.serializable is False

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
from .core.model import Model, get_model
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
    "Provider",
    "ProviderProtocol",
    "ProviderRankingStrategy",
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
