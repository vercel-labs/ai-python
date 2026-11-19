"""Core types for models."""

import warnings
from typing import TYPE_CHECKING

from ...providers.base import Provider
from . import helpers
from .api import (
    Stream,
    generate,
    probe,
    stream,
)
from .model import Model, get_model
from .params import (
    DEFAULT,
    GLOBAL,
    RANDOM,
    UNSET,
    CacheParams,
    CloudRegion,
    ContextManagementParams,
    GeoRegion,
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
)

__all__ = [
    "DEFAULT",
    "GLOBAL",
    "RANDOM",
    "UNSET",
    "CacheParams",
    "CloudRegion",
    "ContextManagementParams",
    "GeoRegion",
    "InferenceRequestParams",
    "MinPSamplerParams",
    "Model",
    "ModelProviderDefault",
    "OutputParams",
    "Provider",
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
    "TemperatureSamplerParams",
    "TokenThreshold",
    "ToolCallingParams",
    "ToolChoiceMode",
    "ToolRef",
    "ToolSelection",
    "TopKSamplerParams",
    "TopPSamplerParams",
    "Unset",
    "experimental_generate",
    "generate",
    "get_model",
    "helpers",
    "probe",
    "stream",
]

if TYPE_CHECKING:
    experimental_generate = generate
else:

    def __getattr__(name: str) -> object:
        if name == "experimental_generate":
            warnings.warn(
                f"{__name__}.experimental_generate is deprecated; "
                "use ai.generate",
                DeprecationWarning,
                stacklevel=2,
            )
            return generate
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
