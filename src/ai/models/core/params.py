import dataclasses
import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Self, final

import pydantic


@final
class ModelProviderDefault:
    """Sentinel for params: default value used by the model/provider."""


DEFAULT = ModelProviderDefault()
"""Sentinel for params: default value used by the model/provider."""


@final
class Unset:
    """Sentinel for a value that should be unset/omitted in a request."""


UNSET = Unset()
"""Sentinel for a value that should be unset/omitted in a request."""


@final
class RandomSeed:
    """Sentinel for explicitly random seed selection."""


RANDOM = RandomSeed()
"""Sentinel requesting provider-random seed selection."""


_PARAMS_CONFIG = pydantic.ConfigDict(frozen=True, populate_by_name=True)


class ImageParams(pydantic.BaseModel):
    """Parameters for image generation (``/image-model`` endpoint)."""

    model_config = _PARAMS_CONFIG

    n: int = 1
    size: str | None = None
    aspect_ratio: str | None = pydantic.Field(
        default=None, serialization_alias="aspectRatio"
    )
    seed: int | None = None
    provider_options: dict[str, Any] = pydantic.Field(
        default_factory=dict, serialization_alias="providerOptions"
    )


class VideoParams(pydantic.BaseModel):
    """Parameters for video generation (``/video-model`` endpoint)."""

    model_config = _PARAMS_CONFIG

    n: int = 1
    aspect_ratio: str | None = pydantic.Field(
        default=None, serialization_alias="aspectRatio"
    )
    resolution: str | None = None
    duration: int | None = None
    fps: int | None = None
    seed: int | None = None
    provider_options: dict[str, Any] = pydantic.Field(
        default_factory=dict, serialization_alias="providerOptions"
    )


GenerateParams = ImageParams | VideoParams


@dataclass(frozen=True, kw_only=True)
class TemperatureSamplerParams:
    """Temperature sampling controls."""

    temperature: float | ModelProviderDefault = DEFAULT


@dataclass(frozen=True, kw_only=True)
class TopKSamplerParams:
    """Top-k sampling controls."""

    top_k: int | ModelProviderDefault | None = DEFAULT


@dataclass(frozen=True, kw_only=True)
class TopPSamplerParams:
    """Nucleus sampling controls."""

    top_p: float | ModelProviderDefault | None = DEFAULT


@dataclass(frozen=True, kw_only=True)
class MinPSamplerParams:
    """Minimum probability sampling controls."""

    min_p: float | ModelProviderDefault | None = DEFAULT


@dataclass(frozen=True, kw_only=True)
class RepetitionPenaltyParams:
    """Penalty controls for repeated or overrepresented tokens."""

    repetition_penalty: float | ModelProviderDefault | None = DEFAULT
    frequency_penalty: float | ModelProviderDefault | None = DEFAULT
    presence_penalty: float | ModelProviderDefault | None = DEFAULT
    consideration_window: int | ModelProviderDefault | None = DEFAULT


@dataclass(frozen=True, kw_only=True)
class SeedSamplerParams:
    """Random seed controls for sampling."""

    seed: int | RandomSeed | ModelProviderDefault | None = DEFAULT


SamplerParams = (
    TemperatureSamplerParams
    | TopKSamplerParams
    | TopPSamplerParams
    | MinPSamplerParams
    | RepetitionPenaltyParams
    | SeedSamplerParams
)


type SamplerParamsMap = dict[type[SamplerParams], SamplerParams]


type ProviderParamsMap = dict[type[Any], Any]


class ToolChoiceMode(enum.StrEnum):
    """Built-in policies for model tool selection."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class ToolRef(str):
    """A reference to a specific tool (by tool name)."""

    def __repr__(self) -> str:
        return f"<ToolRef {super().__repr__()}>"


@dataclass(frozen=True, kw_only=True, init=False)
class ToolSelection:
    """Tool subset paired with a tool choice policy."""

    tools: frozenset[ToolRef]
    mode: ToolChoiceMode

    def __init__(
        self,
        tools: Iterable[ToolRef] | Iterable[str],
        *,
        mode: ToolChoiceMode,
    ) -> None:
        object.__setattr__(self, "tools", frozenset(ToolRef(s) for s in tools))
        object.__setattr__(self, "mode", mode)


@dataclass(frozen=True, kw_only=True)
class ToolCallingParams:
    """Tool calling parameters."""

    max_tool_calls: int | ModelProviderDefault | None = DEFAULT
    """The maximum number of tool calls the model may make."""

    parallel_tool_calls: bool | ModelProviderDefault = DEFAULT
    """Whether the model may call multiple tools in parallel."""

    tool_choice: ToolChoiceMode | ToolRef | ToolSelection
    """Tool choice policy.

    * `ToolChoiceMode.AUTO`: the model can choose whether and what tools to call
    * `ToolChoiceMode.REQUIED`: the model must call (some) tool
    * `ToolChoiceMode.NONE`: the model must not call tools
    * `ToolRef("tool-name")`: the model must call the specified tool
    * `ToolSelection(tools, mode)`: the model should treat the specified set of
      tools according to mode.
    """


@dataclass(frozen=True, kw_only=True)
class ReasoningParams:
    """Model reasoning/thinking options."""

    effort: str | ModelProviderDefault | None = DEFAULT
    """Provider-specific reasoning/thinking effort level.

    None means reasoning is disabled."""


@dataclass(frozen=True, kw_only=True)
class ProviderServiceParams:
    """Provider service parameters (service tier)."""

    service_tier: str | ModelProviderDefault = DEFAULT
    """Provider-specific service tier."""


@dataclass(frozen=True)
class TokenThreshold:
    """Token count used as a trigger threshold."""

    value: int
    """Token count threshold."""


@dataclass(frozen=True, kw_only=True)
class ContextManagementParams:
    """Server-side context management parameters."""

    compaction: TokenThreshold | None = None
    """Compaction trigger threshold."""


@dataclass(frozen=True, kw_only=True)
class OutputParams:
    """Model output options."""

    max_tokens: int | None = None
    """The maximum number of tokens to generate before stopping."""

    include: frozenset[str] | None = None
    """Additional provider-specific data to include in the model response."""

    text_verbosity: str | ModelProviderDefault | None = DEFAULT
    """Provider-specific text verbosity level."""

    reasoning_summary: str | ModelProviderDefault | None = DEFAULT
    """Provider-specific reasoning summary emission level.

    None means "disabled"."""


@dataclass(frozen=True, kw_only=True)
class CacheParams:
    """Provider prompt caching behavior."""

    mode: str | ModelProviderDefault = DEFAULT
    """Provider-specific cache mode."""

    retention: str | ModelProviderDefault = DEFAULT
    """Provider-specific cache retention period (time-to-live)."""

    key: str | None = None
    """Custom cache key component. Support is provider-specific."""


@final
class GlobalRoutingTarget:
    """Sentinel for globally scoped request routing."""

    def __repr__(self) -> str:
        return "GLOBAL"


GLOBAL = GlobalRoutingTarget()
"""Sentinel requesting globally scoped request routing."""


class GeoRegion(str):
    """A broad geography, e.g. ``us`` or ``eu``."""


class CloudRegion(str):
    """A specific cloud/provider region, e.g. ``us-east-1``."""


type RoutingTarget = GlobalRoutingTarget | GeoRegion | CloudRegion


@dataclass(frozen=True, kw_only=True)
class RoutingTargetChain:
    """Separate Gateway and provider routing targets."""

    gateway: RoutingTarget
    provider: RoutingTarget


type RoutingTargetParam = RoutingTarget | RoutingTargetChain


class ProviderRankingStrategy(enum.StrEnum):
    """Provider ranking strategy."""

    COST = "cost"
    TTFT = "ttft"
    TPS = "tps"
    PRICE = "price"
    LATENCY = "latency"
    THROUGHPUT = "throughput"


@dataclass(frozen=True, kw_only=True)
class RoutingParams:
    """Inference request routing options."""

    routing_target: RoutingTargetParam | None = None
    """Request (geo-/region-) routing target."""

    provider_allowlist: frozenset[str] | None = None
    """Restrict gateway routing to these providers."""

    provider_order: tuple[str, ...] | None = None
    """Preferred provider order."""

    provider_ranking: ProviderRankingStrategy | None = None
    """Dynamic provider sorting strategy."""

    fallback_models: tuple[str, ...] | None = None
    """Fallback models to try after the requested model."""


@dataclass(frozen=True, kw_only=True)
class InferenceRequestParams:
    """Model inference request parameters."""

    sampling: SamplerParamsMap | ModelProviderDefault = DEFAULT
    """Advanced token sampling parameters (e.g temperature, max_p etc)."""

    reasoning: ReasoningParams | ModelProviderDefault = DEFAULT
    """Model reasoning parameters."""

    tool_calling: ToolCallingParams | None = None
    """Tool calling parameters."""

    provider_service: ProviderServiceParams | None = None
    """Provider-specific service parameters (service tier etc)."""

    safety_identifier: str | None = None
    """A stable identifier used for safety monitoring and abuse detection."""

    metadata: Mapping[str, str] | None = None
    """User-specified metadata associated with the request.

    Note that not all providers support attaching metadata to inference
    requests, and the ones that do might place restrictions on length of
    metadata both in terms of overall length and in terms of individual items.
    For example, OpenAI and Open Responses-compatible providers specify that
    keys must have a maximum length of 64 characters, values have a maximum
    length of 512 characters, and the total number of metadata items must not
    exceed 16."""

    tags: frozenset[str] | None = None
    """User-specified tags associated with the request.

    Note that not all providers support attaching tags to inference
    requests, and the ones that do might place restrictions on length of
    the tags collections as well as restrictions on individual tag value
    length.  For example, Vercel AI Gateway limits the number of tags to
    10 and the length of each tag to 64 characters.
    """

    output: OutputParams | None = None
    """Model output configuration."""

    cache: CacheParams | None = None
    """Prompt cache parameters."""

    routing: RoutingParams | None = None
    """Request routing parameters."""

    context_management: ContextManagementParams | None = None
    """Context management parameters."""

    provider_params: ProviderParamsMap | None = None
    """Provider-specific typed request parameters keyed by params type."""

    extra_headers: Mapping[str, str | Unset] | None = None
    """Extra headers to pass to the provider API."""

    extra_query: Mapping[str, Any] | None = None
    """Extra URL query string arguments to pass to the provider API."""

    extra_body: Mapping[str, Any] | None = None
    """Extra body arguments to pass to the provider API."""

    def with_temperature(
        self, temperature: float | ModelProviderDefault
    ) -> Self:
        temp_sampling_params: SamplerParamsMap = {
            TemperatureSamplerParams: TemperatureSamplerParams(
                temperature=temperature
            )
        }

        if type(self.sampling) is ModelProviderDefault:
            sampling = temp_sampling_params
        else:
            sampling = self.sampling | temp_sampling_params

        return dataclasses.replace(self, sampling=sampling)

    def with_reasoning_effort(
        self, effort: str | ModelProviderDefault | None
    ) -> Self:
        return dataclasses.replace(
            self,
            reasoning=ReasoningParams(effort=effort),
        )

    def with_provider_params(self, *provider_params: object) -> Self:
        params = dict(self.provider_params or {})
        for value in provider_params:
            params[type(value)] = value
        return dataclasses.replace(self, provider_params=params)
