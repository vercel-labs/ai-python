from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class ProviderTimeoutsParams:
    """Gateway per-provider timeout configuration."""

    byok: Mapping[str, int] | None = None
    """Per-provider BYOK attempt timeouts in milliseconds."""


@dataclass(frozen=True, kw_only=True)
class GatewayParams:
    """Vercel AI Gateway-specific request parameters."""

    quota_entity_id: str | None = None
    """Gateway quota bucket/entity identifier."""

    zero_data_retention: bool | None = None
    """Request zero-data-retention routing."""

    hipaa_compliant: bool | None = None
    """Require HIPAA-compliant providers."""

    disallow_prompt_training: bool | None = None
    """Require providers that do not train on prompts."""

    byok: Mapping[str, Iterable[Mapping[str, Any]]] | None = None
    """Request-supplied BYOK credentials, keyed by provider."""

    provider_timeouts: ProviderTimeoutsParams | None = None
    """Per-provider routing timeout configuration."""


__all__ = ["GatewayParams", "ProviderTimeoutsParams"]
