"""Model metadata types."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Literal, Self, cast, overload

import pydantic

from ... import _modelsdev
from ...errors import ConfigurationError
from ...providers import base

if TYPE_CHECKING:
    from collections.abc import Callable

    import modelsdotdev

_DEFAULT_MODEL_ENV = "AI_SDK_DEFAULT_MODEL"


class Model(pydantic.BaseModel):
    """Lightweight reference to a model on a specific provider.

    * ``id`` — identifier sent to the provider (e.g. ``"claude-sonnet-4-6"``).
    * ``provider_name`` — models.dev provider id used to rebuild the provider.
    * ``provider_args`` — JSON-friendly provider configuration.

    Passing a live ``provider`` makes the model non-serializable.
    """

    id: str
    provider_name: str | None = None
    provider_args: dict[str, Any] = pydantic.Field(default_factory=dict)

    _provider: base.Provider[Any] | None = pydantic.PrivateAttr(default=None)
    _protocol: base.ProviderProtocol[Any] | None = pydantic.PrivateAttr(
        default=None
    )
    _is_serializable: bool = pydantic.PrivateAttr(default=True)

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @overload
    def __init__(
        self,
        id: str,
        *,
        provider_name: str,
        provider_args: dict[str, Any] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        id: str,
        *,
        provider: base.Provider[Any],
        protocol: base.ProviderProtocol[Any] | None = None,
    ) -> None: ...

    def __init__(
        self,
        id: str,
        *,
        provider_name: str | None = None,
        provider_args: dict[str, Any] | None = None,
        provider: base.Provider[Any] | None = None,
        protocol: base.ProviderProtocol[Any] | None = None,
    ) -> None:
        if (provider is None) == (provider_name is None):
            raise ConfigurationError(
                "pass exactly one of provider_name or provider"
            )
        if provider_name == "":
            raise ConfigurationError("provider_name must not be empty")

        if provider is not None:
            if provider_args is not None:
                raise ConfigurationError("provider_args requires provider_name")
            super().__init__(
                id=id,
                provider_name=provider.name,
                provider_args={},
            )
            self._provider = provider
            self._protocol = protocol
            self._is_serializable = False
            return

        if protocol is not None:
            raise ConfigurationError(
                "protocol objects are not serializable; "
                "use provider=... live-object mode"
            )

        super().__init__(
            id=id,
            provider_name=provider_name,
            provider_args=provider_args or {},
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Model):
            return False
        if self._is_serializable and other._is_serializable:
            return (
                self.id == other.id
                and self.provider_name == other.provider_name
                and self.provider_args == other.provider_args
            )
        return (
            not self._is_serializable
            and not other._is_serializable
            and self.id == other.id
            and self._provider is other._provider
            and self._protocol is other._protocol
        )

    def __repr__(self) -> str:
        provider = (
            self._provider if self._provider is not None else self.provider_name
        )
        return f"Model(id={self.id!r}, provider={provider!r})"

    def __hash__(self) -> int:
        if self._is_serializable:
            return hash(
                (
                    self.id,
                    self.provider_name,
                    json.dumps(self.provider_args, sort_keys=True),
                )
            )
        return hash((self.id, id(self._provider), id(self._protocol)))

    @property
    def is_serializable(self) -> bool:
        """Whether this model can be serialized as durable JSON data."""
        return self._is_serializable

    @property
    def provider(self) -> base.Provider[Any]:
        """Provider for this model, lazily rebuilt for durable models."""
        if self._provider is None:
            if self.provider_name is None:
                raise ConfigurationError("model has no provider_name")
            self._provider = base.Provider.from_id(
                self.provider_name,
                model_provider_config=self._model_provider_config(),
                **{
                    key: value
                    for key, value in self.provider_args.items()
                    if value is not None
                },
            )
        return self._provider

    @property
    def protocol(self) -> base.ProviderProtocol[Any] | None:
        """Optional wire-protocol override for this model."""
        return self._protocol

    def with_protocol(self, protocol: base.ProviderProtocol[Any]) -> Self:
        return self.__class__(
            id=self.id,
            provider=self.provider,
            protocol=protocol,
        )

    def _model_provider_config(
        self,
    ) -> modelsdotdev.ModelProviderConfig | None:
        if self.provider_name is None:
            return None
        model_info = _modelsdev.get_model_by_id(
            f"{self.provider_name}:{self.id}"
        )
        return None if model_info is None else model_info.provider_config

    @pydantic.field_serializer("provider_args")
    def _serialize_provider_args(
        self, provider_args: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            key: value
            for key, value in provider_args.items()
            if value is not None
        }

    @pydantic.model_serializer(mode="wrap")
    def _serialize_model(
        self,
        handler: pydantic.SerializerFunctionWrapHandler,
        info: pydantic.SerializationInfo,
    ) -> dict[str, Any]:
        if info.mode == "json" and not self._is_serializable:
            raise ConfigurationError(
                "Model was constructed with a live provider/protocol and "
                "cannot be serialized. Use provider_name/provider_args instead."
            )
        return cast("dict[str, Any]", handler(self))

    def model_dump(
        self,
        *,
        mode: Literal["json", "python"] | str = "python",
        include: Any = None,
        exclude: Any = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_computed_fields: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal["none", "warn", "error"] = True,
        fallback: Callable[[Any], Any] | None = None,
        serialize_as_any: bool = False,
    ) -> dict[str, Any]:
        if mode == "json" and not self._is_serializable:
            raise ConfigurationError(
                "Model was constructed with a live provider/protocol and "
                "cannot be serialized. Use provider_name/provider_args instead."
            )
        return super().model_dump(
            mode=mode,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            exclude_computed_fields=exclude_computed_fields,
            round_trip=round_trip,
            warnings=warnings,
            fallback=fallback,
            serialize_as_any=serialize_as_any,
        )

    def model_dump_json(
        self,
        *,
        indent: int | None = None,
        ensure_ascii: bool = False,
        include: Any = None,
        exclude: Any = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_computed_fields: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal["none", "warn", "error"] = True,
        fallback: Callable[[Any], Any] | None = None,
        serialize_as_any: bool = False,
    ) -> str:
        if not self._is_serializable:
            raise ConfigurationError(
                "Model was constructed with a live provider/protocol and "
                "cannot be serialized. Use provider_name/provider_args instead."
            )
        return super().model_dump_json(
            indent=indent,
            ensure_ascii=ensure_ascii,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            exclude_computed_fields=exclude_computed_fields,
            round_trip=round_trip,
            warnings=warnings,
            fallback=fallback,
            serialize_as_any=serialize_as_any,
        )


def get_model(
    model_id: str | None = None,
    *,
    protocol: base.ProviderProtocol[Any] | None = None,
) -> Model:
    """Resolve a model ID into a :class:`Model`.

    Args:
        model_id:
            Model ID, optionally in the format of ``"provider:model"``.
            When the provider is omitted, the model is routed through
            Vercel AI Gateway. Examples: ``"openai:gpt-5"`` or
            ``"anthropic/claude-sonnet-4"``. When omitted, reads
            ``AI_SDK_DEFAULT_MODEL`` from the environment.
        protocol:
            Optional wire-protocol override for this model. When omitted,
            the provider chooses its default protocol.

    Raises:
        Raises :class:`ai.ConfigurationError` when ``model_id`` and
        ``AI_SDK_DEFAULT_MODEL`` is empty or malformed.
        Raises a :class:`ai.UnsupportedProviderError` when the provider is
        unrecognized or otherwise unsupported.

    """
    if model_id is None:
        model_id = os.environ.get(_DEFAULT_MODEL_ENV)
        if not model_id:
            raise ConfigurationError(
                f"{_DEFAULT_MODEL_ENV} must be set when ai.get_model() "
                "is called without arguments"
            )

    if not model_id:
        raise ConfigurationError(f"get_model: malformed model_id: {model_id!r}")

    if protocol is not None:
        raise ConfigurationError(
            "protocol objects are not serializable; "
            "construct Model with provider=... live-object mode instead"
        )

    if ":" not in model_id:
        model_id = f"gateway:{model_id}"

    ref = _modelsdev.parse_model_id(model_id)
    assert ref.provider_id is not None  # guaranteed to be fully-qualified here
    provider_id = ref.provider_id
    provider_model_id = ref.model_id

    return Model(provider_model_id, provider_name=provider_id)
