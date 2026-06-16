"""Model metadata types."""

import importlib
import inspect
import json
import os
from collections.abc import Callable
from typing import Any, Self, cast

import pydantic

from ... import _modelsdev
from ...errors import ConfigurationError
from ...providers import base

_DEFAULT_MODEL_ENV = "AI_SDK_DEFAULT_MODEL"


def _callable_ref(fn: Callable[..., Any]) -> str:
    """Return the ``"package.module:qualname"`` reference for *fn*.

    Raises :class:`ai.ConfigurationError` when *fn* cannot be found again
    by name — which is exactly what makes a factory unserializable.
    """
    module_name = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module_name or not qualname:
        raise ConfigurationError(
            f"factory {fn!r} has no importable name; it must be a named, "
            "module-level function or class so the model can be serialized"
        )
    if module_name == "__main__":
        raise ConfigurationError(
            f"factory {qualname!r} is defined in __main__ and cannot be "
            "imported by other processes; move it into an importable module"
        )
    if "<" in qualname:
        raise ConfigurationError(
            f"factory {module_name}.{qualname} must be a named, module-level "
            "function or class so the model can be serialized; lambdas and "
            "callables defined inside functions are not importable by name"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ConfigurationError(
            f"cannot import module {module_name!r} of factory "
            f"{qualname!r}: {error}"
        ) from error
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
    if obj is not fn:
        raise ConfigurationError(
            f"factory {module_name}:{qualname} does not import back to the "
            "same object; it must be a named, module-level function or class "
            "(bound methods and decorated wrappers are not supported)"
        )
    return f"{module_name}:{qualname}"


def _import_ref(ref: str) -> Callable[..., Any]:
    """Import a factory from a ``"package.module:qualname"`` reference."""
    module_name, sep, qualname = ref.partition(":")
    if not sep or not module_name or not qualname:
        raise ConfigurationError(
            f"malformed factory reference {ref!r}; "
            "expected 'package.module:name'"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ConfigurationError(
            f"cannot import factory {ref!r}: {error}"
        ) from error
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            raise ConfigurationError(
                f"cannot import factory {ref!r}: module {module_name!r} "
                f"has no attribute {qualname!r}"
            )
    if not callable(obj):
        raise ConfigurationError(f"factory {ref!r} is not callable")
    return cast("Callable[..., Any]", obj)


class Model(pydantic.BaseModel):
    """Reference to a model on a specific provider.

    * ``id`` — identifier sent to the provider (e.g. ``"claude-sonnet-4-6"``).
    * ``provider_factory`` — callable that builds the :class:`Provider`.
    * ``provider_args`` — keyword arguments for the factory.
    * ``protocol_factory`` / ``protocol_args`` — optional wire-protocol
      override for this model, same rules as the provider factory.

    The provider is built lazily on first :attr:`provider` access and
    cached.

    A model is **serializable** when the factory is a named module-level
    function or class (dumped as a ``"package.module:name"`` reference)
    and the args are JSON-friendly — everything :func:`get_model`
    produces qualifies.  Anything that cannot be expressed as JSON args
    (custom clients, shared connection pools) can live inside the
    factory body::

        def my_provider() -> ai.Provider:
            return ai.get_provider("openai", client=_shared_client)

        model = ai.Model("gpt-5", provider_factory=my_provider)

    Any other callable — a lambda, a closure over a live provider — is
    accepted and works normally in-process, but the model then cannot
    cross a JSON boundary: ``model_dump()`` raises with the reason.
    Check :attr:`serializable` to know ahead of time.
    """

    id: str
    provider_factory: Callable[..., Any]
    provider_args: dict[str, Any] = pydantic.Field(default_factory=dict)
    protocol_factory: Callable[..., Any] | None = None
    protocol_args: dict[str, Any] = pydantic.Field(default_factory=dict)

    _provider_instance: base.Provider[Any] | None = pydantic.PrivateAttr(
        default=None
    )
    _protocol_instance: base.ProviderProtocol[Any] | None = (
        pydantic.PrivateAttr(default=None)
    )

    def __init__(
        self,
        id: str,
        *,
        provider_factory: Callable[..., Any] | str,
        provider_args: dict[str, Any] | None = None,
        protocol_factory: Callable[..., Any] | str | None = None,
        protocol_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            id=id,
            provider_factory=provider_factory,
            provider_args={} if provider_args is None else provider_args,
            protocol_factory=protocol_factory,
            protocol_args={} if protocol_args is None else protocol_args,
        )

    @pydantic.field_validator(
        "provider_factory", "protocol_factory", mode="before"
    )
    @classmethod
    def _coerce_factory(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _import_ref(value)
        return value

    @pydantic.model_validator(mode="after")
    def _check_factory_args(self) -> Self:
        for label, factory, args in (
            ("provider_args", self.provider_factory, self.provider_args),
            ("protocol_args", self.protocol_factory, self.protocol_args),
        ):
            if factory is None:
                if args:
                    raise ConfigurationError(
                        "protocol_args given without protocol_factory"
                    )
                continue
            try:
                signature = inspect.signature(factory)
            except (TypeError, ValueError):
                continue  # some builtins have no introspectable signature
            try:
                signature.bind(**args)
            except TypeError as error:
                raise ConfigurationError(
                    f"{label} do not match the signature of "
                    f"{factory!r}: {error}"
                ) from error
        return self

    # Dumps must round-trip or raise: both serializers run for python
    # and JSON dumps alike, so a model built around a closure or live
    # objects fails loudly the moment it tries to cross a boundary.
    @pydantic.field_serializer("provider_factory", "protocol_factory")
    def _serialize_factory(
        self, value: Callable[..., Any] | None
    ) -> str | None:
        return None if value is None else _callable_ref(value)

    @pydantic.field_serializer("provider_args", "protocol_args")
    def _serialize_args(
        self,
        value: dict[str, Any],
        info: pydantic.FieldSerializationInfo,
    ) -> dict[str, Any]:
        try:
            json.dumps(value)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(
                f"{info.field_name} cannot round-trip through JSON "
                f"(put live objects inside a named module-level factory "
                f"instead): {error}"
            ) from error
        return value

    @property
    def serializable(self) -> bool:
        """Whether this model round-trips through ``model_dump``.

        ``True`` for named module-level factories with JSON-friendly
        args — everything :func:`get_model` produces.  ``False`` when
        the model was built around a lambda, closure, or live objects;
        such models work normally in-process but ``model_dump`` raises.
        """
        for factory, args in (
            (self.provider_factory, self.provider_args),
            (self.protocol_factory, self.protocol_args),
        ):
            if factory is None:
                continue
            try:
                _callable_ref(factory)
                json.dumps(args)
            except (ConfigurationError, TypeError, ValueError):
                return False
        return True

    @property
    def provider(self) -> base.Provider[Any]:
        """Provider instance, built lazily from the factory and cached."""
        provider = self._provider_instance
        if provider is None:
            provider = self.provider_factory(**self.provider_args)
            if not isinstance(provider, base.Provider):
                raise ConfigurationError(
                    f"provider factory {self.provider_factory!r} returned "
                    f"{type(provider).__name__}, expected a Provider"
                )
            self._provider_instance = provider
        return provider

    @property
    def protocol(self) -> base.ProviderProtocol[Any] | None:
        """Protocol override instance, built lazily and cached."""
        if self.protocol_factory is None:
            return None
        protocol = self._protocol_instance
        if protocol is None:
            protocol = self.protocol_factory(**self.protocol_args)
            if not isinstance(protocol, base.ProviderProtocol):
                raise ConfigurationError(
                    f"protocol factory {self.protocol_factory!r} returned "
                    f"{type(protocol).__name__}, expected a ProviderProtocol"
                )
            self._protocol_instance = protocol
        return protocol

    async def aclose(self) -> None:
        """Close the provider if this model lazily built one."""
        if self._provider_instance is not None:
            await self._provider_instance.aclose()
            self._provider_instance = None

    def __eq__(self, other: object) -> bool:
        # Pydantic's default __eq__ also compares private attributes,
        # which would make models unequal once one of them lazily built
        # its provider.  Compare the recipe fields only; factories
        # compare by identity (a round-tripped model imports the same
        # factory object back, so equality survives serialization).
        return (
            isinstance(other, Model)
            and self.id == other.id
            and self.provider_factory is other.provider_factory
            and self.provider_args == other.provider_args
            and self.protocol_factory is other.protocol_factory
            and self.protocol_args == other.protocol_args
        )

    def __hash__(self) -> int:
        return hash((self.id, self.provider_factory, self.protocol_factory))

    def with_protocol(
        self,
        protocol_factory: Callable[..., Any] | str,
        **protocol_args: Any,
    ) -> Self:
        model = self.__class__(
            self.id,
            provider_factory=self.provider_factory,
            provider_args=self.provider_args,
            protocol_factory=protocol_factory,
            protocol_args=protocol_args,
        )
        # Keep sharing an already-built provider instance.
        model._provider_instance = self._provider_instance
        return model


def get_model(
    model_id: str | None = None,
    *,
    protocol_factory: Callable[..., Any] | str | None = None,
    protocol_args: dict[str, Any] | None = None,
) -> Model:
    """Resolve a model ID into a :class:`Model`.

    Args:
        model_id:
            Model ID, optionally in the format of ``"provider:model"``.
            When the provider is omitted, the model is routed through
            Vercel AI Gateway. Examples: ``"openai:gpt-5"`` or
            ``"anthropic/claude-sonnet-4"``. When omitted, reads
            ``AI_SDK_DEFAULT_MODEL`` from the environment.
        protocol_factory:
            Optional wire-protocol override for this model — a named,
            module-level callable (usually the protocol class) that builds
            the protocol.  When omitted, the provider chooses its default
            protocol.
        protocol_args:
            JSON-serializable keyword arguments for ``protocol_factory``.

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

    if ":" not in model_id:
        model_id = f"gateway:{model_id}"

    ref = _modelsdev.parse_model_id(model_id)
    assert ref.provider_id is not None  # guaranteed to be fully-qualified here
    provider_id = ref.provider_id
    provider_model_id = ref.model_id

    model_info = _modelsdev.get_model_by_id(
        f"{provider_id}:{provider_model_id}"
    )
    model_provider_config = (
        None if model_info is None else model_info.provider_config
    )

    # Fail early on unknown or unsupported providers without building a
    # provider (and its client); the model only stores the recipe.
    base.resolve_provider_type(
        provider_id, model_provider_config=model_provider_config
    )

    return Model(
        provider_model_id,
        provider_factory=base.provider_for_model,
        provider_args={
            "provider_id": provider_id,
            "model_id": provider_model_id,
        },
        protocol_factory=protocol_factory,
        protocol_args=protocol_args,
    )
