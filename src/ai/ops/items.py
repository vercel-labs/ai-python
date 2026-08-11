"""Result container for model operations."""

from typing import Any

import pydantic

from .. import types


class Warning(pydantic.BaseModel):
    """Warning reported while running a model operation.

    Providers emit these when a requested feature is unsupported or was
    applied only partially (e.g. an ignored ``aspect_ratio``); the SDK adds
    its own when it drops part of a prompt during normalization.
    """

    kind: str = "other"
    """Warning category: ``"unsupported"``, ``"compatibility"``,
    ``"deprecated"``, or ``"other"``."""
    message: str | None = None
    feature: str | None = None
    """The affected feature, for unsupported/compatibility warnings."""
    setting: str | None = None
    """The deprecated setting name, for deprecated warnings."""
    details: str | None = None

    model_config = pydantic.ConfigDict(frozen=True)


class Item[T](pydantic.BaseModel):
    """Result of a model operation.

    ai.ops functions return an ``Item`` instead of a ``Message`` because their
    output isn't meant to enter the message history directly.
    """

    value: T
    usage: types.usage.Usage | None = None
    warnings: list[Warning] = []
    provider_metadata: dict[str, Any] | None = None

    model_config = pydantic.ConfigDict(frozen=True)
