"""Result container for model operations."""

from typing import Any

import pydantic

from .. import types


class Item[T](pydantic.BaseModel):
    """Result of a model operation.

    ai.ops functions return an ``Item`` instead of a ``Message`` because their
    output isn't meant to enter the message history directly.
    """

    value: T
    usage: types.usage.Usage | None = None
    provider_metadata: dict[str, Any] | None = None

    model_config = pydantic.ConfigDict(frozen=True)
