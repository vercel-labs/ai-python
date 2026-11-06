"""Optional provider SDK imports."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .. import errors as ai_errors

if TYPE_CHECKING:
    from types import ModuleType


def import_optional_sdk(
    module_name: str,
    *,
    provider: str | None = None,
    extra: str,
    feature: str | None = None,
) -> ModuleType:
    """Import an optional upstream SDK or raise a helpful installation error.

    ``feature`` names what needs the SDK when it is not a provider.
    """
    root_module = module_name.partition(".")[0]
    feature = feature or f"the {provider} provider"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name not in {module_name, root_module}:
            raise
        raise ai_errors.InstallationError(
            f"could not import `{root_module}`, which is required to use "
            f"{feature}, you can install it with `pip install "
            f'"ai[{extra}]"` or `uv add "ai[{extra}]"`'
        ) from exc
