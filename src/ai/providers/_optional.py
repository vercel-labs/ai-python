"""Optional provider SDK imports."""

from __future__ import annotations

import importlib
from types import ModuleType

from .. import errors as ai_errors


def import_optional_sdk(module_name: str, *, provider: str, extra: str) -> ModuleType:
    """Import an optional upstream SDK or raise a helpful installation error."""
    root_module = module_name.partition(".")[0]
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name not in {module_name, root_module}:
            raise
        raise ai_errors.InstallationError(
            f"could not import `{root_module}`, which is required to use the "
            f"{provider} provider, you can install it with `pip install "
            f'"ai[{extra}]"` or `uv add "ai[{extra}]"`'
        ) from exc
