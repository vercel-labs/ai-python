"""Shared durability context var.

Lives at the package root so that both ``models`` (lower-level) and
``agents3`` (higher-level) can import it without circular dependencies.
The actual ``DurabilityProvider`` protocol and implementations live in
``agents3.durability``; this module only holds the context var and a
thin accessor.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agents3 import durability

# The context var stores Any at runtime to avoid importing the protocol
# at module level.  ``agents3.durability`` provides the typed accessor.
_provider: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "durability_provider", default=None
)


def get_provider() -> durability.DurabilityProvider | None:
    """Return the active durability provider, or ``None``."""
    return _provider.get(None)  # type: ignore[no-any-return]


def set_provider(provider: Any) -> contextvars.Token[Any]:
    """Set the active durability provider. Returns a reset token."""
    return _provider.set(provider)


def reset_provider(token: contextvars.Token[Any]) -> None:
    """Reset the durability provider to its previous value."""
    _provider.reset(token)
