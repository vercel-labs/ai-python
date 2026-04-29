"""Anthropic connection check.

Verifies that the client's credentials are valid and that the model
exists on the Anthropic API by hitting ``GET {base_url}/v1/models/{model_id}``.

The ``anthropic-version`` header is added by the check function itself
so that :class:`~ai.models.core.client.Client` stays provider-agnostic.

This endpoint is free — no tokens or credits are consumed.
"""

from __future__ import annotations

from ..core import client as client_
from ..core import model as model_

_ANTHROPIC_VERSION = "2023-06-01"

# HTTP status codes that indicate bad auth or a missing model.
_FAIL_STATUSES = frozenset({401, 403, 404})


async def check(client: client_.Client, model: model_.Model) -> bool:
    """Return ``True`` if *client* can reach Anthropic and *model* exists."""
    if not client.api_key:
        return False
    url = f"{client.base_url.rstrip('/')}/v1/models/{model.id}"
    headers = {
        "x-api-key": client.api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
    }
    response = await client.http.get(url, headers=headers)
    if response.status_code == 200:
        return True
    if response.status_code in _FAIL_STATUSES:
        return False
    # Unexpected status — let the caller handle it.
    response.raise_for_status()
    return False  # pragma: no cover
