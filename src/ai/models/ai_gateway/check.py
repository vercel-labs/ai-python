"""AI Gateway connection check.

Verifies that the client's credentials are valid and that the model
exists in the gateway's catalogue.

* Auth is validated via ``GET {origin}/v1/credits`` which requires a
  valid API key (returns 401/403 otherwise).
* Model existence is validated via ``GET {baseURL}/config`` which is a
  public endpoint returning ``{"models": [{"id": "...", ...}, ...]}``.

Both endpoints are free — no tokens or credits are consumed.
"""

from typing import Any

from .. import core
from . import sdk

# HTTP status codes that indicate bad auth.
_FAIL_STATUSES = frozenset({401, 403})


async def check(client: core.client.Client, model: core.model.Model) -> bool:
    """Return ``True`` if *client* can reach the gateway and *model* exists."""
    gateway = sdk.GatewayClient(client, model)

    # 1. Verify credentials via /v1/credits (requires valid auth).
    auth_resp = await gateway.get("v1/credits", origin=True)
    if auth_resp.status_code in _FAIL_STATUSES:
        return False
    if auth_resp.status_code != 200:
        auth_resp.raise_for_status()

    # 2. Verify model existence via /config (public catalogue).
    config_resp = await gateway.get("config")
    if config_resp.status_code != 200:
        config_resp.raise_for_status()
        return False  # pragma: no cover

    data: dict[str, Any] = config_resp.json()
    remote_ids: set[str] = {m["id"] for m in data.get("models", [])}
    return model.id in remote_ids
