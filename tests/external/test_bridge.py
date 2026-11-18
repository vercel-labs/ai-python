import urllib.parse
from typing import Any

import httpx2

from ai.external import _bridge


async def test_bridge_requires_auth_and_rejects_browser_origins() -> None:
    async def on_call(name: str, arguments: dict[str, Any]) -> tuple[Any, bool]:
        raise AssertionError("ping must not invoke tools")

    bridge = _bridge.Bridge(on_call)
    other = _bridge.Bridge(on_call)
    await bridge.start()
    try:
        authorization = {"Authorization": f"Bearer {bridge.token}"}
        payload = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
        authority = urllib.parse.urlsplit(bridge.url).netloc
        async with httpx2.AsyncClient(trust_env=False) as client:
            for headers, status in (
                ({}, 401),
                ({"Authorization": f"Bearer {other.token}"}, 401),
                ({**authorization, "Host": "example.test"}, 403),
                ({**authorization, "Origin": f"http://{authority}"}, 403),
                ({**authorization, "Origin": "null"}, 403),
            ):
                response = await client.post(
                    bridge.url, json=payload, headers=headers
                )
                assert response.status_code == status
                assert not response.content
            response = await client.post(
                bridge.url, json=payload, headers=authorization
            )
            assert response.json() == {"jsonrpc": "2.0", "id": 1, "result": {}}
    finally:
        await bridge.close()
