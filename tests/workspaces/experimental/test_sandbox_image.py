"""`VercelSandbox(image=...)`: which root filesystem a new sandbox boots from.

The live tests in tests/harnesses/experimental/e2e/test_sandbox_image.py boot
real VMs from non-default images. These pin the contract offline: what is
refused before anything boots, and what reaches the platform's create call.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai.workspaces.experimental import VercelSandbox


def test_image_and_name_are_refused_together() -> None:
    """A reconnect keeps the image its VM booted from; asking for another
    one there could only be ignored."""
    with pytest.raises(ValueError, match="name= reconnects"):
        VercelSandbox(name="sbx_x", image="vercel/sandbox/ubuntu")


def test_image_and_snapshot_are_refused_together() -> None:
    with pytest.raises(ValueError, match="pass one"):
        VercelSandbox(snapshot="snap_x", image="vercel/sandbox/ubuntu")


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_image_is_refused(blank: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        VercelSandbox(image=blank)


def test_before_open_the_image_is_the_one_asked_for() -> None:
    assert VercelSandbox(image="vercel/sandbox/node:24").image == (
        "vercel/sandbox/node:24"
    )
    assert VercelSandbox().image is None  # the platform default


class _Api:
    """Records what `_open` asks the platform for; boots nothing."""

    def __init__(self, reported_image: str | None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._reported = reported_image

    async def create_sandbox(self, **options: Any) -> Any:
        self.calls.append(options)

        async def mkdir(_path: str) -> None:
            return None

        return SimpleNamespace(
            name="sbx_test",
            image=self._reported,
            fs=SimpleNamespace(mkdir=mkdir),
        )


async def _open_with(
    ws: VercelSandbox, api: _Api, mp: pytest.MonkeyPatch
) -> None:
    mp.setattr(ws, "_api", lambda: api)
    await ws._open()


async def test_the_image_reaches_the_create_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _Api(reported_image="vercel/sandbox/python:3.14")
    await _open_with(
        VercelSandbox(image="vercel/sandbox/python:3.14"), api, monkeypatch
    )
    assert api.calls[0]["image"] == "vercel/sandbox/python:3.14"


async def test_without_an_image_the_platform_default_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No `image` key at all, rather than `image=None`: the default is the
    platform's to choose, and stays its choice when it changes."""
    api = _Api(reported_image=None)
    await _open_with(VercelSandbox(), api, monkeypatch)
    assert "image" not in api.calls[0]


async def test_once_open_the_image_is_the_one_the_platform_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The platform may resolve a tag to something more specific; that is
    what booted, so that is what `image` says."""
    resolved = "vercel/sandbox/python@sha256:" + "0" * 64
    ws = VercelSandbox(image="vercel/sandbox/python:3.14")
    await _open_with(ws, _Api(reported_image=resolved), monkeypatch)
    assert ws.image == resolved


async def test_an_image_travels_with_a_gateway_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image and egress are both creation-time choices; one must not drop
    the other."""
    from ai.workspaces.experimental._gateway import Gateway

    api = _Api(reported_image=None)
    ws = VercelSandbox(
        image="vercel/sandbox/node:24",
        gateway=Gateway(base_url="https://gateway.example", credential="k"),
    )

    async def record(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(ws, "_write_egress_record", record)
    await _open_with(ws, api, monkeypatch)
    assert api.calls[0]["image"] == "vercel/sandbox/node:24"
    assert "network_policy" in api.calls[0]
