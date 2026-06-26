"""Pluggable storage for checkpoints and session data.

Provides a minimal Storage protocol and a FileStorage implementation
that persists data as JSON files on disk. Swap in any backend that
satisfies the protocol (Redis, Postgres, etc.) without changing the
rest of the app.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    """Async key-value storage interface."""

    async def get(self, key: str) -> dict[str, Any] | None: ...
    async def put(self, key: str, value: dict[str, Any]) -> None: ...
    async def delete(self, key: str) -> None: ...


class FileStorage:
    """JSON-file-per-key storage backend.

    Each key is stored as ``{directory}/{key}.json``. Good enough for
    local development; replace with a real database for production.
    """

    def __init__(self, directory: str | pathlib.Path | None = None) -> None:
        if directory is None:
            directory = "/tmp/data" if os.environ.get("VERCEL") else "./data"
        self._dir = pathlib.Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> pathlib.Path:
        # Sanitise the key so it's safe as a filename
        safe = key.replace("/", "__").replace(":", "_")
        return self._dir / f"{safe}.json"

    async def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())  # type: ignore[no-any-return]

    async def put(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.write_text(json.dumps(value, indent=2))

    async def delete(self, key: str) -> None:
        path = self._path(key)
        path.unlink(missing_ok=True)
