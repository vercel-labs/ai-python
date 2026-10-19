"""The example is a plain package next to this directory; make it importable,
and keep its state file out of the developer's home.

Live tests get the repository's `.env.local` (from `vercel env pull`), as
the SDK's own suite does: measured, without it they ran on whatever this
shell had exported, and a stale token there read as afk failing (a 401 from
the gateway). Explicit local config overrides the shell, for those tests
only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent
if str(EXAMPLE) not in sys.path:
    sys.path.insert(0, str(EXAMPLE))

ENV_FILE = EXAMPLE.parents[2] / ".env.local"


def _env_local() -> dict[str, str]:
    values: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"')
    return values


@pytest.fixture(autouse=True)
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AFK_STATE", str(tmp_path / "state.json"))


@pytest.fixture(autouse=True)
def live_credentials(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.node.get_closest_marker("live") is None:
        return
    for key, value in _env_local().items():
        monkeypatch.setenv(key, value)
