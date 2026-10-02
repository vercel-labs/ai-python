"""The example is a plain package next to this directory; make it importable,
and keep its state file out of the developer's home."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent
if str(EXAMPLE) not in sys.path:
    sys.path.insert(0, str(EXAMPLE))


@pytest.fixture(autouse=True)
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AFK_STATE", str(tmp_path / "state.json"))
