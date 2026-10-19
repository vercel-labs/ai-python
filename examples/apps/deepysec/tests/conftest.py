"""The example is a plain package next to this directory; make it importable.

Offline tests check the invariants the spec says must survive a rewrite.
The one marked `live` drives the real pipeline against a real `claude`.

The deliberately vulnerable app the scan tests run on is not in this
repository. Point `DEEPYSEC_FIXTURE` at a copy (harness-sdk keeps one in
`examples/deepysec/fixtures/vulnerable-app`); without it those tests skip.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent
if str(EXAMPLE) not in sys.path:
    sys.path.insert(0, str(EXAMPLE))


@pytest.fixture
def fixture_app() -> Path:
    path = os.environ.get("DEEPYSEC_FIXTURE")
    if not path:
        pytest.skip("set DEEPYSEC_FIXTURE to the vulnerable-app fixture")
    return Path(path)


ENV_FILE = EXAMPLE.parents[2] / ".env.local"


@pytest.fixture(autouse=True)
def live_credentials(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live tests get the repository's `.env.local`, as the SDK suite does,
    over whatever this shell exported (measured: a stale token there made
    the pipeline test fail with a 401 from the gateway)."""
    if request.node.get_closest_marker("live") is None or not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            monkeypatch.setenv(key.strip(), value.strip().strip('"'))
