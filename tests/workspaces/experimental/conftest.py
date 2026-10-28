"""Shared workspace fixtures.

Every test here drives a REAL workspace — no fakes, no recorded doubles.

WHERE it runs (local directory, Vercel Sandbox) must not matter: behavior
must be identical, full stop. The harness fixtures build on these (see
tests/harnesses/experimental/conftest.py).
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TypedDict

import pytest

from ai.workspaces.experimental import Local, VercelSandbox, Workspace, copy
from ai.workspaces.experimental._sandbox import oidc_token_expiry

ENV_FILE = Path(__file__).resolve().parents[3] / ".env.local"
"""`vercel env pull` output at the repo root, so the sandbox axis can run
locally. Deliberately not a dependency: one file, three lines, no dotenv."""

ENV_LOCAL: dict[str, str] = {}
"""What `ENV_FILE` sets. Applied to `os.environ` only for the harness and
workspace tests (see `env_local`), never for the rest of the suite; read it
directly for anything decided at collection time (a skipif marker)."""

if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            ENV_LOCAL[key.strip()] = value.strip().strip('"')


@pytest.fixture(scope="package", autouse=True)
def env_local() -> Iterator[None]:
    """Apply `.env.local` for this package's tests, and undo it after."""
    before = set(os.environ)
    with pytest.MonkeyPatch.context() as mp:
        for key, value in ENV_LOCAL.items():
            # Explicit local config OVERRIDES ambient env: the shell running
            # pytest may already carry another agent's ANTHROPIC_* values.
            mp.setenv(key, value)
        yield
        # `VercelSandbox(env_file=...)` exports the file's credentials into
        # os.environ by design; keep them from outliving these tests too.
        for key in set(os.environ) - before - ENV_LOCAL.keys():
            del os.environ[key]


class SandboxCredentials(TypedDict, total=False):
    """The `VercelSandbox` arguments that authenticate it."""

    token: str
    team_id: str
    project_id: str


def sandbox_credentials() -> SandboxCredentials | None:
    """Vercel credentials, or None when the sandbox axis can't run."""
    exp = oidc_token_expiry(os.environ.get("VERCEL_OIDC_TOKEN"))
    if (
        exp is not None
        and exp < time.time()
        and not os.environ.get("VERCEL_TOKEN")
    ):
        # Fail the RUN, once, with the cause. A run that outlives its token
        # otherwise produces one setup error per sandbox test — 84 of them,
        # 20 minutes in — and the word "expired" appears nowhere.
        when = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(exp))
        pytest.exit(
            f"VERCEL_OIDC_TOKEN expired at {when}; run `vercel env pull`",
            returncode=3,
        )
    token = os.environ.get("VERCEL_TOKEN")
    team = os.environ.get("VERCEL_TEAM_ID")
    project = os.environ.get("VERCEL_PROJECT_ID")
    if token and team and project:
        return {"token": token, "team_id": team, "project_id": project}
    if os.environ.get("VERCEL_OIDC_TOKEN"):
        return {}  # OIDC is picked up ambiently by the SDK
    return None


@pytest.fixture
def project(tmp_path: Path) -> Iterator[Path]:
    """A throwaway project directory with something to look at."""
    (tmp_path / "README.md").write_text(
        "# demo\n\nA tiny project used by the e2e suite.\n"
    )
    (tmp_path / "util.py").write_text("def add(a, b):\n    return a + b\n")
    yield tmp_path


@pytest.fixture
async def workspace(project: Path) -> AsyncIterator[Workspace]:
    async with Local(project) as ws:
        yield ws


@pytest.fixture
async def remote_workspace() -> AsyncIterator[Workspace]:
    """The same contract, in a Vercel Sandbox microVM."""
    creds = sandbox_credentials()
    if creds is None:
        pytest.skip(
            "needs VERCEL_TOKEN/TEAM_ID/PROJECT_ID or VERCEL_OIDC_TOKEN"
        )
    async with VercelSandbox(**creds, execution_time_limit=900) as ws:
        yield ws


@pytest.fixture(
    params=["local", pytest.param("sandbox", marks=pytest.mark.sandbox)]
)
async def any_workspace(
    request: pytest.FixtureRequest, project: Path
) -> AsyncIterator[Workspace]:
    """Both workspace implementations, same contract.

    This fixture is the whole argument for having two backends: every
    test using it must pass identically whether the files and processes live
    on this machine or in a microVM.
    """
    if request.param == "local":
        async with Local(project) as ws:
            yield ws
        return
    creds = sandbox_credentials()
    if creds is None:
        pytest.skip(
            "needs VERCEL_TOKEN/TEAM_ID/PROJECT_ID or VERCEL_OIDC_TOKEN"
        )
    async with VercelSandbox(**creds, execution_time_limit=900) as ws:
        # The SAME project the local branch gets. Without this, every test
        # ported onto any_workspace passed locally and failed in the VM with
        # "util.py does not exist" — a fixture lying about parity.
        await copy(project, ws / ".")
        yield ws
