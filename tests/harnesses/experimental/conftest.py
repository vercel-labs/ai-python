"""Shared harness fixtures.

Every test here drives a REAL harness installation — no fakes, no recorded
doubles. A missing CLI skips; it never silently passes.

Two axes, and the point of the suite is that they are orthogonal:

- WHICH harness (claude-code, codex) — behavior must be identical except
  where a capability is honestly absent, and absence must be advertised.
- WHERE it runs (local directory, Vercel Sandbox) — behavior must be
  identical, full stop.

The workspace fixtures come from tests/workspaces/experimental/conftest.py;
`any_workspace` and `remote_workspace` are overridden here, because a
workspace a harness runs in needs the harness's credentials and config.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from ai.harnesses.experimental import Harness, claude_code, codex
from ai.workspaces.experimental import Local, VercelSandbox, Workspace, copy
from tests.workspaces.experimental import conftest as workspaces

# The workspace suite's fixtures, shared. Assigned, not imported by name:
# ruff reads a fixture parameter of the same name as redefining an import.
env_local = workspaces.env_local
project = workspaces.project
workspace = workspaces.workspace
sandbox_credentials = workspaces.sandbox_credentials

HARNESS_CLI = {"claude": "claude", "codex": "codex"}

SPECS = {"claude": claude_code, "codex": codex}


def _cli_missing(kind: str) -> bool:
    return shutil.which(HARNESS_CLI[kind]) is None


@pytest.fixture(params=["claude", "codex"])
def harness_kind(request: pytest.FixtureRequest) -> str:
    kind: str = request.param
    if _cli_missing(kind):
        pytest.skip(f"needs a local `{HARNESS_CLI[kind]}` installation")
    return kind


# What a harness needs to authenticate once it is inside the microVM. It
# cannot log in interactively there, so credentials arrive as secrets.
HARNESS_SECRET_KEYS = (
    "AI_GATEWAY_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT",
)

CODEX_CONFIG = """model_provider = "vercel"
[model_providers.vercel]
name = "Vercel AI Gateway"
base_url = "https://ai-gateway.vercel.sh/codex/v1"
env_key = "AI_GATEWAY_API_KEY"
wire_api = "responses"
"""


def harness_secrets() -> dict[str, str]:
    return {k: os.environ[k] for k in HARNESS_SECRET_KEYS if k in os.environ}


async def provision(ws: Workspace) -> None:
    """Give the harnesses what they need to authenticate in the VM.

    Installing the CLI is the SDK's job (see Workspace.owner); telling it
    which provider to talk to, and with whose key, is the caller's.
    """
    home = (await ws.exec(["sh", "-c", "echo $HOME"])).stdout.strip() or "/root"
    await ws.exec(["mkdir", "-p", f"{home}/.codex"])
    await ws.write_text(f"{home}/.codex/config.toml", CODEX_CONFIG)


@pytest.fixture
async def remote_workspace() -> AsyncIterator[Workspace]:
    """The same contract, in a Vercel Sandbox microVM."""
    creds = sandbox_credentials()
    if creds is None:
        pytest.skip(
            "needs VERCEL_TOKEN/TEAM_ID/PROJECT_ID or VERCEL_OIDC_TOKEN"
        )
    if not harness_secrets():
        pytest.skip("needs harness credentials to inject into the sandbox")
    # The harness inside the VM needs its credentials, and they are named
    # here rather than arriving because a file happened to be read.
    async with VercelSandbox(
        **creds, env=harness_secrets(), execution_time_limit=900
    ) as ws:
        await provision(ws)
        yield ws


@pytest.fixture
async def harness(
    harness_kind: str, workspace: Workspace
) -> AsyncIterator[Harness]:
    """An open harness on a local workspace, parameterized over both
    harnesses."""
    async with SPECS[harness_kind](workspace=workspace) as h:
        yield h


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
    async with VercelSandbox(
        **creds, env=harness_secrets(), execution_time_limit=900
    ) as ws:
        # Provision here too: any sandbox a harness might open needs to know
        # which provider to talk to, not just the one the remote suite uses.
        await provision(ws)
        # And the SAME project the local branch gets. Without this, every
        # test ported onto any_workspace passed locally and failed in the
        # VM with "util.py does not exist" — a fixture lying about parity.
        await copy(project, ws / ".")
        yield ws


@pytest.fixture
async def any_harness(
    harness_kind: str, any_workspace: Workspace
) -> AsyncIterator[Harness]:
    """Both harnesses x both locations — the SDK's whole claim in one fixture.

    A test on `harness` proves a verb works on this machine. The last three
    bugs that reached George all passed such tests: the failure was at the
    other location, or in a shape the assertion never looked at. Anything
    that is part of the CONTROL SURFACE belongs on this fixture, and pays
    for the sandbox boot.
    """
    async with SPECS[harness_kind](workspace=any_workspace) as h:
        yield h


def exact(text: str, expected: str) -> None:
    """Assert an answer IS the expected word, not merely contains it.

    `"add" in result.text` let 'I\'ll read util.py.add' pass for weeks. When
    the prompt says "reply with the function name only", the contract is
    equality — tolerant of the punctuation and backticks a model wraps a
    single word in, and of nothing else.
    """
    got = text.strip().strip("`'\"*_.").strip()
    assert got == expected, f"expected exactly {expected!r}, got {text!r}"


@pytest.fixture
def make_harness(harness_kind: str, workspace: Workspace) -> Callable[..., Any]:
    """Open a harness of the parameterized kind with extra options.

    Tests that configure a harness (an approval hook, a timeout) use this so
    they still run against BOTH harnesses — uniformity is the contract, so
    nothing may quietly become a single-harness test.
    """

    @asynccontextmanager
    async def _make(
        *, writable: bool = False, **options: Any
    ) -> AsyncIterator[Harness]:
        """`writable=True` means "this test needs the agent to change files".

        The harnesses spell authority differently — Claude gates per call,
        Codex picks a sandbox mode — so the translation lives here rather
        than in every test. A test that hardcoded `sandbox=` would silently
        become a Codex-only test.
        """
        # Codex picks a mode only on your own machine: in a sandbox its own
        # sandbox cannot start, and the default already lets it write.
        if (
            writable
            and harness_kind == "codex"
            and workspace.owner != "provider"
        ):
            options.setdefault("sandbox", "workspace-write")
        async with SPECS[harness_kind](
            workspace=workspace, **options
        ) as harness:
            yield harness

    return _make


@pytest.fixture
def any_make_harness(
    harness_kind: str, any_workspace: Workspace
) -> Callable[..., Any]:
    """`make_harness`, over both locations. Same translation of `writable`."""

    @asynccontextmanager
    async def _make(
        *, writable: bool = False, **options: Any
    ) -> AsyncIterator[Harness]:
        if (
            writable
            and harness_kind == "codex"
            and any_workspace.owner != "provider"
        ):
            options.setdefault("sandbox", "workspace-write")
        async with SPECS[harness_kind](
            workspace=any_workspace, **options
        ) as harness:
            yield harness

    return _make


requires_claude = pytest.mark.skipif(
    _cli_missing("claude"), reason="needs a local `claude` installation"
)


def requires(harness: Harness, capability: str) -> None:
    """Skip when a harness honestly lacks a capability.

    Skipping is only legitimate because the SDK ADVERTISES the absence —
    a test may never paper over a capability the harness claims to have.
    """
    if not getattr(harness.capabilities, capability):
        pytest.skip(f"{harness.name} does not support {capability}")
