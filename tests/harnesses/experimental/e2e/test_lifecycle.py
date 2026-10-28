"""Failure and teardown — the pages that get written last and bite first."""

from __future__ import annotations

import os

import pytest

from ai.harnesses.experimental import claude_code
from ai.harnesses.experimental.errors import (
    ExecutableMissingError,
    HarnessClosedError,
)
from ai.workspaces.experimental import Workspace
from tests.harnesses.experimental.conftest import SPECS, requires_claude

pytestmark = pytest.mark.live


async def test_open_is_explicit_and_idempotent(
    harness_kind: str, workspace: Workspace
) -> None:
    harness = SPECS[harness_kind](workspace=workspace)

    await harness.open()
    await harness.open()  # second open changes nothing
    try:
        assert harness.is_open
        assert (
            harness.version
        ), "opening probes the CLI, so its version is known"
    finally:
        await harness.close()


async def test_a_closed_harness_refuses_work(
    harness_kind: str, workspace: Workspace
) -> None:
    harness = SPECS[harness_kind](workspace=workspace)
    await harness.open()
    await harness.close()

    assert not harness.is_open
    await harness.close()  # closing twice is a no-op, not an error
    with pytest.raises(HarnessClosedError):
        await harness.run("Reply with exactly: READY")


async def test_closing_kills_the_process_tree(
    harness_kind: str, workspace: Workspace
) -> None:
    """Nothing outlives the orchestrator.

    `process_ids` is best effort — a harness library that hides its child
    reports an empty list — so this asserts the guarantee where the pids are
    observable and says so plainly where they are not.
    """
    harness = SPECS[harness_kind](workspace=workspace)
    await harness.open()
    session = harness.session()
    await session.run("Reply with exactly: OK")
    pids = harness.process_ids

    await harness.close()

    if not pids:
        pytest.skip(f"{harness.name} does not expose its process ids")
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


async def test_workspace_close_kills_what_it_spawned(
    any_workspace: Workspace,
) -> None:
    process = await any_workspace.spawn(["sleep", "300"])

    await any_workspace.close()

    assert (
        await process.wait() is not None
    ), "the workspace owns what it spawned"


@requires_claude
async def test_a_missing_executable_names_the_install_command(
    workspace: Workspace,
) -> None:
    harness = claude_code(
        workspace=workspace, executable="claude-does-not-exist"
    )

    with pytest.raises(ExecutableMissingError) as exc:
        await harness.open()

    assert "claude-does-not-exist" in str(exc.value)
    assert "install" in str(exc.value).lower(), "tell the caller how to fix it"


async def test_open_failure_leaks_nothing(
    harness_kind: str, workspace: Workspace
) -> None:
    """A failed open must unwind everything it acquired — the caller's
    `async with` never ran, so nothing else will clean up."""
    harness = SPECS[harness_kind](
        workspace=workspace, executable="nope-not-real"
    )

    with pytest.raises(ExecutableMissingError):
        await harness.open()

    assert not harness.is_open
    assert harness.process_ids == []
