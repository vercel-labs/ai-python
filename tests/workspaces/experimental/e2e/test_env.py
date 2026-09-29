"""Environment variables belong to the workspace.

A task's configuration — DATABASE_URL, NODE_ENV, test credentials — is a
property of the place the work happens, not of the agent doing it. It
reaches the agent's own commands and your `workspace.exec` calls alike.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ai.harnesses.experimental import Allow
from ai.workspaces.experimental import Local
from tests.harnesses.experimental.conftest import SPECS

pytestmark = pytest.mark.live


async def allow_all(ctx: Any) -> Allow:
    return Allow()


async def test_workspace_env_reaches_the_agents_commands(
    harness_kind: str, project: Path
) -> None:
    """The agent inherits it, so the commands IT runs see it too — which is
    the point: the task needs the variable, not the agent."""
    async with Local(
        project, env={"HARNESS_SDK_PROBE": "from-workspace"}
    ) as ws:
        async with SPECS[harness_kind](
            workspace=ws, approve=allow_all
        ) as agent:
            result = await agent.run(
                "Run the shell command `echo $HARNESS_SDK_PROBE` and reply "
                "with " + "its output only."
            )

    assert "from-workspace" in result.text


async def test_workspace_env_reaches_your_own_commands(project: Path) -> None:
    async with Local(
        project, env={"HARNESS_SDK_PROBE": "from-workspace"}
    ) as ws:
        result = await ws.exec(["sh", "-c", "echo $HARNESS_SDK_PROBE"])

    assert result.stdout.strip() == "from-workspace"


async def test_a_per_call_env_still_overrides(project: Path) -> None:
    """The one-off escape hatch, for a command that needs something else."""
    async with Local(
        project, env={"HARNESS_SDK_PROBE": "from-workspace"}
    ) as ws:
        result = await ws.exec(
            ["sh", "-c", "echo $HARNESS_SDK_PROBE"],
            env={"HARNESS_SDK_PROBE": "just-this-once"},
        )

    assert result.stdout.strip() == "just-this-once"


async def test_two_workspaces_over_one_directory_can_differ(
    harness_kind: str, project: Path
) -> None:
    """Different environments means different workspaces, not different
    agents: the environment describes the work, so two agents that need
    different ones are doing different work."""
    async with (
        Local(project, env={"HARNESS_SDK_PROBE": "first"}) as one,
        Local(project, env={"HARNESS_SDK_PROBE": "second"}) as two,
    ):
        async with SPECS[harness_kind](
            workspace=one, approve=allow_all
        ) as agent:
            a = await agent.run(
                "Run `echo $HARNESS_SDK_PROBE`, reply with output only."
            )
        async with SPECS[harness_kind](
            workspace=two, approve=allow_all
        ) as agent:
            b = await agent.run(
                "Run `echo $HARNESS_SDK_PROBE`, reply with output only."
            )

    assert "first" in a.text
    assert "second" in b.text
