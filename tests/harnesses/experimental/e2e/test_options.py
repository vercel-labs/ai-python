"""The day-to-day options, on every harness at every location."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ai.harnesses.experimental.errors import UnsupportedError
from ai.workspaces.experimental import Workspace
from tests.harnesses.experimental import media
from tests.harnesses.experimental.conftest import exact

pytestmark = pytest.mark.live


async def test_instructions_shape_every_reply(
    any_make_harness: Callable[..., Any],
) -> None:
    async with any_make_harness(
        instructions="End every reply with the single word KIWI."
    ) as harness:
        result = await harness.run("Say hello in three words.")
    assert (
        result.text.strip().rstrip(".!").upper().endswith("KIWI")
    ), result.text


async def test_an_mcp_server_gives_the_agent_a_tool(
    any_make_harness: Callable[..., Any], any_workspace: Workspace
) -> None:
    await any_workspace.write_text("mcp_secret.py", media.MCP_SERVER)
    server = {
        "command": "python3",
        "args": [f"{any_workspace.path}/mcp_secret.py"],
    }
    async with any_make_harness(mcp_servers={"secret": server}) as harness:
        result = await harness.run(
            "Call the secret_word tool from the `secret` MCP server and reply "
            + "with exactly the word it returns."
        )
    exact(result.text, "MANGO")


async def test_tools_limits_what_claude_can_do(
    harness_kind: str,
    any_make_harness: Callable[..., Any],
    any_workspace: Workspace,
) -> None:
    read = {"claude": "Read", "pi": "read"}.get(harness_kind)
    if read is None:
        with pytest.raises(UnsupportedError, match="tools"):
            async with any_make_harness(tools=["Read"]):
                pass
        return
    async with any_make_harness(tools=[read]) as harness:
        await harness.run(
            "Write the word NOPE into nope.txt. If you cannot, reply CANNOT."
        )
    assert not await any_workspace.exists("nope.txt")


async def test_max_turns_caps_a_claude_turn(
    harness_kind: str, any_make_harness: Callable[..., Any]
) -> None:
    if harness_kind != "claude":
        with pytest.raises(UnsupportedError, match="max_turns"):
            async with any_make_harness(max_turns=1):
                pass
        return
    async with any_make_harness(max_turns=1) as harness:
        result = await harness.run(
            "Create a.txt, then b.txt, then c.txt, one tool call each, then "
            + "reply DONE."
        )
    assert result.finish_reason == "length"
