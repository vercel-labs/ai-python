"""The workspace contract, proven identical on both implementations.

Every test in this module runs twice: once against a local directory and
once inside a Vercel Sandbox microVM. If a behavior cannot be stated the
same way for both, the abstraction is wrong and this file is where that
shows up.
"""

from __future__ import annotations

import pytest

from ai.workspaces.experimental import Workspace
from tests.harnesses.experimental.conftest import SPECS, exact

pytestmark = pytest.mark.live


async def test_write_then_read_round_trip(any_workspace: Workspace) -> None:
    await any_workspace.write_text("notes/today.md", "hello\n")

    assert await any_workspace.exists("notes/today.md")
    assert await any_workspace.read_text("notes/today.md") == "hello\n"


async def test_missing_file_raises_rather_than_returning_empty(
    any_workspace: Workspace,
) -> None:
    assert not await any_workspace.exists("nope.txt")
    with pytest.raises(FileNotFoundError):
        await any_workspace.read_text("nope.txt")


async def test_exec_returns_output_and_exit_code(
    any_workspace: Workspace,
) -> None:
    await any_workspace.write_text("hello.txt", "from the workspace\n")

    result = await any_workspace.exec(["cat", "hello.txt"])

    assert result.exit_code == 0
    assert "from the workspace" in result.stdout

    failed = await any_workspace.exec(["cat", "does-not-exist"])
    assert failed.exit_code != 0
    assert failed.stderr


async def test_spawn_streams_output(any_workspace: Workspace) -> None:
    process = await any_workspace.spawn(["echo", "ping"])

    line = await process.readline()

    assert line.strip() == "ping"
    await process.terminate()
    assert await process.wait() is not None


async def test_duplex_spawn_works_or_refuses_clearly(
    any_workspace: Workspace,
) -> None:
    """Writing to a spawned process is what lets an adapter drive a harness CLI.

    Where the platform has no stdin channel at all — Vercel Sandbox — the
    workspace must SAY so and refuse, not hand back a process whose writes fail
    somewhere less obvious later.
    """
    process = await any_workspace.spawn(["cat"])
    try:
        if any_workspace.duplex_spawn:
            await process.write("ping\n")
            assert (await process.readline()).strip() == "ping"
        else:
            with pytest.raises(Exception, match="stdin"):
                await process.write("ping\n")
    finally:
        await process.terminate()


async def test_containment_is_decided_by_the_workspace(
    any_workspace: Workspace,
) -> None:
    """Never the caller's pathlib: for a remote workspace those paths do not
    exist on this machine at all."""
    assert any_workspace.contains(f"{any_workspace.path}/util.py")
    assert not any_workspace.contains("/etc/passwd")
    assert not any_workspace.contains(f"{any_workspace.path}/../escape.txt")
    assert not any_workspace.contains(
        "relative/path.txt"
    ), "a relative path cannot be judged from here; deny rather than guess"


async def test_reachable_url_addresses_a_port_from_the_agent_side(
    any_workspace: Workspace,
) -> None:
    url = await any_workspace.reachable_url(8931)

    assert url.startswith("http")
    assert "8931" in url or any_workspace.kind != "local"


@pytest.mark.sandbox
@pytest.mark.slow
async def test_a_harness_runs_the_same_in_a_sandbox(
    harness_kind: str, remote_workspace: Workspace
) -> None:
    """The parity test that matters: identical code, identical result,
    a different machine."""
    await remote_workspace.write_text(
        "util.py", "def add(a, b):\n    return a + b\n"
    )

    async with SPECS[harness_kind](workspace=remote_workspace) as harness:
        result = await harness.run(
            "Read util.py and reply with the function name only."
        )

    exact(result.text, "add")
    assert result.usage.input_tokens > 0
