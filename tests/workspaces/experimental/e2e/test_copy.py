"""Copying trees between your machine and a workspace, in both directions.

One verb, no direction flag: whichever side is a `WorkspacePath` is the
workspace. Going in is batched (0.2s against 75.5s for 200 files); coming
out is one archive, because there is no batched read. What the tree's
`.gitignore` excludes travels neither way unless asked for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai.workspaces.experimental import Workspace, copy
from tests.harnesses.experimental.conftest import exact

pytestmark = pytest.mark.live


def tree(root: Path) -> Path:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "mod.py").write_text("def add(a, b):\n    return a + b\n")
    (root / "pkg" / "data.bin").write_bytes(bytes(range(256)))
    (root / "README.md").write_text("# demo\n")
    (root / ".gitignore").write_text("*.log\n.env\n")
    (root / "debug.log").write_text("noise\n")
    (root / ".env").write_text("SECRET=1\n")
    (root / ".git").mkdir(exist_ok=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / "node_modules").mkdir(exist_ok=True)
    (root / "node_modules" / "junk.js").write_text("// huge\n")
    return root


async def test_copy_copies_a_tree(
    any_workspace: Workspace, tmp_path: Path
) -> None:
    source = tree(tmp_path / "src")

    await copy(source, any_workspace / "project")

    assert await any_workspace.read_text("project/pkg/mod.py") == (
        "def add(a, b):\n    return a + b\n"
    )
    assert await any_workspace.read_text("project/README.md") == "# demo\n"


async def test_copy_preserves_bytes(
    any_workspace: Workspace, tmp_path: Path
) -> None:
    """Binary survives: a source tree is not only text."""
    source = tree(tmp_path / "src")

    await copy(source, any_workspace / "project")

    assert await any_workspace.read_bytes("project/pkg/data.bin") == bytes(
        range(256)
    )


async def test_copy_skips_the_usual_noise(
    any_workspace: Workspace, tmp_path: Path
) -> None:
    """Shipping .git and node_modules is the difference between a second
    and a minute, so they are excluded unless asked for."""
    source = tree(tmp_path / "src")

    await copy(source, any_workspace / "project")

    assert not await any_workspace.exists("project/.git/HEAD")
    assert not await any_workspace.exists("project/node_modules/junk.js")


async def test_copy_can_be_told_what_to_ignore(
    any_workspace: Workspace, tmp_path: Path
) -> None:
    source = tree(tmp_path / "src")

    await copy(source, any_workspace / "project", ignore=["*.md"])

    assert await any_workspace.exists("project/pkg/mod.py")
    assert not await any_workspace.exists("project/README.md")


async def test_copy_honors_gitignore(
    any_workspace: Workspace, tmp_path: Path
) -> None:
    """What a project keeps out of version control — secrets, build output —
    is what nobody means to hand an agent."""
    source = tree(tmp_path / "src")

    count = await copy(source, any_workspace / "project")

    assert not await any_workspace.exists("project/.env")
    assert not await any_workspace.exists("project/debug.log")
    assert await any_workspace.exists("project/.gitignore")
    assert count == 4  # pkg/mod.py, pkg/data.bin, README.md, .gitignore


async def test_copy_can_be_told_to_ship_ignored_files(
    any_workspace: Workspace, tmp_path: Path
) -> None:
    source = tree(tmp_path / "src")

    count = await copy(source, any_workspace / "project", gitignore=False)

    assert await any_workspace.read_text("project/.env") == "SECRET=1\n"
    assert await any_workspace.exists("project/debug.log")
    assert count == 6


def landed_files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )


async def agent_output(ws: Workspace) -> None:
    """A tree as an agent leaves it: source, noise, and a file saying which
    is which."""
    await ws.write_text("project/.gitignore", "*.log\nout/\n")
    await ws.write_text("project/keep.py", "x = 1\n")
    await ws.write_text("project/debug.log", "noise\n")
    await ws.write_text("project/out/bundle.js", "//\n")
    await ws.write_text("project/node_modules/junk.js", "//\n")


async def test_copy_out_honors_gitignore_too(
    any_workspace: Workspace, tmp_path: Path
) -> None:
    """The same rules whichever way the tree travels: build output stays
    where the agent left it."""
    await agent_output(any_workspace)

    landed = tmp_path / "back"
    count = await copy(any_workspace / "project", landed)

    assert landed_files(landed) == [".gitignore", "keep.py"]
    assert count == 2


async def test_copy_out_can_be_told_to_ship_ignored_files(
    any_workspace: Workspace, tmp_path: Path
) -> None:
    await agent_output(any_workspace)

    landed = tmp_path / "back"
    count = await copy(any_workspace / "project", landed, gitignore=False)

    # node_modules is still gone: that is DEFAULT_IGNORE, not .gitignore.
    assert landed_files(landed) == [
        ".gitignore",
        "debug.log",
        "keep.py",
        "out/bundle.js",
    ]
    assert count == 4


async def test_copy_out_can_be_told_what_to_ignore(
    any_workspace: Workspace, tmp_path: Path
) -> None:
    """`ignore` describes the tree, not the direction."""
    await any_workspace.write_text("project/notes.md", "# notes\n")
    await any_workspace.write_text("project/code.py", "x = 1\n")

    landed = tmp_path / "back"
    count = await copy(any_workspace / "project", landed, ignore=["*.md"])

    assert landed_files(landed) == ["code.py"]
    assert count == 1


async def test_copying_a_missing_directory_raises(
    any_workspace: Workspace, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        await copy(tmp_path / "nope", any_workspace / "project")


async def test_the_agent_sees_what_you_copied(
    harness_kind: str, any_workspace: Workspace, tmp_path: Path
) -> None:
    from tests.harnesses.experimental.conftest import SPECS

    await copy(tree(tmp_path / "src"), any_workspace / ".")

    async with SPECS[harness_kind](workspace=any_workspace) as agent:
        result = await agent.run(
            "Read pkg/mod.py and reply with the function name only."
        )

    exact(result.text, "add")


async def test_copy_comes_back_out(
    any_workspace: Workspace, tmp_path: Path
) -> None:
    """The same verb, the other way: the workspace side is the one marked."""
    await copy(tree(tmp_path / "src"), any_workspace / "project")

    landed = tmp_path / "back"
    count = await copy(any_workspace / "project", landed)

    assert count >= 3
    assert (
        landed / "pkg" / "mod.py"
    ).read_text() == "def add(a, b):\n    return a + b\n"
    assert (landed / "pkg" / "data.bin").read_bytes() == bytes(range(256))


async def test_copy_out_brings_back_what_the_agent_wrote(
    harness_kind: str, any_workspace: Workspace, tmp_path: Path
) -> None:
    """The point of copying out: collect what the agent produced."""
    from ai.harnesses.experimental import Allow, ApprovalContext, Decision
    from tests.harnesses.experimental.conftest import SPECS

    async def approve(ctx: ApprovalContext) -> Decision:
        return Allow()

    async with SPECS[harness_kind](
        workspace=any_workspace, approve=approve
    ) as agent:
        await agent.run(
            "Create a directory called findings and write the word DONE into "
            + "findings/report.txt. Then reply DONE."
        )

    # Not tmp_path/"findings": a local workspace IS tmp_path, so that would
    # copy the directory onto itself.
    landed = tmp_path / "collected"
    await copy(any_workspace / "findings", landed)

    assert "DONE" in (landed / "report.txt").read_text()


async def test_two_plain_paths_is_refused(tmp_path: Path) -> None:
    """Local-to-local is shutil's job, and saying so beats doing it badly."""
    with pytest.raises(TypeError, match="workspace"):
        await copy(tmp_path / "a", tmp_path / "b")
