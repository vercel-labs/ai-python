"""`workspace.resolve()`: the real path a path names, symlinks followed.

`contains()` is pure-path on purpose and cannot see a symlink; an approval
hook that must not be fooled by one resolves first.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from ai.workspaces.experimental import Local
from ai.workspaces.experimental._base import Workspace


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_text("x = 1\n")
    outside = tmp_path / "secret.txt"
    outside.write_text("no\n")
    os.symlink(outside, root / "innocent.txt")
    os.symlink(tmp_path, root / "up")
    return root


class ViaExec(Local):
    """`Local`, but resolving the way a remote workspace does: one command."""

    async def resolve(self, path: str) -> str:
        return await Workspace.resolve(self, path)


@pytest.mark.parametrize("kind", [Local, ViaExec])
async def test_resolve_follows_symlinks_out_of_the_workspace(
    tree: Path, kind: Any
) -> None:
    async with kind(tree) as ws:
        link = f"{ws.path}/innocent.txt"
        assert ws.contains(link), "pure-path: a symlink looks inside"
        real = await ws.resolve(link)
        assert not ws.contains(real), "resolved: it is not"
        assert not ws.contains(await ws.resolve(f"{ws.path}/up/secret.txt"))


@pytest.mark.parametrize("kind", [Local, ViaExec])
async def test_resolve_keeps_real_paths_and_relative_ones_are_the_workspaces(
    tree: Path, kind: Any
) -> None:
    async with kind(tree) as ws:
        assert await ws.resolve(f"{ws.path}/src/a.py") == f"{ws.path}/src/a.py"
        assert await ws.resolve("src/a.py") == f"{ws.path}/src/a.py"
        missing = await ws.resolve("src/new.py")  # a file about to be written
        assert missing == f"{ws.path}/src/new.py"
        assert ws.contains(missing)
