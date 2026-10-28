"""`Local` processes, offline."""

from __future__ import annotations

import sys
from pathlib import Path

from ai.workspaces.experimental import Local


async def test_a_process_can_write_a_line_longer_than_asyncios_default(
    tmp_path: Path,
) -> None:
    """The harness protocols are line-delimited JSON, and one line can be
    megabytes (codex `thread/read` of a long thread, `thread/list` on a
    machine with thousands of threads). asyncio's default 64 KiB limit
    failed with "Separator is found, but chunk is longer than limit",
    reported as codex crashing (measured with 3159 local codex threads)."""
    size = 5 * 1024 * 1024
    async with Local(tmp_path) as ws:
        process = await ws.spawn([sys.executable, "-c", f"print('x' * {size})"])
        line = await process.readline()
        await process.wait()
    assert len(line.rstrip("\n")) == size


def test_copy_tree_keeps_permission_bits(tmp_path: Path) -> None:
    import stat

    from ai.workspaces.experimental import _base

    source = tmp_path / "src"
    source.mkdir()
    (source / "run.sh").write_text("#!/bin/sh\n")
    (source / "run.sh").chmod(0o755)
    _base.copy_tree(source, tmp_path / "out", (), gitignore=False)
    assert stat.S_IMODE((tmp_path / "out" / "run.sh").stat().st_mode) == 0o755
