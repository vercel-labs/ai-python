"""One writer per conversation — on both harnesses, at both locations.

Codex enforces this natively; the SDK's workspace lock gives claude the same
behaviour. Every assertion here must hold identically for both, which is
the whole point of enforcing it ourselves.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ai.harnesses.experimental.errors import SessionBusyError
from ai.workspaces.experimental import Workspace
from tests.harnesses.experimental.conftest import SPECS, exact

pytestmark = pytest.mark.live


async def test_a_held_session_refuses_a_second_writer_and_frees_on_close(
    harness_kind: str,
    any_workspace: Workspace,
    any_make_harness: Callable[..., Any],
) -> None:
    async with any_make_harness() as first:
        session = first.session()
        await session.run("Remember the codeword GAMMA. Reply with exactly: OK")
        sid = session.session_id

        second = SPECS[harness_kind](workspace=any_workspace)
        await second.open()
        try:
            with pytest.raises(SessionBusyError):
                await second.resume(sid)
            listed = {i.session_id: i for i in await second.sessions()}
            assert listed[sid].running, "a listing must say who is held open"
            # Forking never writes the original: always allowed.
            forked = await second.fork(sid)
            result = await forked.run(
                "What is the codeword? Reply with the single word only."
            )
            exact(result.text, "GAMMA")
        finally:
            await second.close()

    # The first client closed: its claim is gone, and resume is fine.
    third = SPECS[harness_kind](workspace=any_workspace)
    await third.open()
    try:
        before = {i.session_id: i for i in await third.sessions()}
        assert not before[sid].running, "close() must release the claim"
        picked = await third.resume(sid)
        result = await picked.run(
            "What is the codeword? Reply with the single word only."
        )
        exact(result.text, "GAMMA")
    finally:
        await third.close()


async def test_detach_keeps_the_claim(
    harness_kind: str, any_workspace: Workspace
) -> None:
    """A detached client let go of its connection, not of the conversation:
    the process is still the writer, so a newcomer is still refused."""
    if any_workspace.kind == "local":
        pytest.skip(
            "a local process dies with its client; detach is close there"
        )
    first = SPECS[harness_kind](workspace=any_workspace)
    await first.open()
    session = first.session()
    await session.run("Reply with exactly: OK")
    sid = session.session_id
    await first.detach()

    second = SPECS[harness_kind](workspace=any_workspace)
    await second.open()
    try:
        with pytest.raises(SessionBusyError):
            await second.resume(sid)
    finally:
        await second.close()
