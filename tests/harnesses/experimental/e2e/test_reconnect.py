"""Detach and reconnect: the "close the laptop, come back" contract.

All sandbox-only, because it is the whole point — a sandbox outlives the
client, a Local child does not. Each test leaves a VM running with keep=True,
drops the client, and reconnects by name from a fresh workspace.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai.harnesses.experimental import ApprovalContext, Handle, harness_from
from ai.harnesses.experimental.errors import SessionBusyError
from ai.workspaces.experimental import VercelSandbox, Workspace
from ai.workspaces.experimental._gateway import vercel_ai_gateway
from tests.harnesses.experimental.conftest import SPECS, exact
from tests.workspaces.experimental.conftest import (
    SandboxCredentials,
    sandbox_credentials,
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.sandbox,
    pytest.mark.slow,
    pytest.mark.timeout(900),
]


def _creds_or_skip() -> SandboxCredentials:
    creds = sandbox_credentials()
    if creds is None:
        pytest.skip("needs Vercel Sandbox credentials")
    return creds


async def test_keep_leaves_the_vm_running_and_reconnect_finds_the_work(
    harness_kind: str,
) -> None:
    creds, gw = _creds_or_skip(), vercel_ai_gateway()
    async with VercelSandbox(
        **creds, keep=True, execution_time_limit=900
    ) as ws:
        name = ws.name
        await ws.write_text("marker.txt", "left behind\n")
        async with SPECS[harness_kind](workspace=ws, gateway=gw) as agent:
            s = agent.session()
            await s.run("Remember the codeword NARWHAL. Reply with exactly: OK")
            sid = s.session_id
    assert name

    # A fresh client: the VM is still up, its files and conversation intact.
    async with VercelSandbox(name=name) as back:
        assert (await back.read_text("marker.txt")).strip() == "left behind"
        async with SPECS[harness_kind](workspace=back, gateway=gw) as agent:
            assert sid in {i.session_id for i in await agent.sessions()}
            result = await (await agent.resume(sid)).run(
                "What codeword? Reply with the word only."
            )
        exact(result.text, "NARWHAL")
        async with VercelSandbox(name=name, keep=False):
            pass  # stop it


async def test_resume_handle_reattaches_a_live_sandbox(
    harness_kind: str,
) -> None:
    creds, gw = _creds_or_skip(), vercel_ai_gateway()
    async with VercelSandbox(
        **creds, keep=True, execution_time_limit=900
    ) as ws:
        name = ws.name
        async with SPECS[harness_kind](workspace=ws, gateway=gw) as agent:
            s = agent.session()
            await s.run(
                "Remember the codeword PLATYPUS. Reply with exactly: OK"
            )
            saved = s.handle.model_dump_json()
    async with harness_from(
        Handle.model_validate_json(saved), gateway=gw
    ) as session:
        result = await session.run("What codeword? Reply with the word only.")
    exact(result.text, "PLATYPUS")
    async with VercelSandbox(name=name, keep=False):
        pass


async def test_reconnect_to_a_stopped_sandbox_is_refused_with_a_reason() -> (
    None
):
    creds = _creds_or_skip()
    async with VercelSandbox(
        **creds, execution_time_limit=300
    ) as ws:  # keep defaults False → stopped on close
        name = ws.name
    with pytest.raises(Exception) as caught:
        async with VercelSandbox(name=name):
            pass
    assert (
        "stopped" in str(caught.value).lower()
        or "not running" in str(caught.value).lower()
    )


async def test_two_harnesses_in_one_vm_each_list_their_own_sessions(
    harness_kind: str,
) -> None:
    """Two agents in one VM.

    There is no workspace-level registry: a fresh client instantiates EACH
    harness inside the workspace and asks it for its sessions. The app remembers
    which id is which; the harness lists; the app matches — that is the whole
    discovery model.
    """
    creds, gw = _creds_or_skip(), vercel_ai_gateway()
    other = "codex" if harness_kind == "claude" else "claude"
    remembered: dict[str, str] = {}  # the CONSUMER's state: kind -> session id
    async with VercelSandbox(
        **creds, keep=True, execution_time_limit=900
    ) as ws:
        name = ws.name
        for kind, word in ((harness_kind, "ALPHA"), (other, "BETA")):
            agent = SPECS[kind](workspace=ws, gateway=gw)
            await agent.open()
            session = agent.session()
            await session.run(
                f"Remember the codeword {word}. Reply with exactly: OK"
            )
            remembered[kind] = session.session_id
            await agent.detach()

    async with VercelSandbox(name=name) as back:
        for kind, sid in remembered.items():
            agent = SPECS[kind](workspace=back, gateway=gw)
            await agent.open()
            listed = await agent.sessions()
            assert sid in {
                i.session_id for i in listed
            }, f"{kind} must list its own session"
            assert all(
                i.kind == agent.kind for i in listed
            ), "a harness lists only ITS sessions"
            # The detached harness is still ALIVE and holds this conversation.
            # One writer per conversation, on BOTH harnesses: resume is
            # refused with the same error, and fork is how a fresh client
            # continues — same history, a conversation of its own.
            with pytest.raises(SessionBusyError):
                await agent.resume(sid)
            picked = await agent.fork(sid)
            result = await picked.run(
                "What is the codeword? Reply with the single word only."
            )
            exact(result.text, "ALPHA" if kind == harness_kind else "BETA")
            await agent.detach()
        async with VercelSandbox(name=name, keep=False):
            pass


async def test_a_hooked_harness_survives_and_blocks(harness_kind: str) -> None:
    """Survival and the gate are INDEPENDENT.

    A harness whose hook has parked the turn is alive in the kept VM and has NOT
    acted — observed from a second client reconnecting by name. Liveness is
    checked with the workspace's own primitives, which is all a consumer would
    have too.
    """
    import contextlib

    creds, gw = _creds_or_skip(), vercel_ai_gateway()
    reached = asyncio.Event()
    binary = {"claude": "claude", "codex": "codex"}[harness_kind]

    async def approve(ctx: ApprovalContext) -> None:
        reached.set()
        await asyncio.sleep(3600)  # park the turn at the gate
        return None

    async with VercelSandbox(
        **creds, keep=True, execution_time_limit=900
    ) as ws:
        name = ws.name
        # No sandbox= for codex: in a VM only the default can run commands,
        # and with a hook codex still asks before writing (policy untrusted).
        options: dict[str, Any] = {"approve": approve, "gateway": gw}
        agent = SPECS[harness_kind](workspace=ws, **options)
        await agent.open()
        turn = agent.session().stream(
            "Write the word HELLO into hello.txt, then reply DONE."
        )

        async def drive() -> None:
            async for _ in turn:
                pass

        task = asyncio.create_task(drive())
        try:
            await asyncio.wait_for(reached.wait(), 90)  # the gate was hit
            async with VercelSandbox(name=name) as back:
                alive = await back.exec(["pgrep", "-f", binary])
                assert (
                    alive.exit_code == 0
                ), "the hooked harness must still be running"
                assert not await back.exists(
                    "hello.txt"
                ), "a pending approval must block the write"
        finally:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
            async with VercelSandbox(name=name, keep=False):
                pass


async def test_close_ends_the_harness_and_detach_keeps_it(
    harness_kind: str,
) -> None:
    """The distinction users rely on, made deterministic.

    close() ends the process — graceful EOF first, a bounded kill if it must —
    and leaves no FIFO or pid file behind. detach() leaves it running.
    """
    creds, gw = _creds_or_skip(), vercel_ai_gateway()
    binary = {"claude": "claude", "codex": "codex"}[harness_kind]

    async def alive(ws: Workspace) -> bool:
        return (await ws.exec(["pgrep", "-f", binary])).exit_code == 0

    async with VercelSandbox(
        **creds, keep=True, execution_time_limit=900
    ) as ws:
        name = ws.name
        agent = SPECS[harness_kind](workspace=ws, gateway=gw)
        await agent.open()
        await agent.session().run("Reply with exactly: OK")
        await agent.close()
        assert not await alive(
            ws
        ), "close() must end the harness before it returns"
        leftover = (
            await ws.exec(["sh", "-c", "ls /tmp/harness-* 2>/dev/null | wc -l"])
        ).stdout.strip()
        assert (
            leftover == "0"
        ), f"close() left {leftover} FIFO/pid file(s) behind"

        agent = SPECS[harness_kind](workspace=ws, gateway=gw)
        await agent.open()
        await agent.session().run("Reply with exactly: OK")
        await agent.detach()
        await asyncio.sleep(3)
        assert await alive(ws), "detach() must leave the harness running"
        async with VercelSandbox(name=name, keep=False):
            pass
