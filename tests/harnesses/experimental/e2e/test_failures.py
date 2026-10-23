"""Failure modes: the pages that get written last and bite first.

Every failure here must surface as THIS SDK's error, not the underlying
harness library's. A caller cannot be expected to catch claude-agent-sdk's
ResultError or a stray FileNotFoundError to find out that a session id was
wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai.harnesses.experimental import Harness, claude_code, codex
from ai.harnesses.experimental.errors import (
    AgentCrashedError,
    HarnessClosedError,
    ResumeFailedError,
    TurnFailedError,
    UnsupportedError,
)
from ai.workspaces.experimental import Local
from ai.workspaces.experimental.errors import WorkspaceError
from tests.harnesses.experimental.conftest import SPECS, exact, requires

pytestmark = pytest.mark.live

GHOST = "00000000-0000-0000-0000-000000000000"


async def test_adopt_unknown_session_raises_resume_failed(
    any_harness: Harness,
) -> None:
    requires(any_harness, "resume")
    with pytest.raises(ResumeFailedError) as exc:
        await any_harness.resume(GHOST)
    assert GHOST in str(exc.value)


async def test_fork_unknown_session_raises_resume_failed(
    any_harness: Harness,
) -> None:
    requires(any_harness, "fork")
    with pytest.raises(ResumeFailedError):
        await any_harness.fork(GHOST)


async def test_malformed_session_id_raises_resume_failed(
    any_harness: Harness,
) -> None:
    requires(any_harness, "resume")
    with pytest.raises(ResumeFailedError):
        await any_harness.resume("not-a-uuid")


async def test_a_session_from_another_workspace_is_not_found(
    harness_kind: str, project: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Sessions belong to the place they were created.

    Adopting one from a different workspace is the same failure as resuming a
    ghost — and must not half-succeed by connecting first and discovering it
    later.
    """
    elsewhere = tmp_path_factory.mktemp("elsewhere")

    async with (
        Local(project) as first_ws,
        SPECS[harness_kind](workspace=first_ws) as first,
    ):
        session = first.session()
        await session.run("Reply with exactly: OK")
        foreign_id = session.session_id

    async with (
        Local(elsewhere) as other_ws,
        SPECS[harness_kind](workspace=other_ws) as other,
    ):
        requires(other, "resume")
        with pytest.raises(ResumeFailedError):
            await other.resume(foreign_id)
        with pytest.raises(ResumeFailedError):
            await other.fork(foreign_id)
        with pytest.raises(ResumeFailedError):
            await other.history(foreign_id)


async def test_a_failed_adopt_leaves_no_session_behind(
    any_harness: Harness,
) -> None:
    """A failed take-over must not register a half-connected session."""
    requires(any_harness, "resume")
    before = len(await any_harness.sessions())

    with pytest.raises(ResumeFailedError):
        await any_harness.resume(GHOST)

    assert len(await any_harness.sessions()) == before


class _NoStdinWorkspace(Local):
    """A any_workspace that can run programs but cannot write to them.

    Not a stand-in for a any_harness — it is a any_workspace whose transport is
    deliberately crippled, which is the only way to exercise the refusal
    path now that both real backends support duplex spawn. (Vercel Sandbox
    genuinely behaved this way until `open_interactive` existed.)
    """

    kind = "no-stdin"
    duplex_spawn = False


async def test_a_harness_refuses_a_workspace_it_cannot_drive(
    project: Path,
) -> None:
    """Both harnesses speak newline-delimited JSON on stdin.

    A workspace that cannot carry stdin cannot host them, and that has to fail
    at open() rather than somewhere later and stranger.
    """
    async with _NoStdinWorkspace(project) as crippled:
        for build in (claude_code, codex):
            harness = build(workspace=crippled)
            with pytest.raises((UnsupportedError, WorkspaceError)) as exc:
                await harness.open()
            assert "workspace" in str(exc.value).lower()
            assert not harness.is_open


async def test_history_on_a_closed_harness_raises_harness_closed(
    harness_kind: str, project: Path
) -> None:
    async with Local(project) as ws:
        harness = SPECS[harness_kind](workspace=ws)
        await harness.open()
        await harness.close()
        with pytest.raises(HarnessClosedError):
            await harness.history(GHOST)
        with pytest.raises(HarnessClosedError):
            await harness.sessions()


async def test_a_failed_turn_raises_rather_than_returning_empty(
    project: Path,
) -> None:
    """A turn that failed must not look like a turn that said nothing.

    Codex reports `turn/completed` with status "failed" and an error
    message. Reading only the completion and reporting "stop" turned a 401
    into an empty string with no explanation — which is exactly what it did
    until this test existed.
    """
    # A real endpoint that answers 401, rather than an unreachable host:
    # codex retries a refused connection with a long backoff, and this is
    # the failure people actually hit anyway.
    unauthorized = {
        "model_provider": "unauthorized",
        "model_providers.unauthorized.name": "Unauthorized",
        "model_providers.unauthorized.base_url": "https://ai-gateway.vercel.sh/codex/v1",
        "model_providers.unauthorized.env_key": "HARNESS_SDK_NO_SUCH_KEY",
        "model_providers.unauthorized.wire_api": "responses",
    }
    async with Local(project) as ws:
        async with codex(workspace=ws, config=unauthorized) as agent:
            with pytest.raises(TurnFailedError) as exc:
                await agent.run("Reply with exactly: READY")

    assert str(exc.value), "the failure has to say something about why"


async def test_a_crash_mid_turn_is_agent_crashed_and_the_harness_survives(
    harness: Harness,
) -> None:
    """Thirteen places raise AgentCrashedError; none had ever been reached by a
    test. Kill the CLI while it is mid-turn: the caller must get THIS
    library's error, not the harness library's, and the harness must still
    be closable and able to start a fresh session afterwards.

    Local only, honestly: a remote process is not ours to signal — the
    adapter advertises `process_ids == []` there.
    """
    import os
    import signal

    from ai.types.events import TextDelta

    session = harness.session()
    turn = session.stream(
        "Count slowly from 1 to 400, one number per line, no commentary."
    )
    pids: list[int] = []
    with pytest.raises((AgentCrashedError, TurnFailedError)) as caught:
        async for event in turn:
            if isinstance(event, TextDelta) and not pids:
                pids = list(harness.adapter.process_ids)
                assert pids, "a local harness must expose the process it owns"
                for pid in pids:
                    os.kill(pid, signal.SIGKILL)
    assert not isinstance(
        caught.value, Exception
    ) or caught.value.__class__.__module__.startswith(
        ("ai.harnesses.experimental", "ai.workspaces.experimental")
    ), f"the caller was handed {type(caught.value)!r}, not this SDK's error"
    # The harness is not wedged: a fresh conversation still works.
    result = await harness.run("Reply with exactly: ALIVE")
    exact(result.text, "ALIVE")
