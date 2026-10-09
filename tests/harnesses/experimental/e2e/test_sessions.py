"""Sessions: memory across turns, and resuming one from another process."""

from __future__ import annotations

import json

import pytest

from ai.harnesses.experimental import Handle, Harness, harness_from
from ai.workspaces.experimental import Local, Workspace
from tests.harnesses.experimental.conftest import requires

pytestmark = pytest.mark.live


async def test_a_session_remembers_across_turns(any_harness: Harness) -> None:
    session = any_harness.session()

    await session.run("Remember this codeword: PLATYPUS. Reply OK.")
    second = await session.run(
        "What codeword did I give you? Reply with the word only."
    )

    assert "PLATYPUS" in second.text.upper()


async def test_sessions_are_isolated_from_each_other(
    any_harness: Harness,
) -> None:
    first, second = any_harness.session(), any_harness.session()

    await first.run("Remember this codeword: PLATYPUS. Reply OK.")
    answer = await second.run(
        "What codeword were you given? If none, reply exactly: NONE"
    )

    assert "PLATYPUS" not in answer.text.upper()


async def test_messages_accumulate_as_ai_sdk_messages(
    any_harness: Harness,
) -> None:
    session = any_harness.session()

    await session.run("Reply with exactly: ONE")
    await session.run("Reply with exactly: TWO")

    roles = [m.role for m in session.messages]
    assert roles.count("user") >= 2
    assert roles.count("assistant") >= 2
    assert "ONE" in " ".join(m.text for m in session.messages)


async def test_handle_is_plain_json(any_harness: Harness) -> None:
    session = any_harness.session()
    await session.run("Reply with exactly: READY")

    payload = session.handle.model_dump_json()
    restored = Handle.model_validate_json(payload)

    assert json.loads(payload)["session_id"]
    assert restored == session.handle
    assert "approve" not in json.loads(
        payload
    ), "callbacks are process-local, never serialized"


@pytest.mark.slow
async def test_resume_continues_the_exact_conversation(
    harness: Harness, workspace: Workspace
) -> None:
    requires(harness, "resume")
    session = harness.session()
    await session.run("Remember this codeword: PLATYPUS. Reply OK.")
    handle = session.handle

    # Everything above could have happened in another process, yesterday.
    await harness.close()

    async with harness_from(handle) as resumed:
        answer = await resumed.run(
            "What codeword did I give you? Reply with the word only."
        )

    assert "PLATYPUS" in answer.text.upper()


async def test_resume_rejects_an_unknown_session(workspace: Workspace) -> None:
    handle = Handle(
        kind="claude-code",
        session_id="00000000-0000-0000-0000-000000000000",
        workspace=Local(workspace.path).coords,
        options={},
    )

    with pytest.raises(Exception, match="resume"):
        async with harness_from(handle):
            pass


async def test_session_info_names_its_harness(any_harness: Harness) -> None:
    """A session id alone cannot tell you which any_harness to load it with;
    the listing must. `Handle.kind` already does this for a saved handle."""
    session = any_harness.session()
    await session.run("Reply with exactly: READY")

    listed = await any_harness.sessions()
    mine = next(s for s in listed if s.session_id == session.session_id)
    assert mine.kind == any_harness.kind == session.handle.kind


@pytest.mark.sandbox
async def test_resume_into_a_sandbox_is_refused_with_a_reason(
    harness_kind: str,
) -> None:
    """A handle from a sandbox names a machine that no longer exists.

    Resuming it must refuse — never quietly point at some other VM — and the
    refusal must tell the caller what to do instead (resume in a new workspace).
    """
    import os

    from ai.harnesses.experimental import harness_from
    from ai.workspaces.experimental import VercelSandbox
    from ai.workspaces.experimental._gateway import vercel_ai_gateway
    from ai.workspaces.experimental.errors import WorkspaceError
    from tests.harnesses.experimental.conftest import SPECS
    from tests.workspaces.experimental.conftest import sandbox_credentials

    creds = sandbox_credentials()
    if creds is None:
        pytest.skip("needs Vercel Sandbox credentials")
    if not os.environ.get("AI_GATEWAY_API_KEY"):
        pytest.skip("needs AI_GATEWAY_API_KEY: the VM has no login of its own")
    # A bare VM: the harness must bring its own way to reach a model, or
    # this test fails on authentication before it ever reaches the refusal.
    async with VercelSandbox(**creds, execution_time_limit=900) as ws:
        async with SPECS[harness_kind](
            workspace=ws, gateway=vercel_ai_gateway()
        ) as harness:
            session = harness.session()
            await session.run("Reply with exactly: OK")
            saved = session.handle.model_dump_json()

    from ai.harnesses.experimental import Handle

    with pytest.raises(WorkspaceError, match="resume"):
        async with harness_from(Handle.model_validate_json(saved)):
            pass
