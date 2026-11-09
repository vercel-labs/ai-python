"""Workers against real CLIs; opt-in and never part of offline verification.

Run explicitly: `uv run --frozen pytest -m live tests/workers/test_live.py`.
Each test runs once per CLI installed here, in a fresh directory, with the
CLI's own login. They require user credentials and cost tokens.
"""

from __future__ import annotations

import contextlib
import importlib.util
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import ai
from ai import workers
from ai.types import events

pytestmark = pytest.mark.live

WORKER_CLASSES: dict[str, tuple[str, Callable[..., workers.Worker]]] = {
    "claude-code": ("claude", workers.ClaudeCode),
    "codex": ("codex", workers.Codex),
    "pi": ("pi", workers.Pi),
}


@pytest.fixture(params=list(WORKER_CLASSES))
def worker_class_id(request: pytest.FixtureRequest) -> str:
    worker_class_id = str(request.param)
    executable, _ = WORKER_CLASSES[worker_class_id]
    if shutil.which(executable) is None:
        pytest.skip(f"{executable} is not installed")
    if worker_class_id == "claude-code" and not importlib.util.find_spec(
        "claude_agent_sdk"
    ):
        pytest.skip("claude-agent-sdk is not installed")
    return worker_class_id


@pytest.fixture
def make(worker_class_id: str, tmp_path: Path) -> Callable[..., workers.Worker]:
    """Build this worker class in a fresh directory."""
    _, cls = WORKER_CLASSES[worker_class_id]
    return lambda **fields: cls(cwd=str(tmp_path), **fields)


async def test_a_turn_and_a_resume(worker_class_id: str, make: Any) -> None:
    # None selects the authenticated CLI's default model.
    async with make(model_name=None) as worker:
        first = await ai.experimental_generate(
            worker,
            [ai.user_message("Reply with the word PONG and nothing else.")],
        )
        await worker.sync()
        assert any(
            "PONG" in m.text for m in worker.messages if m.role == "assistant"
        )
    assert "PONG" in first.text
    assert worker.session_id
    saved = worker.model_dump()
    assert saved["worker_class_id"] == worker_class_id
    assert saved["model_name"] is None
    assert not {"kind", "model", "fork"}.intersection(saved)
    again = workers.Worker.model_validate_json(worker.model_dump_json())
    assert again.worker_class_id == worker_class_id
    async with again:
        # Read native storage again: its normalized messages/ids are the prefix
        # to continue, not a pre-sync copy of the streamed mirror.
        await again.sync()
        second = await ai.experimental_generate(
            again,
            [
                *again.messages,
                ai.user_message("Which word did you reply with? One word."),
            ],
        )
    assert "PONG" in second.text.upper()
    assert again.session_id == worker.session_id


async def test_direct_events_and_sync(make: Any) -> None:
    prompt = "Reply with the word PONG and nothing else."
    seen: list[workers.SessionEvent] = []
    async with make() as worker:
        await worker.send(prompt)
        # The feed spans the session; break at the end of this answer and
        # explicitly release its sole reader before sync().
        async with contextlib.aclosing(worker.events()) as feed:
            async for event in feed:
                seen.append(event)
                if isinstance(event, events.StreamEnd):
                    break
        accepted = [
            e for e in seen if isinstance(e, workers.UserMessageAccepted)
        ]
        assert [e.message.text for e in accepted] == [prompt]
        assert any(isinstance(e, events.StreamStart) for e in seen)
        assert isinstance(seen[-1], events.StreamEnd)
        assert "PONG" in "".join(
            e.chunk for e in seen if isinstance(e, events.TextDelta)
        )
        session_id = worker.session_id
    assert session_id
    # A session-id-only worker has no mirror until it reads the CLI's store.
    async with make(session_id=session_id) as again:
        assert again.messages == []
        await again.sync()
        assert any(
            m.role == "user" and prompt in m.text for m in again.messages
        )
        assert any(
            m.role == "assistant" and "PONG" in m.text for m in again.messages
        )


async def test_a_native_fork_keeps_the_past(make: Any) -> None:
    async with make() as worker:
        await ai.experimental_generate(
            worker,
            [ai.user_message("Remember the word HERON. Reply with OK only.")],
        )
    branch = await worker.fork()
    async with branch:
        assert branch.session_id and branch.session_id != worker.session_id
        assert branch.worker_class_id == worker.worker_class_id
        assert any("HERON" in message.text for message in branch.messages)
        reply = await ai.experimental_generate(
            branch,
            [
                *branch.messages,
                ai.user_message("Which word did I ask you to remember?"),
            ],
        )
    assert "HERON" in reply.text.upper()
    assert branch.session_id != worker.session_id


async def test_agent_tools_reach_the_cli(make: Any) -> None:
    @ai.tool
    async def lookup_order(order_id: str) -> str:
        """Look up the status of an order by its id."""
        return f"order {order_id}: shipped on 2026-11-02"

    prompt = (
        "Use the lookup_order tool for order A-17, then state its status in "
        "one short sentence."
    )
    async with make() as worker:
        agent = ai.Agent(tools=[lookup_order])
        async with agent.run(worker, [ai.user_message(prompt)]) as run:
            seen = [event async for event in run]
    assert [m.role for m in run.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert "2026" in run.messages[-1].text or "Nov" in run.messages[-1].text
    hidden = [
        e
        for e in seen
        if isinstance(e, events.BuiltinToolStart) and "lookup" in e.tool_name
    ]
    assert hidden == []  # the CLI's own report of the bridge call


async def test_a_denied_tool_call_does_not_run(
    worker_class_id: str, make: Any, tmp_path: Path
) -> None:
    if worker_class_id == "codex":
        pytest.skip("codex asks only when its own policy escalates")
    pending: list[Any] = []
    prompt = "Create a file named notes.txt containing hello. Use a tool."
    async with make(approve=True) as worker:
        async with ai.Agent().run(worker, [ai.user_message(prompt)]) as run:
            async for event in run:
                if isinstance(event, events.PendingHookEvent):
                    pending.append(event.hook)
                    ai.resolve_hook(
                        event.hook,
                        {"granted": False, "reason": "not allowed here"},
                    )
    assert pending
    assert all(h.hook_type == "ToolApproval" for h in pending)
    assert not (tmp_path / "notes.txt").exists()


async def test_direct_approval_outside_agent(
    worker_class_id: str, make: Any, tmp_path: Path
) -> None:
    if worker_class_id == "codex":
        pytest.skip("codex asks only when its own policy escalates")
    pending: list[Any] = []
    async with make(approve=True) as worker:
        await worker.send(
            "Create a file named notes.txt containing hello. Use a tool."
        )
        async with contextlib.aclosing(worker.events()) as feed:
            async for event in feed:
                if isinstance(event, events.PendingHookEvent):
                    pending.append(event.hook)
                    ai.resolve_hook(
                        event.hook,
                        {"granted": False, "reason": "not allowed here"},
                        registry=worker.hook_registry,
                    )
                elif isinstance(event, events.StreamEnd):
                    break
    assert pending
    assert all(h.hook_type == "ToolApproval" for h in pending)
    assert not (tmp_path / "notes.txt").exists()
