"""Workers against the real CLIs. Opt-in: `uv run pytest -m live tests/workers`.

Each test runs once per CLI installed here, in a fresh directory, with the
CLI's own login. They cost tokens.
"""

from __future__ import annotations

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

KINDS: dict[str, tuple[str, Callable[..., workers.Worker]]] = {
    "claude-code": ("claude", workers.ClaudeCode),
    "codex": ("codex", workers.Codex),
    "pi": ("pi", workers.Pi),
}


@pytest.fixture(params=list(KINDS))
def kind(request: pytest.FixtureRequest) -> str:
    kind = str(request.param)
    executable, _ = KINDS[kind]
    if shutil.which(executable) is None:
        pytest.skip(f"{executable} is not installed")
    if kind == "claude-code" and not importlib.util.find_spec(
        "claude_agent_sdk"
    ):
        pytest.skip("claude-agent-sdk is not installed")
    return kind


@pytest.fixture
def make(kind: str, tmp_path: Path) -> Callable[..., workers.Worker]:
    """Build a worker of this kind, in a fresh directory."""
    _, cls = KINDS[kind]
    return lambda **fields: cls(cwd=str(tmp_path), **fields)


async def test_a_turn_and_a_resume(make: Any) -> None:
    async with make() as worker:
        first = await ai.experimental_generate(
            worker,
            [ai.user_message("Reply with the word PONG and nothing else.")],
        )
    assert "PONG" in first.text
    assert worker.session_id
    again = workers.Worker.model_validate_json(worker.model_dump_json())
    async with again:
        second = await ai.experimental_generate(
            again,
            [
                *again.messages,
                ai.user_message("Which word did you reply with? One word."),
            ],
        )
    assert "PONG" in second.text.upper()
    assert again.session_id == worker.session_id


async def test_a_native_fork_keeps_the_past(make: Any) -> None:
    async with make() as worker:
        await ai.experimental_generate(
            worker,
            [ai.user_message("Remember the word HERON. Reply with OK only.")],
        )
    branch = worker.model_copy(update={"fork": True})
    async with branch:
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
    kind: str, make: Any, tmp_path: Path
) -> None:
    if kind == "codex":
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
