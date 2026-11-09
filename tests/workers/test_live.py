"""Live tests against the real CLIs.

Opt-in: `uv run pytest -m live tests/workers`.

Each test skips when its CLI is not installed. They spend tokens on the
local login of each CLI, so the prompts are small.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import ai
from ai import workers

pytestmark = pytest.mark.live

KINDS: dict[str, tuple[type[workers.Worker], str]] = {
    "codex": (workers.Codex, "codex"),
}


@pytest.fixture(params=sorted(KINDS))
def make(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Callable[..., workers.Worker]:
    worker_class, executable = KINDS[request.param]
    if shutil.which(executable) is None:
        pytest.skip(f"{executable} is not installed")
    (tmp_path / "secret.txt").write_text("the word is HERON\n")
    cwd = os.path.realpath(tmp_path)

    def build(**fields: Any) -> workers.Worker:
        return worker_class(cwd=cwd, **fields)

    return build


async def test_a_turn_with_a_cli_tool_and_a_resume(
    make: Callable[..., workers.Worker],
) -> None:
    worker = make()
    async with asyncio.timeout(240), worker:
        async with ai.stream(
            worker,
            [
                ai.user_message(
                    "Read secret.txt with a tool. Reply with the word only."
                )
            ],
        ) as s:
            async for _ in s:
                pass
        assert "HERON" in s.text.upper()
        assert s.message.builtin_tool_calls
    saved = worker.model_dump_json()
    again = workers.Worker.model_validate_json(saved)
    async with asyncio.timeout(240), again:
        reply = await ai.experimental_generate(
            again,
            [
                *again.messages,
                ai.user_message("Which word did you read? One word."),
            ],
        )
    assert "HERON" in reply.text.upper()
    assert again.session_id == worker.session_id


async def test_an_agent_gives_its_tools_to_the_cli(
    make: Callable[..., workers.Worker],
) -> None:
    @ai.tool
    async def secret_word(topic: str) -> str:
        """Return the secret word for a topic."""
        return "PELICAN" if topic == "birds" else "UNKNOWN"

    worker = make()
    async with asyncio.timeout(240), worker:
        async with ai.Agent(tools=[secret_word]).run(
            worker,
            [
                ai.user_message(
                    "Call the secret_word tool with topic 'birds'. "
                    "Reply with the word only."
                )
            ],
        ) as run:
            async for _ in run:
                pass
    roles = [message.role for message in worker.messages]
    assert "tool" in roles
    assert "PELICAN" in worker.messages[-1].text.upper()


async def test_cancel_interrupts_and_the_session_continues(
    make: Callable[..., workers.Worker],
) -> None:
    worker = make()
    history = [ai.user_message("Write the numbers 1 to 300, one per line.")]
    async with asyncio.timeout(240), worker:
        async with ai.stream(worker, history) as s:
            async for event in s:
                if isinstance(event, ai.events.TextDelta):
                    break  # stop reading: the turn stops
        assert workers.is_interrupted(worker.messages[-1])
        reply = await ai.experimental_generate(
            worker, [*history, ai.user_message("Reply with the word OK only.")]
        )
    assert "OK" in reply.text.upper()


async def test_a_steer_lands_in_the_conversation(
    make: Callable[..., workers.Worker],
) -> None:
    worker = make()
    async with asyncio.timeout(240), worker:
        feed = worker.events()
        await worker.send(
            "Run `sleep 5` with a shell tool, then reply with the word DONE."
        )
        steered = False
        async for event in feed:
            if not steered and isinstance(event, ai.events.BuiltinToolStart):
                steered = True
                await worker.send("After that, also say the word STEERED.")
            if (
                isinstance(event, ai.events.StreamEnd)
                and event.finish_reason == "stop"
            ):
                if any(
                    m.role == "user" and "STEERED" in m.text
                    for m in worker.messages
                ):
                    break
        with contextlib.suppress(Exception):
            await feed.aclose()
    user_texts = [m.text for m in worker.messages if m.role == "user"]
    assert any("STEERED" in t for t in user_texts)
    assert (
        "STEERED"
        in "".join(
            m.text for m in worker.messages if m.role == "assistant"
        ).upper()
    )
