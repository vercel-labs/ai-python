"""Live tests against the real CLIs.

Opt-in: `AI_EXTERNAL_LIVE=1 uv run pytest tests/external/test_live.py`.

Each test skips when its CLI is not installed. They spend tokens on the
local login of each CLI, so the prompts are small.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import ai
from ai import external

pytestmark = pytest.mark.skipif(
    not os.environ.get("AI_EXTERNAL_LIVE"),
    reason="drives real CLIs: set AI_EXTERNAL_LIVE=1 to run",
)

KINDS: dict[str, tuple[type[external.ExternalHarness], str]] = {
    "claude-code": (external.ClaudeCode, "claude"),
    "codex": (external.Codex, "codex"),
    "fx": (external.Fx, "fx"),
    "pi": (external.Pi, "pi"),
}


@pytest.fixture(params=sorted(KINDS))
def make(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Callable[..., external.ExternalHarness]:
    harness_class, executable = KINDS[request.param]
    if shutil.which(executable) is None:
        pytest.skip(f"{executable} is not installed")
    (tmp_path / "secret.txt").write_text("the word is HERON\n")
    cwd = os.path.realpath(tmp_path)

    def build(**fields: Any) -> external.ExternalHarness:
        return harness_class(cwd=cwd, **fields)

    return build


async def test_a_turn_with_a_cli_tool_and_a_resume(
    make: Callable[..., external.ExternalHarness],
) -> None:
    harness = make()
    async with asyncio.timeout(240), harness.session() as session:
        async with ai.stream(
            harness,
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
    assert session.id is not None
    assert harness.session_id == session.id  # the external harness remembers it
    # The external harness's data is all that a later process needs. No block:
    # the call starts the CLI and continues the session.
    again = external.ExternalHarness.model_validate_json(
        harness.model_dump_json()
    )
    async with asyncio.timeout(240):
        reply = await ai.experimental_generate(
            again, [ai.user_message("Which word did you read? One word.")]
        )
    assert "HERON" in reply.text.upper()
    assert again.session_id == session.id


async def test_model_calls_without_a_session_continue_the_conversation(
    make: Callable[..., external.ExternalHarness],
) -> None:
    harness = make()
    history = [
        ai.user_message(
            "Read secret.txt with a tool. Reply with the word only."
        )
    ]
    async with asyncio.timeout(240):
        async with ai.stream(harness, history) as s:
            async for _ in s:
                pass
        assert "HERON" in s.text.upper()
        # The CLI stopped after the turn; this call continues its session.
        reply = await ai.experimental_generate(
            harness,
            [
                *history,
                s.message,
                ai.user_message("Which file did you read? The name only."),
            ],
        )
    assert "SECRET" in reply.text.upper()


async def test_an_agent_gives_its_tools_to_the_cli(
    make: Callable[..., external.ExternalHarness],
) -> None:
    @ai.tool
    async def secret_word(topic: str) -> str:
        """Return the secret word for a topic."""
        return "PELICAN" if topic == "birds" else "UNKNOWN"

    harness = make()
    async with asyncio.timeout(240):  # no session: the CLI runs for the turn
        async with ai.Agent(tools=[secret_word]).run(
            harness,
            [
                ai.user_message(
                    "Call the secret_word tool with topic 'birds'. "
                    "Reply with the word only."
                )
            ],
        ) as run:
            async for _ in run:
                pass
    roles = [message.role for message in run.messages]
    assert "tool" in roles
    assert "PELICAN" in run.messages[-1].text.upper()


async def test_cancel_interrupts_and_the_session_continues(
    make: Callable[..., external.ExternalHarness],
) -> None:
    harness = make()
    history = [ai.user_message("Write the numbers 1 to 300, one per line.")]
    async with asyncio.timeout(240), harness.session() as session:
        async with ai.stream(harness, history) as s:
            async for event in s:
                if isinstance(event, ai.events.TextDelta):
                    break  # stop reading: the turn stops
        cut = session.messages[-1].provider_metadata or {}
        assert cut.get("external") == {"stop_reason": "interrupted"}
        reply = await ai.experimental_generate(
            harness, [*history, ai.user_message("Reply with the word OK only.")]
        )
    assert "OK" in reply.text.upper()


async def test_a_steer_lands_in_the_conversation(
    make: Callable[..., external.ExternalHarness],
) -> None:
    harness = make()
    async with asyncio.timeout(240), harness.session() as session:
        await session.send(
            ai.user_message(
                "Run `sleep 5` with a shell tool, "
                "then reply with the word DONE."
            )
        )
        steered = False
        async for event in session.events():
            if not steered and isinstance(event, ai.events.BuiltinToolStart):
                steered = True
                await session.send(
                    ai.user_message("After that, also say the word STEERED.")
                )
            if (
                isinstance(event, ai.events.StreamEnd)
                and event.finish_reason == "stop"
            ):
                if any(
                    m.role == "user" and "STEERED" in m.text
                    for m in session.messages
                ):
                    break
    user_texts = [m.text for m in session.messages if m.role == "user"]
    assert any("STEERED" in t for t in user_texts)
    assert (
        "STEERED"
        in "".join(
            m.text for m in session.messages if m.role == "assistant"
        ).upper()
    )


async def test_a_new_session_uses_the_given_past(
    make: Callable[..., external.ExternalHarness],
) -> None:
    harness = make()
    assert harness.session_id is None
    history = [
        ai.user_message("The secret word is GENTIAN-31. Remember it."),
        ai.assistant_message("Noted."),
        ai.user_message(
            "What is the secret word from our conversation? Reply with it only."
        ),
    ]
    async with asyncio.timeout(240):
        reply = await ai.experimental_generate(harness, history)
    assert "GENTIAN-31" in reply.text.upper()
    assert harness.session_id is not None
