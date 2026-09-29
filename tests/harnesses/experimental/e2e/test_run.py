"""One-shot runs: the simplest thing the SDK must do, on every harness."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel

from ai.harnesses.experimental import Harness
from ai.types.messages import Message, TextPart, ToolCallPart
from tests.harnesses.experimental.conftest import exact

pytestmark = pytest.mark.live


async def test_run_returns_text_usage_and_finish_reason(
    any_harness: Harness,
) -> None:
    result = await any_harness.run("Reply with exactly: READY")

    exact(result.text, "READY")
    # The AI SDK's vocabulary (gen_ai semconv), not one we invented.
    assert result.finish_reason == "stop"
    # Token accounting is the AI SDK's Usage model. 0 means "not disclosed";
    # both harnesses do disclose input/output, so this must be real.
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0


async def test_run_result_carries_ai_sdk_messages(any_harness: Harness) -> None:
    result = await any_harness.run("Reply with exactly: READY")

    assert result.messages, "a settled turn always carries its conversation"
    assert all(isinstance(m, Message) for m in result.messages)
    assert result.messages[0].role == "user"
    assert result.messages[-1].role == "assistant"


async def test_run_is_stateless_between_calls(any_harness: Harness) -> None:
    """`any_harness.run()` is a throwaway conversation. Two runs share
    nothing."""
    await any_harness.run("Remember this codeword and reply OK: PLATYPUS")
    second = await any_harness.run(
        "What codeword were you just asked to remember? "
        + "If you were not given one, reply exactly: NONE"
    )

    assert "PLATYPUS" not in second.text.upper()


class RepoSummary(BaseModel):
    name: str
    languages: list[str]


async def test_structured_output_returns_a_validated_instance(
    any_harness: Harness,
) -> None:
    result = await any_harness.run(
        "Summarize this project. The name is 'demo'.", output_type=RepoSummary
    )

    assert isinstance(result.output, RepoSummary)
    assert result.output.name
    assert isinstance(result.output.languages, list)


async def test_structured_output_is_opt_in(any_harness: Harness) -> None:
    """Without output_type the plain path is untouched: no schema, no output."""
    result = await any_harness.run("Reply with exactly: READY")

    assert result.output is None
    assert (
        "schema" not in result.prompt_sent.lower()
    ), "the SDK must not edit a prompt it was not asked to annotate"


async def test_structured_output_annotation_is_inspectable(
    any_harness: Harness,
) -> None:
    """When the SDK DOES edit the prompt, the caller can see exactly what
    went."""
    result = await any_harness.run(
        "Summarize this project.", output_type=RepoSummary
    )

    assert "Summarize this project." in result.prompt_sent
    assert (
        "languages" in result.prompt_sent
    ), "the schema the agent actually saw"


@pytest.mark.slow
async def test_timeout_settles_as_cancelled_rather_than_raising(
    any_harness: Harness,
) -> None:
    """A deadline is a settlement, not an exception.

    The prompt has to genuinely outlive the deadline: asking a model to
    "count slowly" does not — it delegates to a shell command and answers in
    six seconds. Emitting thousands of tokens directly is what actually
    holds a turn open.
    """
    result = await any_harness.run(
        "Write the numbers 1 through 2000 separated by newlines, "
        + "typing them out yourself. Output nothing else.",
        timeout=10,
    )

    assert result.finish_reason == "cancelled"
    assert not result.approval_errors


@pytest.mark.slow
async def test_an_interrupted_turn_keeps_what_it_streamed(
    any_harness: Harness,
) -> None:
    """Partial work survives an interruption — the whole reason a cancelled
    turn is a settlement rather than a loss.

    The interrupt is triggered by OBSERVING streamed text rather than by a
    wall-clock guess: whether a model has started typing within N seconds is
    not something a test should bet on.
    """
    from ai.types.events import TextDelta

    session = any_harness.session()
    turn = session.stream(
        "Write the numbers 1 through 2000 separated by newlines, "
        + "typing them out yourself. Output nothing else."
    )

    async for event in turn:
        if isinstance(event, TextDelta) and event.chunk.strip():
            await session.stop()
            break

    assert turn.result.cancelled
    assert turn.result.text, "whatever the any_harness streamed before the stop"


async def test_text_is_the_final_message_not_the_narration(
    any_make_harness: Callable[..., Any],
) -> None:
    """`result.text` is what the agent said AFTER its last action.

    Agents narrate before acting — "I'll write the file now" — and that
    narration is real, it belongs in the transcript. It does not belong
    glued onto the answer. Measured before the fix on claude:
    'I\'ll write "HELLO" to greeting.txt…for you.Done! The file…'
    Codex drops its `commentary` in the adapter; the session has to draw
    the same line for every any_harness.
    """
    async with any_make_harness(writable=True) as any_harness:
        result = await any_harness.run(
            "First, in one short sentence, tell me you are about to start. "
            + "Then write the word HELLO into hi.txt. "
            + "Then reply with exactly: DONE"
        )

    final = result.messages[-1]
    assert any(
        isinstance(p, ToolCallPart) for p in final.parts
    ), "the prompt is meant to make the agent use a tool"
    assert result.text.strip() == "DONE", result.text
    # The invariant, stated directly: nothing said BEFORE the last tool call
    # is in `.text`. Whether that narration is kept in the transcript is the
    # any_harness's business — claude keeps it, codex labels it `commentary`
    # and the adapter drops it — and `.text` must be the same either way.
    last_call = max(
        i for i, p in enumerate(final.parts) if isinstance(p, ToolCallPart)
    )
    for narration in final.parts[:last_call]:
        if isinstance(narration, TextPart) and narration.text.strip():
            assert narration.text.strip() not in result.text, narration.text


@pytest.mark.parametrize("effort", ["low", "max"])
async def test_effort_is_accepted_on_both_harnesses(
    any_make_harness: Callable[..., Any], effort: str
) -> None:
    """`effort=` reaches the CLI: a turn with it settles normally, and the
    handle remembers it so a resume reproduces the configuration."""
    async with any_make_harness(effort=effort) as harness:
        session = harness.session()
        result = await session.run("Reply with exactly: READY")
        exact(result.text, "READY")
        assert result.finish_reason == "stop"
        assert session.handle.options["effort"] == effort
