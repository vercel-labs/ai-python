"""Prompts with files: checked before anything runs, carried as each CLI
takes them."""

from __future__ import annotations

import base64

import pydantic
import pytest

from ai.harnesses.experimental import _capabilities, _input
from ai.harnesses.experimental._adapters import codex
from ai.harnesses.experimental.errors import UnsupportedError
from ai.types import messages as m

from . import fakes

PNG = b"\x89PNG\r\n\x1a\nfake"
IMAGE = m.FilePart(data=PNG, media_type="image/png")
PDF = m.FilePart(data=b"%PDF-1.4 fake", media_type="application/pdf")
NOTE = m.FilePart(data=b"remember the milk", media_type="text/plain")
AUDIO = m.FilePart(data=b"RIFF", media_type="audio/wav")


async def test_a_prompt_with_files_reaches_the_adapter_and_the_history() -> (
    None
):
    harness, adapter = await fakes.open_harness()
    session = harness.session()
    result = await session.run(["what is this?", IMAGE])
    text, image = adapter.prompts[0]
    assert isinstance(text, m.TextPart) and text.text == "what is this?"
    assert image == IMAGE
    assert result.messages[0].parts[1] == IMAGE
    assert result.prompt_sent == "what is this?"


@pytest.mark.parametrize("part", [AUDIO, PDF])
async def test_a_file_the_harness_cannot_take_is_refused_up_front(
    part: m.FilePart,
) -> None:
    caps = _capabilities.Capabilities(images=True, documents=False)
    harness, adapter = await fakes.open_harness(
        fakes.FakeAdapter(capabilities=caps)
    )
    session = harness.session()
    with pytest.raises(UnsupportedError, match=part.media_type):
        session.stream(["read this", part])
    assert adapter.prompts == [] and session.messages == []


async def test_text_files_are_taken_everywhere() -> None:
    caps = _capabilities.Capabilities()  # no images, no documents
    harness, adapter = await fakes.open_harness(
        fakes.FakeAdapter(capabilities=caps)
    )
    await harness.session().run(["summarize", NOTE])
    assert adapter.prompts[0][1] == NOTE


async def test_structured_output_adds_its_schema_to_a_file_prompt() -> None:
    class Answer(pydantic.BaseModel):
        word: str

    harness, adapter = await fakes.open_harness(
        fakes.FakeAdapter(lambda prompt: fakes.reply('{"word": "red"}'))
    )
    result = await harness.session().run([IMAGE], output_type=Answer, retries=0)
    assert result.output == Answer(word="red")
    sent = adapter.prompts[0]
    assert sent[0] == IMAGE
    assert isinstance(sent[1], m.TextPart) and "schema" in sent[1].text


@pytest.mark.parametrize(
    "data",
    [
        PNG,
        base64.b64encode(PNG).decode(),
        "data:image/png;base64," + base64.b64encode(PNG).decode(),
    ],
)
async def test_file_bytes_come_from_any_spelling(data: str | bytes) -> None:
    part = m.FilePart(data=data, media_type="image/png")
    assert await _input.data(part) == PNG


async def test_a_text_file_is_inlined_with_its_name() -> None:
    part = m.FilePart(data=b"hi", media_type="text/plain", filename="a.txt")
    assert await _input.as_text(part) == "<file name='a.txt'>\nhi\n</file>"


async def test_codex_gets_images_as_data_urls_and_text_inline() -> None:
    items = await codex._input_items([m.TextPart(text="look"), IMAGE, NOTE])
    assert items[0] == {"type": "text", "text": "look"}
    assert items[1] == {
        "type": "image",
        "url": "data:image/png;base64," + base64.b64encode(PNG).decode(),
    }
    assert (
        items[2]["type"] == "text" and "remember the milk" in items[2]["text"]
    )


async def test_claude_gets_base64_blocks() -> None:
    claude = pytest.importorskip("ai.harnesses.experimental._adapters.claude")
    assert await claude._content([m.TextPart(text="only text")]) == "only text"
    blocks = await claude._content([m.TextPart(text="look"), IMAGE, PDF])
    assert [b["type"] for b in blocks] == ["text", "image", "document"]
    assert blocks[1]["source"]["data"] == base64.b64encode(PNG).decode()
    assert blocks[2]["source"]["media_type"] == "application/pdf"


# -- messages in, like the rest of `ai` ---------------------------------------


async def test_a_user_message_is_a_prompt() -> None:
    harness, adapter = await fakes.open_harness()
    result = await harness.session().run(
        m.Message(role="user", parts=[m.TextPart(text="hi"), IMAGE])
    )
    assert [type(p) for p in adapter.prompts[0]] == [m.TextPart, m.FilePart]
    assert result.text == "hi"


async def test_a_prompt_message_must_be_a_user_message() -> None:
    harness, _ = await fakes.open_harness()
    assistant = m.Message(role="assistant", parts=[m.TextPart(text="hi")])
    with pytest.raises(ValueError, match="user message"):
        harness.session().stream(assistant)


async def test_harness_run_takes_a_conversation_like_agent_run() -> None:
    harness, adapter = await fakes.open_harness()
    earlier = [
        m.Message(role="user", parts=[m.TextPart(text="one")]),
        m.Message(role="assistant", parts=[m.TextPart(text="one")]),
    ]
    prompt = m.Message(role="user", parts=[m.TextPart(text="two")])
    result = await harness.run([*earlier, prompt])
    assert adapter.histories == [earlier], "the past became the history"
    assert result.text == "two"
    assert [msg.role for msg in result.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


async def test_a_conversation_must_end_with_a_user_message() -> None:
    harness, _ = await fakes.open_harness()
    with pytest.raises(ValueError, match="end with a user message"):
        await harness.run(
            [m.Message(role="assistant", parts=[m.TextPart(text="hi")])]
        )


async def test_a_conversation_and_history_together_are_refused() -> None:
    harness, _ = await fakes.open_harness()
    prompt = m.Message(role="user", parts=[m.TextPart(text="hi")])
    with pytest.raises(ValueError, match="not both"):
        await harness.run([prompt], history=[prompt])


async def test_a_session_turn_does_not_take_a_conversation() -> None:
    harness, _ = await fakes.open_harness()
    prompt = m.Message(role="user", parts=[m.TextPart(text="hi")])
    with pytest.raises(TypeError, match=r"Harness\.run"):
        harness.session().stream([prompt])  # type: ignore[list-item]  # ty: ignore[invalid-argument-type]
