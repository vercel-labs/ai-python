"""Prompts with files, on every harness at every location."""

from __future__ import annotations

import pytest

from ai.harnesses.experimental import Harness
from ai.harnesses.experimental.errors import UnsupportedError
from ai.types import messages as m
from tests.harnesses.experimental import media
from tests.harnesses.experimental.conftest import exact

pytestmark = pytest.mark.live


async def test_an_image_in_the_prompt_is_seen(any_harness: Harness) -> None:
    session = any_harness.session()
    result = await session.run(
        [
            "What colour fills this image? Reply with one word.",
            m.FilePart(data=media.png((220, 20, 20)), media_type="image/png"),
        ]
    )
    assert "red" in result.text.lower(), result.text
    # The stored conversation keeps the image, as a file.
    history = await any_harness.history(session.session_id)
    first = next(msg for msg in history if msg.role == "user")
    assert any(
        isinstance(p, m.FilePart) and p.media_type.startswith("image/")
        for p in first.parts
    ), first.parts


async def test_a_pdf_where_the_harness_takes_documents(
    any_harness: Harness,
) -> None:
    prompt: list[str | m.FilePart] = [
        "Which word is written in this PDF? Reply with the word only.",
        m.FilePart(data=media.pdf("PINEAPPLE"), media_type="application/pdf"),
    ]
    if not any_harness.capabilities.documents:
        with pytest.raises(UnsupportedError, match="application/pdf"):
            await any_harness.run(prompt)
        return
    exact((await any_harness.run(prompt)).text, "PINEAPPLE")


async def test_a_text_file_is_read_on_every_harness(
    any_harness: Harness,
) -> None:
    result = await any_harness.run(
        [
            "What is the password in this file? Reply with the word only.",
            m.FilePart(
                data=b"The password is OSPREY.\n",
                media_type="text/plain",
                filename="notes.txt",
            ),
        ]
    )
    exact(result.text, "OSPREY")
