"""What a turn's prompt may hold: text, and the files a harness can take.

A prompt is a string, or a list of strings and `ai` parts — `TextPart` and
`FilePart` — the same parts a user message holds anywhere in `ai`:

    await session.run(["What is wrong in this screenshot?", ai.file_part(png)])

Which files a harness takes is declared in its capabilities and checked
here, before anything runs: images on every harness (`images`), PDFs where
`documents` is set, and text files (`text/*`) everywhere — those travel as
text. Anything else raises `UnsupportedError`.

Files are turned into bytes HERE, on the machine running this SDK: a URL is
downloaded with the SDK's own SSRF-checked downloader, not handed to the
CLI. A CLI in a sandbox has its egress locked down and could not fetch it,
and measured: Claude Code drops an image given by URL without a word.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...models.core.helpers import files
from ...types import media
from ...types import messages as messages_
from . import errors

if TYPE_CHECKING:
    from . import _capabilities

Part = messages_.TextPart | messages_.FilePart
Prompt = str | Sequence[str | messages_.TextPart | messages_.FilePart]


def parts(prompt: Prompt) -> list[Part]:
    """Normalize a prompt into parts: bare strings become `TextPart`s."""
    if isinstance(prompt, str):
        return [messages_.TextPart(text=prompt)]
    out: list[Part] = []
    for item in prompt:
        if isinstance(item, str):
            out.append(messages_.TextPart(text=item))
        elif isinstance(item, messages_.TextPart | messages_.FilePart):
            out.append(item)
        else:
            raise TypeError(
                "a prompt holds strings, TextPart and FilePart; got "
                + type(item).__name__
            )
    if not out:
        raise ValueError("a prompt needs at least one part")
    return out


def check(
    prompt: list[Part], capabilities: _capabilities.Capabilities, harness: str
) -> None:
    """Raise UnsupportedError for a file this harness cannot take."""
    for part in prompt:
        if not isinstance(part, messages_.FilePart):
            continue
        kind = part.media_type
        if kind.startswith("text/"):
            continue
        if kind.startswith("image/") and capabilities.images:
            continue
        if kind == "application/pdf" and capabilities.documents:
            continue
        raise errors.UnsupportedError(harness, f"{kind} input")


def text(prompt: list[Part]) -> str:
    """Return the prompt's text, files left out: what `prompt_sent` shows."""
    return "\n\n".join(
        p.text for p in prompt if isinstance(p, messages_.TextPart)
    )


async def data(part: messages_.FilePart) -> bytes:
    """Return a file's bytes, downloading a URL here (see the module doc)."""
    if isinstance(part.data, bytes):
        return part.data
    if media.is_downloadable_url(part.data):
        body, _ = await files.download(part.data)
        return body
    return base64.b64decode(media.data_to_base64(part.data))


async def as_text(part: messages_.FilePart) -> str:
    """Return a text file as text a CLI can read inline, named if it has one."""
    body = (await data(part)).decode("utf-8", errors="replace")
    name = part.filename or "attached file"
    return f"<file name={name!r}>\n{body}\n</file>"
