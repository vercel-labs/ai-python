"""Tool that returns a ContentOutput so the model can see image files directly.

The ``read_file`` tool reads a path from disk and inspects the bytes:

* If the file is an image, it returns a :class:`ContentOutput` carrying
  a summary line and an image :class:`FilePart`. All three providers
  turn that into a real image content block on the next model turn, so
  the model actually *sees* the picture.
* Otherwise it returns the decoded text -- a plain ``str`` result the
  provider sends to the model verbatim.

A single tool covers both code-reading and image-reading duties in an
agentic loop.
"""

import asyncio
import json
import pathlib

import ai
from ai.types import media

# Restrict the tool to a directory we trust the model to roam in.
# `.resolve()` collapses symlinks so a path inside ALLOWED_ROOT cannot
# escape via a symlink that points elsewhere.
ALLOWED_ROOT = pathlib.Path(__file__).parent.resolve()


def _resolve_within_allowed(path: str) -> pathlib.Path:
    resolved = pathlib.Path(path).resolve()
    if not resolved.is_relative_to(ALLOWED_ROOT):
        raise ValueError(
            f"Refusing to read {path!r}: outside allowed root {ALLOWED_ROOT}"
        )
    return resolved


@ai.tool
async def read_file(path: str) -> str | ai.messages.ContentOutput:
    """Read a file from disk.

    Image files come back as a ContentOutput so the model can view them.
    """
    data = _resolve_within_allowed(path).read_bytes()
    image_type = media.detect_image_media_type(data)
    if image_type is not None:
        return ai.content_output(
            f"Loaded {path} ({image_type}, {len(data)} bytes).",
            ai.file_part(data, media_type=image_type),
        )
    return data.decode("utf-8", errors="replace")


async def main() -> None:
    model = ai.get_model("gateway:anthropic/claude-sonnet-4.6")
    my_agent = ai.agent(tools=[read_file])

    here = pathlib.Path(__file__).parent
    image_path = here / "sample_image.jpg"
    text_path = here / "agent_simple.py"

    messages = [
        ai.system_message(
            "Use the read_file tool to inspect any files the user mentions."
        ),
        ai.user_message(
            f"First read {image_path} and describe what you see in the "
            f"picture. Then read {text_path} and summarize what the "
            f"script does in one sentence."
        ),
    ]

    async with my_agent.run(model, messages) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
            elif isinstance(event, ai.events.ToolEnd):
                args = json.loads(event.tool_call.tool_args or "{}")
                print(f"\n[read_file({args.get('path')!r})]")
            elif isinstance(event, ai.events.StreamEnd):
                print()
    print()


if __name__ == "__main__":
    asyncio.run(main())
