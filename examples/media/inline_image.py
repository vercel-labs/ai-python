"""Inline image generation — LLM that outputs images alongside text.

Models like Gemini 3 Pro Image can generate images as part of their
language model response. The images arrive as ``FileEvent`` events
during the stream and end up as ``FilePart``s on the aggregated
``Stream.message``.
"""

import asyncio
import base64
import pathlib

import ai

model = ai.get_model("google/gemini-3-pro-image")

messages = [
    ai.system_message(
        "You are an art assistant. When asked to draw or create an image, "
        "generate it."
    ),
    ai.user_message(
        "Draw a cat sitting in a field of cherry blossoms at sunset."
    ),
]


async def main() -> None:
    # Stream — text deltas arrive as TextDelta events, generated images
    # arrive as FileEvent events and accumulate on s.message.
    async with ai.stream(model, messages) as s:
        async for event in s:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)

    print()

    # Check for images in the aggregated message.
    if s.message.images:
        for i, img in enumerate(s.message.images):
            filename = f"inline_{i}.png"
            data = (
                img.data
                if isinstance(img.data, bytes)
                else base64.b64decode(img.data)
            )
            pathlib.Path(filename).write_bytes(data)
            print(f"Saved {filename} ({img.media_type}, {len(data)} bytes)")
    else:
        print("No images were generated in this response.")


if __name__ == "__main__":
    asyncio.run(main())
