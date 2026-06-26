"""Multimodal input — send a local image to the model and ask about it."""

import asyncio
import pathlib

import ai

model = ai.get_model("gateway:anthropic/claude-sonnet-4.6")

image_path = pathlib.Path(__file__).parent / "sample_image.jpg"
image_data = image_path.read_bytes()

messages = [
    ai.user_message(
        "Describe this image in detail.",
        ai.file_part(image_data, media_type="image/jpeg"),
    ),
]


async def main() -> None:
    async with ai.stream(model, messages) as s:
        async for event in s:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
