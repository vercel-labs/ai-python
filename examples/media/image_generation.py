"""Image generation — dedicated image model via experimental_generate()."""

import asyncio
import base64
import pathlib

import ai
from ai.models.core import api

model = ai.get_model("google/imagen-4.0-generate-001")

messages = [
    ai.user_message(
        "A watercolor painting of a cozy cabin in the mountains at sunset, "
        "with warm light spilling from the windows and smoke rising from "
        "the chimney."
    ),
]


async def main() -> None:
    result = await api.experimental_generate(
        model, messages, ai.ImageParams(n=2, aspect_ratio="16:9")
    )

    print(f"Generated {len(result.images)} image(s)")
    for i, img in enumerate(result.images):
        filename = f"generated_{i}.png"
        data = (
            img.data
            if isinstance(img.data, bytes)
            else base64.b64decode(img.data)
        )
        pathlib.Path(filename).write_bytes(data)
        print(f"  {filename}: {img.media_type}, {len(data)} bytes")

    if result.usage:
        print(
            f"Usage: {result.usage.input_tokens} input, "
            f"{result.usage.output_tokens} output tokens"
        )


if __name__ == "__main__":
    asyncio.run(main())
