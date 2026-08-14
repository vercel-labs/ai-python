"""Image generation — dedicated image model via ai.ops.generate_image()."""

import asyncio
import base64
import pathlib

import ai

model = ai.get_model("google/imagen-4.0-generate-001")

prompt = (
    "A watercolor painting of a cozy cabin in the mountains at sunset, "
    "with warm light spilling from the windows and smoke rising from "
    "the chimney."
)


async def main() -> None:
    result = await ai.ops.generate_image(
        model, prompt, params=ai.ops.ImageParams(n=2, aspect_ratio="16:9")
    )

    print(f"Generated {len(result.value)} image(s)")
    for i, img in enumerate(result.value):
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
