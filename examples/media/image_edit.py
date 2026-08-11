"""Image editing with a dedicated image model.

Demonstrates sending an input image to be edited/transformed by the
image model. The input image goes in the ImagePrompt alongside the
text, and the model returns the edited version.
"""

import asyncio
import base64
import pathlib

import ai

model = ai.get_model("openai/gpt-image-1")


async def main() -> None:
    # An input image can be a URL, raw bytes, base-64 data, or a FilePart.
    prompt = ai.ops.ImagePrompt(
        text=(
            "Transform this photo into a soft watercolor painting style. "
            "Keep the composition and subject the same but make it look "
            "like a hand-painted watercolor."
        ),
        images=["https://picsum.photos/id/237/400/300.jpg"],
    )

    result = await ai.ops.generate_image(
        model, prompt, params=ai.ops.ImageParams(size="1024x1024")
    )

    print(f"Generated {len(result.value)} edited image(s)")
    for i, img in enumerate(result.value):
        filename = f"watercolor_edit_{i}.png"
        data = (
            img.data
            if isinstance(img.data, bytes)
            else base64.b64decode(img.data)
        )
        pathlib.Path(filename).write_bytes(data)
        print(f"  {filename}: {img.media_type}, {len(data)} bytes")


if __name__ == "__main__":
    asyncio.run(main())
