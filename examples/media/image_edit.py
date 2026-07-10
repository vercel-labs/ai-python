"""Image editing with a dedicated image model.

Demonstrates sending an input image to be edited/transformed by the
image model. The input image is passed as a FilePart in the user
message, and the model returns the edited version.
"""

import asyncio
import base64
import pathlib

import ai
from ai.models.core import api

model = ai.get_model("openai/gpt-image-1")


async def main() -> None:
    # Load an existing image to use as input for editing.
    # In practice you would load a real image file and pass its bytes
    # with media_type="image/png".
    input_image = ai.messages.FilePart(
        data="https://picsum.photos/id/237/400/300.jpg",
        media_type="image/jpeg",
    )

    messages = [
        ai.user_message(
            "Transform this photo into a soft watercolor painting style. "
            "Keep the composition and subject the same but make it look "
            "like a hand-painted watercolor.",
            input_image,
        ),
    ]

    result = await api.experimental_generate(
        model, messages, ai.ImageParams(size="1024x1024")
    )

    print(f"Generated {len(result.images)} edited image(s)")
    for i, img in enumerate(result.images):
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
