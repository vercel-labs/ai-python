"""Video generation — dedicated video model via ai.ops.generate_video()."""

import asyncio
import base64
import pathlib

import ai

model = ai.get_model("google/veo-3.1-fast-generate-001")

messages = [
    ai.user_message("A paper boat drifting across a puddle in the rain."),
]


async def main() -> None:
    print("Generating video (this may take a minute or two)...")

    result = await ai.ops.generate_video(
        model,
        messages,
        params=ai.ops.VideoParams(aspect_ratio="16:9", duration=4),
    )

    print(f"Generated {len(result.videos)} video(s)")
    for i, vid in enumerate(result.videos):
        ext = "mp4" if "mp4" in vid.media_type else "webm"
        filename = f"generated_{i}.{ext}"
        data = (
            vid.data
            if isinstance(vid.data, bytes)
            else base64.b64decode(vid.data)
        )
        pathlib.Path(filename).write_bytes(data)
        print(f"  {filename}: {vid.media_type}, {len(data)} bytes")


if __name__ == "__main__":
    asyncio.run(main())
