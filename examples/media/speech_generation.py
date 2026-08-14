"""Speech generation — dedicated speech model via ai.ops.generate_audio()."""

import asyncio
import base64
import pathlib

import ai

model = ai.get_model("openai/tts-1")

prompt = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs."
)


async def main() -> None:
    result = await ai.ops.generate_audio(
        model, prompt, params=ai.ops.AudioParams(voice="alloy")
    )

    print(f"Generated {len(result.value)} audio file(s)")
    for i, clip in enumerate(result.value):
        ext = "mp3" if clip.media_type == "audio/mpeg" else "bin"
        filename = f"generated_{i}.{ext}"
        data = (
            clip.data
            if isinstance(clip.data, bytes)
            else base64.b64decode(clip.data)
        )
        pathlib.Path(filename).write_bytes(data)
        print(f"  {filename}: {clip.media_type}, {len(data)} bytes")


if __name__ == "__main__":
    asyncio.run(main())
