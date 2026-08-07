"""Speech-to-text — dedicated transcription model via ai.ops.transcribe()."""

import asyncio
import pathlib
import sys

import ai

model = ai.get_model("openai/whisper-1")


async def main() -> None:
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "generated_0.mp3")
    result = await ai.ops.transcribe(model, path.read_bytes())

    print(f"Transcript: {result.value.text}")
    if result.value.language:
        print(f"Language: {result.value.language}")
    if result.value.duration_seconds is not None:
        print(f"Duration: {result.value.duration_seconds:.1f}s")
    for segment in result.value.segments:
        print(
            f"  [{segment.start_second:6.2f} - {segment.end_second:6.2f}] "
            f"{segment.text}"
        )


if __name__ == "__main__":
    asyncio.run(main())
