"""Text embeddings — dedicated embedding model via ai.ops.embed()."""

import asyncio
import math

import ai

model = ai.get_model("openai/text-embedding-3-small")

values = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn fox leaps above a sleepy canine.",
    "The stock market closed higher today.",
]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


async def main() -> None:
    result = await ai.ops.embed(model, values)

    print(f"Embedded {len(result.value)} values")
    for text, vector in zip(values, result.value, strict=True):
        print(f"  {len(vector)} dims: {text!r}")
    if result.usage:
        print(f"Token usage: {result.usage.input_tokens}")

    query, *candidates = values
    print(f"\nSimilarity to {query!r}:")
    for text, vector in zip(candidates, result.value[1:], strict=True):
        score = cosine_similarity(result.value[0], vector)
        print(f"  {score:.3f}: {text!r}")


if __name__ == "__main__":
    asyncio.run(main())
