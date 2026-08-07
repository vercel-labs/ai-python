"""Document reranking — dedicated reranking model via ai.ops.rerank()."""

import asyncio

import ai

model = ai.get_model("cohere/rerank-v3.5")

query = "How do I reset my password?"

documents = [
    "Our office is open Monday through Friday, 9am to 5pm.",
    "To reset your password, click 'Forgot password' on the login page.",
    "The quarterly report shows strong growth in the APAC region.",
    "Password requirements: at least 12 characters, one number.",
    "Contact support via the chat widget in the bottom right corner.",
]


async def main() -> None:
    result = await ai.ops.rerank(
        model,
        documents,
        query,
        params=ai.ops.RerankParams(top_n=3),
    )

    print(f"Query: {query!r}\n")
    for ranked in result.value:
        print(f"  {ranked.score:.3f}: {documents[ranked.index]!r}")


if __name__ == "__main__":
    asyncio.run(main())
