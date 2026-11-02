"""Ask decision questions about images with OpenAI ``gpt-6-luna``.

Set ``OPENAI_API_KEY`` and run:

    uv run python examples/models/openai/decisions_images.py

OpenAI decisions take images in two places:

* in the state, as a list of text and file parts (one user message), and
* in a question's instructions, to give one question its own image.

OpenAI does not accept images nested inside JSON state or in criteria.
"""

import asyncio
import pathlib

import pydantic

import ai


class DeliveryQuestions(pydantic.BaseModel):
    damaged: ai.ops.experimental.PredicateQuestion
    condition: ai.ops.experimental.ScoreQuestion
    matches_reference: ai.ops.experimental.PredicateQuestion


class DeliveryAnswers(pydantic.BaseModel):
    # OpenAI can refuse a question. This field accepts a refusal; a refusal
    # for the other fields raises ValueError.
    damaged: (
        ai.ops.experimental.PredicateAnswer | ai.ops.experimental.RefusalAnswer
    )
    condition: ai.ops.experimental.ScoreAnswer
    matches_reference: ai.ops.experimental.PredicateAnswer


async def main() -> None:
    model = ai.get_model("openai:gpt-6-luna")
    if not model.provider.is_configured():
        print("Set OPENAI_API_KEY to run this example.")
        return

    image_path = (
        pathlib.Path(__file__).parents[2] / "media" / "sample_image.jpg"
    )

    # Shared evidence for all questions: text and an image, like the content
    # of a user message.
    state: ai.ops.experimental.DecisionInput = [
        "A customer uploaded this photo of a delivered item. This is photo A.",
        ai.file_part(image_path.read_bytes()),
    ]

    result = await ai.ops.experimental.decide(
        model,
        state,
        DeliveryQuestions(
            damaged=ai.ops.experimental.PredicateQuestion(
                instructions=(
                    "Does the item in photo A have visible damage, such as a "
                    "crack, tear, or dent? Ignore shadows."
                ),
            ),
            condition=ai.ops.experimental.ScoreQuestion(
                instructions="What is the condition of the item in photo A?",
                criteria=["Broken", "Worn", "Like new"],
            ),
            # This question has its own image. Here it is the same file, so
            # the answer should be close to 100%.
            matches_reference=ai.ops.experimental.PredicateQuestion(
                instructions=[
                    "Is candidate photo B a copy or variant of photo A, "
                    "including recompression or resizing?",
                    ai.file_part(image_path.read_bytes()),
                ],
            ),
        ),
        output_type=DeliveryAnswers,
    )

    answers = result.value
    if isinstance(answers.damaged, ai.ops.experimental.RefusalAnswer):
        print("damaged: refused")
    else:
        print("damaged:", f"{answers.damaged.probability:.0%}")
    print("condition:", answers.condition.score)
    print("matches reference:", f"{answers.matches_reference.probability:.0%}")
    print("usage:", result.usage)


if __name__ == "__main__":
    asyncio.run(main())
