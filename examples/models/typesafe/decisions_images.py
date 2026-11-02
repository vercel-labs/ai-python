"""Ask decision questions about images with TypeSafe.

Set ``TYPESAFE_API_KEY`` and run:

    uv run python examples/models/typesafe/decisions_images.py

Images are file parts. TypeSafe accepts them anywhere text can go: inside
JSON state, in a question's instructions, and in its criteria. Questions
refer to state fields by name in backticks.
"""

import asyncio
import pathlib

import pydantic

import ai
from ai.providers.typesafe import TypeSafeProvider


class ListingQuestions(pydantic.BaseModel):
    matches_listing: ai.ops.experimental.PredicateQuestion
    lighting: ai.ops.experimental.ChoiceQuestion
    photo_quality: ai.ops.experimental.ScoreQuestion


class ListingAnswers(pydantic.BaseModel):
    matches_listing: ai.ops.experimental.PredicateAnswer
    lighting: ai.ops.experimental.ChoiceAnswer
    photo_quality: ai.ops.experimental.ScoreAnswer


async def main() -> None:
    model = ai.Model(id="jev-latest", provider=TypeSafeProvider())
    if not model.provider.is_configured():
        print("Set TYPESAFE_API_KEY to run this example.")
        return

    image_path = (
        pathlib.Path(__file__).parents[2] / "media" / "sample_image.jpg"
    )
    photo = ai.file_part(image_path.read_bytes(), media_type="image/jpeg")

    # The image sits inside structured state, next to the data it belongs to.
    listing: ai.ops.experimental.DecisionInput = {
        "listing": {
            "title": "Scenic print, framed",
            "price_usd": 120,
            "photo": photo,
        },
        "buyer_report": {
            "message": "This is what arrived.",
            "photo": photo,
        },
    }

    result = await ai.ops.experimental.decide(
        model,
        listing,
        ListingQuestions(
            matches_listing=ai.ops.experimental.PredicateQuestion(
                instructions=(
                    "Does `buyer_report.photo` show the same item as "
                    "`listing.photo`?"
                ),
            ),
            lighting=ai.ops.experimental.ChoiceQuestion(
                instructions="What is the lighting in `listing.photo` like?",
                criteria={
                    "daylight": "Natural daylight",
                    "artificial": "Lamps or other artificial light",
                    "dark": "Little or no light",
                },
            ),
            # A criterion can be an image: here, the listing photo is the
            # example of the top quality level.
            photo_quality=ai.ops.experimental.ScoreQuestion(
                instructions="How good is `buyer_report.photo` as evidence?",
                criteria=[
                    "Unusable: blurry, dark, or the item is not visible",
                    "Usable: the item is visible but details are unclear",
                    ["As clear as this reference photo:", photo],
                ],
            ),
        ),
        output_type=ListingAnswers,
    )

    answers = result.value
    print("matches listing:", f"{answers.matches_listing.probability:.0%}")
    print("lighting:", answers.lighting.choice)
    print("photo quality:", answers.photo_quality.score)
    print("usage:", result.usage)


if __name__ == "__main__":
    asyncio.run(main())
