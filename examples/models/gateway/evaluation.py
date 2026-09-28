"""Evaluate plain text and structured data with Jev through AI Gateway.

Set ``AI_GATEWAY_API_KEY`` and run:

    uv run python examples/models/gateway/evaluation.py

The two calls below are independent examples. The first asks one question about
plain text. The second asks choice, score, and boolean questions about shared
structured state in one request.
"""

import asyncio

import pydantic

import ai


class RefundQuestions(pydantic.BaseModel):
    requests_refund: ai.ops.experimental.BooleanQuestion


class RefundAnswers(pydantic.BaseModel):
    requests_refund: ai.ops.experimental.BooleanAnswer


class TicketQuestions(pydantic.BaseModel):
    queue: ai.ops.experimental.ChoiceQuestion
    urgency: ai.ops.experimental.ScoreQuestion
    refund_warranted: ai.ops.experimental.BooleanQuestion


class TicketAnswers(pydantic.BaseModel):
    queue: ai.ops.experimental.ChoiceAnswer
    urgency: ai.ops.experimental.ScoreAnswer
    refund_warranted: ai.ops.experimental.BooleanAnswer


async def main() -> None:
    model = ai.get_model("typesafe-ai/jev")
    if not model.provider.is_configured():
        print("Set AI_GATEWAY_API_KEY to run this example.")
        return

    # Ask a single boolean question about plain text. Boolean answers are
    # probabilities rather than only true or false.
    refund_result = await ai.ops.experimental.evaluate(
        model,
        "Please refund the duplicate charge on my account.",
        RefundQuestions(
            requests_refund=ai.ops.experimental.BooleanQuestion(
                instructions="Is the customer asking for a refund?",
            )
        ),
        output_type=RefundAnswers,
    )

    print(
        "refund requested:",
        f"{refund_result.value.requests_refund.probability:.0%}",
    )

    # Ask several question types about the same structured state. This is useful
    # when related decisions should use exactly the same source information.
    ticket: ai.ops.experimental.EvaluationInput = {
        "message": (
            "I was charged $240 twice for the same renewal. The service works, "
            "but please refund the duplicate charge."
        ),
        "customer": {
            "plan": "pro-annual",
            "account_age_days": 812,
        },
        "payments": [
            {
                "id": "pay_01",
                "amount_usd": 240,
                "status": "settled",
                "renewal_id": "ren_42",
            },
            {
                "id": "pay_02",
                "amount_usd": 240,
                "status": "settled",
                "renewal_id": "ren_42",
            },
        ],
        "service_status": "operational",
    }

    result = await ai.ops.experimental.evaluate(
        model,
        ticket,
        TicketQuestions(
            queue=ai.ops.experimental.ChoiceQuestion(
                instructions="Which support queue should handle this ticket?",
                criteria={
                    "billing": "Charges, duplicate payments, and refunds",
                    "technical": "Bugs, outages, and performance problems",
                    "trust_and_safety": "Fraud or account compromise",
                    "other": None,
                },
            ),
            urgency=ai.ops.experimental.ScoreQuestion(
                instructions="How urgent is the customer's primary problem?",
                criteria=[
                    "Low: no active customer impact",
                    "Normal: limited impact with a workaround",
                    "High: material financial or workflow impact",
                    "Critical: security incident, outage, or ongoing loss",
                ],
            ),
            refund_warranted=ai.ops.experimental.BooleanQuestion(
                instructions="Does the evidence warrant a refund?",
                criteria={
                    "true": "A duplicate settlement or billing error is shown",
                    "false": "The charge is valid or evidence is insufficient",
                },
            ),
        ),
        output_type=TicketAnswers,
        # Provider options are optional and apply only to this request.
        params=ai.ops.experimental.EvaluationParams(
            provider_options={
                "gateway": {
                    "zeroDataRetention": True,
                    "disallowPromptTraining": True,
                }
            }
        ),
    )

    answers = result.value
    print("\nqueue:", answers.queue.choice)
    print("queue probabilities:", answers.queue.probabilities)

    print("\nurgency score:", answers.urgency.score)
    print("urgency probabilities:", answers.urgency.probabilities)

    print(
        "\nrefund warranted:",
        f"{answers.refund_warranted.probability:.0%}",
    )

    # Every operation also exposes framework and provider metadata.
    print("\nusage:", result.usage)
    print("warnings:", result.warnings)
    print("metadata:", result.metadata)
    print("provider metadata:", result.provider_metadata)


if __name__ == "__main__":
    asyncio.run(main())
