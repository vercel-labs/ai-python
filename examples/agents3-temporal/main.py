"""Entry point — starts a Temporal worker and executes the agent workflow.

Two examples in one project:
  - ``direct``   — custom loop, each I/O call is a Temporal activity
  - ``provider`` — default loop, DurabilityProvider routes I/O to activities

Prerequisites:
    1. Temporal dev server: temporal server start-dev
    2. AI_GATEWAY_API_KEY environment variable set

Usage:
    uv run python main.py direct
    uv run python main.py provider
    uv run python main.py direct "What is the weather in Tokyo?"
    uv run python main.py provider "Compare weather in NYC and LA"
"""

from __future__ import annotations

import asyncio
import sys
import uuid

import activities
import direct
import provider
import temporalio.client
import temporalio.worker

TASK_QUEUE = "agents3-durable"


async def main(mode: str, user_query: str) -> None:
    temporal = await temporalio.client.Client.connect("localhost:7233")

    workflows = {
        "direct": direct.DirectWorkflow,
        "provider": provider.ProviderWorkflow,
    }
    workflow_cls = workflows[mode]

    async with temporalio.worker.Worker(
        temporal,
        task_queue=TASK_QUEUE,
        workflows=[workflow_cls],
        activities=[
            activities.llm_call_activity,
            activities.get_weather_activity,
            activities.get_population_activity,
            activities.tool_dispatch_activity,
        ],
    ):
        workflow_id = f"agents3-{mode}-{uuid.uuid4().hex[:8]}"
        print(f"Mode:     {mode}")
        print(f"Workflow: {workflow_id}")
        print(f"Query:    {user_query}\n")

        result = await temporal.execute_workflow(
            workflow_cls.run,
            user_query,
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
        print(result)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("direct", "provider"):
        print("Usage: python main.py <direct|provider> [query]")
        sys.exit(1)

    mode = sys.argv[1]
    query = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "What's the weather and population of New York and Los Angeles?"
    )
    asyncio.run(main(mode, query))
