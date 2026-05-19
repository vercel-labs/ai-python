"""Agent with Context7 MCP integration for live documentation."""

import asyncio
import os

import ai


async def main() -> None:
    model = ai.get_model("gateway:anthropic/claude-sonnet-4.6")

    context7_tools: list[ai.AgentTool] = await ai.mcp.get_http_tools(
        "https://mcp.context7.com/mcp",
        headers={"CONTEXT7_API_KEY": os.environ.get("CONTEXT7_API_KEY", "")},
        tool_prefix="context7",
    )

    my_agent = ai.agent(tools=context7_tools)

    messages = [
        ai.system_message(
            "You are a helpful assistant. "
            "Use context7 to look up documentation."
        ),
        ai.user_message("How do I create middleware in Next.js?"),
    ]

    async with my_agent.run(model, messages) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
