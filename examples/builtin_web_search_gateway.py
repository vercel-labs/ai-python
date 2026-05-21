"""Built-in web search through AI Gateway."""

import asyncio
import json

import ai
from ai.providers.ai_gateway import tools as gateway_tools
from ai.providers.anthropic import tools as anthropic_tools

messages = [
    ai.system_message("Be concise. Cite sources you use. The year is 2026"),
    ai.user_message(
        "Who won the most recent Formula 1 Grand Prix, and where was it held?"
    ),
]

model = ai.get_model("gateway:anthropic/claude-sonnet-4.6")


def _strip_encrypted(value: object) -> object:
    """Some fields are encrypted, strip them for clarity"""
    if isinstance(value, dict):
        return {
            k: _strip_encrypted(v)
            for k, v in value.items()
            if k not in ["encryptedContent", "encryptedStdout"]
        }
    if isinstance(value, list):
        return [_strip_encrypted(v) for v in value]
    return value


def format(value: object) -> str:
    try:
        return json.dumps(
            _strip_encrypted(value), indent=2, ensure_ascii=False, default=str
        )
    except (TypeError, ValueError):
        return repr(value)


async def main() -> None:
    print("anthropic web search")
    async with ai.stream(
        model,
        messages,
        tools=[anthropic_tools.web_search(max_uses=3)],
    ) as s:
        async for event in s:
            match event:
                case ai.events.TextDelta():
                    print(event.chunk, end="", flush=True)
                case ai.events.BuiltinToolEnd():
                    args = json.loads(event.tool_call.tool_args or "{}")
                    print(f"\n[{event.tool_call.tool_name}] input:")
                    print(format(args))
                case ai.events.BuiltinToolResult():
                    print(f"\n[{event.result.tool_name}] result:")
                    print(format(event.result.result))
        print()

    print("perplexity web search")
    async with ai.stream(
        model,
        messages,
        tools=[gateway_tools.perplexity_search(max_results=5)],
    ) as s:
        async for event in s:
            match event:
                case ai.events.TextDelta():
                    print(event.chunk, end="", flush=True)
                case ai.events.BuiltinToolEnd():
                    args = json.loads(event.tool_call.tool_args or "{}")
                    print(f"\n[{event.tool_call.tool_name}] input:")
                    print(format(args))
                case ai.events.BuiltinToolResult():
                    print(f"\n[{event.result.tool_name}] result:")
                    print(format(event.result.result))
        print()


if __name__ == "__main__":
    asyncio.run(main())
