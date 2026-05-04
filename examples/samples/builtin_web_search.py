"""Anthropic built-in web search.
https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool
"""

import asyncio
import json

import ai

model = ai.anthropic("claude-sonnet-4-6")

messages = [
    ai.system_message("Be concise. Cite sources you use. The year is 2026"),
    ai.user_message(
        "Who won the most recent Formula 1 Grand Prix, and where was it held?"
    ),
]

tools = [ai.anthropic.tools.web_search(max_uses=3)]


_ENCRYPTED_KEYS = frozenset({"encrypted_content", "encrypted_stdout"})


def _strip_encrypted(value: object) -> object:
    """Some fields are encrypted, strip them for clarity"""
    if isinstance(value, dict):
        return {
            k: _strip_encrypted(v) for k, v in value.items() if k not in _ENCRYPTED_KEYS
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
    async with ai.stream(model, messages, tools=tools) as s:
        async for event in s:
            match event:
                case ai.TextDelta():
                    print(event.chunk, end="", flush=True)
                case ai.types.BuiltinToolEnd():
                    args = json.loads(event.tool_call.tool_args or "{}")
                    print(f"\n[{event.tool_call.tool_name}] input:")
                    print(format(args))
                case ai.types.BuiltinToolResult():
                    print(f"\n[{event.result.tool_name}] result:")
                    print(format(event.result.result))
        print()


if __name__ == "__main__":
    asyncio.run(main())
