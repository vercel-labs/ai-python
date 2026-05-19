"""Absolutely minimal 'coding agent'."""

import asyncio
import json
import sys

import ai

SYSTEM_PROMPT = """
You are a coding agent in an incredibly minimal harness.

Your only tool is executing shell commands. Use `cat -n` to read
files, `cat` with a heredoc to write new files, and `sed -i
'<lineno>s/old/new/'` to make edits to files. After making an edit
with sed, make sure to double check the result.
"""

STREAM_PARAMS = ai.InferenceRequestParams(cache=ai.CacheParams(mode="auto"))


@ai.tool
async def shell(cmd: str) -> str:
    """Run a command using sh -c.

    Returns stdin and stdout, joined.
    """
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    stdout, _ = await proc.communicate()

    return stdout.decode("utf-8", errors="replace")


model = ai.get_model("gateway:anthropic/claude-opus-4.6")
agent = ai.agent(tools=[shell])


async def step(
    messages: list[ai.messages.Message],
) -> list[ai.messages.Message]:
    async with agent.run(model, messages, params=STREAM_PARAMS) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
            elif isinstance(event, ai.events.StreamEnd):
                print()
            elif isinstance(event, ai.events.ToolEnd):
                args = json.loads(event.tool_call.tool_args)
                print("\nSHELL:", args["cmd"])
            elif isinstance(event, ai.events.ToolCallResult):
                for res in event.results:
                    print("RESULT:\n", res.result)
                    print("=======")

        print()
        return stream.messages


def main() -> None:
    messages = [ai.system_message(SYSTEM_PROMPT)]

    while True:
        print("> ", end="", flush=True)
        s = sys.stdin.readline()
        if not s:
            return
        if not s.strip():
            continue

        messages += [ai.user_message(s)]
        messages = asyncio.run(step(messages))


if __name__ == "__main__":
    main()
