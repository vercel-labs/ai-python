"""A minimal Slack agent: mention it, it streams a reply in the thread.

The whole app is this file: one ``app_mention`` listener that rebuilds
the conversation from the Slack thread, runs an agent turn, and
forwards ``TextDelta`` chunks into Slack's native message-streaming
API (``chat.startStream`` / ``chat.appendStream`` / ``chat.stopStream``).

Socket Mode: the process dials out to Slack over a WebSocket, so there
is no public URL, tunnel, or hosting involved — the bot is online while
this process runs. There is no local state either: the thread itself is
the conversation store, so restarts lose nothing.

Run it from this directory:

    SLACK_BOT_TOKEN=xoxb-... SLACK_APP_TOKEN=xapp-... uv run main.py
"""

import asyncio
import json
import os
import re
from typing import Any

import slack_bolt.adapter.socket_mode.async_handler
import slack_bolt.app.async_app
import slack_bolt.context.async_context
import slack_sdk.web.async_client

import ai

MODEL_ID = os.environ.get("MODEL_ID", "anthropic/claude-sonnet-4.6")

SYSTEM_PROMPT = """\
You are a helpful assistant in a Slack channel.  Keep replies short and
conversational; use markdown sparingly.  The conversation may involve
several people — reply to the last message, using the rest of the
thread as context.
"""

# How much streamed text to buffer per Slack API call. Larger values
# make fewer, chunkier updates; the stream methods are Tier 2 rate
# limited (~20 calls/min), so don't go much lower.
STREAM_BUFFER = 400


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72F in {city}"


app = slack_bolt.app.async_app.AsyncApp(token=os.environ["SLACK_BOT_TOKEN"])
model = ai.get_model(MODEL_ID)
agent = ai.Agent(tools=[get_weather])

# One turn at a time per thread: a second mention while the first is
# still streaming waits its turn (and then sees the finished reply in
# the thread history it fetches).
_thread_locks: dict[str, asyncio.Lock] = {}


def _tool_line(part: ai.messages.ToolCallPart) -> str:
    try:
        args: dict[str, Any] = json.loads(part.tool_args or "{}")
    except json.JSONDecodeError:
        args = {}
    rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"\n_→ {part.tool_name}({rendered})_\n"


async def _thread_messages(
    client: slack_sdk.web.async_client.AsyncWebClient,
    channel: str,
    thread_ts: str,
    bot_user_id: str | None,
) -> list[ai.messages.Message]:
    """Rebuild agent messages from the Slack thread.

    The bot's own messages become assistant messages, everyone else's
    become user messages. For a top-level mention the "thread" is just
    the mention itself.
    """
    replies = await client.conversations_replies(
        channel=channel, ts=thread_ts, limit=50
    )
    messages = [ai.system_message(SYSTEM_PROMPT)]
    for reply in replies.get("messages") or []:
        text = re.sub(rf"<@{bot_user_id}>", "", reply.get("text", "")).strip()
        if not text:
            continue
        if reply.get("user") == bot_user_id:
            messages.append(ai.assistant_message(text))
        else:
            messages.append(ai.user_message(text))
    return messages


@app.event("app_mention")
async def on_mention(
    event: dict[str, Any],
    client: slack_sdk.web.async_client.AsyncWebClient,
    context: slack_bolt.context.async_context.AsyncBoltContext,
) -> None:
    channel = event["channel"]
    # A mention inside a thread replies to that thread; a top-level
    # mention starts a thread rooted at the mention itself.
    thread_ts = event.get("thread_ts") or event["ts"]

    lock = _thread_locks.setdefault(f"{channel}:{thread_ts}", asyncio.Lock())
    async with lock:
        messages = await _thread_messages(
            client, channel, thread_ts, context.bot_user_id
        )

        # Streamed messages are always threaded replies; streaming into
        # a channel requires the recipient ids (the "responding to @…"
        # attribution shown while streaming).
        streamer = await client.chat_stream(
            channel=channel,
            thread_ts=thread_ts,
            recipient_user_id=event["user"],
            recipient_team_id=context.team_id,
            buffer_size=STREAM_BUFFER,
        )
        try:
            async with agent.run(model, messages) as run:
                async for run_event in run:
                    if isinstance(run_event, ai.events.TextDelta):
                        await streamer.append(markdown_text=run_event.chunk)
                    elif isinstance(run_event, ai.events.ToolEnd):
                        await streamer.append(
                            markdown_text=_tool_line(run_event.tool_call)
                        )
        except Exception as error:
            await streamer.stop(
                markdown_text=f"\n:warning: `{type(error).__name__}: {error}`"
            )
            raise
        await streamer.stop()


async def main() -> None:
    handler = (
        slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler(
            app, os.environ["SLACK_APP_TOKEN"]
        )
    )
    await handler.start_async()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    asyncio.run(main())
