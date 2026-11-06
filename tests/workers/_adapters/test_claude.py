"""The claude adapter against a scripted claude-agent-sdk client, offline."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

claude_agent_sdk = pytest.importorskip("claude_agent_sdk")

from ai import workers  # noqa: E402
from ai.types import events  # noqa: E402
from ai.types import messages as m  # noqa: E402
from ai.workers._adapters import claude  # noqa: E402


def _stream(event: dict[str, Any]) -> Any:
    return claude_agent_sdk.StreamEvent(uuid="u", session_id="s", event=event)


def _result(**fields: Any) -> Any:
    return claude_agent_sdk.ResultMessage(
        subtype=fields.pop("subtype", "success"),
        duration_ms=1,
        duration_api_ms=1,
        is_error=fields.pop("is_error", False),
        num_turns=1,
        session_id="s",
        usage={"input_tokens": 7, "output_tokens": 3},
        total_cost_usd=0.01,
        **fields,
    )


TURN = [
    _stream(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking"},
        }
    ),
    _stream(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "plan"},
        }
    ),
    _stream({"type": "content_block_stop", "index": 0}),
    claude_agent_sdk.AssistantMessage(
        content=[
            claude_agent_sdk.ToolUseBlock(
                id="t1", name="Bash", input={"command": "ls"}
            )
        ],
        model="claude",
    ),
    claude_agent_sdk.UserMessage(
        content=[
            claude_agent_sdk.ToolResultBlock(tool_use_id="t1", content="a.py")
        ]
    ),
    _stream(
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text"},
        }
    ),
    _stream(
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "one file"},
        }
    ),
    _stream({"type": "content_block_stop", "index": 1}),
    claude_agent_sdk.AssistantMessage(
        content=[claude_agent_sdk.TextBlock(text="one file")], model="claude"
    ),
    _result(),
]


class Client:
    """Answers like `ClaudeSDKClient`: a query, then the scripted reply."""

    def __init__(self, *replies: list[Any]) -> None:
        self.replies = list(replies)
        self.queries: list[Any] = []
        self.interrupted = False

    async def query(self, prompt: Any) -> None:
        self.queries.append(prompt)

    async def receive_response(self) -> AsyncIterator[Any]:
        for message in self.replies.pop(0):
            yield message

    async def interrupt(self) -> None:
        self.interrupted = True

    async def disconnect(self) -> None:
        pass


def _adapter(tmp_path: Path, **options: Any) -> claude.ClaudeAdapter:
    return claude.ClaudeAdapter(
        cwd=str(tmp_path),
        env={"CLAUDE_CONFIG_DIR": str(tmp_path / "config")},
        executable="claude",
        model=options.get("model"),
        effort=None,
        mcp_servers=options.get("mcp_servers", {}),
        approve=None,
        allowed_tools=None,
        disallowed_tools=None,
        max_turns=None,
    )


async def _turn(adapter: claude.ClaudeAdapter, *prompt: Any) -> list[Any]:
    async with asyncio.timeout(5):
        return [e async for e in adapter.turn(list(prompt))]


async def test_a_turn_streams_in_the_model_view(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter._client = Client(TURN)
    seen = await _turn(adapter, m.TextPart(text="what is here?"))
    assert [type(e).__name__ for e in seen] == [
        "StreamStart",
        "ReasoningStart",
        "ReasoningDelta",
        "ReasoningEnd",
        "BuiltinToolStart",
        "BuiltinToolDelta",
        "BuiltinToolEnd",
        "BuiltinToolResult",
        "TextStart",
        "TextDelta",
        "TextEnd",
        "StreamEnd",
    ]
    assert adapter._client.queries == ["what is here?"]
    end = seen[-1]
    assert end.finish_reason == "stop"
    assert end.usage.input_tokens == 7
    assert end.usage.raw["cost_usd"] == 0.01
    delta = next(e for e in seen if isinstance(e, events.BuiltinToolDelta))
    assert json.loads(delta.chunk) == {"command": "ls"}
    result = next(e for e in seen if isinstance(e, events.BuiltinToolResult))
    assert (result.result.tool_name, result.result.result) == ("Bash", "a.py")


async def test_files_travel_as_base64_blocks(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter._client = Client([_result()])
    png = m.FilePart(data=b"\x89PNG", media_type="image/png")
    await _turn(adapter, m.TextPart(text="look"), png)
    (sent,) = adapter._client.queries
    (record,) = [record async for record in sent]
    message = record["message"]
    assert message["content"][1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORw==",
        },
    }


async def test_an_unknown_file_is_refused(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter._client = Client([_result()])
    zipped = m.FilePart(data=b"PK", media_type="application/zip")
    with pytest.raises(workers.WorkerError, match="application/zip"):
        await _turn(adapter, zipped)


async def test_finish_reasons(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter._client = Client([_result(subtype="error_max_turns")])
    assert (await _turn(adapter, m.TextPart(text="x")))[-1].finish_reason == (
        "length"
    )
    adapter._client = Client(
        [_result(is_error=True, result="Invalid API key · Please run /login")]
    )
    with pytest.raises(workers.NotAuthenticatedError):
        await _turn(adapter, m.TextPart(text="x"))
    adapter._client = Client([_result(is_error=True, result="overloaded")])
    with pytest.raises(workers.WorkerError, match="overloaded"):
        await _turn(adapter, m.TextPart(text="x"))


async def test_an_interrupted_turn_is_cancelled_not_failed(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    # The CLI's error result after the interrupt is read first, then the
    # next turn settles on its own result.
    client = Client([_result(is_error=True, result="interrupted")], [_result()])
    adapter._client = client
    adapter._running = True  # a turn was running when it was interrupted
    await adapter.interrupt()
    assert client.interrupted
    seen = await _turn(adapter, m.TextPart(text="again"))
    assert seen[-1].finish_reason == "stop"
    assert client.replies == []


async def test_a_seed_is_written_where_the_cli_reads(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    seed = [
        m.Message(role="user", parts=[m.TextPart(text="earlier")]),
        m.Message(
            role="assistant",
            parts=[
                m.BuiltinToolCallPart(
                    tool_call_id="t1", tool_name="Read", tool_args="{}"
                ),
                m.BuiltinToolReturnPart(
                    tool_call_id="t1", tool_name="Read", result="milk"
                ),
                m.TextPart(text="It says milk."),
            ],
        ),
    ]
    session_id = await adapter._stage(seed)
    key = claude_agent_sdk.project_key_for_directory(str(tmp_path))
    path = tmp_path / "config" / "projects" / key / f"{session_id}.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["type"] for r in records] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert records[1]["message"]["content"][0]["type"] == "tool_use"
    assert records[2]["message"]["content"][0]["type"] == "tool_result"


def test_a_stored_transcript_reads_as_the_model_view() -> None:
    records = [
        claude_agent_sdk.SessionMessage(
            type="user",
            uuid="1",
            session_id="s",
            message={"role": "user", "content": "q"},
            parent_tool_use_id=None,
        ),
        claude_agent_sdk.SessionMessage(
            type="assistant",
            uuid="2",
            session_id="s",
            message={
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Read",
                        "input": {},
                    }
                ],
            },
            parent_tool_use_id=None,
        ),
        claude_agent_sdk.SessionMessage(
            type="user",
            uuid="3",
            session_id="s",
            message={
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "x"}
                ],
            },
            parent_tool_use_id=None,
        ),
    ]
    from ai.workers._adapters import _handoff

    stored = [msg for r in records for msg in claude._stored_to_messages(r)]
    history = _handoff.model_view(stored)
    assert [x.role for x in history] == ["user", "assistant"]
    call, result = history[1].parts
    assert isinstance(call, m.BuiltinToolCallPart)
    assert isinstance(result, m.BuiltinToolReturnPart)
    assert (result.tool_name, result.result) == ("Read", "x")


def test_the_tui_command(tmp_path: Path) -> None:
    adapter = _adapter(
        tmp_path,
        model="sonnet",
        mcp_servers={
            "ai_sdk": {"type": "http", "url": "http://127.0.0.1:1/mcp"},
            "docs": {"command": "docs-mcp"},
        },
    )
    argv, env, minted = adapter.tui_command("abc")
    assert argv[:3] == ["claude", "--resume", "abc"]
    assert argv[argv.index("--model") + 1] == "sonnet"
    config = json.loads(argv[argv.index("--mcp-config") + 1])
    assert list(config["mcpServers"]) == ["docs"]  # no bridge in a TUI
    assert env["DISABLE_AUTOUPDATER"] == "1"
    assert minted == "abc"
    argv, _, minted = adapter.tui_command(None)
    assert argv[1] == "--session-id" and argv[2] == minted
