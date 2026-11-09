"""The Claude worker against a scripted claude-agent-sdk client, offline."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pydantic
import pytest

claude_agent_sdk = pytest.importorskip("claude_agent_sdk")

import ai  # noqa: E402
from ai import workers  # noqa: E402
from ai.types import events  # noqa: E402
from ai.types import messages as m  # noqa: E402
from ai.workers import _adapter  # noqa: E402
from ai.workers.claude_code import worker as worker_  # noqa: E402


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

    async def receive_messages(self) -> AsyncIterator[Any]:
        while self.replies:
            for message in self.replies.pop(0):
                yield message

    async def interrupt(self) -> None:
        self.interrupted = True

    async def disconnect(self) -> None:
        pass


def _worker(tmp_path: Path, **options: Any) -> workers.ClaudeCode:
    return workers.ClaudeCode(
        cwd=str(tmp_path),
        env={"CLAUDE_CONFIG_DIR": str(tmp_path / "config")},
        **options,
    )


async def _turn(worker: workers.ClaudeCode, *prompt: Any) -> list[Any]:
    async with asyncio.timeout(5):
        return [e async for e in worker._cli_turn(list(prompt))]


async def test_a_turn_streams_in_the_model_view(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    worker._client = Client(TURN)
    seen = await _turn(worker, m.TextPart(text="what is here?"))
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
    assert worker._client.queries == ["what is here?"]
    end = seen[-1]
    assert end.finish_reason == "stop"
    assert end.usage.input_tokens == 7
    assert end.usage.raw["cost_usd"] == 0.01
    delta = next(e for e in seen if isinstance(e, events.BuiltinToolDelta))
    assert json.loads(delta.chunk) == {"command": "ls"}
    result = next(e for e in seen if isinstance(e, events.BuiltinToolResult))
    assert (result.result.tool_name, result.result.result) == ("Bash", "a.py")


async def test_files_travel_as_base64_blocks(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    worker._client = Client([_result()])
    png = m.FilePart(data=b"\x89PNG", media_type="image/png")
    await _turn(worker, m.TextPart(text="look"), png)
    (sent,) = worker._client.queries
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
    worker = _worker(tmp_path)
    worker._client = Client([_result()])
    zipped = m.FilePart(data=b"PK", media_type="application/zip")
    with pytest.raises(workers.WorkerError, match="application/zip"):
        await _turn(worker, zipped)


async def test_finish_reasons(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    worker._client = Client([_result(subtype="error_max_turns")])
    assert (await _turn(worker, m.TextPart(text="x")))[-1].finish_reason == (
        "length"
    )
    worker._client = Client(
        [_result(is_error=True, result="Invalid API key · Please run /login")]
    )
    with pytest.raises(workers.NotAuthenticatedError):
        await _turn(worker, m.TextPart(text="x"))
    worker._client = Client([_result(is_error=True, result="overloaded")])
    with pytest.raises(workers.WorkerError, match="overloaded"):
        await _turn(worker, m.TextPart(text="x"))


@pytest.mark.parametrize(
    ("detail", "error_type"),
    [
        ("Invalid API key · Please run /login", workers.NotAuthenticatedError),
        ("overloaded", workers.WorkerError),
    ],
)
@pytest.mark.parametrize("partial", [False, True])
async def test_public_generate_raises_native_failure(
    tmp_path: Path,
    detail: str,
    error_type: type[workers.WorkerError],
    partial: bool,
) -> None:
    worker = _worker(tmp_path)
    # Exercise the real public model/worker path. Only the CLI lifecycle
    # and output are mocked; a terminal event must not hide its failure.
    replies = [*TURN[:-1]] if partial else []
    replies.append(_result(is_error=True, result=detail))
    client = Client(replies)
    worker._client = client
    with (
        mock.patch.object(
            workers.ClaudeCode, "_cli_start", new_callable=mock.AsyncMock
        ),
        mock.patch.object(
            workers.ClaudeCode,
            "_cli_open",
            new_callable=mock.AsyncMock,
            return_value="session",
        ),
    ):
        async with asyncio.timeout(5), worker:
            with pytest.raises(error_type, match=detail):
                await ai.experimental_generate(
                    worker, [ai.user_message("hello")]
                )
            assert client.queries == ["hello"]
            assert worker._pending_results == 0
            assert not worker._running


async def test_an_interrupted_turn_is_cancelled_not_failed(
    tmp_path: Path,
) -> None:
    worker = _worker(tmp_path)
    # The CLI's error result after the interrupt is read first, then the
    # next turn settles on its own result.
    client = Client([_result(is_error=True, result="interrupted")], [_result()])
    worker._client = client
    worker._running = True  # a turn was running when it was interrupted
    await worker._cli_interrupt()
    assert client.interrupted
    seen = await _turn(worker, m.TextPart(text="again"))
    assert seen[-1].finish_reason == "stop"
    assert client.replies == []


async def test_a_seed_is_written_where_the_cli_reads(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
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
    session_id = await worker._stage(seed)
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


def test_a_stored_result_joins_the_message_of_its_call() -> None:
    """On disk the result is in the next user record, without a name."""
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
    history = worker_._read(records)
    assert [x.role for x in history] == ["user", "assistant"]
    call, result = history[1].parts
    assert isinstance(call, m.BuiltinToolCallPart)
    assert isinstance(result, m.BuiltinToolReturnPart)
    assert (result.tool_name, result.result) == ("Read", "x")


def test_the_tui_command(tmp_path: Path) -> None:
    worker = _worker(
        tmp_path,
        model_name="sonnet",
        mcp_servers={
            "ai_sdk": {"type": "http", "url": "http://127.0.0.1:1/mcp"},
            "docs": {"command": "docs-mcp"},
        },
    )
    argv, env, minted = worker._cli_tui_command("abc")
    assert argv[:3] == ["claude", "--resume", "abc"]
    assert argv[argv.index("--model") + 1] == "sonnet"
    config = json.loads(argv[argv.index("--mcp-config") + 1])
    assert list(config["mcpServers"]) == ["docs"]  # no bridge in a TUI
    assert env["DISABLE_AUTOUPDATER"] == "1"
    assert minted == "abc"
    argv, _, minted = worker._cli_tui_command(None)
    assert argv[1] == "--session-id" and argv[2] == minted


def test_worker_fields_and_private_state(tmp_path: Path) -> None:
    worker = _worker(tmp_path, model_name="sonnet", allowed_tools=[])
    worker._client = Client()
    worker._running = True
    dumped = worker.model_dump()
    assert dumped["worker_class_id"] == "claude-code"
    assert dumped["model_name"] == "sonnet"
    assert dumped["allowed_tools"] == []
    assert "kind" not in dumped and "model" not in dumped
    assert not any(key.startswith("_") for key in dumped)
    restored = workers.ClaudeCode.model_validate(dumped)
    assert restored._client is None and not restored._running
    assert "model_name" not in workers.ClaudeCode.__annotations__
    assert "_make_adapter" not in workers.ClaudeCode.__dict__
    assert worker._approve is not None  # still the base's method
    with pytest.raises(pydantic.ValidationError):
        _worker(tmp_path, worker_class_id="codex")


async def test_stream_start_acknowledges_query_not_response(
    tmp_path: Path,
) -> None:
    worker = _worker(tmp_path)
    entered, accepted = asyncio.Event(), asyncio.Event()

    class SlowQuery(Client):
        async def query(self, prompt: Any) -> None:
            entered.set()
            await accepted.wait()
            await super().query(prompt)

    client = SlowQuery([_result()])
    worker._client = client
    stream = worker._cli_turn([m.TextPart(text="hello")])
    async with asyncio.timeout(5):
        first = asyncio.create_task(anext(stream))
        try:
            await entered.wait()
            assert not first.done()
            assert not worker._running
            accepted.set()
            assert isinstance(await first, events.StreamStart)
            assert client.queries == ["hello"]
            # No response was read to acknowledge the send.
            assert len(client.replies) == 1
            assert worker._running
            assert isinstance(await anext(stream), events.StreamEnd)
        finally:
            accepted.set()
            await first
            await stream.aclose()
    assert not worker._running


async def test_query_failure_does_not_acknowledge(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    worker._client = mock.Mock(
        query=mock.AsyncMock(
            side_effect=claude_agent_sdk.CLIConnectionError("gone")
        )
    )
    stream = worker._cli_turn([m.TextPart(text="hello")])
    with pytest.raises(workers.WorkerError, match="lost the CLI connection"):
        await anext(stream)
    assert not worker._running


async def test_closing_turn_clears_running_flag(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    worker._client = Client()
    stream = worker._cli_turn([m.TextPart(text="hello")])
    assert isinstance(await anext(stream), events.StreamStart)
    assert worker._running
    await stream.aclose()
    assert not worker._running


async def test_running_send_preserves_text_and_files(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    worker._client = Client()
    worker._running = True
    assert await worker._cli_send_running([m.TextPart(text="first")])
    assert await worker._cli_send_running(
        [
            m.TextPart(text="look"),
            m.FilePart(data=b"notes", media_type="text/plain"),
            m.FilePart(data=b"\x89PNG", media_type="image/png"),
            m.FilePart(data=b"%PDF", media_type="application/pdf"),
        ]
    )
    text, blocks = worker._client.queries
    assert text == "first"
    (record,) = [record async for record in blocks]
    assert record["message"]["content"] == [
        {"type": "text", "text": "look"},
        {"type": "text", "text": "<file name='attached file'>\nnotes\n</file>"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "iVBORw==",
            },
        },
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBERg==",
            },
        },
    ]


async def test_running_send_rejects_unsupported_files(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    worker._client = Client()
    worker._running = True
    with pytest.raises(workers.WorkerError, match="application/zip"):
        await worker._cli_send_running(
            [m.FilePart(data=b"PK", media_type="application/zip")]
        )
    assert worker._client.queries == []


async def test_running_send_returns_false_only_before_writing(
    tmp_path: Path,
) -> None:
    worker = _worker(tmp_path)
    client = Client()
    worker._client = client
    prompt: _adapter.Prompt = [m.TextPart(text="next")]
    assert not await worker._cli_send_running(prompt)
    assert client.queries == []
    worker._running = True

    async def finish_while_reading(prompt: Any) -> str:
        worker._running = False
        return "next"

    with mock.patch.object(worker_, "_content", finish_while_reading):
        assert not await worker._cli_send_running(prompt)
    assert client.queries == []


@pytest.mark.parametrize("accepted", [False, True])
async def test_result_during_query_keeps_reader_until_accepted_input_settles(
    tmp_path: Path,
    accepted: bool,
) -> None:
    worker = _worker(tmp_path)
    writing, release, result_read = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    output: asyncio.Queue[Any] = asyncio.Queue()

    class DuplexClient(Client):
        closed_readers = 0

        async def query(self, prompt: Any) -> None:
            if prompt == "second":
                writing.set()
                await release.wait()
                if not accepted:
                    raise claude_agent_sdk.CLIConnectionError("write failed")
            await super().query(prompt)

        async def receive_messages(self) -> AsyncIterator[Any]:
            try:
                while True:
                    message = await output.get()
                    if isinstance(message, claude_agent_sdk.ResultMessage):
                        result_read.set()
                    yield message
            finally:
                self.closed_readers += 1

    client = DuplexClient()
    worker._client = client
    stream = worker._cli_turn([m.TextPart(text="first")])
    assert isinstance(await anext(stream), events.StreamStart)
    seen: list[events.Event] = []

    async def pump() -> None:
        async for event in stream:
            seen.append(event)

    async with asyncio.timeout(5):
        reader = asyncio.create_task(pump())
        sender = asyncio.create_task(
            worker._cli_send_running([m.TextPart(text="second")])
        )
        try:
            await writing.wait()
            output.put_nowait(_result())
            await result_read.wait()
            # The native result has arrived while query() is suspended.
            # Its accounting must wait for the write's acceptance decision.
            assert not reader.done() and not sender.done()
            assert worker._running and client.closed_readers == 0
            release.set()
            if accepted:
                assert await sender
                for message in TURN:
                    output.put_nowait(message)
            else:
                with pytest.raises(workers.WorkerError, match="write failed"):
                    await sender
            await reader
        finally:
            release.set()
            for task in (sender, reader):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sender, reader, return_exceptions=True)
            await stream.aclose()

    assert client.queries == (["first", "second"] if accepted else ["first"])
    boundaries = [
        type(e).__name__
        for e in seen
        if isinstance(e, events.StreamStart | events.StreamEnd)
    ]
    assert boundaries == (
        ["StreamEnd", "StreamStart", "StreamEnd"] if accepted else ["StreamEnd"]
    )
    if accepted:
        assert [e.chunk for e in seen if isinstance(e, events.TextDelta)] == [
            "one file"
        ]
    assert output.empty() and client.closed_readers == 1
    assert worker._pending_results == 0 and not worker._running
    assert not await worker._cli_send_running([m.TextPart(text="third")])
    output.put_nowait(_result())
    assert [
        type(e).__name__ for e in await _turn(worker, m.TextPart(text="third"))
    ] == ["StreamStart", "StreamEnd"]
    assert client.queries[-1] == "third"
    assert client.closed_readers == 2 and output.empty()


async def test_multiple_accepted_queries_have_separate_boundaries(
    tmp_path: Path,
) -> None:
    worker = _worker(tmp_path)
    client = Client([_result()], TURN, [_result()])
    worker._client = client
    stream = worker._cli_turn([m.TextPart(text="first")])
    assert isinstance(await anext(stream), events.StreamStart)
    assert await worker._cli_send_running([m.TextPart(text="second")])
    assert await worker._cli_send_running([m.TextPart(text="third")])
    assert worker._pending_results == 3
    async with asyncio.timeout(5):
        seen = [event async for event in stream]
    assert [
        type(e).__name__
        for e in seen
        if isinstance(e, events.StreamStart | events.StreamEnd)
    ] == ["StreamEnd", "StreamStart", "StreamEnd", "StreamStart", "StreamEnd"]
    assert client.queries == ["first", "second", "third"]
    assert client.replies == []
    assert worker._pending_results == 0 and not worker._running


@pytest.mark.parametrize("pending", [1, 2])
async def test_failed_response_raises_before_terminal_event_and_preserves_drain(
    tmp_path: Path,
    pending: int,
) -> None:
    worker = _worker(tmp_path)
    # Failed response, accepted follow-ups, then the NEXT query's response.
    client = Client(
        [_result(is_error=True, result="overloaded")],
        *[[_result()] for _ in range(pending)],
        TURN,
    )
    worker._client = client
    stream = worker._cli_turn([m.TextPart(text="first")])
    assert isinstance(await anext(stream), events.StreamStart)
    for index in range(pending):
        assert await worker._cli_send_running(
            [m.TextPart(text=f"follow-up {index}")]
        )
    seen: list[events.Event] = []
    with pytest.raises(workers.WorkerError, match="overloaded"):
        async for event in stream:
            seen.append(event)
    # No terminal event can hide the exception from a public stream reader.
    assert seen == []
    assert not worker._running and worker._interrupted
    assert worker._pending_results == pending
    assert len(client.replies) == pending + 1
    assert not await worker._cli_send_running([m.TextPart(text="next")])
    assert len(client.queries) == pending + 1

    # Resume drains only the already accepted replies, then sends once.
    next_turn = await _turn(worker, m.TextPart(text="next"))
    assert [e.chunk for e in next_turn if isinstance(e, events.TextDelta)] == [
        "one file"
    ]
    assert next_turn[-1].finish_reason == "stop"
    assert client.queries[-1] == "next" and len(client.queries) == pending + 2
    assert client.replies == [] and worker._pending_results == 0
    assert not worker._running and not worker._interrupted


async def test_early_output_end_does_not_silently_drop_accepted_input(
    tmp_path: Path,
) -> None:
    worker = _worker(tmp_path)
    worker._client = Client([_result()])
    stream = worker._cli_turn([m.TextPart(text="first")])
    assert isinstance(await anext(stream), events.StreamStart)
    assert await worker._cli_send_running([m.TextPart(text="second")])
    with pytest.raises(
        workers.WorkerError, match="1 response\\(s\\) outstanding"
    ):
        async for _ in stream:
            pass
    assert not worker._running


async def test_send_after_result_before_reader_exit_writes_nothing(
    tmp_path: Path,
) -> None:
    worker = _worker(tmp_path)
    client = Client([_result()], [_result()])
    worker._client = client
    stream = worker._cli_turn([m.TextPart(text="first")])
    assert isinstance(await anext(stream), events.StreamStart)
    assert isinstance(await anext(stream), events.StreamEnd)
    assert not await worker._cli_send_running([m.TextPart(text="second")])
    assert client.queries == ["first"]
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    await _turn(worker, m.TextPart(text="second"))
    assert client.queries == ["first", "second"]
    assert client.replies == [] and worker._pending_results == 0


async def test_interrupted_reader_drains_all_previously_accepted_responses(
    tmp_path: Path,
) -> None:
    worker = _worker(tmp_path)
    client = Client(
        [_result(is_error=True, result="interrupted")], [_result()], [_result()]
    )
    worker._client = client
    stream = worker._cli_turn([m.TextPart(text="first")])
    assert isinstance(await anext(stream), events.StreamStart)
    assert await worker._cli_send_running([m.TextPart(text="second")])
    await worker._cli_interrupt()
    await stream.aclose()
    assert worker._pending_results == 2
    seen = await _turn(worker, m.TextPart(text="third"))
    assert [type(event).__name__ for event in seen] == [
        "StreamStart",
        "StreamEnd",
    ]
    assert client.replies == [] and worker._pending_results == 0
    assert not worker._interrupted


async def test_running_send_failure_is_not_reported_as_idle(
    tmp_path: Path,
) -> None:
    worker = _worker(tmp_path)
    worker._running = True
    with pytest.raises(workers.WorkerError, match="no CLI connection"):
        await worker._cli_send_running([m.TextPart(text="x")])
    worker._client = mock.Mock(
        query=mock.AsyncMock(
            side_effect=claude_agent_sdk.CLIConnectionError("gone")
        )
    )
    with pytest.raises(workers.WorkerError, match="lost the CLI connection"):
        await worker._cli_send_running([m.TextPart(text="x")])


async def test_interrupt_reports_cancelled_on_current_stream(
    tmp_path: Path,
) -> None:
    worker = _worker(tmp_path)
    client = Client([_result(is_error=True, result="interrupted")])
    worker._client = client
    stream = worker._cli_turn([m.TextPart(text="x")])
    try:
        assert isinstance(await anext(stream), events.StreamStart)
        await worker._cli_interrupt()
        end = await anext(stream)
        assert isinstance(end, events.StreamEnd)
        assert end.finish_reason == "cancelled"
        assert client.interrupted
        assert not worker._interrupted
    finally:
        await stream.aclose()


@pytest.mark.parametrize("approve", [False, True])
async def test_start_connect_and_close(
    tmp_path: Path,
    approve: bool,
) -> None:
    worker = _worker(
        tmp_path,
        approve=approve,
        executable="/custom/claude",
        model_name="sonnet",
        effort="high",
        allowed_tools=[],
        disallowed_tools=["Bash"],
        max_turns=3,
    )
    servers = {"docs": {"command": "docs-mcp"}}
    client = mock.Mock(connect=mock.AsyncMock(), disconnect=mock.AsyncMock())
    with (
        mock.patch.object(
            _adapter, "probe", new_callable=mock.AsyncMock
        ) as probe,
        mock.patch.object(
            claude_agent_sdk, "ClaudeSDKClient", return_value=client
        ) as factory,
    ):
        probe.return_value = "2.1.289"
        await worker._cli_start(servers)
        assert worker._cli_version == "2.1.289"
        assert worker._approval_callback == (
            worker._approve if approve else None
        )
        probe.assert_awaited_once_with(
            "claude-code",
            "/custom/claude",
            install=worker_.INSTALL,
            verified=worker_.VERIFIED_VERSION,
            cwd=str(tmp_path.resolve()),
            env=worker.env,
        )
        sid = await worker._cli_open(
            session_id=None, fork=False, seed=None, instructions="Be brief"
        )
        options = factory.call_args.kwargs["options"]
        assert options.session_id == sid and options.resume is None
        assert options.cli_path == "/custom/claude"
        assert options.cwd == str(tmp_path.resolve())
        assert options.env == worker.env
        assert options.model == "sonnet"
        assert options.effort == "high"
        assert options.thinking == {"type": "adaptive"}
        assert options.tools == []
        assert options.disallowed_tools == ["Bash"]
        assert options.max_turns == 3
        assert options.mcp_servers == servers
        assert options.system_prompt["append"] == "Be brief"
        if approve:
            assert options.can_use_tool == worker._can_use_tool
            assert options.hooks["PreToolUse"][0].hooks == [
                worker._pre_tool_use
            ]
        else:
            assert options.permission_mode == "bypassPermissions"
            assert options.can_use_tool is None
        client.connect.assert_awaited_once()
        worker._running = worker._interrupted = True
        await worker._cli_close()
        await worker._cli_close()
        client.disconnect.assert_awaited_once()
        assert worker._client is None
        assert not worker._running and not worker._interrupted
        assert worker._approval_callback is None
        assert worker._instructions is None and worker._mcp_servers == {}


@pytest.mark.parametrize("granted", [False, True])
async def test_approval_callbacks_do_not_shadow_base_method(
    tmp_path: Path,
    granted: bool,
) -> None:
    worker = _worker(tmp_path, approve=True)
    approve = mock.AsyncMock(return_value=(granted, "policy"))
    worker._approval_callback = approve
    output = await worker._pre_tool_use(
        {"tool_name": "Read", "tool_input": {"path": "a"}}, "t1", None
    )
    assert output["hookSpecificOutput"]["permissionDecision"] == (
        "allow" if granted else "deny"
    )
    approve.assert_awaited_once_with("t1", "Read", {"path": "a"})
    result = await worker._can_use_tool(
        "Read", {"path": "a"}, claude_agent_sdk.ToolPermissionContext()
    )
    assert result.behavior == ("allow" if granted else "deny")
    assert callable(worker._approve)
    if not granted:
        assert (
            output["hookSpecificOutput"]["permissionDecisionReason"] == "policy"
        )
        assert isinstance(result, claude_agent_sdk.PermissionResultDeny)
        assert result.message == "policy"


@pytest.mark.parametrize("fork", [False, True])
async def test_open_resumes_or_natively_forks(
    tmp_path: Path, fork: bool
) -> None:
    worker = _worker(tmp_path)
    with (
        mock.patch.object(
            claude_agent_sdk, "get_session_info", return_value=object()
        ) as known,
        mock.patch.object(
            claude_agent_sdk,
            "fork_session",
            return_value=SimpleNamespace(session_id="child"),
        ) as native_fork,
        mock.patch.object(
            workers.ClaudeCode, "_connect", new_callable=mock.AsyncMock
        ) as connect,
    ):
        sid = await worker._cli_open(
            session_id="parent", fork=fork, seed=None, instructions=None
        )
        assert sid == ("child" if fork else "parent")
        known.assert_called_once_with("parent", str(tmp_path.resolve()))
        connect.assert_awaited_once_with(sid, resume=sid)
        if fork:
            native_fork.assert_called_once_with(
                "parent", str(tmp_path.resolve())
            )
        else:
            native_fork.assert_not_called()


async def test_unknown_session_is_rejected_before_connect(
    tmp_path: Path,
) -> None:
    worker = _worker(tmp_path)
    with (
        mock.patch.object(
            claude_agent_sdk, "get_session_info", return_value=None
        ),
        mock.patch.object(
            workers.ClaudeCode, "_connect", new_callable=mock.AsyncMock
        ) as connect,
    ):
        with pytest.raises(workers.WorkerError, match="no session ghost"):
            await worker._cli_open(
                session_id="ghost", fork=False, seed=None, instructions=None
            )
        connect.assert_not_awaited()


async def test_history_and_session_listing_use_the_cli_store(
    tmp_path: Path,
) -> None:
    worker = _worker(tmp_path)
    record = SimpleNamespace(
        type="user", message={"role": "user", "content": "hello"}
    )
    info = SimpleNamespace(
        session_id="s",
        custom_title=None,
        summary="summary",
        first_prompt="hello",
        cwd=str(tmp_path),
        created_at=10,
        last_modified=20,
    )
    with (
        mock.patch.object(
            claude_agent_sdk, "get_session_info", return_value=info
        ),
        mock.patch.object(
            claude_agent_sdk, "get_session_messages", return_value=[record]
        ) as history,
        mock.patch.object(
            claude_agent_sdk, "list_sessions", return_value=[info]
        ) as sessions,
    ):
        (message,) = await worker._cli_history("s")
        assert message.role == "user"
        (part,) = message.parts
        assert isinstance(part, m.TextPart) and part.text == "hello"
        (session,) = await worker._cli_list_sessions()
        assert session.session_id == "s" and session.title == "summary"
        assert session.created_at == 10 and session.updated_at == 20
        history.assert_called_once_with("s", str(tmp_path.resolve()))
        sessions.assert_called_once_with(str(tmp_path.resolve()))


def test_tui_needs_no_open_or_optional_sdk(tmp_path: Path) -> None:
    worker = _worker(
        tmp_path,
        executable="/custom/claude",
        allowed_tools=[],
        disallowed_tools=["Bash"],
    )
    with mock.patch.object(
        worker_, "_sdk", side_effect=AssertionError("SDK imported")
    ):
        argv, env, sid = worker._cli_tui_command(None)
    assert argv[:3] == ["/custom/claude", "--session-id", sid]
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--disallowedTools") + 1] == "Bash"
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "config")
    assert worker._client is None
