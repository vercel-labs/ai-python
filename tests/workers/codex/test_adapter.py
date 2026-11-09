"""The Codex worker transport against a scripted `codex app-server`, offline."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, cast

import pytest

from ai import workers
from ai.types import events
from ai.types import messages as m
from ai.workers import _adapter
from ai.workers.codex import jsonrpc

from .. import fakes

THREAD = "T"
CWD = "/w"


class AppServer:
    """Answers like `codex app-server`. `turn` plays out after turn/start."""

    def __init__(self, turn: list[dict[str, Any]] | None = None) -> None:
        self.process = fakes.FakeProcess(self.answer)
        self.turn = turn or []
        self.on_turn: Callable[[], None] | None = None
        self.threads: dict[str, dict[str, Any]] = {
            THREAD: {"id": THREAD, "cwd": CWD, "turns": []}
        }

    def says(self, method: str, **params: Any) -> None:
        self.process.server_says({"method": method, "params": params})

    def answer(self, message: dict[str, Any]) -> Any:
        method = message.get("method")
        params = message.get("params") or {}
        if method == "turn/start":
            if self.on_turn is not None:
                self.on_turn()
            for record in self.turn:
                self.process.server_says(record)
            return {"turn": {"id": "turn-1"}}
        if method == "turn/interrupt":
            self.says(
                "turn/completed",
                threadId=THREAD,
                turn={"id": "turn-1", "status": "interrupted"},
            )
            return {}
        if method == "thread/read":
            return {"thread": self.threads.get(str(params.get("threadId")), {})}
        if method in ("thread/start", "thread/fork"):
            return {"thread": {"id": "new"}}
        if method == "thread/resume":
            return {"thread": {"id": params["threadId"]}}
        if "method" in message and "id" in message:
            return {}
        return None

    def sent(self, method: str) -> list[dict[str, Any]]:
        return [w for w in self.process.written if w.get("method") == method]


def note(method: str, **params: Any) -> dict[str, Any]:
    return {
        "method": method,
        "params": {"threadId": THREAD, "turnId": "turn-1", **params},
    }


COMMAND = {
    "type": "commandExecution",
    "id": "c1",
    "command": "/bin/zsh -lc 'cat secret.txt'",
    "aggregatedOutput": "the word is HERON\n",
    "exitCode": 0,
    "status": "completed",
}

TURN = [
    note("item/reasoning/summaryTextDelta", itemId="r1", delta="plan"),
    note("item/completed", item={"type": "reasoning", "id": "r1"}),
    note("item/started", item={**COMMAND, "aggregatedOutput": None}),
    note("item/completed", item=COMMAND),
    note("item/agentMessage/delta", itemId="a1", delta="HER"),
    note("item/agentMessage/delta", itemId="a1", delta="ON"),
    note("item/completed", item={"type": "agentMessage", "id": "a1"}),
    note(
        "thread/tokenUsage/updated",
        tokenUsage={"lastTurn": {"inputTokens": 10, "outputTokens": 4}},
    ),
    note("turn/completed", turn={"id": "turn-1", "status": "completed"}),
]


def _worker(
    server: AppServer, approve: _adapter.Approve | None = None, **options: Any
) -> workers.Codex:
    worker = workers.Codex(cwd=CWD, approve=approve is not None, **options)
    worker._approval_callback = approve
    peer = jsonrpc.JsonRpcPeer(cast("Any", server.process), kind="codex")
    worker._process = cast("Any", server.process)
    worker._peer = peer
    worker._thread = THREAD
    peer.start(worker._on_notify, worker._on_request, on_crash=worker._on_crash)
    return worker


async def _turn(worker: workers.Codex, text: str = "go") -> list[Any]:
    async with asyncio.timeout(5):
        return [e async for e in worker._cli_turn([m.TextPart(text=text)])]


async def test_a_turn_streams_in_the_model_view() -> None:
    worker = _worker(AppServer(TURN))
    seen = await _turn(worker)
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
        "TextDelta",
        "TextEnd",
        "StreamEnd",
    ]
    result = next(e for e in seen if isinstance(e, events.BuiltinToolResult))
    # codex 0.160 reports a command's output as `aggregatedOutput`.
    assert result.result.result == "the word is HERON\n"
    end = seen[-1]
    assert end.finish_reason == "stop"
    assert end.usage.input_tokens == 10
    hydrator = events.MessageHydrator()
    for event in seen:
        hydrator.feed(event)
    assert [type(p).__name__ for p in hydrator.message.parts] == [
        "ReasoningPart",
        "BuiltinToolCallPart",
        "BuiltinToolReturnPart",
        "TextPart",
    ]


async def test_a_message_that_was_never_streamed_still_arrives() -> None:
    server = AppServer(
        [
            note(
                "item/completed",
                item={"type": "agentMessage", "id": "a1", "text": "PONG"},
            ),
            note(
                "turn/completed", turn={"id": "turn-1", "status": "completed"}
            ),
        ]
    )
    seen = await _turn(_worker(server))
    assert [e.chunk for e in seen if isinstance(e, events.TextDelta)] == [
        "PONG"
    ]


async def test_another_turns_notifications_are_not_this_turns() -> None:
    server = AppServer(
        [
            {
                "method": "turn/completed",
                "params": {"threadId": THREAD, "turn": {"id": "old"}},
            },
            {
                "method": "item/agentMessage/delta",
                "params": {"threadId": "other", "itemId": "x", "delta": "no"},
            },
            note(
                "turn/completed", turn={"id": "turn-1", "status": "completed"}
            ),
        ]
    )
    seen = await _turn(_worker(server))
    assert [type(e).__name__ for e in seen] == ["StreamStart", "StreamEnd"]


async def test_a_failed_turn_without_a_key_is_not_authenticated() -> None:
    server = AppServer(
        [
            note("error", error={"message": "401 Unauthorized"}),
            note("turn/completed", turn={"id": "turn-1", "status": "failed"}),
        ]
    )
    with pytest.raises(workers.NotAuthenticatedError):
        await _turn(_worker(server))


@pytest.mark.parametrize("granted", [True, False])
async def test_approvals_ask_and_answer(granted: bool) -> None:
    server = AppServer()
    asked: list[tuple[str, str, dict[str, Any]]] = []

    async def approve(
        call_id: str, name: str, tool_input: dict[str, Any]
    ) -> tuple[bool, str]:
        asked.append((call_id, name, tool_input))
        return granted, "no"

    def ask() -> None:
        server.process.server_says(
            {
                "id": 500,
                "method": "item/commandExecution/requestApproval",
                "params": {
                    "threadId": THREAD,
                    "turnId": "turn-1",
                    "itemId": "i1",
                    "command": "rm -rf /",
                },
            }
        )

    def complete(message: dict[str, Any]) -> Any:
        if message.get("id") == 500:
            server.says(
                "turn/completed",
                threadId=THREAD,
                turn={"id": "turn-1", "status": "completed"},
            )
            return None
        return AppServer.answer(server, message)

    server.on_turn = ask
    server.process.answer = complete
    worker = _worker(server, approve)
    assert worker._thread_params()["approvalPolicy"] == "untrusted"
    await _turn(worker)
    assert asked == [("i1", "commandExecution", {"command": "rm -rf /"})]
    answer = next(w for w in server.process.written if w.get("id") == 500)
    assert answer["result"] == {"decision": "accept" if granted else "decline"}


async def test_without_a_gate_codex_never_asks() -> None:
    worker = _worker(AppServer())
    assert worker._thread_params() == {
        "sandbox": "danger-full-access",
        "approvalPolicy": "never",
    }


async def test_an_interrupted_turn_ends_as_cancelled() -> None:
    server = AppServer()
    worker = _worker(server)
    turn = worker._cli_turn([m.TextPart(text="go")])
    assert isinstance(await anext(turn), events.StreamStart)
    reader = asyncio.ensure_future(_rest(turn))
    await asyncio.sleep(0)
    await worker._cli_interrupt()
    seen = await asyncio.wait_for(reader, 5)
    assert seen[-1].finish_reason == "cancelled"
    assert server.sent("turn/interrupt")[0]["params"] == {
        "threadId": THREAD,
        "turnId": "turn-1",
    }


async def _rest(turn: Any) -> list[Any]:
    return [e async for e in turn]


async def test_opening_a_new_thread_seeds_it() -> None:
    server = AppServer()
    worker = _worker(server)
    seed = [
        m.Message(role="user", parts=[m.TextPart(text="earlier")]),
        m.Message(role="assistant", parts=[m.TextPart(text="answer")]),
    ]
    assert (
        await worker._cli_open(
            session_id=None, fork=False, seed=seed, instructions="Be terse."
        )
        == "new"
    )
    started = server.sent("thread/start")[0]["params"]
    assert started["cwd"] == CWD
    assert started["developerInstructions"] == "Be terse."
    injected = server.sent("thread/inject_items")[0]["params"]
    assert injected["threadId"] == "new"
    assert [i["role"] for i in injected["items"]] == ["user", "assistant"]


async def test_a_fork_is_native() -> None:
    server = AppServer()
    worker = _worker(server)
    forked = await worker._cli_open(
        session_id=THREAD, fork=True, seed=None, instructions=None
    )
    assert forked == "new"
    assert server.sent("thread/fork")[0]["params"]["threadId"] == THREAD


async def test_a_thread_from_another_directory_is_refused() -> None:
    server = AppServer()
    server.threads["elsewhere"] = {"id": "elsewhere", "cwd": "/other"}
    worker = _worker(server)
    with pytest.raises(workers.WorkerError, match="/other"):
        await worker._cli_open(
            session_id="elsewhere", fork=False, seed=None, instructions=None
        )


async def test_history_has_the_cli_tools_as_builtin_parts() -> None:
    server = AppServer()
    server.threads[THREAD]["turns"] = [
        {
            "items": [
                {
                    "type": "userMessage",
                    "content": [{"type": "text", "text": "q"}],
                },
                COMMAND,
                {"type": "agentMessage", "text": "HERON"},
            ]
        }
    ]
    history = await _worker(server)._cli_history(THREAD)
    assert [x.role for x in history] == ["user", "assistant", "assistant"]
    call, result = history[1].parts
    assert isinstance(call, m.BuiltinToolCallPart)
    assert json.loads(call.tool_args)["command"].endswith("'cat secret.txt'")
    assert isinstance(result, m.BuiltinToolReturnPart)
    assert result.result == "the word is HERON\n"


async def test_listing_asks_for_this_directory_and_reads_every_page() -> None:
    """`thread/list` is paged, and the store is the whole machine's."""
    asked: list[dict[str, Any]] = []
    pages = {
        None: ({"id": "t1", "cwd": CWD}, "p2"),
        "p2": ({"id": "t2", "cwd": CWD}, None),
    }
    server = AppServer()

    def answer(message: dict[str, Any]) -> Any:
        if message.get("method") != "thread/list":
            return AppServer.answer(server, message)
        params = message["params"]
        asked.append(params)
        thread, cursor = pages[params.get("cursor")]
        return {"data": [thread], "nextCursor": cursor}

    server.process.answer = answer
    infos = await _worker(server)._cli_list_sessions()
    assert [i.session_id for i in infos] == ["t1", "t2"]
    assert all(p["cwd"] == CWD for p in asked)
    assert [p.get("cursor") for p in asked] == [None, "p2"]


def test_the_tui_gets_no_bridge_before_open() -> None:
    worker = workers.Codex(
        mcp_servers={
            "ai_sdk": {"url": "http://127.0.0.1:1/mcp"},
            "docs": {"command": "docs-mcp"},
        },
    )
    argv, _, minted = worker._cli_tui_command(THREAD)
    assert argv[:3] == ["codex", "resume", THREAD]
    assert 'mcp_servers.docs.command="docs-mcp"' in argv
    assert not any("ai_sdk" in arg for arg in argv)
    assert minted is None  # codex picks its own ids
    assert worker._process is None


def test_tui_uses_public_settings_before_open() -> None:
    worker = workers.Codex(
        executable="/custom/codex",
        env={"CODEX_HOME": "/custom/home"},
        model_name="test-model",
        effort="max",
        config={"model_reasoning_effort": "low"},
    )
    argv, env, minted = worker._cli_tui_command(None)
    assert argv[0] == "/custom/codex"
    assert argv[-2:] == ["--model", "test-model"]
    assert 'model_reasoning_effort="low"' in argv
    assert env == worker.env
    assert env is not worker.env
    assert minted is None
    assert worker._process is None


@pytest.mark.parametrize("approve", [False, True])
async def test_start_initializes_runtime_from_worker_fields(
    monkeypatch: pytest.MonkeyPatch, approve: bool
) -> None:
    server = AppServer()
    probed: list[tuple[str, str, dict[str, Any]]] = []
    spawned: list[tuple[list[str], dict[str, Any]]] = []

    async def probe(kind: str, executable: str, **kwargs: Any) -> str:
        probed.append((kind, executable, kwargs))
        return "codex 0.160.1"

    async def spawn(argv: list[str], **kwargs: Any) -> jsonrpc.Process:
        spawned.append((argv, kwargs))
        return cast("jsonrpc.Process", server.process)

    monkeypatch.setattr(_adapter, "probe", probe)
    monkeypatch.setattr(jsonrpc, "spawn", spawn)
    worker = workers.Codex(
        cwd=CWD,
        executable="/custom/codex",
        env={"CODEX_HOME": "/custom/home"},
        model_name="test-model",
        effort="max",
        approve=approve,
        sandbox="workspace-write",
        config={"mcp_servers.docs.command": "custom-mcp"},
    )
    settings = worker.model_dump()
    servers = {
        "ai_sdk": {"url": "http://127.0.0.1:1/mcp"},
        "docs": {"command": "docs-mcp"},
    }
    try:
        await worker._cli_start(servers)
        assert worker._cli_version == "codex 0.160.1"
        assert worker._process is cast("jsonrpc.Process", server.process)
        assert worker._peer is not None
        assert probed[0][:2] == ("codex", "/custom/codex")
        assert probed[0][2]["cwd"] == CWD
        assert probed[0][2]["env"] == worker.env
        argv, kwargs = spawned[0]
        assert argv[:2] == ["/custom/codex", "app-server"]
        assert 'model_reasoning_effort="xhigh"' in argv
        assert 'mcp_servers.docs.command="custom-mcp"' in argv
        assert "mcp_servers.ai_sdk.tool_timeout_sec=86400" in argv
        assert kwargs == {"cwd": CWD, "env": worker.env}
        assert server.sent("initialize")
        assert server.sent("initialized")
        assert worker._thread_params() == {
            "sandbox": "workspace-write",
            "approvalPolicy": "untrusted" if approve else "never",
            "model": "test-model",
        }
        assert worker._approval_callback == (
            worker._approve if approve else None
        )
        assert worker.model_dump() == settings
        assert settings["worker_class_id"] == "codex"
        assert settings["model_name"] == "test-model"
        assert "kind" not in settings and "model" not in settings
        assert not any(k.startswith("_") for k in settings)
    finally:
        await worker._cli_close()
    assert worker._process is None
    assert worker._peer is None
    assert worker._approval_callback is None
    await worker._cli_close()  # idempotent


async def test_a_failed_handshake_closes_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = AppServer()
    terminated = False

    async def probe(*args: Any, **kwargs: Any) -> str:
        return "codex 0.160.1"

    async def spawn(*args: Any, **kwargs: Any) -> jsonrpc.Process:
        return cast("jsonrpc.Process", server.process)

    async def terminate() -> None:
        nonlocal terminated
        terminated = True

    def answer(message: dict[str, Any]) -> None:
        server.process.server_says(
            {"id": message["id"], "error": {"message": "bad handshake"}}
        )

    monkeypatch.setattr(_adapter, "probe", probe)
    monkeypatch.setattr(jsonrpc, "spawn", spawn)
    monkeypatch.setattr(server.process, "terminate", terminate)
    server.process.answer = answer
    worker = workers.Codex(cwd=CWD)
    with pytest.raises(jsonrpc.JsonRpcError, match="bad handshake"):
        await worker._cli_start({})
    assert terminated
    assert worker._process is None
    assert worker._peer is None


async def test_stream_start_waits_for_cli_acceptance() -> None:
    server = AppServer()
    received = asyncio.Event()

    def answer(message: dict[str, Any]) -> None:
        assert message["method"] == "turn/start"
        received.set()

    server.process.answer = answer
    worker = _worker(server)
    turn = worker._cli_turn([m.TextPart(text="go")])
    first = asyncio.ensure_future(anext(turn))
    try:
        await asyncio.wait_for(received.wait(), 5)
        assert not first.done()
        request = server.sent("turn/start")[0]
        server.process.server_says(
            {"id": request["id"], "result": {"turn": {"id": "turn-1"}}}
        )
        assert isinstance(await asyncio.wait_for(first, 5), events.StreamStart)
        assert worker._turn == "turn-1"
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        await turn.aclose()
        await worker._cli_close()


@pytest.mark.parametrize("malformed", [False, True])
async def test_rejected_or_malformed_start_never_acknowledges(
    malformed: bool,
) -> None:
    server = AppServer()

    def answer(message: dict[str, Any]) -> Any:
        if malformed:
            return {"turn": {}}
        server.process.server_says(
            {
                "id": message["id"],
                "error": {"code": -32600, "message": "start rejected"},
            }
        )
        return None

    server.process.answer = answer
    worker = _worker(server)
    turn = worker._cli_turn([m.TextPart(text="go")])
    try:
        with pytest.raises(workers.WorkerError):
            await anext(turn)
        assert worker._turn is None
        assert worker._queue is None
    finally:
        await turn.aclose()
        await worker._cli_close()


async def test_running_send_preserves_text_and_images() -> None:
    server = AppServer()
    worker = _worker(server)
    turn = worker._cli_turn([m.TextPart(text="go")])
    try:
        assert isinstance(await anext(turn), events.StreamStart)
        image = "data:image/png;base64,aW1hZ2U="
        assert await worker._cli_send_running(
            [m.TextPart(text="look"), m.FilePart.from_url(image)]
        )
        assert server.sent("turn/steer")[0]["params"] == {
            "threadId": THREAD,
            "expectedTurnId": "turn-1",
            "input": [
                {"type": "text", "text": "look"},
                {"type": "image", "url": image},
            ],
        }
        assert len(server.sent("turn/start")) == 1
    finally:
        await turn.aclose()
        await worker._cli_close()


async def test_idle_send_reports_no_running_turn_without_sending() -> None:
    server = AppServer()
    worker = _worker(server)
    try:
        assert not await worker._cli_send_running([m.TextPart(text="next")])
        assert not server.sent("turn/steer")
        assert not server.sent("turn/start")
    finally:
        await worker._cli_close()


@pytest.mark.parametrize(
    "detail", ["no active turn", "no active turn to steer"]
)
async def test_steer_racing_turn_end_can_start_a_new_turn(detail: str) -> None:
    server = AppServer()

    def answer(message: dict[str, Any]) -> Any:
        if message.get("method") == "turn/steer":
            server.says(
                "turn/completed",
                threadId=THREAD,
                turn={"id": "turn-1", "status": "completed"},
            )
            server.process.server_says(
                {
                    "id": message["id"],
                    "error": {"code": -32600, "message": detail},
                }
            )
            return None
        return AppServer.answer(server, message)

    server.process.answer = answer
    worker = _worker(server)
    turn = worker._cli_turn([m.TextPart(text="go")])
    try:
        assert isinstance(await anext(turn), events.StreamStart)
        # The event consumer has not drained the completed turn yet.
        assert not await worker._cli_send_running([m.TextPart(text="next")])
        assert len(server.sent("turn/start")) == 1
        seen = await asyncio.wait_for(_rest(turn), 5)
        assert isinstance(seen[-1], events.StreamEnd)
        assert worker._turn is None
        # The base drains the old turn before starting the new one.
        server.turn = TURN
        await _turn(worker, "next")
        assert len(server.sent("turn/start")) == 2
        assert len(server.sent("turn/steer")) == 1
    finally:
        await turn.aclose()
        await worker._cli_close()


@pytest.mark.parametrize(
    ("code", "detail"),
    [
        (-32600, "expected turn id does not match active turn id"),
        (-32600, "invalid input"),
        (-32603, "internal error"),
        (-32603, "no active turn"),
    ],
)
async def test_other_steer_rejections_are_not_retried(
    code: int,
    detail: str,
) -> None:
    server = AppServer()

    def answer(message: dict[str, Any]) -> Any:
        if message.get("method") == "turn/steer":
            # Even a concurrent completion must not hide unrelated failures.
            server.says(
                "turn/completed",
                threadId=THREAD,
                turn={"id": "turn-1", "status": "completed"},
            )
            server.process.server_says(
                {
                    "id": message["id"],
                    "error": {"code": code, "message": detail},
                }
            )
            return None
        return AppServer.answer(server, message)

    server.process.answer = answer
    worker = _worker(server)
    turn = worker._cli_turn([m.TextPart(text="go")])
    try:
        assert isinstance(await anext(turn), events.StreamStart)
        with pytest.raises(workers.WorkerError, match="steer rejected"):
            await worker._cli_send_running([m.TextPart(text="next")])
        assert len(server.sent("turn/start")) == 1
        assert len(server.sent("turn/steer")) == 1
    finally:
        await turn.aclose()
        await worker._cli_close()


@pytest.mark.parametrize("error", [TimeoutError(), workers.WorkerError("lost")])
async def test_ambiguous_steer_failures_are_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    server = AppServer()
    worker = _worker(server)
    turn = worker._cli_turn([m.TextPart(text="go")])

    async def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    try:
        assert isinstance(await anext(turn), events.StreamStart)
        assert worker._peer is not None
        monkeypatch.setattr(worker._peer, "call", fail)
        with pytest.raises(type(error)) as caught:
            await worker._cli_send_running([m.TextPart(text="next")])
        assert caught.value is error
        assert len(server.sent("turn/start")) == 1
    finally:
        await turn.aclose()
        await worker._cli_close()
