"""The codex adapter against a scripted `codex app-server`, offline."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, cast

import pytest

from ai import workers
from ai.types import events
from ai.types import messages as m
from ai.workers._adapters import codex, jsonrpc

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


def _adapter(
    server: AppServer, approve: Any = None, **options: Any
) -> codex.CodexAdapter:
    adapter = codex.CodexAdapter(
        cwd=CWD,
        env={},
        executable="codex",
        model=None,
        mcp_servers=options.pop("mcp_servers", {}),
        approve=approve,
        sandbox=None,
        config={},
    )
    peer = jsonrpc.JsonRpcPeer(cast("Any", server.process), kind="codex")
    adapter._process = cast("Any", server.process)
    adapter._peer = peer
    adapter._thread = THREAD
    peer.start(
        adapter._on_notify, adapter._on_request, on_crash=adapter._on_crash
    )
    return adapter


async def _turn(adapter: codex.CodexAdapter, text: str = "go") -> list[Any]:
    async with asyncio.timeout(5):
        return [e async for e in adapter.turn([m.TextPart(text=text)])]


async def test_a_turn_streams_in_the_model_view() -> None:
    adapter = _adapter(AppServer(TURN))
    seen = await _turn(adapter)
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
    seen = await _turn(_adapter(server))
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
    seen = await _turn(_adapter(server))
    assert [type(e).__name__ for e in seen] == ["StreamStart", "StreamEnd"]


async def test_a_failed_turn_without_a_key_is_not_authenticated() -> None:
    server = AppServer(
        [
            note("error", error={"message": "401 Unauthorized"}),
            note("turn/completed", turn={"id": "turn-1", "status": "failed"}),
        ]
    )
    with pytest.raises(workers.NotAuthenticatedError):
        await _turn(_adapter(server))


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
    adapter = _adapter(server, approve)
    assert adapter._thread_params()["approvalPolicy"] == "untrusted"
    await _turn(adapter)
    assert asked == [("i1", "commandExecution", {"command": "rm -rf /"})]
    answer = next(w for w in server.process.written if w.get("id") == 500)
    assert answer["result"] == {"decision": "accept" if granted else "decline"}


async def test_without_a_gate_codex_never_asks() -> None:
    adapter = _adapter(AppServer())
    assert adapter._thread_params() == {
        "sandbox": "danger-full-access",
        "approvalPolicy": "never",
    }


async def test_an_interrupted_turn_ends_as_cancelled() -> None:
    server = AppServer()
    adapter = _adapter(server)
    turn = adapter.turn([m.TextPart(text="go")])
    assert isinstance(await anext(turn), events.StreamStart)
    reader = asyncio.ensure_future(_rest(turn))
    await asyncio.sleep(0)
    await adapter.interrupt()
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
    adapter = _adapter(server)
    seed = [
        m.Message(role="user", parts=[m.TextPart(text="earlier")]),
        m.Message(role="assistant", parts=[m.TextPart(text="answer")]),
    ]
    assert (
        await adapter.open(
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
    adapter = _adapter(server)
    forked = await adapter.open(
        session_id=THREAD, fork=True, seed=None, instructions=None
    )
    assert forked == "new"
    assert server.sent("thread/fork")[0]["params"]["threadId"] == THREAD


async def test_a_thread_from_another_directory_is_refused() -> None:
    server = AppServer()
    server.threads["elsewhere"] = {"id": "elsewhere", "cwd": "/other"}
    adapter = _adapter(server)
    with pytest.raises(workers.WorkerError, match="/other"):
        await adapter.open(
            session_id="elsewhere", fork=False, seed=None, instructions=None
        )


async def test_history_is_the_model_view() -> None:
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
    history = await _adapter(server).history(THREAD)
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
    infos = await _adapter(server).list_sessions()
    assert [i.session_id for i in infos] == ["t1", "t2"]
    assert all(p["cwd"] == CWD for p in asked)
    assert [p.get("cursor") for p in asked] == [None, "p2"]


async def test_the_tui_gets_no_bridge() -> None:
    adapter = _adapter(
        AppServer(),
        mcp_servers={
            "ai_sdk": {"url": "http://127.0.0.1:1/mcp"},
            "docs": {"command": "docs-mcp"},
        },
    )
    argv, _, minted = adapter.tui_command(THREAD)
    assert argv[:3] == ["codex", "resume", THREAD]
    assert 'mcp_servers.docs.command="docs-mcp"' in argv
    assert not any("ai_sdk" in arg for arg in argv)
    assert minted is None  # codex picks its own ids
