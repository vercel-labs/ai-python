"""The Codex worker against a scripted `codex app-server`, offline."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, cast

import pytest

import ai
from ai import workers
from ai.types import messages as m
from ai.workers import _process
from ai.workers.codex import jsonrpc

from .. import fakes

THREAD = "T"


class AppServer:
    """Answers like `codex app-server`.

    `turns[i]` plays out after the i-th turn/start, BEFORE the response, as
    the real server can: the worker must still order acceptance first.
    """

    def __init__(self, cwd: str) -> None:
        self.process = fakes.FakeProcess(self.answer)
        self.cwd = cwd
        self.turns: list[list[dict[str, Any]]] = []
        self.started = 0
        self.steer_error: dict[str, Any] | None = None
        self.threads: dict[str, dict[str, Any]] = {
            THREAD: {"id": THREAD, "cwd": cwd},
            "elsewhere": {"id": "elsewhere", "cwd": "/other"},
        }
        self.on_request_answer: list[dict[str, Any]] = []
        self.spawned: list[tuple[list[str], dict[str, Any]]] = []

    def says(self, method: str, **params: Any) -> None:
        self.process.server_says({"method": method, "params": params})

    def answer(self, message: dict[str, Any]) -> Any:
        method = message.get("method")
        params = message.get("params") or {}
        if method == "turn/start":
            self.started += 1
            turn = f"turn-{self.started}"
            for record in self.turns[self.started - 1] if self.turns else []:
                self.process.server_says(record)
            return {"turn": {"id": turn}}
        if method == "turn/steer":
            if self.steer_error is not None:
                self.process.server_says(
                    {"id": message["id"], "error": self.steer_error}
                )
                return None
            return {}
        if method == "turn/interrupt":
            self.says(
                "turn/completed",
                threadId=THREAD,
                turn={"id": params["turnId"], "status": "interrupted"},
            )
            return {}
        if method == "thread/read":
            return {"thread": self.threads.get(str(params.get("threadId")), {})}
        if method == "thread/start":
            return {"thread": {"id": THREAD}}
        if method == "thread/resume":
            return {"thread": {"id": params["threadId"]}}
        if "method" not in message:  # our answer to a server request
            self.on_request_answer.append(message)
            return None
        if "id" in message:
            return {}
        return None

    def sent(self, method: str) -> list[dict[str, Any]]:
        return [w for w in self.process.written if w.get("method") == method]


def note(method: str, turn_id: str = "turn-1", **params: Any) -> dict[str, Any]:
    return {
        "method": method,
        "params": {"threadId": THREAD, "turnId": turn_id, **params},
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


@pytest.fixture
def cwd(tmp_path: Path) -> str:
    return os.path.realpath(tmp_path)


@pytest.fixture
def server(cwd: str, monkeypatch: pytest.MonkeyPatch) -> AppServer:
    server = AppServer(cwd)

    async def probe(kind: str, executable: str, **kwargs: Any) -> str:
        return "codex 0.160.1"

    async def spawn(argv: list[str], **kwargs: Any) -> _process.Process:
        server.spawned.append((argv, kwargs))
        if server.process.terminated:
            server.process = fakes.FakeProcess(server.process.answer)
        return cast("_process.Process", server.process)

    monkeypatch.setattr(_process, "probe", probe)
    monkeypatch.setattr(_process, "spawn", spawn)
    return server


async def run(
    worker: workers.Codex, messages: list[m.Message]
) -> ai.Stream[str]:
    async with asyncio.timeout(5):
        async with ai.stream(worker, messages) as s:
            async for _ in s:
                pass
    return s


async def test_a_turn_streams_in_the_model_view(
    server: AppServer, cwd: str
) -> None:
    server.turns = [TURN]
    user = ai.user_message("go")
    async with workers.Codex(cwd=cwd) as worker:
        s = await run(worker, [user])
    message = s.message
    assert message.reasoning == "plan"
    assert message.text == "HERON"
    (call,) = message.builtin_tool_calls
    assert call.tool_name == "commandExecution"
    assert "secret.txt" in call.tool_args
    (result,) = message.builtin_tool_returns
    assert result.result == "the word is HERON\n"
    assert message.usage is not None and message.usage.input_tokens == 10
    # Acceptance came first, although the notes came before the response.
    assert [msg.role for msg in worker.messages] == ["user", "assistant"]
    assert worker.messages[0].id == user.id
    assert worker.session_id == THREAD


async def test_start_runs_app_server_with_the_bridge(
    server: AppServer, cwd: str
) -> None:
    worker = workers.Codex(
        cwd=cwd,
        executable="/custom/codex",
        env={"CODEX_HOME": "/h"},
        model_name="gpt-test",
        instructions="Be brief.",
        sandbox="workspace-write",
        config={"model_reasoning_effort": "high"},
    )
    async with worker:
        (argv, kwargs) = server.spawned[0]
        assert argv[:2] == ["/custom/codex", "app-server"]
        assert worker._session is not None
        bridge = worker._session.bridge.url
        assert f'mcp_servers.ai_sdk.url="{bridge}"' in argv
        assert "mcp_servers.ai_sdk.tool_timeout_sec=86400" in argv
        assert 'model_reasoning_effort="high"' in argv
        assert kwargs == {"cwd": cwd, "env": {"CODEX_HOME": "/h"}}
        (start,) = server.sent("thread/start")
        assert start["params"] == {
            "cwd": cwd,
            "sandbox": "workspace-write",
            "approvalPolicy": "never",
            "model": "gpt-test",
            "developerInstructions": "Be brief.",
        }
    assert server.process.terminated


async def test_a_new_thread_is_seeded_with_the_past(
    server: AppServer, cwd: str
) -> None:
    server.turns = [
        [note("turn/completed", turn={"id": "turn-1", "status": "completed"})]
    ]
    history = [
        ai.user_message("one"),
        ai.assistant_message("done"),
        ai.user_message("two"),
    ]
    worker = workers.Codex(cwd=cwd)  # not started: stream() seeds the start
    await run(worker, history)
    await worker.close()
    (inject,) = server.sent("thread/inject_items")
    texts = [item["content"][0]["text"] for item in inject["params"]["items"]]
    assert texts == ["one", "done"]
    (start,) = server.sent("turn/start")
    assert start["params"]["input"] == [{"type": "text", "text": "two"}]


async def test_a_session_resumes_here_only(server: AppServer, cwd: str) -> None:
    async with workers.Codex(cwd=cwd, session_id=THREAD):
        assert server.sent("thread/resume")[0]["params"]["threadId"] == THREAD
    with pytest.raises(workers.WorkerError, match="belongs to /other"):
        await workers.Codex(cwd=cwd, session_id="elsewhere").start()


async def test_another_turns_notifications_are_not_this_turns(
    server: AppServer, cwd: str
) -> None:
    server.turns = [
        [
            note(
                "turn/completed",
                turn_id="old",
                turn={"id": "old", "status": "completed"},
            ),
            note("item/agentMessage/delta", itemId="a1", delta="mine"),
            note("item/completed", item={"type": "agentMessage", "id": "a1"}),
            note(
                "turn/completed", turn={"id": "turn-1", "status": "completed"}
            ),
        ]
    ]
    async with workers.Codex(cwd=cwd) as worker:
        s = await run(worker, [ai.user_message("go")])
    assert s.text == "mine"


async def test_a_message_that_was_never_streamed_still_arrives(
    server: AppServer, cwd: str
) -> None:
    server.turns = [
        [
            note(
                "item/completed",
                item={"type": "agentMessage", "id": "a1", "text": "whole"},
            ),
            note(
                "turn/completed", turn={"id": "turn-1", "status": "completed"}
            ),
        ]
    ]
    async with workers.Codex(cwd=cwd) as worker:
        s = await run(worker, [ai.user_message("go")])
    assert s.text == "whole"


async def test_a_failed_turn_without_a_key_is_not_authenticated(
    server: AppServer, cwd: str
) -> None:
    server.turns = [
        [
            note(
                "turn/completed",
                turn={
                    "id": "turn-1",
                    "status": "failed",
                    "error": {"message": "401 Unauthorized"},
                },
            )
        ]
    ]
    async with workers.Codex(cwd=cwd) as worker:
        with pytest.raises(workers.NotAuthenticatedError):
            await run(worker, [ai.user_message("go")])


async def test_interrupt_ends_the_turn_as_interrupted(
    server: AppServer, cwd: str
) -> None:
    server.turns = [
        [note("item/agentMessage/delta", itemId="a1", delta="work")]
    ]
    async with workers.Codex(cwd=cwd) as worker:
        feed = worker.events()
        await worker.send("long job")
        async with asyncio.timeout(5):
            async for event in feed:
                if isinstance(event, ai.events.TextDelta):
                    break
            await worker.interrupt()
        (interrupt,) = server.sent("turn/interrupt")
        assert interrupt["params"] == {"threadId": THREAD, "turnId": "turn-1"}
        assert workers.is_interrupted(worker.messages[-1])
        assert worker.messages[-1].text == "work"
        await feed.aclose()


async def test_a_send_during_a_turn_steers_it(
    server: AppServer, cwd: str
) -> None:
    server.turns = [
        [note("item/agentMessage/delta", itemId="a1", delta="Look")]
    ]
    async with workers.Codex(cwd=cwd) as worker:
        feed = worker.events()
        await worker.send("fix")
        await worker.send("skip slow ones")
        (steer,) = server.sent("turn/steer")
        assert steer["params"]["expectedTurnId"] == "turn-1"
        assert steer["params"]["input"] == [
            {"type": "text", "text": "skip slow ones"}
        ]
        server.process.server_says(
            note("item/completed", item={"type": "agentMessage", "id": "a1"})
        )
        server.process.server_says(
            note("item/agentMessage/delta", itemId="a2", delta="OK")
        )
        server.process.server_says(
            note("item/completed", item={"type": "agentMessage", "id": "a2"})
        )
        server.process.server_says(
            note("turn/completed", turn={"id": "turn-1", "status": "completed"})
        )
        async with asyncio.timeout(5):
            while len(worker.messages) < 4:
                await anext(feed)
        assert [(msg.role, msg.text) for msg in worker.messages] == [
            ("user", "fix"),
            ("assistant", "Look"),
            ("user", "skip slow ones"),
            ("assistant", "OK"),
        ]
        await feed.aclose()


async def test_a_steer_that_lost_the_race_starts_a_turn(
    server: AppServer, cwd: str
) -> None:
    server.turns = [
        [],
        [
            note(
                "turn/completed",
                turn_id="turn-2",
                turn={"id": "turn-2", "status": "completed"},
            )
        ],
    ]
    server.steer_error = {"code": -32600, "message": "no active turn to steer"}
    async with workers.Codex(cwd=cwd) as worker:
        await worker.send("one")
        await worker.send("two")
        assert len(server.sent("turn/start")) == 2


async def test_other_steer_rejections_are_errors(
    server: AppServer, cwd: str
) -> None:
    server.turns = [[]]
    server.steer_error = {"code": -32600, "message": "expected turn mismatch"}
    async with workers.Codex(cwd=cwd) as worker:
        await worker.send("one")
        with pytest.raises(workers.WorkerError, match="steer rejected"):
            await worker.send("two")
        assert len(server.sent("turn/start")) == 1


async def test_bridge_calls_are_ai_tool_calls_not_builtin(
    server: AppServer, cwd: str
) -> None:
    bridge_item = {
        "type": "mcpToolCall",
        "id": "m1",
        "server": "ai_sdk",
        "tool": "lookup",
    }
    server.turns = [
        [
            note("item/started", item=bridge_item),
            note("item/completed", item={**bridge_item, "result": "x"}),
            note("item/agentMessage/delta", itemId="a1", delta="done"),
            note("item/completed", item={"type": "agentMessage", "id": "a1"}),
            note(
                "turn/completed", turn={"id": "turn-1", "status": "completed"}
            ),
        ]
    ]
    async with workers.Codex(cwd=cwd) as worker:
        s = await run(worker, [ai.user_message("go")])
    assert s.message.builtin_tool_calls == []
    assert s.text == "done"


async def test_an_approval_request_is_answered(
    server: AppServer, cwd: str
) -> None:
    async with workers.Codex(cwd=cwd):
        server.process.server_says(
            {
                "id": 99,
                "method": "item/commandExecution/requestApproval",
                "params": {},
            }
        )
        async with asyncio.timeout(5):
            while not server.on_request_answer:
                await asyncio.sleep(0.01)
    assert server.on_request_answer[0] == {
        "jsonrpc": "2.0",
        "id": 99,
        "result": {"decision": "accept"},
    }


async def test_a_failed_handshake_closes_the_process(
    server: AppServer, cwd: str
) -> None:
    def refuse(message: dict[str, Any]) -> Any:
        if message.get("method") == "initialize":
            server.process.server_says(
                {"id": message["id"], "error": {"code": 1, "message": "no"}}
            )
            return None
        return {}

    server.process.answer = refuse
    with pytest.raises(jsonrpc.JsonRpcError):
        await workers.Codex(cwd=cwd).start()
    assert server.process.terminated


async def test_a_crash_reaches_the_stream(server: AppServer, cwd: str) -> None:
    server.turns = [
        [note("item/agentMessage/delta", itemId="a1", delta="half")]
    ]

    async def die() -> None:
        await asyncio.sleep(0.05)
        await server.process.terminate()

    async with workers.Codex(cwd=cwd) as worker:
        task = asyncio.create_task(die())
        with pytest.raises(workers.WorkerError, match=r"crashed|stopped"):
            await run(worker, [ai.user_message("go")])
        await task
