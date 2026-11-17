"""The Fx external harness against a scripted `fx acp` 0.0.17, offline.

Native fields and event order follow recorded ACP sessions. Prompt replies
are delayed until the turn ends. History replays before the load reply.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest

import ai
from ai import external
from ai.external import _process
from ai.external.fx import wire as wire_
from ai.types import messages as m

from .. import fakes

SESSION = "HLBbbJxgqBxq"
PREFS = {
    "provider": "gateway",
    "model": "openai/gpt-test",
    "effort": "auto",
    "fast_mode": False,
}


def update(update_kind: str, **fields: Any) -> dict[str, Any]:
    return {
        "method": "session/update",
        "params": {
            "sessionId": SESSION,
            "update": {"sessionUpdate": update_kind, **fields},
        },
    }


def text(value: str, message_id: str = "a1") -> dict[str, Any]:
    return update(
        "agent_message_chunk",
        messageId=message_id,
        content={"type": "text", "text": value},
    )


CALL = update(
    "tool_call",
    toolCallId="c1",
    name="shell",
    title="Running",
    kind="execute",
    status="pending",
    rawInput={"action": "run", "command": "cat secret.txt"},
    _meta={"fx": {"toolCall": {"internal": False}}},
)
RESULT = update(
    "tool_call_update",
    toolCallId="c1",
    status="completed",
    content=[
        {"type": "content", "content": {"type": "text", "text": "HERON\n"}}
    ],
)
END = {"stopReason": "end_turn"}


class YieldingProcess(fakes.FakeProcess):
    """Let the peer read early records while a prompt write drains."""

    yield_prompt_write = False
    fail_next_prompt = False

    async def write(self, data: str) -> None:
        if (
            self.fail_next_prompt
            and json.loads(data).get("method") == "session/prompt"
        ):
            self.fail_next_prompt = False
            raise OSError("scripted write failed")
        await super().write(data)
        if (
            self.yield_prompt_write
            and json.loads(data).get("method") == "session/prompt"
        ):
            await asyncio.sleep(0)
            await asyncio.sleep(0)


class AcpServer:
    def __init__(self, cwd: str, home: Path) -> None:
        self.cwd = cwd
        self.home = home
        self.process = YieldingProcess(self.answer)
        self.spawned: list[tuple[list[str], dict[str, Any]]] = []
        self.turns: list[
            tuple[list[dict[str, Any]], dict[str, Any] | None]
        ] = []
        self.prompts: list[dict[str, Any]] = []
        self.steers: list[dict[str, Any]] = []
        self.initialize_error: dict[str, Any] | None = None
        self.load_error = False
        self.hold_load = False
        self.closed_folders: list[tuple[str, bool]] = []
        self.active_requests: set[int] = set()
        self.rejected_prompts: list[dict[str, Any]] = []
        self.cancel_updates: list[dict[str, Any]] = []
        self.replay_users: int | None = None
        self.loaded: Path | None = None
        self.loaded_records: list[dict[str, Any]] = []
        self.pages: list[list[str]] = [[SESSION]]
        self.replies: list[dict[str, Any]] = []
        self.cancel_result = END
        self.active_session = SESSION
        self.listed_cwd: str | None = None
        self.cancel_request: dict[str, Any] | None = None

    def says(self, *records: dict[str, Any]) -> None:
        for record in records:
            if "id" in record and ("result" in record or "error" in record):
                self.active_requests.discard(record["id"])
            if record.get("method") == "session/update":
                record = {
                    **record,
                    "params": {
                        **record["params"],
                        "sessionId": self.active_session,
                    },
                }
            self.process.server_says(record)

    def finish(self, request: dict[str, Any], result: dict[str, Any]) -> None:
        self.says({"id": request["id"], "result": result})

    def sent(self, method: str) -> list[dict[str, Any]]:
        return [w for w in self.process.written if w.get("method") == method]

    async def wait_for(self, method: str, count: int = 1) -> None:
        async with asyncio.timeout(5):
            while len(self.sent(method)) < count:
                await asyncio.sleep(0)

    def answer(self, request: dict[str, Any]) -> Any:
        method = request.get("method")
        params = request.get("params", {})
        options = [
            {"id": key, "type": "select", "currentValue": value}
            for key, value in (
                ("provider", PREFS["provider"]),
                ("model", PREFS["model"]),
                ("effort", PREFS["effort"]),
                ("fast", "false"),
            )
        ]
        if method == "initialize":
            if self.initialize_error:
                self.says({"id": request["id"], "error": self.initialize_error})
                return None
            return {
                "protocolVersion": 1,
                "agentInfo": {"name": "fx", "version": "0.0.17"},
                "agentCapabilities": {
                    "loadSession": True,
                    "sessionCapabilities": {
                        "list": {},
                        "resume": {},
                        "close": {},
                        "systemPrompt": {},
                    },
                    "_meta": {"fx": {"steering": True}},
                },
                "authMethods": [],
            }
        if method == "session/new":
            return {"sessionId": SESSION, "configOptions": options}
        if method == "session/list":
            assert params["cwd"] == self.cwd
            page = int(params.get("cursor", "0"))
            result: dict[str, Any] = {
                "sessions": [
                    {"sessionId": sid, "cwd": self.listed_cwd or self.cwd}
                    for sid in self.pages[page]
                ]
            }
            if page + 1 < len(self.pages):
                result["nextCursor"] = str(page + 1)
            return result
        if method == "session/load":
            self.active_session = params["sessionId"]
            self.loaded = self.home / "sessions/v2" / params["sessionId"]
            self.loaded_records = [
                json.loads(line)
                for line in (self.loaded / "log.jsonl").read_text().splitlines()
            ]
            if self.hold_load:
                return None
            if self.load_error:
                self.says(
                    {
                        "id": request["id"],
                        "error": {
                            "code": -32603,
                            "message": "Session could not be loaded",
                        },
                    }
                )
                return None
            users = 0
            for index, record in enumerate(self.loaded_records):
                kind = record.get("type")
                if kind not in ("user", "assistant"):
                    continue
                if kind == "user":
                    users += 1
                    if (
                        self.replay_users is not None
                        and users > self.replay_users
                    ):
                        continue
                native = update(
                    f"{'user' if kind == 'user' else 'agent'}_message_chunk",
                    messageId=f"replay-{index}",
                    content={"type": "text", "text": record["data"]["text"]},
                )
                native["params"]["sessionId"] = params["sessionId"]
                self.says(native)
                # An image is a second chunk of the SAME user message.
                for image in record["data"].get("images", []):
                    self.says(
                        {
                            **native,
                            "params": {
                                **native["params"],
                                "update": {
                                    **native["params"]["update"],
                                    "content": {
                                        "type": "image",
                                        "mimeType": image["media_type"],
                                        "data": base64.b64encode(
                                            (
                                                self.loaded
                                                / "blobs"
                                                / image["blob"]
                                            ).read_bytes()
                                        ).decode(),
                                    },
                                },
                            },
                        }
                    )
            return {"configOptions": options}
        if method == "session/resume":
            return {"configOptions": options}
        if method == "session/prompt":
            if params.get("_meta", {}).get("fx", {}).get("steer"):
                self.active_requests.add(request["id"])
                self.steers.append(request)
                return None
            if self.active_requests:
                self.rejected_prompts.append(request)
                self.says(
                    {
                        "id": request["id"],
                        "error": {
                            "code": -32600,
                            "message": "Prompt already in progress",
                        },
                    }
                )
                return None
            self.active_requests.add(request["id"])
            index = len(self.prompts)
            self.prompts.append(request)
            records, response = (
                self.turns[index] if index < len(self.turns) else ([], END)
            )
            self.says(*records)
            if response is not None:
                self.finish(request, response)
            return None
        if method == "session/close":
            sid = params["sessionId"]
            folder = self.home / "sessions/v2" / sid
            self.closed_folders.append((sid, folder.exists()))
            return {}
        if method == "session/cancel":
            if self.prompts:
                self.finish(
                    self.cancel_request or self.prompts[-1], self.cancel_result
                )
                self.says(*self.cancel_updates)
            return None
        if method is None:
            self.replies.append(request)
            return None
        return {} if "id" in request else None


@pytest.fixture
def cwd(tmp_path: Path) -> str:
    project = tmp_path / "project"
    project.mkdir()
    return os.path.realpath(project)


@pytest.fixture
def server(
    cwd: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AcpServer:
    server = AcpServer(cwd, tmp_path / "fx-home")

    async def probe(kind: str, executable: str, **kwargs: Any) -> str:
        return "0.0.17"

    async def spawn(argv: list[str], **kwargs: Any) -> _process.Process:
        server.spawned.append((argv, kwargs))
        return cast("_process.Process", server.process)

    monkeypatch.setattr(_process, "probe", probe)
    monkeypatch.setattr(_process, "spawn", spawn)
    return server


@pytest.fixture
def harness(server: AcpServer) -> external.Fx:
    return external.Fx(
        cwd=server.cwd,
        provider=external.FxProvider(env={"FX_HOME": str(server.home)}),
    )


async def run(
    harness: external.Fx, messages: list[m.Message]
) -> ai.Stream[str]:
    async with asyncio.timeout(5), ai.stream(harness, messages) as stream:
        async for _ in stream:
            pass
    return stream


async def test_a_turn_streams_in_the_model_view(
    server: AcpServer, harness: external.Fx
) -> None:
    server.turns = [
        (
            [
                update(
                    "agent_thought_chunk",
                    content={"type": "text", "text": "plan"},
                ),
                CALL,
                update(
                    "tool_call_update", toolCallId="c1", status="in_progress"
                ),
                RESULT,
                RESULT,
                text("HER"),
                text("ON"),
                update("usage_update", used=999, size=1000000),
            ],
            {
                "stopReason": "end_turn",
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 4,
                    "cacheReadTokens": 2,
                    "cacheWriteTokens": 1,
                    "reasoningTokens": 3,
                },
            },
        )
    ]
    user = ai.user_message("go")
    async with harness:
        stream = await run(harness, [user])
        assert harness.session.id == SESSION
        assert [msg.role for msg in harness.session.messages] == [
            "user",
            "assistant",
        ]
        assert harness.session.messages[0].id == user.id
    message = stream.message
    assert message.text == "HERON"
    assert message.reasoning == "plan"
    (call,) = message.builtin_tool_calls
    assert call.tool_name == "shell"
    assert json.loads(call.tool_args) == CALL["params"]["update"]["rawInput"]
    (result,) = message.builtin_tool_returns
    assert result.result == "HERON\n"
    assert message.usage is not None
    assert message.usage.input_tokens == 10
    assert message.usage.output_tokens == 4
    assert message.usage.cache_read_tokens == 2
    assert message.usage.cache_write_tokens == 1
    assert message.usage.reasoning_tokens == 3
    metadata = message.provider_metadata or {}
    assert metadata["model"] == PREFS["model"]
    assert metadata["fx_version"] == "0.0.17"


async def test_start_gives_the_bridge_and_settings(
    server: AcpServer, harness: external.Fx
) -> None:
    harness.provider = external.FxProvider(
        executable="/custom/fx", env={"FX_HOME": str(server.home)}
    )
    harness.model_provider = "gateway"
    harness.model_name = "openai/gpt-test"
    harness.effort = "high"
    harness.instructions = "Be brief."
    async with harness:
        await harness.session.start()
        argv, kwargs = server.spawned[0]
        assert argv == [
            "/custom/fx",
            "acp",
            "--permission-mode",
            "full-access",
            "--provider",
            "gateway",
            "--model",
            "openai/gpt-test",
            "--effort",
            "high",
        ]
        assert kwargs == {
            "cwd": server.cwd,
            "env": {"FX_HOME": str(server.home)},
        }
        (init,) = server.sent("initialize")
        assert init["params"]["protocolVersion"] == 1
        assert init["params"]["clientCapabilities"]["session"]["notices"] == {}
        (new,) = server.sent("session/new")
        assert new["params"]["cwd"] == server.cwd
        assert new["params"]["systemPrompt"] == [
            {"type": "text", "text": "Be brief."}
        ]
        (bridge,) = new["params"]["mcpServers"]
        assert bridge == {
            "type": "http",
            "name": "ai_sdk",
            "url": bridge["url"],
            "headers": [],
        }
        assert bridge["url"].startswith("http://127.0.0.1:")
    assert server.process.terminated


@pytest.mark.parametrize("instructions", [None, ""])
async def test_default_start_omits_optional_flags_and_empty_instructions(
    server: AcpServer, harness: external.Fx, instructions: str | None
) -> None:
    harness.instructions = instructions
    async with harness:
        await harness.session.start()
        assert server.spawned[0][0] == [
            "fx",
            "acp",
            "--permission-mode",
            "full-access",
        ]
        assert "systemPrompt" not in server.sent("session/new")[0]["params"]


async def test_a_new_session_loads_the_written_past_without_streaming_replay(
    server: AcpServer, harness: external.Fx
) -> None:
    server.turns = [([text("fresh")], END)]
    harness.instructions = "Keep the seed instructions."
    history = [
        ai.user_message("one"),
        ai.assistant_message("old reply"),
        ai.user_message("two"),
    ]
    async with harness:
        stream = await run(harness, history)
        assert (
            harness.session.id
            == server.sent("session/load")[0]["params"]["sessionId"]
        )
    assert stream.text == "fresh"
    assert [
        r["data"]["text"]
        for r in server.loaded_records
        if r.get("type") in ("user", "assistant")
    ] == ["one", "old reply"]
    prefs = next(
        r["value"] for r in server.loaded_records if r.get("key") == "prefs"
    )
    assert prefs == PREFS
    assert (
        next(
            r["value"]
            for r in server.loaded_records
            if r.get("key") == "client_prompt"
        )
        == "Keep the seed instructions."
    )
    assert server.loaded_records[0]["workspace"] == server.cwd
    methods = [r.get("method") for r in server.process.written]
    assert (
        methods.index("initialize")
        < methods.index("session/new")
        < methods.index("session/load")
        < methods.index("session/prompt")
    )
    assert not server.sent("session/list")
    assert (
        server.sent("session/load")[0]["params"]["mcpServers"]
        == server.sent("session/new")[0]["params"]["mcpServers"]
    )
    assert server.prompts[0]["params"]["prompt"] == [
        {"type": "text", "text": "two"}
    ]


@pytest.mark.parametrize("failure", ["load", "missing replay"])
async def test_a_refused_seed_is_removed(
    server: AcpServer, harness: external.Fx, failure: str
) -> None:
    sibling = server.home / "sessions/v2/UserSession01"
    sibling.mkdir(parents=True)
    sibling_log = sibling / "log.jsonl"
    sibling_log.write_bytes(b"user session must stay unchanged\n")
    index = sibling.parent / "index.jsonl"
    index.write_bytes(b"user index must stay unchanged\n")
    server.load_error = failure == "load"
    server.replay_users = 0 if failure == "missing replay" else None
    with pytest.raises(
        external.ExternalHarnessError, match=r"0\.0\.17"
    ) as error:
        async with harness:
            await run(
                harness,
                [
                    ai.user_message("old"),
                    ai.assistant_message("reply"),
                    ai.user_message("new"),
                ],
            )
    assert str(error.value).startswith(
        "fx 0.0.17: this version of ai cannot seed an fx session: "
    )
    assert sibling_log.read_bytes() == b"user session must stay unchanged\n"
    assert index.read_bytes() == b"user index must stay unchanged\n"
    assert server.loaded is not None and not server.loaded.exists()
    assert server.closed_folders[0] == (server.loaded.name, True)
    assert len(server.sent("session/new")) == 1
    assert not server.prompts
    assert server.process.terminated


async def test_a_resume_reads_all_pages_and_opens_here_only(
    server: AcpServer, harness: external.Fx
) -> None:
    server.pages = [["other"], [SESSION]]
    harness = external.Fx(
        cwd=harness.cwd, provider=harness.provider, resume=SESSION
    )
    async with harness:
        await harness.session.start()
        (resume,) = server.sent("session/resume")
        assert resume["params"]["sessionId"] == SESSION
        assert resume["params"]["cwd"] == server.cwd
        assert resume["params"]["mcpServers"][0]["name"] == "ai_sdk"
        assert len(server.sent("session/list")) == 2


@pytest.mark.parametrize("session_id", ["elsewhere", "unknown"])
async def test_a_resume_refuses_an_id_not_in_this_directory(
    server: AcpServer, harness: external.Fx, session_id: str
) -> None:
    with pytest.raises(external.ExternalHarnessError):
        async with external.Fx(
            cwd=harness.cwd, provider=harness.provider, resume=session_id
        ) as harness:
            await harness.session.start()
    assert not server.sent("session/resume")


async def test_bridge_calls_are_ai_tool_calls_not_builtin(
    server: AcpServer, harness: external.Fx
) -> None:
    @ai.tool
    async def lookup() -> str:
        """Return the word."""
        return "PELICAN"

    async def call_bridge() -> None:
        await server.wait_for("session/prompt")
        bridge = server.sent("session/new")[0]["params"]["mcpServers"][0]["url"]
        server.says(
            update(
                "tool_call",
                toolCallId="bridge-1",
                name="mcp_ai_sdk_lookup",
                title="lookup",
                kind="other",
                status="pending",
                rawInput={},
                _meta={
                    "fx": {
                        "toolCall": {
                            "internal": False,
                            "mcp": {"server": "ai_sdk", "tool": "lookup"},
                        }
                    }
                },
            )
        )
        reply = await fakes.mcp_call(
            bridge,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "lookup", "arguments": {}},
            },
        )
        assert "PELICAN" in reply["result"]["content"][0]["text"]
        server.says(
            update(
                "tool_call_update",
                toolCallId="bridge-1",
                status="completed",
                content=[
                    {
                        "type": "content",
                        "content": {"type": "text", "text": "PELICAN"},
                    }
                ],
            ),
            text("done"),
        )
        server.finish(server.prompts[0], END)

    server.turns = [([], None)]
    task = asyncio.create_task(call_bridge())
    async with (
        asyncio.timeout(5),
        harness,
        ai.Agent(tools=[lookup]).run(
            harness, [ai.user_message("go")]
        ) as result,
    ):
        async for _ in result:
            pass
    await task
    assert any(
        msg.role == "tool"
        and any("PELICAN" in str(part.result) for part in msg.tool_results)
        for msg in result.messages
    )
    assert not any(
        msg.builtin_tool_calls or msg.builtin_tool_returns
        for msg in result.messages
    )


async def test_a_send_during_a_turn_waits_for_its_correlated_echo(
    server: AcpServer, harness: external.Fx
) -> None:
    server.turns = [([text("Look")], None)]
    async with asyncio.timeout(5), harness:
        feed = harness.session.events()
        events = []
        await harness.session.send(ai.user_message("fix"))
        async for event in feed:
            events.append(event)
            if isinstance(event, ai.events.TextDelta):
                break
        steer = asyncio.create_task(
            harness.session.send(ai.user_message("skip slow ones"))
        )
        await server.wait_for("session/prompt", 2)
        request = server.steers[0]
        assert request["params"]["_meta"] == {"fx": {"steer": True}}
        assert request["params"]["prompt"] == [
            {"type": "text", "text": "skip slow ones"}
        ]
        server.says(
            update(
                "user_message_chunk",
                messageId="unrelated",
                content={"type": "text", "text": "skip slow ones"},
                _meta={"fx": {"steering": {"requestId": -1}}},
            ),
            text(" before", "a1"),
        )
        async for event in feed:
            events.append(event)
            if (
                isinstance(event, ai.events.TextDelta)
                and event.chunk == " before"
            ):
                break
        assert [
            msg.text for msg in harness.session.messages if msg.role == "user"
        ] == ["fix"]
        server.says(
            update(
                "user_message_chunk",
                messageId="u2",
                content={"type": "text", "text": "skip slow ones"},
                _meta={"fx": {"steering": {"requestId": request["id"]}}},
            ),
            text("OK", "a2"),
        )
        server.finish(server.prompts[0], END)
        server.finish(
            request, {**END, "_meta": {"fx": {"steering": "absorbed"}}}
        )
        await steer
        events += await through_turns(feed, 2)
        assert (
            len([e for e in events if isinstance(e, ai.events.StreamStart)])
            == 2
        )
        assert [
            e.finish_reason
            for e in events
            if isinstance(e, ai.events.StreamEnd)
        ] == ["other", "stop"]
        assert [(msg.role, msg.text) for msg in harness.session.messages] == [
            ("user", "fix"),
            ("assistant", "Look before"),
            ("user", "skip slow ones"),
            ("assistant", "OK"),
        ]


@pytest.mark.parametrize("outcome", ["dropped", "not delivered"])
async def test_a_dropped_steer_starts_a_new_prompt(
    server: AcpServer, harness: external.Fx, outcome: str
) -> None:
    server.turns = [
        ([text("first", "a1")], None),
        ([text("second", "a2")], END),
    ]
    async with asyncio.timeout(5), harness:
        feed = harness.session.events()
        await harness.session.send(ai.user_message("one"))
        task = asyncio.create_task(harness.session.send(ai.user_message("two")))
        await server.wait_for("session/prompt", 2)
        server.finish(server.prompts[0], END)
        request = server.steers[0]
        if outcome == "dropped":
            server.finish(
                request,
                {
                    "stopReason": "cancelled",
                    "_meta": {"fx": {"steering": "dropped"}},
                },
            )
        else:
            server.says(
                {
                    "id": request["id"],
                    "error": {
                        "code": -32600,
                        "message": (
                            "Steering prompt was not delivered because "
                            "the turn ended first; send it as a new prompt"
                        ),
                    },
                }
            )
        await task
        await server.wait_for("session/prompt", 3)
        events = await through_turns(feed, 2)
        assert [(msg.role, msg.text) for msg in harness.session.messages] == [
            ("user", "one"),
            ("assistant", "first"),
            ("user", "two"),
            ("assistant", "second"),
        ]
        assert (
            len([e for e in events if isinstance(e, ai.events.StreamStart)])
            == 2
        )
        assert [
            e.finish_reason
            for e in events
            if isinstance(e, ai.events.StreamEnd)
        ] == ["stop", "stop"]
        assert server.prompts[1]["params"]["prompt"] == [
            {"type": "text", "text": "two"}
        ]
        assert (
            not server.prompts[1]["params"]
            .get("_meta", {})
            .get("fx", {})
            .get("steer")
        )


@pytest.mark.parametrize("stop_reason", ["cancelled", "end_turn", "refused"])
async def test_interrupt_ends_as_interrupted(
    server: AcpServer, harness: external.Fx, stop_reason: str
) -> None:
    server.turns = [([text("work")], None)]
    server.cancel_result = {"stopReason": stop_reason}
    async with asyncio.timeout(5), harness:
        feed = harness.session.events()
        await harness.session.send(ai.user_message("long job"))
        async for event in feed:
            if isinstance(event, ai.events.TextDelta):
                break
        await harness.session.interrupt()
        (cancel,) = server.sent("session/cancel")
        assert "id" not in cancel
        assert cancel["params"] == {"sessionId": SESSION}
        assert (harness.session.messages[-1].provider_metadata or {})[
            "external"
        ] == {"stop_reason": "interrupted"}
        assert harness.session.messages[-1].text == "work"


async def test_a_failed_initialize_without_sign_in_is_not_authenticated(
    server: AcpServer, harness: external.Fx
) -> None:
    server.initialize_error = {
        "code": -32600,
        "message": (
            "fx needs access to Vercel AI Gateway. Run fx login to "
            "sign in, fx login vercel --api-key to use an API key, "
            "or set AI_GATEWAY_API_KEY."
        ),
    }
    with pytest.raises(external.NotAuthenticatedError):
        async with harness:
            await run(harness, [ai.user_message("go")])
    assert server.process.terminated


@pytest.mark.parametrize("recovery", [True, False])
async def test_a_refused_turn_raises_with_the_last_error_detail(
    server: AcpServer, harness: external.Fx, recovery: bool
) -> None:
    records = [text("first failure", "a1"), text("last failure", "a2")]
    if recovery:
        records += [
            update(
                "session_info_update",
                _meta={
                    "fx": {
                        "modelResponseRecovery": {
                            "state": "waiting",
                            "attempt": 1,
                            "message": "old recovery",
                        }
                    }
                },
            ),
            update(
                "session_info_update",
                _meta={
                    "fx": {
                        "modelResponseRecovery": {
                            "state": "waiting",
                            "attempt": 2,
                            "message": "final recovery",
                        }
                    }
                },
            ),
        ]
    server.turns = [(records, {"stopReason": "refused"})]
    with pytest.raises(
        external.ExternalHarnessError,
    ) as error:
        async with harness:
            await run(harness, [ai.user_message("go")])
    detail = "final recovery" if recovery else "last failure"
    assert str(error.value) == f"fx turn failed: {detail}"


async def test_a_crash_reaches_the_stream(
    server: AcpServer, harness: external.Fx
) -> None:
    server.turns = [([text("half")], None)]

    async def die() -> None:
        await server.wait_for("session/prompt")
        await server.process.terminate()

    task = asyncio.create_task(die())
    with pytest.raises(external.ExternalHarnessError, match=r"crashed|stopped"):
        async with harness:
            await run(harness, [ai.user_message("go")])
    await task


async def test_close_is_safe_twice(
    server: AcpServer, harness: external.Fx
) -> None:
    async with harness:
        await harness.session.start()
        await harness.session._close()
        await harness.session._close()
    assert server.process.terminated
    assert len(server.sent("session/close")) == 1


async def test_a_failed_handshake_stops_the_process(
    server: AcpServer, harness: external.Fx
) -> None:
    server.initialize_error = {"code": -32602, "message": "unsupported effort"}
    with pytest.raises(
        external.ExternalHarnessError, match="unsupported effort"
    ):
        async with harness:
            await harness.session.start()
    assert server.process.terminated


async def test_text_and_reasoning_blocks_close_without_end_markers(
    server: AcpServer, harness: external.Fx
) -> None:
    server.turns = [
        (
            [
                text("one"),
                text("two", "a2"),
                update(
                    "agent_thought_chunk",
                    content={"type": "text", "text": "think"},
                ),
                update(
                    "agent_thought_chunk",
                    content={"type": "text", "text": " more"},
                ),
                text("three", "a3"),
            ],
            END,
        )
    ]
    events = []
    async with (
        asyncio.timeout(5),
        harness,
        ai.stream(harness, [ai.user_message("go")]) as stream,
    ):
        async for event in stream:
            events.append(event)
    blocks = [
        e
        for e in events
        if isinstance(
            e,
            ai.events.TextStart
            | ai.events.TextEnd
            | ai.events.ReasoningStart
            | ai.events.ReasoningEnd,
        )
    ]
    assert [type(e) for e in blocks] == [
        ai.events.TextStart,
        ai.events.TextEnd,
        ai.events.TextStart,
        ai.events.TextEnd,
        ai.events.ReasoningStart,
        ai.events.ReasoningEnd,
        ai.events.TextStart,
        ai.events.TextEnd,
    ]
    for start, end in zip(blocks[::2], blocks[1::2], strict=True):
        assert start.block_id == end.block_id
    assert len({e.block_id for e in blocks[::2]}) == 4
    assert sum(isinstance(e, ai.events.StreamEnd) for e in events) == 1


async def test_seed_replay_counts_distinct_user_ids_not_image_chunks(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    image = m.FilePart(data=b"image bytes", media_type="image/png")
    history = [
        m.Message(role="user", parts=[m.TextPart(text="past picture"), image]),
        ai.assistant_message("past answer"),
        ai.user_message("now"),
    ]
    server.turns = [([text("fresh")], END)]
    async with harness:
        stream = await run(harness, history)
    assert stream.text == "fresh"
    assert server.loaded is not None and server.loaded.exists()


@pytest.mark.parametrize(
    "native, expected",
    [
        ("end_turn", "stop"),
        ("max_output_tokens", "length"),
        ("max_model_turns", "length"),
        ("cancelled", "other"),
    ],
)
async def test_prompt_stop_reasons_have_one_stream_end(
    server: AcpServer,
    harness: external.Fx,
    native: str,
    expected: str,
) -> None:
    server.turns = [([text("done")], {"stopReason": native})]
    ends = []
    async with (
        asyncio.timeout(5),
        harness,
        ai.stream(harness, [ai.user_message("go")]) as stream,
    ):
        async for event in stream:
            if isinstance(event, ai.events.StreamEnd):
                ends.append(event)
    assert len(ends) == 1
    assert ends[0].finish_reason == expected


async def test_an_unexpected_permission_request_gets_an_answer(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    async with asyncio.timeout(5), harness:
        await harness.session.start()
        server.says(
            {
                "id": 99,
                "method": "session/request_permission",
                "params": {
                    "sessionId": SESSION,
                    "toolCall": {
                        "toolCallId": "c",
                        "name": "shell",
                        "title": "Running",
                        "kind": "execute",
                        "status": "pending",
                        "rawInput": {},
                    },
                    "options": [
                        {
                            "optionId": "allow_once",
                            "name": "Allow once",
                            "kind": "allow_once",
                        }
                    ],
                },
            }
        )
        while not server.replies:
            await asyncio.sleep(0)
        assert server.replies[0]["id"] == 99
        assert server.replies[0]["result"] == {
            "outcome": {"outcome": "selected", "optionId": "allow_once"}
        }


async def test_a_nontext_steer_waits_and_becomes_a_new_prompt(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    server.turns = [([], None), ([text("seen")], END)]
    image = m.FilePart(data=b"image bytes", media_type="image/png")
    async with asyncio.timeout(5), harness:
        await harness.session.send(ai.user_message("first"))
        task = asyncio.create_task(
            harness.session.send(
                m.Message(role="user", parts=[m.TextPart(text="look"), image])
            )
        )
        await asyncio.sleep(0.01)
        assert len(server.sent("session/prompt")) == 1
        server.finish(server.prompts[0], END)
        await task
        await server.wait_for("session/prompt", 2)
        assert not server.steers
        assert server.prompts[1]["params"]["prompt"] == [
            {"type": "text", "text": "look"},
            {
                "type": "image",
                "mimeType": "image/png",
                "data": base64.b64encode(b"image bytes").decode(),
            },
        ]


async def test_a_prompt_rpc_error_reaches_the_stream(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    server.turns = [([], None)]

    async def reject() -> None:
        await server.wait_for("session/prompt")
        server.says(
            {
                "id": server.prompts[0]["id"],
                "error": {"code": -32603, "message": "model request failed"},
            }
        )

    task = asyncio.create_task(reject())
    with pytest.raises(
        external.ExternalHarnessError, match="model request failed"
    ):
        async with harness:
            await run(harness, [ai.user_message("go")])
    await task


async def test_a_refused_turn_with_bad_credentials_is_not_authenticated(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    server.turns = [
        (
            [text("gateway authentication failed")],
            {"stopReason": "refused"},
        )
    ]
    with pytest.raises(external.NotAuthenticatedError):
        async with harness:
            await run(harness, [ai.user_message("go")])


async def through_turns(feed: AsyncIterator[Any], count: int) -> list[Any]:
    events = []
    async with asyncio.timeout(5):
        async for event in feed:
            events.append(event)
            if isinstance(event, ai.events.StreamEnd):
                count -= 1
                if count == 0:
                    break
        try:
            async with asyncio.timeout(0.02):
                async for event in feed:
                    events.append(event)
        except TimeoutError:
            pass
    return events


@pytest.mark.parametrize("has_content", [True, False])
async def test_a_late_steer_runs_as_its_own_turn(
    server: AcpServer,
    harness: external.Fx,
    has_content: bool,
) -> None:
    # Native order from fx 0.0.17: first reply, second content, plain reply.
    server.turns = [([text("ONE", "a1")], None)]
    async with asyncio.timeout(5), harness:
        feed = harness.session.events()
        await harness.session.send(ai.user_message("ONE"))
        await harness.session.send(ai.user_message("TWO"))
        request = server.steers[0]
        server.finish(
            server.prompts[0],
            {**END, "usage": {"inputTokens": 11, "outputTokens": 1}},
        )
        if has_content:
            server.says(text("TWO", "a2"))
        server.finish(
            request, {**END, "usage": {"inputTokens": 22, "outputTokens": 2}}
        )
        events = await through_turns(feed, 2)
        assert [(msg.role, msg.text) for msg in harness.session.messages] == [
            ("user", "ONE"),
            ("assistant", "ONE"),
            ("user", "TWO"),
            ("assistant", "TWO" if has_content else ""),
        ]
        boundaries = [
            e
            for e in events
            if isinstance(e, ai.events.StreamStart | ai.events.StreamEnd)
        ]
        assert [type(e) for e in boundaries] == [
            ai.events.StreamStart,
            ai.events.StreamEnd,
            ai.events.StreamStart,
            ai.events.StreamEnd,
        ]
        ends = [e for e in events if isinstance(e, ai.events.StreamEnd)]
        assert [e.finish_reason for e in ends] == ["stop", "stop"]
        usages = [e.usage for e in ends]
        assert all(usage is not None for usage in usages)
        assert [
            usage.input_tokens for usage in usages if usage is not None
        ] == [11, 22]
        assert [
            usage.output_tokens for usage in usages if usage is not None
        ] == [1, 2]
        if has_content:
            second_start = events.index(boundaries[2])
            second_delta = next(
                i
                for i, e in enumerate(events)
                if isinstance(e, ai.events.TextDelta) and e.chunk == "TWO"
            )
            assert (
                events.index(boundaries[1])
                < second_start
                < second_delta
                < events.index(boundaries[3])
            )
        assert (
            not server.sent("session/prompt")[1]["params"].get("prompt") == []
        )
        assert len(server.sent("session/prompt")) == 2


async def test_interrupt_cancels_the_turn_of_a_late_steer(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    server.turns = [([text("ONE", "a1")], None)]
    async with asyncio.timeout(5), harness:
        feed = harness.session.events()
        await harness.session.send(ai.user_message("ONE"))
        await harness.session.send(ai.user_message("TWO"))
        request = server.steers[0]
        server.cancel_request = request
        server.finish(server.prompts[0], END)
        server.says(text("TWO", "a2"))
        events = []
        async for event in feed:
            events.append(event)
            if isinstance(event, ai.events.TextDelta) and event.chunk == "TWO":
                break
        await harness.session.interrupt()
        (cancel,) = server.sent("session/cancel")
        assert cancel["params"] == {"sessionId": SESSION}
        assert [(msg.role, msg.text) for msg in harness.session.messages] == [
            ("user", "ONE"),
            ("assistant", "ONE"),
            ("user", "TWO"),
            ("assistant", "TWO"),
        ]
        assert (harness.session.messages[-1].provider_metadata or {})[
            "external"
        ] == {"stop_reason": "interrupted"}


async def test_a_rejected_steer_waits_for_the_running_turn_then_retries(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    server.turns = [([text("ONE", "a1")], None), ([text("TWO", "a2")], END)]
    async with asyncio.timeout(5), harness:
        feed = harness.session.events()
        await harness.session.send(ai.user_message("ONE"))
        await harness.session.send(ai.user_message("TWO"))
        request = server.steers[0]
        server.says(
            {
                "id": request["id"],
                "error": {
                    "code": -32602,
                    "message": "Steering prompt too long",
                },
            },
            text(" CONTINUES", "a1"),
        )
        events = []
        async for event in feed:
            events.append(event)
            if (
                isinstance(event, ai.events.TextDelta)
                and event.chunk == " CONTINUES"
            ):
                break
        assert not any(isinstance(e, ai.events.StreamEnd) for e in events)
        assert [
            msg.text for msg in harness.session.messages if msg.role == "user"
        ] == ["ONE"]
        assert len(server.sent("session/prompt")) == 2
        server.finish(server.prompts[0], END)
        events += await through_turns(feed, 2)
        assert [(msg.role, msg.text) for msg in harness.session.messages] == [
            ("user", "ONE"),
            ("assistant", "ONE CONTINUES"),
            ("user", "TWO"),
            ("assistant", "TWO"),
        ]
        assert (
            len([e for e in events if isinstance(e, ai.events.StreamStart)])
            == 2
        )
        assert [
            e.finish_reason
            for e in events
            if isinstance(e, ai.events.StreamEnd)
        ] == ["stop", "stop"]
        assert len(server.prompts) == 2 and len(server.steers) == 1
        assert server.prompts[1]["params"]["prompt"] == [
            {"type": "text", "text": "TWO"}
        ]


async def test_updates_after_a_prompt_response_do_not_join_the_ended_turn(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    server.turns = [([CALL], None)]
    async with asyncio.timeout(5), harness:
        feed = harness.session.events()
        await harness.session.send(ai.user_message("ONE"))
        server.finish(server.prompts[0], END)
        server.says(RESULT, text("late text", "late"))
        events = await through_turns(feed, 1)
        assert not any(
            isinstance(e, ai.events.BuiltinToolResult | ai.events.TextDelta)
            for e in events
        ), ", ".join(type(e).__name__ for e in events)
        assert harness.session.messages[-1].builtin_tool_returns == []
        assert harness.session.messages[-1].text == ""


@pytest.mark.parametrize(
    "native", ["session_info_update", "notice", "usage_update"]
)
async def test_noncontent_updates_do_not_split_a_text_block(
    server: AcpServer,
    harness: external.Fx,
    native: str,
) -> None:
    fields_by_kind: dict[str, dict[str, Any]] = {
        "session_info_update": {"title": "title"},
        "notice": {"title": "notice", "severity": "info"},
        "usage_update": {"used": 1, "size": 100},
    }
    fields = fields_by_kind[native]
    server.turns = [([text("HE"), update(native, **fields), text("RON")], END)]
    events = []
    async with (
        harness,
        ai.stream(harness, [ai.user_message("go")]) as stream,
    ):
        async for event in stream:
            events.append(event)
    assert stream.text == "HERON"
    assert len([e for e in events if isinstance(e, ai.events.TextStart)]) == 1
    assert len([e for e in events if isinstance(e, ai.events.TextEnd)]) == 1
    assert len(stream.message.parts) == 1


async def test_a_final_tool_update_without_a_call_is_ignored(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    server.turns = [([RESULT, text("done")], END)]
    events = []
    async with (
        harness,
        ai.stream(harness, [ai.user_message("go")]) as stream,
    ):
        async for event in stream:
            events.append(event)
    assert not any(isinstance(e, ai.events.BuiltinToolResult) for e in events)
    assert stream.message.builtin_tool_returns == []


async def test_an_unknown_server_request_gets_an_error_reply(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    async with asyncio.timeout(5), harness:
        await harness.session.start()
        server.says({"id": 98, "method": "unknown/request", "params": {}})
        while not server.replies:
            await asyncio.sleep(0)
        reply = server.replies[0]
        assert reply["id"] == 98 and "result" not in reply
        assert isinstance(reply["error"], dict)
        assert "code" in reply["error"] and "message" in reply["error"]


async def test_credential_refresh_failure_is_not_authenticated(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    server.turns = [
        ([text("credential refresh failed")], {"stopReason": "refused"})
    ]
    with pytest.raises(external.NotAuthenticatedError):
        async with harness:
            await run(harness, [ai.user_message("go")])


async def test_a_crash_while_idle_reaches_session_events(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    async with asyncio.timeout(5), harness:
        await harness.session.start()
        await server.process.terminate()
        with pytest.raises(
            external.ExternalHarnessError, match=r"^fx crashed:"
        ):
            await anext(harness.session.events())


async def test_early_content_waits_for_acceptance_while_write_yields(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    server.process.yield_prompt_write = True
    server.turns = [([text("reply")], END)]
    async with harness:
        stream = await run(harness, [ai.user_message("question")])
        assert [(msg.role, msg.text) for msg in harness.session.messages] == [
            ("user", "question"),
            ("assistant", "reply"),
        ]
    assert stream.text == "reply"


async def test_a_resume_refuses_a_listed_session_with_a_different_cwd(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    server.listed_cwd = "/other"
    with pytest.raises(
        external.ExternalHarnessError,
        match="unknown or belongs to another directory",
    ):
        async with external.Fx(
            cwd=harness.cwd, provider=harness.provider, resume=SESSION
        ) as harness:
            await harness.session.start()
    assert not server.sent("session/resume")


async def test_wire_start_omits_empty_instructions_and_close_is_safe_twice(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    # The close of the session has its own guard. Check the Wire contract
    # separately.
    wire = wire_.FxWire(harness)
    await wire.start(
        session_id=None,
        bridge_url="http://127.0.0.1:1/mcp",
        instructions="",
        seed=[],
    )
    assert "systemPrompt" not in server.sent("session/new")[0]["params"]
    await wire.close()
    await wire.close()
    assert server.process.terminated
    assert len(server.sent("session/close")) == 1


@pytest.mark.parametrize("outcome", ["dropped", "absorbed"])
async def test_a_no_echo_steer_always_ends_and_the_next_send_is_normal(
    server: AcpServer,
    harness: external.Fx,
    outcome: str,
) -> None:
    server.turns = [([text("ONE", "a1")], None), ([text("THREE", "a3")], END)]
    messages = [ai.user_message(value) for value in ("ONE", "TWO", "THREE")]
    async with asyncio.timeout(5), harness:
        feed = harness.session.events()
        await harness.session.send(messages[0])
        await harness.session.send(messages[1])
        request = server.steers[0]
        server.finish(server.prompts[0], END)
        server.says(text("TWO", "a2"))
        server.finish(
            request,
            {
                "stopReason": "cancelled"
                if outcome == "dropped"
                else "end_turn",
                "_meta": {"fx": {"steering": outcome}},
            },
        )
        events = []
        async for event in feed:
            events.append(event)
            if sum(isinstance(e, ai.events.StreamEnd) for e in events) == 2:
                break
        assert [(msg.role, msg.text) for msg in harness.session.messages] == [
            ("user", "ONE"),
            ("assistant", "ONE"),
            ("user", "TWO"),
            ("assistant", "TWO"),
        ]
        await harness.session.send(messages[2])
        events += await through_turns(feed, 1)
        assert [(msg.role, msg.text) for msg in harness.session.messages] == [
            ("user", "ONE"),
            ("assistant", "ONE"),
            ("user", "TWO"),
            ("assistant", "TWO"),
            ("user", "THREE"),
            ("assistant", "THREE"),
        ]
        assert [
            msg.id for msg in harness.session.messages if msg.role == "user"
        ] == [msg.id for msg in messages]
        assert len(server.prompts) == 2 and len(server.steers) == 1
        assert not server.rejected_prompts
        assert (
            len([e for e in events if isinstance(e, ai.events.StreamStart)])
            == 3
        )
        assert [
            e.finish_reason
            for e in events
            if isinstance(e, ai.events.StreamEnd)
        ] == ["stop", "other" if outcome == "dropped" else "stop", "stop"]


async def test_a_send_in_the_gap_before_late_steer_content_keeps_send_order(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    server.turns = [([text("ONE", "a1")], None), ([text("THREE", "a3")], END)]
    messages = [ai.user_message(value) for value in ("ONE", "TWO", "THREE")]
    async with asyncio.timeout(5), harness:
        feed = harness.session.events()
        await harness.session.send(messages[0])
        await harness.session.send(messages[1])
        second = server.steers[0]
        server.finish(server.prompts[0], END)
        events = []
        async for event in feed:
            events.append(event)
            if isinstance(event, ai.events.StreamEnd):
                break
        third_send = asyncio.create_task(harness.session.send(messages[2]))
        # Let send() write, without requiring it to bypass a queue.
        await asyncio.sleep(0.01)
        assert not server.rejected_prompts
        server.says(text("TWO", "a2"))
        server.finish(second, END)
        await server.wait_for("session/prompt", 3)
        third = server.sent("session/prompt")[2]
        if third in server.steers:
            server.says(text("THREE", "a3"))
            server.finish(third, END)
        await third_send
        events += await through_turns(feed, 2)
        assert [(msg.role, msg.text) for msg in harness.session.messages] == [
            ("user", "ONE"),
            ("assistant", "ONE"),
            ("user", "TWO"),
            ("assistant", "TWO"),
            ("user", "THREE"),
            ("assistant", "THREE"),
        ]
        assert [
            msg.id for msg in harness.session.messages if msg.role == "user"
        ] == [msg.id for msg in messages]
        assert not server.rejected_prompts
        assert (
            len([e for e in events if isinstance(e, ai.events.StreamStart)])
            == 3
        )
        assert [
            e.finish_reason
            for e in events
            if isinstance(e, ai.events.StreamEnd)
        ] == ["stop", "stop", "stop"]


@pytest.mark.parametrize("late", ["text", "tool result"])
async def test_a_late_update_after_interrupt_does_not_accept_a_dropped_steer(
    server: AcpServer,
    harness: external.Fx,
    late: str,
) -> None:
    server.turns = [
        ([text("ONE", "a1"), CALL], None),
        ([text("TWO", "a2")], END),
    ]
    server.cancel_result = {"stopReason": "cancelled"}
    server.cancel_updates = [text("LATE", "late") if late == "text" else RESULT]
    messages = [ai.user_message(value) for value in ("ONE", "TWO")]
    async with asyncio.timeout(5), harness:
        feed = harness.session.events()
        await harness.session.send(messages[0])
        await harness.session.send(messages[1])
        request = server.steers[0]
        await harness.session.interrupt()
        server.finish(
            request,
            {
                "stopReason": "cancelled",
                "_meta": {"fx": {"steering": "dropped"}},
            },
        )
        events = await through_turns(feed, 2)
        assert [(msg.role, msg.text) for msg in harness.session.messages] == [
            ("user", "ONE"),
            ("assistant", "ONE"),
            ("user", "TWO"),
            ("assistant", "TWO"),
        ]
        assert [
            msg.id for msg in harness.session.messages if msg.role == "user"
        ] == [msg.id for msg in messages]
        assert not any(
            isinstance(e, ai.events.TextDelta) and e.chunk == "LATE"
            for e in events
        )
        assert not any(
            isinstance(e, ai.events.BuiltinToolResult) for e in events
        )
        assert len(server.prompts) == 2 and len(server.steers) == 1
        assert not server.rejected_prompts
        assert (
            len([e for e in events if isinstance(e, ai.events.StreamStart)])
            == 2
        )
        assert [
            e.finish_reason
            for e in events
            if isinstance(e, ai.events.StreamEnd)
        ] == ["other", "stop"]


async def test_a_cancelled_seed_load_closes_before_removing_its_folder(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    server.hold_load = True
    sibling = server.home / "sessions/v2/UserSession01"
    sibling.mkdir(parents=True)
    (sibling / "log.jsonl").write_bytes(b"keep")
    index = sibling.parent / "index.jsonl"
    index.write_bytes(b"keep index")
    history = [ai.user_message("old"), ai.assistant_message("old answer")]
    harness = external.Fx(
        cwd=harness.cwd, provider=harness.provider, history=history
    )
    async with asyncio.timeout(5), harness:
        task = asyncio.create_task(harness.session.start())
        await server.wait_for("session/load")
        assert server.loaded is not None and server.loaded.exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not server.loaded.exists()
    assert server.closed_folders[0] == (server.loaded.name, True)
    assert (sibling / "log.jsonl").read_bytes() == b"keep"
    assert index.read_bytes() == b"keep index"
    assert server.process.terminated


@pytest.mark.parametrize("ending", ["crash", "close"])
async def test_an_image_send_waiting_for_a_turn_raises_when_the_process_stops(
    server: AcpServer,
    harness: external.Fx,
    ending: str,
) -> None:
    server.turns = [([text("working")], None)]
    image_message = m.Message(
        role="user",
        parts=[
            m.TextPart(text="look"),
            m.FilePart(data=b"image", media_type="image/png"),
        ],
    )
    async with asyncio.timeout(5), harness:
        await harness.session.send(ai.user_message("first"))
        pending = asyncio.create_task(harness.session.send(image_message))
        try:
            await asyncio.sleep(0.01)
            assert not pending.done()
            assert len(server.sent("session/prompt")) == 1
            if ending == "crash":
                await server.process.terminate()
            else:
                await harness.session._close()
            with pytest.raises(external.ExternalHarnessError):
                async with asyncio.timeout(1):
                    await pending
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


async def test_a_newer_steer_echo_does_not_lose_an_older_refused_steer(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    server.turns = [
        ([text("T reply", "a1")], None),
        ([text("OLDER reply", "a3")], END),
    ]
    initial, older, newer = [
        ai.user_message(value) for value in ("T", "OLDER", "NEWER")
    ]
    async with asyncio.timeout(5), harness:
        feed = harness.session.events()
        for message in (initial, older, newer):
            await harness.session.send(message)
        old_request, new_request = server.steers
        server.finish(server.prompts[0], END)
        server.says(
            update(
                "user_message_chunk",
                messageId="newer",
                content={"type": "text", "text": "NEWER"},
                _meta={"fx": {"steering": {"requestId": new_request["id"]}}},
            ),
            text("NEWER reply", "a2"),
        )
        server.finish(new_request, END)
        server.says(
            {
                "id": old_request["id"],
                "error": {
                    "code": -32600,
                    "message": (
                        "Steering prompt was not delivered because "
                        "the turn ended first; send it as a new prompt"
                    ),
                },
            }
        )
        events = await through_turns(feed, 3)
        assert [(msg.role, msg.text) for msg in harness.session.messages] == [
            ("user", "T"),
            ("assistant", "T reply"),
            ("user", "NEWER"),
            ("assistant", "NEWER reply"),
            ("user", "OLDER"),
            ("assistant", "OLDER reply"),
        ]
        assert [
            msg.id for msg in harness.session.messages if msg.role == "user"
        ] == [
            initial.id,
            newer.id,
            older.id,
        ]
        assert [
            request["params"]["prompt"][0]["text"] for request in server.prompts
        ] == ["T", "OLDER"]
        assert not server.rejected_prompts
        assert (
            len([e for e in events if isinstance(e, ai.events.StreamStart)])
            == 3
        )
        assert [
            e.finish_reason
            for e in events
            if isinstance(e, ai.events.StreamEnd)
        ] == ["stop", "stop", "stop"]


@pytest.mark.parametrize(
    ("accepted_outcome", "echo_before_end"),
    [("absorbed", False), ("dropped", False), ("dropped", True)],
)
async def test_three_steers_keep_retry_order_and_accept_each_message_once(
    server: AcpServer,
    harness: external.Fx,
    accepted_outcome: str,
    echo_before_end: bool,
) -> None:
    server.turns = [
        ([text("T reply", "a1")], None),
        ([text("S1 reply", "a3")], None),
        ([text("S3 reply", "a4")], None),
        ([text("NEW reply", "a5")], END),
    ]
    initial, refused, accepted, dropped, queued = [
        ai.user_message(value) for value in ("T", "S1", "S2", "S3", "NEW")
    ]
    async with asyncio.timeout(5), harness:
        feed = harness.session.events()
        server.process.fail_next_prompt = True
        failed = ai.user_message("not written")
        with pytest.raises(
            (OSError, external.ExternalHarnessError),
            match="scripted write failed",
        ):
            await harness.session.send(failed)
        await harness.session.send(initial)
        assert len(server.prompts) == 1 and not server.steers
        for message in (refused, accepted, dropped):
            await harness.session.send(message)
        s1, s2, s3 = server.steers
        server.cancel_request = s2
        if not echo_before_end:
            server.finish(server.prompts[0], END)
        server.says(
            update(
                "user_message_chunk",
                messageId="s2",
                content={"type": "text", "text": "S2"},
                _meta={"fx": {"steering": {"requestId": s2["id"]}}},
            ),
            text("S2 reply", "a2"),
        )
        # Reverse reply order must not reverse the retry queue (S1, then S3).
        server.finish(
            s3,
            {
                "stopReason": "cancelled",
                "_meta": {"fx": {"steering": "dropped"}},
            },
        )
        server.says(
            {
                "id": s1["id"],
                "error": {
                    "code": -32600,
                    "message": (
                        "Steering prompt was not delivered because "
                        "the turn ended first; send it as a new prompt"
                    ),
                },
            },
            text(" READY", "a2"),
        )
        events = []
        async for event in feed:
            events.append(event)
            if (
                isinstance(event, ai.events.TextDelta)
                and event.chunk == " READY"
            ):
                break
        new_send = asyncio.create_task(harness.session.send(queued))
        await asyncio.sleep(0.01)
        assert [
            request["params"]["prompt"][0]["text"]
            for request in server.sent("session/prompt")
        ] == ["T", "S1", "S2", "S3"]
        if echo_before_end:
            server.finish(server.prompts[0], END)
        server.finish(
            s2,
            {
                "stopReason": "end_turn"
                if accepted_outcome == "absorbed"
                else "cancelled",
                "_meta": {"fx": {"steering": accepted_outcome}},
            },
        )
        await server.wait_for("session/prompt", 5)
        first_retry = server.prompts[1]
        assert first_retry["params"]["prompt"] == [
            {"type": "text", "text": "S1"}
        ]
        assert not server.rejected_prompts
        server.finish(first_retry, END)
        await server.wait_for("session/prompt", 6)
        second_retry = server.prompts[2]
        assert second_retry["params"]["prompt"] == [
            {"type": "text", "text": "S3"}
        ]
        server.finish(second_retry, END)
        await server.wait_for("session/prompt", 7)
        new_request = server.sent("session/prompt")[6]
        if new_request in server.steers:
            server.says(text("NEW reply", "a5"))
            server.finish(new_request, END)
        await new_send
        events += await through_turns(feed, 4)
        assert [(msg.role, msg.text) for msg in harness.session.messages] == [
            ("user", "T"),
            ("assistant", "T reply"),
            ("user", "S2"),
            ("assistant", "S2 reply READY"),
            ("user", "S1"),
            ("assistant", "S1 reply"),
            ("user", "S3"),
            ("assistant", "S3 reply"),
            ("user", "NEW"),
            ("assistant", "NEW reply"),
        ]
        assert [
            msg.id for msg in harness.session.messages if msg.role == "user"
        ] == [
            initial.id,
            accepted.id,
            refused.id,
            dropped.id,
            queued.id,
        ]
        sent_texts = [
            request["params"]["prompt"][0]["text"]
            for request in server.sent("session/prompt")
        ]
        assert sent_texts == ["T", "S1", "S2", "S3", "S1", "S3", "NEW"]
        assert sent_texts.count("S2") == 1
        assert not server.rejected_prompts
        assert (
            len([e for e in events if isinstance(e, ai.events.StreamStart)])
            == 5
        )
        assert [
            e.finish_reason
            for e in events
            if isinstance(e, ai.events.StreamEnd)
        ] == [
            "other" if echo_before_end else "stop",
            "stop"
            if echo_before_end or accepted_outcome == "absorbed"
            else "other",
            "stop",
            "stop",
            "stop",
        ]


async def test_seeding_warns_when_reasoning_is_not_carried_into_fx(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    history = [
        ai.user_message("old"),
        m.Message(
            role="assistant",
            parts=[
                m.ReasoningPart(text="private thought"),
                m.TextPart(text="old reply"),
            ],
        ),
        ai.user_message("new"),
    ]
    server.turns = [([text("fresh")], END)]
    with pytest.warns(UserWarning, match=r"1 reasoning part\(s\).*"):
        async with harness:
            stream = await run(harness, history)
    assert stream.text == "fresh"
    assert "private thought" not in json.dumps(server.loaded_records)


async def test_close_fails_a_send_waiting_behind_the_retry_queue(
    server: AcpServer,
    harness: external.Fx,
) -> None:
    server.turns = [
        ([text("first", "a1")], None),
        ([text("retry", "a2")], None),
    ]
    async with asyncio.timeout(5), harness:
        await harness.session.send(ai.user_message("T"))
        await harness.session.send(ai.user_message("S1"))
        await harness.session.send(ai.user_message("S2"))
        server.finish(server.prompts[0], END)
        for request in server.steers:
            server.finish(
                request,
                {
                    "stopReason": "cancelled",
                    "_meta": {"fx": {"steering": "dropped"}},
                },
            )
        await server.wait_for("session/prompt", 4)
        pending = asyncio.create_task(
            harness.session.send(ai.user_message("NEW"))
        )
        try:
            await asyncio.sleep(0.01)
            assert not pending.done()
            assert len(server.sent("session/prompt")) == 4
            await harness.session._close()
            with pytest.raises(external.ExternalHarnessError):
                async with asyncio.timeout(1):
                    await pending
            assert all(
                request["params"]["prompt"][0]["text"] != "NEW"
                for request in server.sent("session/prompt")
            )
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
