"""The seeding rule, pinned without a CLI.

Text verbatim; tool calls as native items with the ORIGINAL name and arguments;
tool results as strings bound to their call; reasoning as a user-role note after
the turn it belonged to (codex), or not at all (claude, fx, pi). An external
harness holds a tool an `ai.Agent` ran as a call and a `tool` message, and a
tool the CLI ran as built-in parts in one message: each renderer takes both, and
both seed the same.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import pydantic
import pytest

from ai.external import _utils
from ai.external.claude_code import handoff as claude_handoff
from ai.external.codex import handoff as codex_handoff
from ai.external.fx import handoff as fx_handoff
from ai.external.pi import handoff as pi_handoff
from ai.types import messages as m
from ai.types.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    Message,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)

# Write as a tool an `ai.Agent` ran: its result in a `tool` message.
AGENT_TOOL = [
    Message(role="user", parts=[TextPart(text="Create recipe.txt")]),
    Message(
        role="assistant",
        parts=[
            ReasoningPart(text="The user wants a file; Write is the tool."),
            TextPart(text="Creating it."),
            ToolCallPart(
                tool_call_id="call_1",
                tool_name="Write",
                tool_args=json.dumps(
                    {"file_path": "/w/recipe.txt", "content": "two eggs"}
                ),
            ),
        ],
    ),
    Message(
        role="tool",
        parts=[
            ToolResultPart(
                tool_call_id="call_1",
                tool_name="Write",
                result={"ok": True},
                result_kind="json",
            )
        ],
    ),
    Message(role="assistant", parts=[TextPart(text="Done.")]),
]

# The same conversation, Write as a tool the CLI ran: built-in parts.
CLI_TOOL = [
    AGENT_TOOL[0],
    Message(
        role="assistant",
        parts=[
            ReasoningPart(text="The user wants a file; Write is the tool."),
            TextPart(text="Creating it."),
            BuiltinToolCallPart(
                tool_call_id="call_1",
                tool_name="Write",
                tool_args=json.dumps(
                    {"file_path": "/w/recipe.txt", "content": "two eggs"}
                ),
            ),
            BuiltinToolReturnPart(
                tool_call_id="call_1", tool_name="Write", result={"ok": True}
            ),
        ],
    ),
    AGENT_TOOL[3],
]


def test_codex_items_follow_the_rule() -> None:
    items = codex_handoff.to_codex_items(AGENT_TOOL)
    kinds = [(i["type"], i.get("role")) for i in items]
    assert kinds == [
        ("message", "user"),
        ("message", "assistant"),  # "Creating it."
        ("function_call", None),  # Write, name untouched
        ("message", "user"),  # the reasoning note, after its turn
        ("function_call_output", None),
        ("message", "assistant"),  # "Done."
    ]
    call = items[2]
    assert call["name"] == "Write"
    assert json.loads(call["arguments"])["content"] == "two eggs"
    assert call["call_id"] == "call_1"
    out = items[4]
    assert out["call_id"] == "call_1"
    assert out["output"] == json.dumps({"ok": True})
    note = items[3]["content"][0]["text"]
    assert note.startswith(codex_handoff.NOTE_PREFIX)
    assert "Write is the tool" in note


async def test_both_shapes_seed_the_same() -> None:
    assert codex_handoff.to_codex_items(CLI_TOOL) == (
        codex_handoff.to_codex_items(AGENT_TOOL)
    )
    assert pi_handoff.to_pi_messages(CLI_TOOL, timestamp=1) == (
        pi_handoff.to_pi_messages(AGENT_TOOL, timestamp=1)
    )
    cli = await claude_handoff.to_claude_records(CLI_TOOL, "s", "/w")
    agent = await claude_handoff.to_claude_records(AGENT_TOOL, "s", "/w")
    assert [(r["type"], r["message"]) for r in cli] == [
        (r["type"], r["message"]) for r in agent
    ]


async def test_claude_records_follow_the_rule() -> None:
    recs = await claude_handoff.to_claude_records(CLI_TOOL, "sid-1", "/w")
    # user prompt, assistant (text + tool_use), tool result, assistant; NO
    # reasoning note: Claude's safeguards refuse replayed reasoning.
    assert [r["type"] for r in recs] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assistant = recs[1]["message"]["content"]
    assert [b["type"] for b in assistant] == ["text", "tool_use"]
    assert assistant[1]["name"] == "Write"
    assert assistant[1]["input"]["content"] == "two eggs"
    assert assistant[1]["id"] == "call_1"
    result = recs[2]["message"]["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "call_1"
    assert _utils.count_reasoning_parts(CLI_TOOL) == 1  # what is warned about
    # a real chain, in the CLI's own store shape
    assert recs[0]["parentUuid"] is None
    assert all(
        recs[i]["parentUuid"] == recs[i - 1]["uuid"]
        for i in range(1, len(recs))
    )
    assert all(r["sessionId"] == "sid-1" and r["cwd"] == "/w" for r in recs)


def test_pi_merges_assistant_messages_and_keeps_tool_results() -> None:
    rendered = pi_handoff.to_pi_messages(
        [
            Message(role="user", parts=[TextPart(text="hi")]),
            Message(role="assistant", parts=[TextPart(text="one")]),
            Message(
                role="assistant",
                parts=[
                    BuiltinToolCallPart(
                        tool_call_id="c1", tool_name="Bash", tool_args="{}"
                    ),
                    BuiltinToolReturnPart(
                        tool_call_id="c1",
                        tool_name="Bash",
                        result={"ok": True},
                        is_error=True,
                    ),
                ],
            ),
        ],
        timestamp=1,
    )
    assert [r["role"] for r in rendered] == ["user", "assistant", "toolResult"]
    assert [b["type"] for b in rendered[1]["content"]] == ["text", "toolCall"]
    assert rendered[1]["stopReason"] == "toolUse"
    assert rendered[2]["content"] == [{"type": "text", "text": '{"ok": true}'}]
    assert rendered[2]["isError"] is True


def test_text_after_a_cli_tool_stays_after_its_result() -> None:
    one = Message(
        role="assistant",
        parts=[
            BuiltinToolCallPart(tool_call_id="c", tool_name="Read"),
            BuiltinToolReturnPart(
                tool_call_id="c", tool_name="Read", result="x"
            ),
            TextPart(text="It says x."),
        ],
    )
    items = codex_handoff.to_codex_items([one])
    assert [i["type"] for i in items] == [
        "function_call",
        "function_call_output",
        "message",
    ]
    rendered = pi_handoff.to_pi_messages([one], timestamp=1)
    assert [r["role"] for r in rendered] == [
        "assistant",
        "toolResult",
        "assistant",
    ]


def test_reasoning_is_never_the_agents_own_thought() -> None:
    """No reasoning item: measured to be ignored. It travels as a note."""
    items = codex_handoff.to_codex_items(AGENT_TOOL)
    assert not any(i["type"] == "reasoning" for i in items)


def test_empty_parts_produce_nothing() -> None:
    blank = [Message(role="assistant", parts=[TextPart(text="   ")])]
    assert codex_handoff.to_codex_items(blank) == []


FX_PREFS = {
    "provider": "gateway",
    "model": "openai/gpt-5.4-nano",
    "effort": "auto",
    "fast_mode": False,
}


def read_fx_log(home: Path, session_id: str) -> list[dict[str, Any]]:
    """Check the byte frame independently of the writer's checksum code."""
    folder = home / "sessions/v2" / session_id
    assert (folder / "lock").read_bytes() == b""
    assert not (home / "sessions/v2/index.jsonl").exists()
    records = []
    for number, line in enumerate(
        (folder / "log.jsonl").read_bytes().splitlines(keepends=True), start=1
    ):
        assert line.endswith(b"}\n")
        body, checksum = line.rsplit(b',"crc":"', 1)
        record = json.loads(line)
        assert list(record)[:4] == ["v", "seq", "ts", "kind"]
        assert list(record)[-1] == "crc"
        assert record["v"] == 1 and record["seq"] == number
        # Bitwise reference implementation: no table from the source.
        crc = 0xFFFFFFFF
        for byte in body:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
        assert checksum == f"{crc ^ 0xFFFFFFFF:08x}".encode() + b'"}\n'
        records.append(record)
    return records


class ProjectedResult(pydantic.BaseModel):
    published_on: datetime.date
    hidden: str = pydantic.Field(default="do not replay", exclude=True)


class WrappedBundle(pydantic.BaseModel):
    bundles: list[m.MessageBundle]


PROJECTED_BUNDLE = m.MessageBundle(
    messages=(
        Message(
            role="tool",
            parts=[
                ToolResultPart(
                    tool_call_id="inner",
                    tool_name="Read",
                    result="private nested result",
                    model_input="nested model input",
                ),
            ],
        ),
        Message(role="assistant", parts=[TextPart(text="Public reply")]),
    )
)


@pytest.mark.parametrize(
    ("model_input", "expected"),
    [
        pytest.param("public", "public", id="text"),
        pytest.param({"public": True}, '{"public": true}', id="json"),
        pytest.param(None, "null", id="explicit-null"),
        pytest.param(
            ProjectedResult(published_on=datetime.date(2026, 11, 17)),
            '{"published_on": "2026-11-17"}',
            id="pydantic-model",
        ),
        pytest.param(
            m.ContentOutput(
                value=[
                    TextPart(text="Visible description"),
                    m.FilePart(data=b"not inlined", media_type="image/png"),
                ]
            ),
            "Visible description\n[file: image/png]",
            id="multipart",
        ),
        pytest.param(
            PROJECTED_BUNDLE,
            "Public reply",
            id="message-bundle",
        ),
        pytest.param(
            WrappedBundle(bundles=[PROJECTED_BUNDLE]),
            '"[unsupported structured output omitted]"',
            id="wrapped-message-bundle",
        ),
    ],
)
async def test_handoffs_use_model_input_for_tool_results(
    tmp_path: Path, model_input: object, expected: str
) -> None:
    raw_result = {"private": "not for the model"}
    part = ToolResultPart(
        tool_call_id="call_1",
        tool_name="Write",
        result=raw_result,
        model_input=model_input,
    )
    messages = [
        *AGENT_TOOL[:2],
        Message(role="tool", parts=[part]),
        AGENT_TOOL[-1],
    ]

    claude = await claude_handoff.to_claude_records(messages, "sid", "/w")
    claude_result = claude[2]["message"]["content"][0]
    assert claude_result["tool_use_id"] == "call_1"
    assert claude_result["content"] == expected

    codex = codex_handoff.to_codex_items(messages)
    codex_result = next(i for i in codex if i["type"] == "function_call_output")
    assert codex_result["call_id"] == "call_1"
    assert codex_result["output"] == expected

    pi = pi_handoff.to_pi_messages(messages, timestamp=1)
    pi_result = next(i for i in pi if i["role"] == "toolResult")
    assert pi_result["toolCallId"] == "call_1"
    assert pi_result["content"] == [{"type": "text", "text": expected}]

    session_id, _ = fx_handoff.write_session(
        messages,
        fx_home=str(tmp_path),
        cwd="/w",
        prefs=FX_PREFS,
        instructions=None,
    )
    fx_result = next(
        r
        for r in read_fx_log(tmp_path, session_id)
        if r.get("type") == "tool_result"
    )
    assert fx_result["data"]["call_id"] == "call_1"
    assert fx_result["data"]["preview"] == expected
    (digest,) = fx_result["blobs"]
    blobs = tmp_path / "sessions/v2" / session_id / "blobs"
    assert (blobs / digest).read_bytes() == expected.encode()
    assert part.result == raw_result


def test_fx_crc32c_has_the_standard_check_value() -> None:
    assert fx_handoff.crc32c(b"123456789") == 0xE3069283
    assert fx_handoff.crc32c(b"") == 0


def test_fx_log_matches_the_recorded_text_session(tmp_path: Path) -> None:
    # These records are the first turn in the real ExampleSeed01 fixture.
    instructions = (
        "Seeded session. Be brief. If asked for the pass phrase "
        "of the instructions, it is ORCHID-22."
    )
    session_id, turns = fx_handoff.write_session(
        [
            Message(
                role="user",
                parts=[
                    TextPart(text="My secret word is PLUMBAGO-7. Remember it.")
                ],
            ),
            Message(role="assistant", parts=[TextPart(text="Noted.")]),
        ],
        fx_home=str(tmp_path),
        cwd="/work/project",
        prefs=FX_PREFS,
        instructions=instructions,
        session_id="ExampleSeed01",
        now_ms=1794857000000,
    )
    assert session_id == "ExampleSeed01" and turns == 1
    records = read_fx_log(tmp_path, session_id)
    assert [r["kind"] for r in records] == [
        "session_created",
        "set",
        "set",
        "set",
        "set",
        "turn_started",
        "item",
        "item",
        "item",
        "turn_committed",
        "closed",
    ]
    created = {
        k: v
        for k, v in records[0].items()
        if k not in ("v", "seq", "ts", "crc")
    }
    assert created == {
        "kind": "session_created",
        "id": session_id,
        "workspace": "/work/project",
        "role": "root",
        "host": "acp",
    }
    assert [(r["key"], r["value"]) for r in records if r["kind"] == "set"] == [
        ("prefs", FX_PREFS),
        (
            "permissions",
            {"schema_version": 2, "next_generation": 1, "rules": []},
        ),
        ("language", "und"),
        ("client_prompt", instructions),
    ]
    assert [(r["type"], r["data"]) for r in records if r["kind"] == "item"] == [
        (
            "user",
            {
                "text": "My secret word is PLUMBAGO-7. Remember it.",
                "images": [],
                "work_id": None,
            },
        ),
        (
            "assistant",
            {
                "text": "Noted.",
                "provider_replay": None,
                "standalone_response": False,
            },
        ),
        ("turn_end", {"files": [], "turn_summary": None}),
    ]


def test_fx_both_tool_shapes_keep_names_arguments_and_results(
    tmp_path: Path,
) -> None:
    rendered = []
    for index, messages in enumerate((AGENT_TOOL, CLI_TOOL)):
        home = tmp_path / str(index)
        session_id, turns = fx_handoff.write_session(
            messages,
            fx_home=str(home),
            cwd="/w",
            prefs=FX_PREFS,
            instructions=None,
            now_ms=1000,
        )
        assert turns == 1
        records = read_fx_log(home, session_id)
        items = [r for r in records if r["kind"] == "item"]
        assert [r["type"] for r in items] == [
            "user",
            "assistant",
            "tool_call",
            "tool_result",
            "assistant",
            "turn_end",
        ]
        call, result = items[2]["data"], items[3]["data"]
        assert call == {
            "call_id": "call_1",
            "tool_name": "Write",
            "arguments_json": call["arguments_json"],
            "argument_integrity": "valid",
            "provisional_id": None,
            "provider_result": None,
            "final_identity": "valid",
            "provenance": "fx_local",
        }
        assert json.loads(call["arguments_json"]) == {
            "file_path": "/w/recipe.txt",
            "content": "two eggs",
        }
        assert result["call_id"] == call["call_id"]
        assert result["tool_name"] == call["tool_name"]
        assert json.loads(result["preview"]) == {"ok": True}
        assert result["status"] == "success"
        assert set(result) == {
            "call_id",
            "tool_name",
            "status",
            "artifact_ref",
            "tool_image_handle",
            "output_bytes",
            "stored_bytes",
            "completeness",
            "preview",
            "provider_native",
            "created_at_ms",
            "permission_feedback",
            "committed_file_presentation",
            "command_replay_ref",
            "command_replay_bytes",
            "command_process_presentation",
            "terminal_action_presentation",
        }
        assert "Write is the tool" not in json.dumps(items)
        rendered.append(
            [
                (
                    r["type"],
                    {
                        k: v
                        for k, v in r["data"].items()
                        if k != "created_at_ms"
                    },
                )
                for r in items
            ]
        )
    assert rendered[0] == rendered[1]


def test_fx_long_tool_results_are_complete_in_a_blob(tmp_path: Path) -> None:
    body = "é" * 3000
    messages = [
        Message(role="user", parts=[TextPart(text="read")]),
        Message(
            role="assistant",
            parts=[
                BuiltinToolCallPart(
                    tool_call_id="c", tool_name="Read", tool_args="{}"
                ),
                BuiltinToolReturnPart(
                    tool_call_id="c",
                    tool_name="Read",
                    result=body,
                    is_error=True,
                ),
                TextPart(text="After the result."),
            ],
        ),
    ]
    session_id, _ = fx_handoff.write_session(
        messages,
        fx_home=str(tmp_path),
        cwd="/w",
        prefs=FX_PREFS,
        instructions=None,
    )
    records = read_fx_log(tmp_path, session_id)
    items = [r for r in records if r["kind"] == "item"]
    assert [r["type"] for r in items] == [
        "user",
        "assistant",
        "tool_call",
        "tool_result",
        "assistant",
        "turn_end",
    ]
    result = next(r for r in items if r["type"] == "tool_result")
    digest = hashlib.sha256(body.encode()).hexdigest()
    assert result["blobs"] == [digest]
    assert (
        tmp_path / "sessions/v2" / session_id / "blobs" / digest
    ).read_bytes() == body.encode()
    data = result["data"]
    assert data["artifact_ref"] == f"result-{digest}.txt"
    assert data["status"] == "failure" and data["completeness"] == "complete"
    assert data["stored_bytes"] == len(body.encode())
    assert 0 < len(data["preview"].encode()) <= 4096
    assert body.startswith(data["preview"])
    assert items[-2]["data"]["text"] == "After the result."


def test_fx_images_use_the_recorded_blob_shape(tmp_path: Path) -> None:
    picture = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKElEQVR4nO3NsQ0AAAzCMP5/un0CNkuZ41wybXsHAAAAAAAAAAAAxR4yw/wuPL6QkAAAAABJRU5ErkJggg=="
    )
    messages = [
        Message(
            role="user",
            parts=[
                TextPart(text="Here is a picture."),
                m.FilePart(data=picture, media_type="image/png"),
            ],
        )
    ]
    session_id, turns = fx_handoff.write_session(
        messages,
        fx_home=str(tmp_path),
        cwd="/w",
        prefs=FX_PREFS,
        instructions=None,
    )
    assert turns == 1
    records = read_fx_log(tmp_path, session_id)
    user = next(r for r in records if r.get("type") == "user")
    digest = hashlib.sha256(picture).hexdigest()
    assert (
        digest
        == "becf0cf30d6978325d9ccdea315923e2dbbf7e591c2857b92b5f82e41957bb98"
    )
    (image,) = user["data"]["images"]
    assert set(image) == {"id", "path", "media_type", "blob"}
    assert (
        image["id"] == 1
        and image["media_type"] == "image/png"
        and image["blob"] == digest
    )
    assert user["data"]["text"] == "Here is a picture.\n[Image #1]"
    assert user["blobs"] == [digest]
    assert (
        tmp_path / "sessions/v2" / session_id / "blobs" / digest
    ).read_bytes() == picture


def test_fx_assistant_first_past_gets_a_nonempty_user(tmp_path: Path) -> None:
    session_id, turns = fx_handoff.write_session(
        [Message(role="assistant", parts=[TextPart(text="earlier answer")])],
        fx_home=str(tmp_path),
        cwd="/w",
        prefs=FX_PREFS,
        instructions=None,
    )
    assert turns == 1
    items = [
        r for r in read_fx_log(tmp_path, session_id) if r["kind"] == "item"
    ]
    assert [r["type"] for r in items] == ["user", "assistant", "turn_end"]
    assert items[0]["data"]["text"].strip()
    assert items[1]["data"]["text"] == "earlier answer"


def test_fx_consecutive_users_merge_into_one_turn(tmp_path: Path) -> None:
    session_id, turns = fx_handoff.write_session(
        [
            Message(role="user", parts=[TextPart(text="one")]),
            Message(role="user", parts=[TextPart(text="two")]),
        ],
        fx_home=str(tmp_path),
        cwd="/w",
        prefs=FX_PREFS,
        instructions=None,
    )
    assert turns == 1
    items = [
        r for r in read_fx_log(tmp_path, session_id) if r["kind"] == "item"
    ]
    assert [r["type"] for r in items] == ["user", "turn_end"]
    assert (
        "one" in items[0]["data"]["text"] and "two" in items[0]["data"]["text"]
    )
    assert items[0]["data"]["text"].index("one") < items[0]["data"][
        "text"
    ].index("two")


def test_fx_unanswered_calls_get_a_failed_result(tmp_path: Path) -> None:
    session_id, _ = fx_handoff.write_session(
        [
            Message(role="user", parts=[TextPart(text="read")]),
            Message(
                role="assistant",
                parts=[
                    ToolCallPart(
                        tool_call_id="missing",
                        tool_name="foreign_tool",
                        tool_args='{"x":1}',
                    )
                ],
            ),
        ],
        fx_home=str(tmp_path),
        cwd="/w",
        prefs=FX_PREFS,
        instructions=None,
    )
    items = [
        r for r in read_fx_log(tmp_path, session_id) if r["kind"] == "item"
    ]
    call = next(r["data"] for r in items if r["type"] == "tool_call")
    result = next(r["data"] for r in items if r["type"] == "tool_result")
    assert call["call_id"] == result["call_id"] == "missing"
    assert call["tool_name"] == result["tool_name"] == "foreign_tool"
    assert result["status"] == "failure" and result["preview"].strip()


def test_fx_does_not_replace_an_existing_session(tmp_path: Path) -> None:
    kwargs: dict[str, Any] = {
        "fx_home": str(tmp_path),
        "cwd": "/w",
        "prefs": FX_PREFS,
        "instructions": None,
        "session_id": "existing",
    }
    messages = [Message(role="user", parts=[TextPart(text="keep")])]
    fx_handoff.write_session(messages, **kwargs)
    log = tmp_path / "sessions/v2/existing/log.jsonl"
    before = log.read_bytes()
    with pytest.raises(FileExistsError):
        fx_handoff.write_session(messages, **kwargs)
    assert log.read_bytes() == before


def test_fx_parallel_calls_precede_their_paired_results(tmp_path: Path) -> None:
    # The two-call step follows the recorded ExampleSeed01 second turn.
    messages = [
        Message(
            role="user",
            parts=[
                TextPart(
                    text=(
                        "Look up the release code in the docs, "
                        "and read notes.txt."
                    )
                )
            ],
        ),
        Message(
            role="assistant",
            parts=[
                TextPart(text="I will search the docs."),
                ToolCallPart(
                    tool_call_id="call_seed_1",
                    tool_name="search_docs",
                    tool_args='{"query":"release code"}',
                ),
                ToolCallPart(
                    tool_call_id="call_seed_2",
                    tool_name="read_file",
                    tool_args='{"path":"notes.txt"}',
                ),
            ],
        ),
        Message(
            role="tool",
            parts=[
                ToolResultPart(
                    tool_call_id="call_seed_1",
                    tool_name="search_docs",
                    result="The release code is ZX-4417.",
                ),
                ToolResultPart(
                    tool_call_id="call_seed_2",
                    tool_name="read_file",
                    result="notes.txt: the build colour is teal.",
                ),
            ],
        ),
        Message(role="assistant", parts=[TextPart(text="Done.")]),
        Message(role="user", parts=[TextPart(text="Next turn")]),
    ]
    session_id, turns = fx_handoff.write_session(
        messages,
        fx_home=str(tmp_path),
        cwd="/w",
        prefs=FX_PREFS,
        instructions=None,
    )
    assert turns == 2
    records = read_fx_log(tmp_path, session_id)
    items = [r for r in records if r["kind"] == "item" and r["turn"] == 1]
    assert [r["type"] for r in items] == [
        "user",
        "assistant",
        "tool_call",
        "tool_call",
        "tool_result",
        "tool_result",
        "assistant",
        "turn_end",
    ]
    for call, result in zip(items[2:4], items[4:6], strict=True):
        assert call["data"]["call_id"] == result["data"]["call_id"]
        assert call["data"]["tool_name"] == result["data"]["tool_name"]
    assert [
        (r["kind"], r["turn"])
        for r in records
        if r["kind"] in ("turn_started", "turn_committed")
    ] == [
        ("turn_started", 1),
        ("turn_committed", 1),
        ("turn_started", 2),
        ("turn_committed", 2),
    ]


def test_fx_new_store_directories_are_private_with_a_permissive_umask(
    tmp_path: Path,
) -> None:
    home = tmp_path / "new-home"
    previous = os.umask(0o022)
    try:
        session_id, _ = fx_handoff.write_session(
            [Message(role="user", parts=[TextPart(text="private")])],
            fx_home=str(home),
            cwd="/w",
            prefs=FX_PREFS,
            instructions=None,
        )
    finally:
        os.umask(previous)
    for directory in [
        home,
        home / "sessions",
        home / "sessions/v2",
        home / "sessions/v2/.tmp",
        home / "sessions/v2" / session_id,
    ]:
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700, directory
    folder = home / "sessions/v2" / session_id
    assert stat.S_IMODE((folder / "log.jsonl").stat().st_mode) == 0o600
    assert stat.S_IMODE((folder / "lock").stat().st_mode) == 0o600


def test_fx_failed_publish_removes_only_its_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sibling = tmp_path / "sessions/v2/existing"
    sibling.mkdir(parents=True)
    (sibling / "log.jsonl").write_bytes(b"keep")

    def fail_publish(src: str, dst: str) -> None:
        raise OSError("publish failed")

    monkeypatch.setattr(os, "rename", fail_publish)
    with pytest.raises(OSError, match=r"^publish failed$"):
        fx_handoff.write_session(
            [Message(role="user", parts=[TextPart(text="private")])],
            fx_home=str(tmp_path),
            cwd="/w",
            prefs=FX_PREFS,
            instructions=None,
            session_id="failed-publish",
        )
    assert not (tmp_path / "sessions/v2/.tmp/failed-publish").exists()
    assert not (tmp_path / "sessions/v2/failed-publish").exists()
    assert (sibling / "log.jsonl").read_bytes() == b"keep"
