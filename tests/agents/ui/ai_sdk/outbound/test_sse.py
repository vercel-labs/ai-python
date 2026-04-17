from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from ai.agents.ui.ai_sdk import protocol, to_sse
from ai.agents.ui.ai_sdk.outbound.sse import format_sse, serialize_part
from ai.types import messages as messages_


def test_serialize_part_camelcases_keys() -> None:
    part = protocol.StartPart(message_id="m1")
    payload = json.loads(serialize_part(part))
    assert payload == {"type": "start", "messageId": "m1"}


def test_format_sse_wraps_data_line() -> None:
    part = protocol.TextDeltaPart(id="t1", delta="hi")
    line = format_sse(part)
    assert line.startswith("data: ")
    assert line.endswith("\n\n")


def test_serialize_data_part_uses_type_with_prefix() -> None:
    part = protocol.DataPart(data_type="custom", data={"k": 1})
    payload = json.loads(serialize_part(part))
    assert payload["type"] == "data-custom"
    assert "dataType" not in payload


async def _gen(
    msgs: list[messages_.Message],
) -> AsyncGenerator[messages_.Message]:
    for m in msgs:
        yield m


async def test_to_sse_emits_data_prefixed_lines() -> None:
    msgs = [
        messages_.Message(
            id="m1",
            role="assistant",
            turn_id="t1",
            parts=[messages_.TextPart(text="hi")],
            stream=messages_.MessageStreamState(events=[], is_done=True),
        )
    ]
    lines = [line async for line in to_sse(_gen(msgs))]
    assert all(line.startswith("data: ") for line in lines)
    # first line is the start part
    first = json.loads(lines[0].removeprefix("data: ").rstrip())
    assert first["type"] == "start"
