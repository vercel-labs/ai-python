from __future__ import annotations

from collections import Counter

from ai.agents.ui import ai_sdk
from ai.agents.ui.ai_sdk import outbound_messages, to_ui_messages
from ai.agents.ui.ai_sdk.ui_messages import (
    UIDynamicToolPart,
    UIFilePart,
    UIReasoningPart,
    UITextPart,
    UIToolApproval,
    UIToolPart,
)
from ai.types import integrity
from ai.types import messages as messages_


def _parallel_tool_turn(
    *,
    turn_id: str,
    assistant_prefix: str | None = None,
    tool_call_ids: tuple[str, str] = ("tc-bash", "tc-web"),
) -> list[messages_.Message]:
    prefix = assistant_prefix or turn_id
    tc_bash, tc_web = tool_call_ids

    return [
        messages_.Message(
            id=f"{prefix}:assistant:0",
            turn_id=turn_id,
            role="assistant",
            parts=[
                messages_.TextPart(
                    id=f"{prefix}:text:0",
                    text="I will run two tools.",
                ),
                messages_.ToolCallPart(
                    id=f"{prefix}:call:bash",
                    tool_call_id=tc_bash,
                    tool_name="bash",
                    tool_args='{"command":"date"}',
                ),
                messages_.ToolCallPart(
                    id=f"{prefix}:call:web",
                    tool_call_id=tc_web,
                    tool_name="web_fetch",
                    tool_args='{"url":"https://httpbin.org/get"}',
                ),
            ],
        ),
        messages_.Message(
            id=f"{prefix}:tool:0",
            turn_id=turn_id,
            role="tool",
            parts=[
                messages_.ToolResultPart(
                    id=f"{prefix}:result:bash",
                    tool_call_id=tc_bash,
                    tool_name="bash",
                    result="Tue May 19 2026",
                ),
                messages_.ToolResultPart(
                    id=f"{prefix}:result:web",
                    tool_call_id=tc_web,
                    tool_name="web_fetch",
                    result={"status": 200},
                ),
            ],
        ),
        messages_.Message(
            id=f"{prefix}:assistant:1",
            turn_id=turn_id,
            role="assistant",
            parts=[
                messages_.TextPart(
                    id=f"{prefix}:text:1",
                    text="Both tools finished.",
                ),
            ],
        ),
    ]


def test_to_ui_parts_text_and_reasoning() -> None:
    parts: list[messages_.Part] = [
        messages_.ReasoningPart(text="thinking"),
        messages_.TextPart(text="hi"),
    ]
    ui_parts = outbound_messages.to_ui_parts(parts)
    assert isinstance(ui_parts[0], UIReasoningPart)
    assert ui_parts[0].text == "thinking"
    assert isinstance(ui_parts[1], UITextPart)
    assert ui_parts[1].text == "hi"


def test_to_ui_parts_tool_call_parses_json_args() -> None:
    parts: list[messages_.Part] = [
        messages_.ToolCallPart(
            tool_call_id="tc1",
            tool_name="search",
            tool_args='{"q": "x"}',
        )
    ]
    ui_parts = outbound_messages.to_ui_parts(parts)
    assert isinstance(ui_parts[0], UIToolPart)
    assert ui_parts[0].type == "tool-search"
    assert ui_parts[0].input == {"q": "x"}
    assert ui_parts[0].state == "input-available"


def test_merge_tool_results_updates_state_and_output() -> None:
    parts: list[messages_.Part] = [
        messages_.ToolCallPart(
            tool_call_id="tc1",
            tool_name="search",
            tool_args="{}",
        )
    ]
    ui_parts = outbound_messages.to_ui_parts(parts)
    outbound_messages.merge_tool_results(
        ui_parts,
        [
            messages_.ToolResultPart(
                tool_call_id="tc1",
                tool_name="search",
                result={"hits": 3},
            )
        ],
    )
    merged = ui_parts[0]
    assert isinstance(merged, UIToolPart)
    assert merged.state == "output-available"
    assert merged.output == {"hits": 3}


def test_merge_approval_signals_pending_then_resolved() -> None:
    parts: list[messages_.Part] = [
        messages_.ToolCallPart(
            tool_call_id="tc1",
            tool_name="delete",
            tool_args="{}",
        )
    ]
    ui_parts = outbound_messages.to_ui_parts(parts)

    outbound_messages.merge_approval_signals(
        ui_parts,
        [
            messages_.HookPart(
                hook_id="approve_tc1",
                hook_type="ToolApproval",
                status="pending",
            )
        ],
    )
    requested = ui_parts[0]
    assert isinstance(requested, UIToolPart)
    assert requested.state == "approval-requested"
    assert isinstance(requested.approval, UIToolApproval)

    outbound_messages.merge_approval_signals(
        ui_parts,
        [
            messages_.HookPart(
                hook_id="approve_tc1",
                hook_type="ToolApproval",
                status="resolved",
                resolution={"granted": True, "reason": None},
            )
        ],
    )
    responded = ui_parts[0]
    assert isinstance(responded, UIToolPart)
    assert responded.state == "approval-responded"
    assert responded.approval is not None
    assert responded.approval.approved is True


def _tool_counts(
    messages: list[messages_.Message],
) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for message in messages:
        for part in message.parts:
            if isinstance(part, messages_.ToolCallPart):
                counts["tool_call", part.tool_call_id] += 1
            elif isinstance(part, messages_.ToolResultPart):
                counts["tool_result", part.tool_call_id] += 1
    return counts


class IdUpsertStore:
    """Small app-like store: persist full history by message id."""

    def __init__(self) -> None:
        self._rows: list[messages_.Message] = []

    def save_full_history(self, messages: list[messages_.Message]) -> None:
        for message in messages:
            if message.role == "system":
                continue

            for index, existing in enumerate(self._rows):
                if existing.id == message.id:
                    self._rows[index] = message
                    break
            else:
                self._rows.append(message)

    def load(self) -> list[messages_.Message]:
        return list(self._rows)


def test_to_ui_messages_user_and_assistant() -> None:
    msgs = [
        messages_.Message(
            id="u1", role="user", parts=[messages_.TextPart(text="hi")]
        ),
        messages_.Message(
            id="a1",
            role="assistant",
            parts=[messages_.TextPart(text="hello back")],
        ),
    ]
    result = to_ui_messages(msgs)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[1].role == "assistant"
    assert result[1].id == "a1"


def test_to_ui_messages_merges_assistant_tool_internal() -> None:
    msgs = [
        messages_.Message(
            id="a1",
            role="assistant",
            parts=[
                messages_.TextPart(text="calling"),
                messages_.ToolCallPart(
                    tool_call_id="tc1",
                    tool_name="search",
                    tool_args='{"q":"x"}',
                ),
            ],
        ),
        messages_.Message(
            role="tool",
            parts=[
                messages_.ToolResultPart(
                    tool_call_id="tc1",
                    tool_name="search",
                    result={"hits": 2},
                )
            ],
        ),
        messages_.Message(
            role="assistant",
            parts=[messages_.TextPart(text="done")],
        ),
    ]
    result = to_ui_messages(msgs)
    assert len(result) == 1
    ui_msg = result[0]
    assert ui_msg.role == "assistant"
    assert ui_msg.id == "a1"
    assert isinstance(ui_msg.parts[0], UITextPart)
    assert ui_msg.parts[0].text == "calling"
    assert isinstance(ui_msg.parts[1], UIToolPart)
    assert ui_msg.parts[1].state == "output-available"
    assert ui_msg.parts[1].output == {"hits": 2}
    assert isinstance(ui_msg.parts[2], UITextPart)
    assert ui_msg.parts[2].text == "done"


def test_to_ui_messages_records_source_messages_in_metadata() -> None:
    msgs = [
        messages_.Message(
            id="turn-1:assistant:0",
            turn_id="turn-1",
            role="assistant",
            parts=[
                messages_.TextPart(id="text-0", text="calling"),
                messages_.ToolCallPart(
                    id="call-0",
                    tool_call_id="tc1",
                    tool_name="search",
                    tool_args="{}",
                ),
            ],
        ),
        messages_.Message(
            id="turn-1:tool:0",
            turn_id="turn-1",
            role="tool",
            parts=[
                messages_.ToolResultPart(
                    id="result-0",
                    tool_call_id="tc1",
                    tool_name="search",
                    result={"hits": 2},
                )
            ],
        ),
        messages_.Message(
            id="turn-1:assistant:1",
            turn_id="turn-1",
            role="assistant",
            parts=[messages_.TextPart(id="text-1", text="done")],
        ),
    ]

    [ui_msg] = to_ui_messages(msgs)

    assert ui_msg.id == "turn-1"
    assert ui_msg.metadata == {
        "aiPython": {
            "sourceMessages": [
                {
                    "id": "turn-1:assistant:0",
                    "role": "assistant",
                    "turnId": "turn-1",
                    "partIds": ["text-0", "call-0"],
                },
                {
                    "id": "turn-1:tool:0",
                    "role": "tool",
                    "turnId": "turn-1",
                    "partIds": ["result-0"],
                },
                {
                    "id": "turn-1:assistant:1",
                    "role": "assistant",
                    "turnId": "turn-1",
                    "partIds": ["text-1"],
                },
            ]
        }
    }


def test_to_ui_messages_internal_role_merges_approval() -> None:
    msgs = [
        messages_.Message(
            id="a1",
            role="assistant",
            parts=[
                messages_.ToolCallPart(
                    tool_call_id="tc1",
                    tool_name="delete",
                    tool_args="{}",
                )
            ],
        ),
        messages_.Message(
            role="internal",
            parts=[
                messages_.HookPart(
                    hook_id="approve_tc1",
                    hook_type="ToolApproval",
                    status="pending",
                )
            ],
        ),
    ]
    result = to_ui_messages(msgs)
    ui_msg = result[0]
    tool_part = ui_msg.parts[0]
    assert isinstance(tool_part, UIToolPart)
    assert tool_part.state == "approval-requested"
    assert tool_part.approval is not None
    assert tool_part.approval.id == "approve_tc1"


def test_to_ui_messages_user_message_uses_own_id() -> None:
    msgs = [
        messages_.Message(
            id="u1", role="user", parts=[messages_.TextPart(text="a")]
        )
    ]
    result = to_ui_messages(msgs)
    assert result[0].id == "u1"


def test_to_ui_messages_uses_first_assistant_id_as_bubble_id() -> None:
    msgs = [
        messages_.Message(
            id="a1",
            role="assistant",
            parts=[messages_.TextPart(text="first")],
        ),
        messages_.Message(
            id="a2",
            role="assistant",
            parts=[messages_.TextPart(text="second")],
        ),
    ]
    result = to_ui_messages(msgs)
    assert len(result) == 1
    assert result[0].id == "a1"


def test_to_ui_messages_preserves_provider_metadata_and_files() -> None:
    msgs = [
        messages_.Message(
            id="a1",
            role="assistant",
            parts=[
                messages_.TextPart(
                    text="hello",
                    provider_metadata={"provider": {"text": True}},
                ),
                messages_.FilePart(
                    data=b"abc",
                    media_type="image/png",
                    filename="image.png",
                    provider_metadata={"provider": {"file": True}},
                ),
            ],
        )
    ]

    result = to_ui_messages(msgs)

    text_part = result[0].parts[0]
    assert isinstance(text_part, UITextPart)
    assert text_part.provider_metadata == {"provider": {"text": True}}

    file_part = result[0].parts[1]
    assert isinstance(file_part, UIFilePart)
    assert file_part.url == "data:image/png;base64,YWJj"
    assert file_part.filename == "image.png"
    assert file_part.provider_metadata == {"provider": {"file": True}}


def test_to_ui_messages_maps_builtin_tools_to_dynamic_parts() -> None:
    msgs = [
        messages_.Message(
            id="a1",
            role="assistant",
            parts=[
                messages_.BuiltinToolCallPart(
                    tool_call_id="tc1",
                    tool_name="web_search",
                    tool_args='{"q":"ai"}',
                    provider_metadata={"provider": {"call": True}},
                ),
                messages_.BuiltinToolReturnPart(
                    tool_call_id="tc1",
                    tool_name="web_search",
                    result={"hits": 1},
                    provider_metadata={"provider": {"result": True}},
                ),
            ],
        )
    ]

    result = to_ui_messages(msgs)

    assert len(result[0].parts) == 1
    tool_part = result[0].parts[0]
    assert isinstance(tool_part, UIDynamicToolPart)
    assert tool_part.provider_executed is True
    assert tool_part.state == "output-available"
    assert tool_part.input == {"q": "ai"}
    assert tool_part.output == {"hits": 1}
    assert tool_part.call_provider_metadata == {"provider": {"call": True}}
    assert tool_part.result_provider_metadata == {"provider": {"result": True}}


def test_collapsed_assistant_turn_roundtrips_internal_ids() -> None:
    original = [
        messages_.Message(
            id="assistant-alpha",
            turn_id="turn-arbitrary",
            role="assistant",
            parts=[
                messages_.TextPart(id="text-alpha", text="calling first"),
                messages_.ToolCallPart(
                    id="call-alpha",
                    tool_call_id="tc-first",
                    tool_name="search",
                    tool_args='{"q":"first"}',
                ),
            ],
        ),
        messages_.Message(
            id="tool-beta",
            turn_id="turn-arbitrary",
            role="tool",
            parts=[
                messages_.ToolResultPart(
                    id="result-beta",
                    tool_call_id="tc-first",
                    tool_name="search",
                    result={"hits": 1},
                )
            ],
        ),
        messages_.Message(
            id="assistant-gamma",
            turn_id="turn-arbitrary",
            role="assistant",
            parts=[
                messages_.TextPart(id="text-gamma", text="calling second"),
                messages_.ToolCallPart(
                    id="call-gamma",
                    tool_call_id="tc-second",
                    tool_name="lookup",
                    tool_args='{"id":2}',
                ),
            ],
        ),
        messages_.Message(
            id="tool-delta",
            turn_id="turn-arbitrary",
            role="tool",
            parts=[
                messages_.ToolResultPart(
                    id="result-delta",
                    tool_call_id="tc-second",
                    tool_name="lookup",
                    result={"value": 2},
                )
            ],
        ),
        messages_.Message(
            id="assistant-epsilon",
            turn_id="turn-arbitrary",
            role="assistant",
            parts=[
                messages_.TextPart(id="text-epsilon", text="all done"),
            ],
        ),
    ]

    [ui_msg] = ai_sdk.to_ui_messages(original)
    roundtripped, approvals = ai_sdk.to_messages([ui_msg])

    assert approvals == []
    assert ui_msg.role == "assistant"
    assert ui_msg.id == "turn-arbitrary"
    assert [m.role for m in roundtripped] == [m.role for m in original]
    assert [m.id for m in roundtripped] == [m.id for m in original]
    assert [m.turn_id for m in roundtripped] == [m.turn_id for m in original]
    assert [[p.id for p in m.parts] for m in roundtripped] == [
        [p.id for p in m.parts] for m in original
    ]


def test_common_id_upsert_persistence_is_idempotent_after_reload() -> None:
    store = IdUpsertStore()

    first_run = [
        messages_.Message(
            id="user-1",
            role="user",
            parts=[messages_.TextPart(id="user-1:text", text="run two tools")],
        ),
        *_parallel_tool_turn(turn_id="turn-1"),
    ]
    store.save_full_history(first_run)

    reloaded_ui = ai_sdk.to_ui_messages(store.load())
    request_history, _ = ai_sdk.to_messages(reloaded_ui)

    second_run_result = [
        *request_history,
        messages_.Message(
            id="user-2",
            role="user",
            parts=[messages_.TextPart(id="user-2:text", text="do nothing")],
        ),
        messages_.Message(
            id="turn-2:assistant:0",
            turn_id="turn-2",
            role="assistant",
            parts=[messages_.TextPart(id="turn-2:text:0", text="standing by")],
        ),
    ]
    store.save_full_history(second_run_result)

    loaded = store.load()
    integrity.prepare_messages(loaded)

    counts = _tool_counts(loaded)
    assert counts["tool_call", "tc-bash"] == 1
    assert counts["tool_result", "tc-bash"] == 1
    assert counts["tool_call", "tc-web"] == 1
    assert counts["tool_result", "tc-web"] == 1


def test_duplicate_tool_copies_do_not_reach_model_integrity() -> None:
    history = [
        *_parallel_tool_turn(turn_id="turn-1", assistant_prefix="server"),
        *_parallel_tool_turn(turn_id="turn-1", assistant_prefix="client"),
    ]

    reloaded_ui = ai_sdk.to_ui_messages(history)
    next_request_history, _ = ai_sdk.to_messages(reloaded_ui)

    integrity.prepare_messages(next_request_history)


def _parked_turn(*, hook_id: str, tool_call_id: str) -> list[messages_.Message]:
    """History as a run parked on a gated tool records it."""
    return [
        messages_.Message(
            id="assistant-1",
            turn_id="turn-1",
            role="assistant",
            parts=[
                messages_.ToolCallPart(
                    id="call-1",
                    tool_call_id=tool_call_id,
                    tool_name="bash",
                    tool_args='{"command": "ls"}',
                )
            ],
        ),
        messages_.Message(
            id="hook-1",
            turn_id="turn-1",
            role="internal",
            parts=[
                messages_.HookPart(
                    id="part-1",
                    hook_id=hook_id,
                    hook_type="ToolApproval",
                    status="pending",
                    metadata={"tool": "bash", "kwargs": {"command": "ls"}},
                    tool_call_id=tool_call_id,
                )
            ],
        ),
    ]


def test_pending_approval_hook_roundtrips_tool_call_id() -> None:
    # Both the conventional approve_<id> label and a custom label must
    # survive: the approval rides the UI tool part via the hook's
    # tool_call_id field, and inbound reconstruction restores it.
    for hook_id in ("approve_tc-1", "my_custom_gate"):
        ui = ai_sdk.to_ui_messages(
            _parked_turn(hook_id=hook_id, tool_call_id="tc-1")
        )
        [tool_part] = [
            p
            for m in ui
            for p in m.parts
            if isinstance(p, UIToolPart | UIDynamicToolPart)
        ]
        assert tool_part.state == "approval-requested"

        back, approval_responses = ai_sdk.to_messages(ui)
        assert approval_responses == []
        [hook] = [
            p
            for m in back
            for p in m.parts
            if isinstance(p, messages_.HookPart)
        ]
        assert hook.hook_id == hook_id
        assert hook.status == "pending"
        assert hook.tool_call_id == "tc-1"
