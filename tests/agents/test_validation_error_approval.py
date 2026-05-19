"""Tool with require_approval=True and invalid args from the model.

When the model sends arguments that fail Pydantic validation for a tool
that requires approval, the agent should:
1. Still fire the approval hook (with best-effort kwargs)
2. Return an error tool result (not crash)
3. Allow the agent loop to continue so the model can retry
"""

from __future__ import annotations

import pydantic

import ai
from ai.types import events as events_
from ai.types import messages as messages_

from ..conftest import MOCK_MODEL, mock_llm, text_msg


class TextEdit(pydantic.BaseModel):
    old_text: str = pydantic.Field(alias="oldText")
    new_text: str = pydantic.Field(alias="newText")


@ai.tool(require_approval=True)
async def edit(path: str, edits: list[TextEdit]) -> str:
    """Edit a file."""
    return f"Edited {len(edits)} block(s) in {path}"


async def test_invalid_args_with_approval_returns_error_result() -> None:
    """Invalid tool args on an approval-gated tool should produce an error
    tool result without firing the approval hook."""
    my_agent = ai.agent(tools=[edit])

    # Model sends edits as a dict instead of a list — this will fail validation
    bad_call = messages_.Message(
        id="msg-1",
        role="assistant",
        parts=[
            messages_.ToolCallPart(
                tool_call_id="tc-1",
                tool_name="edit",
                tool_args=(
                    '{"path": "test.txt", '
                    '"edits": {"oldText": "foo", "newText": "bar"}}'
                ),
            ),
        ],
    )
    # After getting the error, model responds with text
    final = text_msg("Sorry, I made an error with the edit format.", id="msg-2")
    llm = mock_llm([[bad_call], [final]])

    hook_events: list[events_.HookEvent] = []

    async with my_agent.run(
        MOCK_MODEL, [ai.user_message("edit something")]
    ) as stream:
        async for event in stream:
            if (
                isinstance(event, events_.HookEvent)
                and event.hook.status == "pending"
            ):
                hook_events.append(event)
                ai.resolve_hook(
                    event.hook.hook_id,
                    ai.tools.ToolApproval(granted=True, reason="auto"),
                )

    # The agent should have completed without raising
    assert llm.call_count == 2

    # No approval hook should have fired — bad args skip the hook
    assert len(hook_events) == 0

    # There should be an error tool result in the messages
    tool_msgs = [m for m in stream.messages if m.role == "tool"]
    assert len(tool_msgs) >= 1
    error_results = [r for m in tool_msgs for r in m.tool_results if r.is_error]
    assert len(error_results) >= 1
    assert "ValidationError" in str(error_results[0].result)


async def test_invalid_args_skips_approval_hook() -> None:
    """Invalid args should produce a validation error result without
    ever prompting for approval."""
    my_agent = ai.agent(tools=[edit])

    bad_call = messages_.Message(
        id="msg-1",
        role="assistant",
        parts=[
            messages_.ToolCallPart(
                tool_call_id="tc-1",
                tool_name="edit",
                tool_args=(
                    '{"path": "test.txt", '
                    '"edits": {"oldText": "foo", "newText": "bar"}}'
                ),
            ),
        ],
    )
    final = text_msg("OK, let me fix that.", id="msg-2")
    llm = mock_llm([[bad_call], [final]])

    hook_fired = False

    async with my_agent.run(
        MOCK_MODEL, [ai.user_message("edit something")]
    ) as stream:
        async for event in stream:
            if (
                isinstance(event, events_.HookEvent)
                and event.hook.status == "pending"
            ):
                hook_fired = True
                ai.resolve_hook(
                    event.hook.hook_id,
                    ai.tools.ToolApproval(granted=True, reason="auto"),
                )

    assert not hook_fired, "Approval hook should not fire for invalid args"
    assert llm.call_count == 2
    tool_msgs = [m for m in stream.messages if m.role == "tool"]
    assert len(tool_msgs) >= 1
    error_results = [r for m in tool_msgs for r in m.tool_results if r.is_error]
    assert len(error_results) >= 1
    assert "ValidationError" in str(error_results[0].result)


async def test_completely_invalid_json_with_approval() -> None:
    """Completely unparseable tool_args should also be handled gracefully."""
    my_agent = ai.agent(tools=[edit])

    bad_call = messages_.Message(
        id="msg-1",
        role="assistant",
        parts=[
            messages_.ToolCallPart(
                tool_call_id="tc-1",
                tool_name="edit",
                tool_args='{"path": "test.txt", "edits": ',  # truncated JSON
            ),
        ],
    )
    final = text_msg("Let me try again.", id="msg-2")
    llm = mock_llm([[bad_call], [final]])

    async with my_agent.run(
        MOCK_MODEL, [ai.user_message("edit something")]
    ) as stream:
        async for event in stream:
            if (
                isinstance(event, events_.HookEvent)
                and event.hook.status == "pending"
            ):
                ai.resolve_hook(
                    event.hook.hook_id,
                    ai.tools.ToolApproval(granted=True, reason="auto"),
                )

    assert llm.call_count == 2
    tool_msgs = [m for m in stream.messages if m.role == "tool"]
    assert len(tool_msgs) >= 1
    error_results = [r for m in tool_msgs for r in m.tool_results if r.is_error]
    assert len(error_results) >= 1
