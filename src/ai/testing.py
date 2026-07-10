"""Deterministic local model for agent and example tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

import pydantic

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

from .models.core import model as model_
from .models.core import params as params_
from .providers import base
from .types import events
from .types import messages as messages_
from .types import tools as tools_

_CALL_PREFIX = "call:"
_PING_PREFIX = "ping:"
_TEXT_BLOCK_ID = "test-text-0"
_MODEL_ID = "test"
_DEFAULT_MAX_COMMAND_CHARS = 64 * 1024
_DEFAULT_MAX_RESULT_CHARS = 1024 * 1024
_MAX_JSON_NESTING_DEPTH = 64
_JSON_ADAPTER = pydantic.TypeAdapter(dict[str, Any])


class _TestClient:
    """Credential-free placeholder used by :class:`TestProvider`."""


_TEST_CLIENT = _TestClient()


def _build_unique_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                "call: tool arguments must not contain duplicate JSON keys"
            )
        result[key] = value
    return result


def _validate_json_nesting(value: Any) -> None:
    pending_values = [(value, 1)]
    while pending_values:
        current_value, depth = pending_values.pop()
        if depth > _MAX_JSON_NESTING_DEPTH:
            raise ValueError("call: tool argument nesting is too deep")

        next_depth = depth + 1
        if isinstance(current_value, dict):
            pending_values.extend(
                (nested_value, next_depth)
                for nested_value in current_value.values()
            )
        elif isinstance(current_value, list):
            pending_values.extend(
                (nested_value, next_depth) for nested_value in current_value
            )


def _parse_tool_call(command: str) -> tuple[str, str]:
    payload = command.removeprefix(_CALL_PREFIX).strip()
    if not payload:
        raise ValueError("call: requires a tool name")

    parts = payload.split(maxsplit=1)
    tool_name = parts[0]
    if len(parts) == 1:
        return tool_name, "{}"

    try:
        tool_args = json.loads(
            parts[1], object_pairs_hook=_build_unique_json_object
        )
    except json.JSONDecodeError as exc:
        raise ValueError("call: tool arguments must be valid JSON") from exc
    except RecursionError as exc:
        raise ValueError("call: tool argument nesting is too deep") from exc
    if not isinstance(tool_args, dict):
        raise ValueError("call: tool arguments must be a JSON object")
    _validate_json_nesting(tool_args)
    try:
        serialized = json.dumps(
            tool_args,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except ValueError as exc:
        raise ValueError("call: tool arguments must be valid JSON") from exc
    return tool_name, serialized


def _validate_tool_name(
    tool_name: str,
    tools: Sequence[tools_.Tool] | None,
) -> None:
    available_names = sorted(
        tool.name for tool in tools or () if tool.kind == "function"
    )
    if tool_name in available_names:
        return

    available = ", ".join(available_names) or "none"
    raise ValueError(
        f"unknown test tool {tool_name!r}; "
        f"available function tools: {available}"
    )


def _next_tool_call_id(messages: Sequence[messages_.Message]) -> str:
    existing_ids = {
        tool_call.tool_call_id
        for message in messages
        for tool_call in message.tool_calls
    }
    tool_call_index = 0
    while True:
        tool_call_id = f"test-tool-call-{tool_call_index}"
        if tool_call_id not in existing_ids:
            return tool_call_id
        tool_call_index += 1


def _format_tool_result(
    message: messages_.Message,
    *,
    max_chars: int,
) -> str:
    tool_results = message.tool_results
    if len(tool_results) != 1:
        raise ValueError(
            "test model requires exactly one tool result; "
            f"received {len(tool_results)}"
        )

    tool_result = tool_results[0]
    payload = {
        "is_error": tool_result.is_error,
        "result": tool_result.get_model_input(),
        "tool_call_id": tool_result.tool_call_id,
        "tool_name": tool_result.tool_name,
    }
    try:
        json_payload = _JSON_ADAPTER.dump_python(payload, mode="json")
        serialized = json.dumps(
            json_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError(
            "test model tool result must be JSON serializable"
        ) from exc
    if len(serialized) > max_chars:
        raise ValueError(
            "test model tool result exceeds max_result_chars "
            f"({len(serialized)} > {max_chars})"
        )
    return f"return: {serialized}"


async def _stream_text(text: str) -> AsyncGenerator[events.Event]:
    yield events.StreamStart()
    yield events.TextStart(block_id=_TEXT_BLOCK_ID)
    yield events.TextDelta(block_id=_TEXT_BLOCK_ID, chunk=text)
    yield events.TextEnd(block_id=_TEXT_BLOCK_ID)
    yield events.StreamEnd()


async def _stream_tool_call(
    tool_call_id: str,
    tool_name: str,
    tool_args: str,
) -> AsyncGenerator[events.Event]:
    yield events.StreamStart()
    yield events.ToolStart(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
    )
    yield events.ToolDelta(tool_call_id=tool_call_id, chunk=tool_args)
    yield events.ToolEnd(
        tool_call_id=tool_call_id,
        tool_call=messages_.DUMMY_TOOL_CALL,
    )
    yield events.StreamEnd()


class TestProtocol(base.ProviderProtocol[_TestClient]):
    """Parse bounded local commands and emit normal model stream events."""

    protocol_class_id: Literal["test_model"] = "test_model"
    max_command_chars: int = pydantic.Field(
        default=_DEFAULT_MAX_COMMAND_CHARS,
        gt=0,
    )
    max_result_chars: int = pydantic.Field(
        default=_DEFAULT_MAX_RESULT_CHARS,
        gt=0,
    )

    def stream(
        self,
        client: _TestClient,
        model: model_.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
        provider: str,
    ) -> AsyncGenerator[events.Event]:
        _ = client, model, output_type, params, provider
        if not messages:
            raise ValueError("test model requires at least one message")

        latest_message = messages[-1]
        if latest_message.role == "tool":
            return _stream_text(
                _format_tool_result(
                    latest_message,
                    max_chars=self.max_result_chars,
                )
            )
        if latest_message.role != "user":
            raise ValueError(
                "test model expects a user command or tool result; "
                f"received role {latest_message.role!r}"
            )

        command = latest_message.text
        if len(command) > self.max_command_chars:
            raise ValueError(
                "test model command exceeds max_command_chars "
                f"({len(command)} > {self.max_command_chars})"
            )
        if command.startswith(_PING_PREFIX):
            return _stream_text(f"pong:{command.removeprefix(_PING_PREFIX)}")
        if command.startswith(_CALL_PREFIX):
            tool_name, tool_args = _parse_tool_call(command)
            _validate_tool_name(tool_name, tools)
            tool_call_id = _next_tool_call_id(messages)
            return _stream_tool_call(tool_call_id, tool_name, tool_args)
        raise ValueError(
            "unsupported test model command; expected 'ping: <text>' or "
            "'call: <tool> [<json object>]'"
        )


class TestProvider(base.Provider[_TestClient]):
    """Credential-free provider backed by :class:`TestProtocol`."""

    provider_class_id: Literal["test"] = "test"
    name: str = "test"
    default_base_url: str = "test://local"

    def model_post_init(self, __context: Any) -> None:
        self._set_client(_TEST_CLIENT)

    def default_protocol(self) -> TestProtocol:
        return TestProtocol()

    async def list_models(self) -> list[str]:
        return [_MODEL_ID]

    async def probe(self, model: model_.Model) -> None:
        _ = model


def test_model() -> model_.Model:
    """Create a deterministic local model that requires no credentials."""
    return model_.Model(id=_MODEL_ID, provider=TestProvider())


__all__ = ["TestProtocol", "TestProvider", "test_model"]
