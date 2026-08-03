"""Test a model / agent interaction against a script.

:class:`FakeModel` is a drop-in :class:`~ai.models.core.model.Model`
that replays scripted assistant messages::

    calls = [ai.testing.tool_call(lookup, topic="mars")]
    model = ai.testing.FakeModel([
        (ai.user_message("hi"), [
            ai.assistant_message("checking", *calls),
            ai.assistant_message("done"),
        ]),
    ])
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any, Literal, cast

import pydantic

from . import agents, models
from .types import events
from .types import messages as messages_

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from .models.core import params as params_
    from .types import tools as tools_

__all__ = ["FakeModel", "fingerprint", "tool_call"]


_VOLATILE_MESSAGE_FIELDS = ("id", "turn_id", "usage", "provider_metadata")
_VOLATILE_PART_FIELDS = (
    "id",
    "tool_call_id",
    "provider_metadata",
    "model_input",
    "cached_result",
)


def fingerprint(message: messages_.Message) -> str:
    """Serialize message and drop volatile fields.

    :class:`FakeModel` uses this to match incoming conversations against script
    keys; the same dump appears in its error messages.
    """
    data = message.model_dump(mode="json")
    for field in _VOLATILE_MESSAGE_FIELDS:
        data.pop(field, None)
    for part in data.get("parts", []):
        for field in _VOLATILE_PART_FIELDS:
            part.pop(field, None)
    return json.dumps(data, sort_keys=True, indent=2)


def tool_call(
    tool: agents.AgentTool | str, **kwargs: Any
) -> messages_.ToolCallPart:
    """Build a :class:`ToolCallPart` for a scripted assistant message.

    Validates ``kwargs`` against the tool's signature (when given an
    :class:`~ai.AgentTool`) and generates a fresh ``tool_call_id``.
    """
    if isinstance(tool, str):
        name = tool
    else:
        name = tool.name
        if tool.validator is not None:
            tool.validator.model_validate(kwargs)
    return messages_.ToolCallPart(
        tool_call_id=messages_.generate_id("call"),
        tool_name=name,
        tool_args=json.dumps(kwargs),
    )


@dataclasses.dataclass
class _Entry:
    key: messages_.Message
    key_fingerprint: str
    responses: list[messages_.Message]
    played: int = 0

    @property
    def exhausted(self) -> bool:
        return self.played >= len(self.responses)


type _Script = Sequence[
    tuple[
        messages_.Message,
        messages_.Message | Sequence[messages_.Message],
    ]
]


def _parse_script(
    script: _Script,
) -> tuple[list[_Entry], dict[str, str]]:
    """Validate the script; return entries and a tool_call_id -> name map."""
    entries: list[_Entry] = []
    tool_names: dict[str, str] = {}
    for index, (key, value) in enumerate(script):
        responses = (
            [value] if isinstance(value, messages_.Message) else list(value)
        )
        where = f"script entry {index} ({key.text[:60]!r})"
        if not responses:
            raise ValueError(f"{where}: no response messages")
        for position, message in enumerate(responses):
            if message.role != "assistant":
                raise ValueError(
                    f"{where}: response {position} has role "
                    f"{message.role!r}; scripted responses must be "
                    "assistant messages"
                )
            is_last = position == len(responses) - 1
            if not is_last and not message.tool_calls:
                raise ValueError(
                    f"{where}: response {position} has no tool calls, so "
                    "the turn ends there and later responses can never "
                    "play"
                )
            for part in message.tool_calls:
                if part.tool_call_id in tool_names:
                    raise ValueError(
                        f"{where}: duplicate tool_call_id "
                        f"{part.tool_call_id!r}; build each call with its "
                        "own ai.testing.tool_call(...)"
                    )
                tool_names[part.tool_call_id] = part.tool_name
        entries.append(
            _Entry(
                key=key,
                key_fingerprint=fingerprint(key),
                responses=responses,
            )
        )
    return entries, tool_names


class _FakeProvider(models.Provider):
    """Provider backing :class:`FakeModel`; replays the script."""

    provider_class_id: Literal["testing-fake-model"] = "testing-fake-model"
    name: str = "fake"
    default_base_url: str = "http://fake.invalid"

    _entries: list[_Entry] = pydantic.PrivateAttr(default_factory=list)
    _tool_names: dict[str, str] = pydantic.PrivateAttr(default_factory=dict)
    _pending: dict[frozenset[str], _Entry] = pydantic.PrivateAttr(
        default_factory=dict
    )
    _calls: list[list[messages_.Message]] = pydantic.PrivateAttr(
        default_factory=list
    )

    async def list_models(self) -> list[str]:
        return []

    def stream(
        self,
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
    ) -> AsyncGenerator[events.Event]:
        response = self._next_response(messages)
        return events._replay_message_events(response)

    async def generate(
        self,
        model: models.Model,
        messages: list[messages_.Message],
        params: params_.GenerateParams,
    ) -> messages_.Message:
        return self._next_response(messages)

    def _next_response(
        self, messages: list[messages_.Message]
    ) -> messages_.Message:
        self._calls.append([m.model_copy(deep=True) for m in messages])
        if not messages:
            raise AssertionError("FakeModel was called with no messages")
        last = messages[-1]

        # middle of a turn: answer tool calls with a tool message,
        # route by tool_call_id.
        results = last.tool_results
        if results:
            ids = frozenset(part.tool_call_id for part in results)
            entry = self._pending.pop(ids, None)
            if entry is None:
                raise AssertionError(self._mismatched_results_error(results))
            return self._play(entry)

        # start of a turn: match the injected message against the keys.
        key_fingerprint = fingerprint(last)
        for entry in self._entries:
            if entry.played == 0 and entry.key_fingerprint == key_fingerprint:
                return self._play(entry)
        raise AssertionError(self._no_entry_error(key_fingerprint))

    def _play(self, entry: _Entry) -> messages_.Message:
        if entry.exhausted:
            raise AssertionError(
                f"FakeModel: entry {entry.key.text[:60]!r} already played "
                f"all {len(entry.responses)} scripted messages, but the "
                "model was called again in that conversation"
            )
        message = entry.responses[entry.played]
        entry.played += 1
        ids = frozenset(part.tool_call_id for part in message.tool_calls)
        if ids:
            self._pending[ids] = entry
        return message

    def _describe(self, ids: frozenset[str]) -> str:
        return ", ".join(
            f"{self._tool_names.get(i, '?')}#{i}" for i in sorted(ids)
        )

    def _mismatched_results_error(
        self, results: Sequence[messages_.ToolResultPart]
    ) -> str:
        got = ", ".join(
            f"{part.tool_name}#{part.tool_call_id}"
            for part in sorted(results, key=lambda p: p.tool_call_id)
        )
        expected = [self._describe(ids) for ids in self._pending] or [
            "(none pending)"
        ]
        return (
            "FakeModel: got tool results answering [{got}], but the "
            "pending scripted tool calls are: {expected}".format(
                got=got, expected="; ".join(expected)
            )
        )

    def _no_entry_error(self, key_fingerprint: str) -> str:
        unused = [
            entry.key_fingerprint
            for entry in self._entries
            if entry.played == 0
        ]
        return (
            "FakeModel: no scripted entry matches the last message.\n"
            f"Got:\n{key_fingerprint}\n"
            "Unused keys:\n" + ("\n".join(unused) if unused else "(none)")
        )


class FakeModel(models.Model):
    """A model that replays a script instead of calling a provider.

    ``script`` is a sequence of ``(input message, response(s))`` pairs.
    """

    def __init__(
        self,
        script: _Script,
        *,
        id: str = "fake-model",
    ) -> None:
        provider = _FakeProvider()
        provider._entries, provider._tool_names = _parse_script(script)
        super().__init__(id=id, provider=provider)

    @property
    def _fake(self) -> _FakeProvider:
        return cast("_FakeProvider", self.provider)

    @property
    def calls(self) -> list[list[messages_.Message]]:
        """The exact input messages of every model call, in call order."""
        return self._fake._calls

    @property
    def unused(self) -> list[messages_.Message]:
        """Keys of entries with scripted messages that never played.

        A finished test normally asserts ``not model.unused``.
        """
        return [
            entry.key for entry in self._fake._entries if not entry.exhausted
        ]
