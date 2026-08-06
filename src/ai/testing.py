"""Test a model / agent interaction against scripted conversations.

A script is a plain list of messages. :class:`FakeModel` is a drop-in
:class:`~ai.Model` that replays it::

    model = ai.testing.FakeModel([
        ai.user_message("hi"),
        ai.assistant_message(
            "checking", ai.testing.tool_call(lookup, topic="mars")
        ),
        ai.assistant_message("done"),
    ])

* tool messages may be left out of a script.
* include a tool message ``ai.tool_message(...)`` to additionally assert
  the exact results.
* unscripted system messages are ignored, so system prompt doesn't need
  to appear in the script.
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
    "provider_metadata",
    "model_input",
    "cached_result",
)


def fingerprint(message: messages_.Message) -> str:
    """Serialize a message, dropping volatile fields.

    :class:`FakeModel` compares fingerprints to match incoming
    conversations against scripts.
    """
    data = message.model_dump(mode="json")
    for field in _VOLATILE_MESSAGE_FIELDS:
        data.pop(field, None)
    parts = data.get("parts", [])
    for part in parts:
        for field in _VOLATILE_PART_FIELDS:
            part.pop(field, None)
    if data.get("role") == "tool":
        # parallel tools finish in arbitrary order
        parts.sort(key=lambda part: part.get("tool_call_id", ""))
    return json.dumps(data, sort_keys=True, indent=2)


def tool_call(
    tool: agents.AgentTool | str, /, **kwargs: Any
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
class _Script:
    messages: list[messages_.Message]
    fingerprints: list[str]
    played: set[int] = dataclasses.field(default_factory=set)

    def label(self) -> str:
        return f"script starting with {self.messages[0].text[:60]!r}"


@dataclasses.dataclass
class _NoMatch:
    matched: int  # incoming messages matched before giving up
    reason: str


def _parse_scripts(
    scripts: Sequence[Sequence[messages_.Message]],
) -> list[_Script]:
    parsed: list[_Script] = []
    seen_call_ids: set[str] = set()
    for index, messages in enumerate(scripts):
        where = f"script {index}"
        messages = list(messages)
        if not messages:
            raise ValueError(f"{where}: empty script")
        if not any(m.role == "assistant" for m in messages):
            raise ValueError(f"{where}: no assistant messages to play")
        for position, message in enumerate(messages):
            for part in message.tool_calls:
                if part.tool_call_id in seen_call_ids:
                    raise ValueError(
                        f"{where}: duplicate tool_call_id "
                        f"{part.tool_call_id!r}; build each call with its "
                        "own ai.testing.tool_call(...)"
                    )
                seen_call_ids.add(part.tool_call_id)
            if (
                position > 0
                and message.role == "assistant"
                and messages[position - 1].role == "assistant"
                and not messages[position - 1].tool_calls
            ):
                raise ValueError(
                    f"{where}: message {position - 1} is an assistant "
                    "message with no tool calls, so the turn ends there "
                    f"and the assistant message {position} can never play"
                )
        parsed.append(
            _Script(
                messages=messages,
                fingerprints=[fingerprint(m) for m in messages],
            )
        )
    return parsed


def _walk(
    script: _Script,
    got: Sequence[messages_.Message],
    got_fingerprints: Sequence[str],
) -> int | _NoMatch:
    """Match an incoming conversation against a script.

    Returns the index of the next scripted message to play, or a
    :class:`_NoMatch` explaining where the conversation diverges.
    """
    expected = script.messages
    at = 0  # index into expected
    matched = 0
    pending: set[str] = set()  # scripted tool calls awaiting results
    for position, (message, print_) in enumerate(
        zip(got, got_fingerprints, strict=True)
    ):
        if at < len(expected) and script.fingerprints[at] == print_:
            if message.role == "tool":
                pending -= {p.tool_call_id for p in message.tool_results}
            pending |= {p.tool_call_id for p in expected[at].tool_calls}
            at += 1
            matched += 1
        elif message.role == "tool":
            # unscripted tool message: results are not asserted, but they
            # must answer tool calls this script played.
            ids = {p.tool_call_id for p in message.tool_results}
            if not ids <= pending:
                return _NoMatch(
                    matched,
                    f"message {position} answers tool calls "
                    f"{sorted(ids - pending)} that this script never made",
                )
            pending -= ids
        elif message.role == "system":
            continue  # e.g. an agent's system prompt
        elif at < len(expected):
            return _NoMatch(
                matched,
                f"message {position} does not match.\n"
                f"expected:\n{script.fingerprints[at]}\n"
                f"got:\n{print_}",
            )
        else:
            return _NoMatch(
                matched, f"the script ended before message {position}"
            )
    if at >= len(expected):
        return _NoMatch(
            matched,
            "all scripted messages played, but the model was called again",
        )
    nxt = expected[at]
    if nxt.role != "assistant":
        return _NoMatch(
            matched,
            f"the next scripted message has role {nxt.role!r} and nothing "
            f"in the conversation matches it:\n{script.fingerprints[at]}",
        )
    if pending:
        return _NoMatch(
            matched,
            f"scripted tool calls {sorted(pending)} have no results yet, "
            f"so the next assistant message cannot play",
        )
    return at


class _FakeProvider(models.Provider):
    """Provider backing :class:`FakeModel`; replays the scripts."""

    provider_class_id: Literal["testing-fake-model"] = "testing-fake-model"
    name: str = "fake"
    default_base_url: str = "http://fake.invalid"

    _scripts: list[_Script] = pydantic.PrivateAttr(default_factory=list)
    _calls: list[list[messages_.Message]] = pydantic.PrivateAttr(
        default_factory=list
    )

    def __init__(
        self,
        scripts: Sequence[Sequence[messages_.Message]] = (),
        **data: Any,
    ) -> None:
        super().__init__(**data)
        self._scripts = _parse_scripts(scripts)

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
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
    ) -> messages_.Message:
        return self._next_response(messages)

    def _next_response(
        self, messages: list[messages_.Message]
    ) -> messages_.Message:
        self._calls.append([m.model_copy(deep=True) for m in messages])
        if not messages:
            raise AssertionError("FakeModel was called with no messages")
        fingerprints = [fingerprint(m) for m in messages]
        failures: list[tuple[_Script, _NoMatch]] = []
        for script in self._scripts:
            result = _walk(script, messages, fingerprints)
            if isinstance(result, int):
                script.played.add(result)
                return script.messages[result]
            failures.append((script, result))
        raise AssertionError(self._no_match_error(fingerprints, failures))

    def _no_match_error(
        self,
        fingerprints: list[str],
        failures: list[tuple[_Script, _NoMatch]],
    ) -> str:
        lines = [
            "FakeModel: no script continues this conversation.",
            f"The model was called with {len(fingerprints)} message(s); "
            "the last one:",
            fingerprints[-1],
        ]
        if failures:
            script, no_match = max(failures, key=lambda f: f[1].matched)
            lines.append(
                f"Closest is the {script.label()} "
                f"(matched {no_match.matched} message(s)): {no_match.reason}"
            )
        return "\n".join(lines)


class FakeModel(models.Model):
    """A model that replays scripted conversations."""

    def __init__(
        self,
        *scripts: Sequence[messages_.Message],
        id: str = "fake-model",
    ) -> None:
        super().__init__(id=id, provider=_FakeProvider(scripts=scripts))

    @property
    def _fake(self) -> _FakeProvider:
        return cast("_FakeProvider", self.provider)

    @property
    def calls(self) -> list[list[messages_.Message]]:
        """The exact input messages of every model call, in call order."""
        return self._fake._calls

    @property
    def unused(self) -> list[messages_.Message]:
        """Scripted assistant messages that never played.

        A finished test normally asserts ``not model.unused``.
        """
        return [
            message
            for script in self._fake._scripts
            for index, message in enumerate(script.messages)
            if message.role == "assistant" and index not in script.played
        ]
