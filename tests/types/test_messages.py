"""Focused tests for message model branches with real logic."""

from __future__ import annotations

import asyncio
import dataclasses
import random
import subprocess
import sys
from typing import Any

import pydantic
import pytest

from ai.types import messages, usage


@dataclasses.dataclass
class ToolForResultValidation:
    name: str
    return_type: Any


def test_usage_add_merges_optional_fields() -> None:
    a = usage.Usage(
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=20,
    )
    b = usage.Usage(
        input_tokens=200,
        output_tokens=80,
        reasoning_tokens=10,
    )
    total = a + b

    assert total.input_tokens == 300
    assert total.output_tokens == 130
    assert total.total_tokens == 430
    assert total.reasoning_tokens == 10
    assert total.cache_read_tokens == 20
    assert total.cache_write_tokens is None
    assert total.raw is None


def test_file_part_in_part_union() -> None:
    msg = messages.Message(
        id="m1",
        role="user",
        parts=[
            messages.TextPart(text="look at this"),
            messages.FilePart(
                data="https://example.com/cat.jpg", media_type="image/jpeg"
            ),
        ],
    )
    dumped = msg.model_dump()
    restored = messages.Message.model_validate(dumped)
    assert len(restored.parts) == 2
    assert isinstance(restored.parts[1], messages.FilePart)
    assert restored.parts[1].media_type == "image/jpeg"


def test_provider_metadata_round_trips_as_dict() -> None:
    msg = messages.Message(
        id="m1",
        role="assistant",
        provider_metadata={"provider": "test", "nested": {"value": 1}},
        parts=[
            messages.TextPart(
                text="hello",
                provider_metadata={"provider": "test", "block": "text"},
            )
        ],
    )

    restored = messages.Message.model_validate(msg.model_dump())

    assert restored.provider_metadata == {
        "provider": "test",
        "nested": {"value": 1},
    }
    part = restored.parts[0]
    assert isinstance(part, messages.TextPart)
    assert part.provider_metadata == {"provider": "test", "block": "text"}


def test_from_url_infers_from_data_url() -> None:
    fp = messages.FilePart.from_url("data:audio/wav;base64,AAAA")
    assert fp.media_type == "audio/wav"


def test_from_url_explicit_media_type_overrides() -> None:
    fp = messages.FilePart.from_url(
        "https://example.com/img", media_type="image/webp"
    )
    assert fp.media_type == "image/webp"


def test_from_url_unknown_extension_raises() -> None:
    with pytest.raises(ValueError, match="Cannot infer media_type"):
        messages.FilePart.from_url("https://example.com/blob")


def test_from_bytes_detects_image_and_preserves_filename() -> None:
    data = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A])
    fp = messages.FilePart.from_bytes(data, filename="photo.png")
    assert fp.media_type == "image/png"
    assert fp.data == data
    assert fp.filename == "photo.png"


def test_from_bytes_detects_audio() -> None:
    data = bytes(
        [0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45]
    )
    fp = messages.FilePart.from_bytes(data)
    assert fp.media_type == "audio/wav"


def test_from_bytes_explicit_overrides() -> None:
    fp = messages.FilePart.from_bytes(b"\x00\x00", media_type="video/mp4")
    assert fp.media_type == "video/mp4"


def test_from_bytes_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Cannot detect media_type"):
        messages.FilePart.from_bytes(b"\x00\x01\x02\x03")


# ---------------------------------------------------------------------------
# ToolResultPart -- typed result coercion and round-trip
# ---------------------------------------------------------------------------


def test_models_fully_defined_at_import() -> None:
    """No model may escape ``import ai`` with an unbuilt schema.

    ``MessageBundle`` forward-references ``Message``, which is defined later
    in the module, so its schema is deferred at class creation and must be
    completed by the module-end ``model_rebuild()``.  Without that it leaks
    out of the import incomplete, and whether a downstream consumer works
    then depends on *when* and *where* the lazy rebuild fires: embedding the
    class in another model usually heals it, but e.g. a durable-workflow
    module re-import broke with "``SessionState`` is not fully defined; you
    should define ``Message``" in the seal backend.

    This must run in a fresh interpreter: pydantic heals an incomplete model
    on its first direct validation, so any earlier test that builds a
    ``MessageBundle`` (e.g. the ai_sdk adapter tests, which sort first)
    silently masks the regression in-process.
    """
    code = (
        "import inspect\n"
        "import pydantic\n"
        "import ai\n"
        "import ai.types.messages as m\n"
        "bad = [\n"
        "    name\n"
        "    for name, obj in vars(m).items()\n"
        "    if inspect.isclass(obj)\n"
        "    and issubclass(obj, pydantic.BaseModel)\n"
        "    and obj.__module__ == m.__name__\n"
        "    and not obj.__pydantic_complete__\n"
        "]\n"
        "assert not bad, f'models escaped import with unbuilt schemas: {bad}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_message_bundle_embeddable_without_sys_modules_entry() -> None:
    """Embedding ``MessageBundle`` must not require a ``sys.modules`` lookup.

    If ``MessageBundle`` escapes import incomplete, a consumer model
    embedding it makes pydantic re-evaluate the deferred ``"Message"``
    annotation via ``sys.modules[MessageBundle.__module__]``.  Environments
    that rebuild ``sys.modules`` break then: the vercel workflow sandbox
    clears ``sys.modules`` and re-registers only what sandboxed code
    explicitly imports — a bare ``import ai`` does *not* re-register the
    ``ai.types.messages`` key — so the lookup misses, the consumer fails
    with "name 'Message' is not defined", and every model embedding it is
    poisoned (this took down the seal backend's ``SessionState``).  A
    complete-at-import ``MessageBundle`` never needs the lookup.

    Runs in a fresh interpreter for the same masking reason as above.
    """
    code = (
        "import sys\n"
        "import pydantic\n"
        "import ai\n"
        "from ai.types import messages\n"
        "del sys.modules['ai.types.messages']\n"
        "class Holder(pydantic.BaseModel):\n"
        "    x: messages.MessageBundle | None = None\n"
        "Holder(x=messages.MessageBundle(messages=()))\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_tool_result_content_output_with_file_part_round_trip() -> None:
    """FilePart inside ContentOutput survives JSON round-trip."""
    fp = messages.FilePart(data=b"fake-image-data", media_type="image/png")
    trp = messages.ToolResultPart(
        tool_call_id="tc1",
        tool_name="read",
        result=messages.ContentOutput(
            value=[messages.TextPart(text="label"), fp]
        ),
        result_kind="special",
    )
    j = trp.model_dump_json()
    restored = messages.ToolResultPart.model_validate_json(j)
    assert isinstance(restored.result, messages.ContentOutput)
    assert len(restored.result.value) == 2
    text_part, file_part = restored.result.value
    assert isinstance(text_part, messages.TextPart)
    assert text_part.text == "label"
    assert isinstance(file_part, messages.FilePart)
    assert file_part.media_type == "image/png"


def test_tool_result_validates_result_from_context_tools() -> None:
    class Weather(pydantic.BaseModel):
        temp: int
        city: str

    weather_tool = ToolForResultValidation(name="weather", return_type=Weather)

    restored = messages.ToolResultPart.model_validate(
        {
            "tool_call_id": "tc",
            "tool_name": "weather",
            "result": {"temp": "72", "city": "SF"},
        },
        context=messages.tool_context([weather_tool]),
    )

    assert isinstance(restored.result, Weather)
    assert restored.result.temp == 72
    assert restored.result.city == "SF"


def test_tool_result_uses_tool_name_for_context_lookup() -> None:
    class Weather(pydantic.BaseModel):
        temp: int

    class Search(pydantic.BaseModel):
        hits: list[str]

    weather_tool = ToolForResultValidation(name="weather", return_type=Weather)
    search_tool = ToolForResultValidation(name="search", return_type=Search)

    restored = messages.ToolResultPart.model_validate(
        {
            "tool_call_id": "tc",
            "tool_name": "search",
            "result": {"hits": ["a", "b"]},
        },
        context=messages.tool_context([weather_tool, search_tool]),
    )

    assert isinstance(restored.result, Search)
    assert restored.result.hits == ["a", "b"]


def test_message_validates_tool_result_parts_with_context_tools() -> None:
    class Weather(pydantic.BaseModel):
        temp: int

    weather_tool = ToolForResultValidation(name="weather", return_type=Weather)

    msg = messages.Message.model_validate(
        {
            "role": "tool",
            "parts": [
                {
                    "kind": "tool_result",
                    "tool_call_id": "tc",
                    "tool_name": "weather",
                    "result": {"temp": "72"},
                }
            ],
        },
        context=messages.tool_context([weather_tool]),
    )

    part = msg.tool_results[0]
    assert isinstance(part.result, Weather)
    assert part.result.temp == 72


def test_tool_result_without_context_stores_raw() -> None:
    part = messages.ToolResultPart.model_validate(
        {
            "tool_call_id": "tc",
            "tool_name": "weather",
            "result": {"temp": "72"},
        }
    )

    assert part.result == {"temp": "72"}


def test_tool_result_context_validation_error() -> None:
    class Weather(pydantic.BaseModel):
        temp: int

    weather_tool = ToolForResultValidation(name="weather", return_type=Weather)

    with pytest.raises(pydantic.ValidationError, match="temp"):
        messages.ToolResultPart.model_validate(
            {
                "tool_call_id": "tc",
                "tool_name": "weather",
                "result": {"temp": "hot"},
            },
            context=messages.tool_context([weather_tool]),
        )


def test_tool_result_plain_values_stored_raw() -> None:
    """Plain str / dict / list / None results are stored as-is and round-trip.

    ``result`` is ``Any`` -- there is no wrapper type, so a tool's return
    value lands on the part unchanged and survives a JSON round-trip.
    """
    cases: list[Any] = ["hello", None, [1, 2, 3], {"key": "val"}]
    for raw in cases:
        trp = messages.ToolResultPart(
            tool_call_id="tc", tool_name="t", result=raw
        )
        assert trp.result == raw
        assert trp.result_kind == "json"
        restored = messages.ToolResultPart.model_validate_json(
            trp.model_dump_json()
        )
        assert restored.result == raw
        assert restored.result_kind == "json"


def test_tool_result_content_in_message_round_trip() -> None:
    """ContentOutput with a FilePart survives Message round-trip."""
    fp = messages.FilePart(data=b"img-data", media_type="image/webp")
    msg = messages.Message(
        role="tool",
        parts=[
            messages.ToolResultPart(
                tool_call_id="tc",
                tool_name="read",
                result=messages.ContentOutput(
                    value=[messages.TextPart(text="Read image"), fp]
                ),
                result_kind="special",
            )
        ],
    )
    j = msg.model_dump_json()
    restored = messages.Message.model_validate_json(j)
    part = restored.parts[0]
    assert isinstance(part, messages.ToolResultPart)
    assert isinstance(part.result, messages.ContentOutput)
    fp2 = part.result.value[1]
    assert isinstance(fp2, messages.FilePart)
    assert fp2.media_type == "image/webp"


def test_tool_result_file_part_base64_valid_after_round_trip() -> None:
    """After round-trip, data_to_base64 produces standard base-64."""
    import base64

    from ai.types import media as media_

    raw = b"\xff\xd8\xff\xe0\x00\x10JFIF" * 10
    fp = messages.FilePart(data=raw, media_type="image/jpeg")
    trp = messages.ToolResultPart(
        tool_call_id="tc",
        tool_name="read",
        result=messages.ContentOutput(
            value=[messages.TextPart(text="label"), fp]
        ),
        result_kind="special",
    )
    restored = messages.ToolResultPart.model_validate_json(
        trp.model_dump_json()
    )
    assert isinstance(restored.result, messages.ContentOutput)
    fp2 = restored.result.value[1]
    assert isinstance(fp2, messages.FilePart)

    b64 = media_.data_to_base64(fp2.data)
    assert "_" not in b64
    assert "-" not in b64
    decoded = base64.b64decode(b64)
    assert decoded == raw


def test_tool_result_without_model_input_serializes_after_deep_copy() -> None:
    """A deep-copied ToolResultPart with no model_input still serializes.

    ``model_input`` defaults to the ``_MODEL_INPUT_UNSET`` singleton, and
    the sentinel checks (``exclude_if``, ``has_model_input``,
    ``get_model_input``) test for it by type.  ``model_copy(deep=True)``
    rebuilds the sentinel into a *new* instance: an identity (``is``) check
    would miss it, leave the field un-excluded, and make pydantic choke on
    the bare sentinel.  Client apps that deep-copy messages hit this.
    """
    msg = messages.Message(
        role="tool",
        parts=[
            messages.ToolResultPart(
                tool_call_id="tc", tool_name="t", result={"ok": 1}
            )
        ],
    )
    part = msg.parts[0]
    assert isinstance(part, messages.ToolResultPart)
    assert not part.has_model_input

    cloned = msg.model_copy(deep=True)

    # The clone's sentinel is a fresh instance; the type-based checks must
    # still treat it as unset.
    cpart = cloned.parts[0]
    assert isinstance(cpart, messages.ToolResultPart)
    assert not cpart.has_model_input
    assert cpart.get_model_input() == {"ok": 1}

    j = cloned.model_dump_json()
    restored = messages.Message.model_validate_json(j)
    rpart = restored.parts[0]
    assert isinstance(rpart, messages.ToolResultPart)
    assert rpart.result == {"ok": 1}
    assert not rpart.has_model_input


def test_use_random_overrides_and_restores() -> None:
    # A seeded Random gives a deterministic id sequence; the override
    # drives generate_id and the model default factories alike.
    with messages.use_random(random.Random(0)):
        a_msg = messages.generate_id("msg")
        a_part = messages.TextPart(text="hi").id
    with messages.use_random(random.Random(0)):
        b_msg = messages.generate_id("msg")
        b_part = messages.TextPart(text="hi").id

    assert (a_msg, a_part) == (b_msg, b_part)
    assert a_msg.startswith("msg_")
    assert a_part.startswith("part_")

    # A factory is resolved on entry (so e.g. workflow.random works).
    with messages.use_random(lambda: random.Random(0)):
        assert messages.generate_id("msg") == a_msg

    # Restored on exit -- back to the default Random.
    assert messages.generate_id("msg").startswith("msg_")


async def test_use_random_overrides_and_restores_async() -> None:
    with messages.use_random(random.Random(0)):
        expected = messages.generate_id("msg")

    # Works across an await...
    async with messages.use_random(random.Random(0)):
        await asyncio.sleep(0)
        assert messages.generate_id("msg") == expected

    # Works as a decorator on an async fn, resolving the factory per call.
    @messages.use_random(lambda: random.Random(0))
    async def build() -> str:
        await asyncio.sleep(0)
        return messages.generate_id("msg")

    assert await build() == expected
    assert await build() == expected
