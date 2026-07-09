"""Guard against the pydantic-core ``exclude_if`` GC leak.

pydantic-core's ``SchemaSerializer`` stores ``Field(exclude_if=...)``
callables but omits them from ``tp_traverse``.  A callable whose
``__globals__`` can reach its own model class therefore forms a
reference cycle the garbage collector can never prove unreachable, and
the module's entire object graph leaks.  Passing ``weakref.proxy`` of a
named module-level function keeps the serializer's (untraversed)
reference weak, so no strong cycle forms.
"""

from __future__ import annotations

import gc
import importlib
import pathlib
import sys
import weakref

import ai.types.messages as messages_mod
from ai.types.messages import Message, ToolCallPart, ToolResultPart

_AI_SRC = pathlib.Path(messages_mod.__file__).parents[2]


def test_exclude_if_behavior_through_proxy() -> None:
    """The weakref.proxy indirection must not change serialization."""
    part = ToolResultPart(tool_call_id="tc", tool_name="t", result=1)
    dumped = part.model_dump()
    # model_input is unset -> excluded by _exclude_if_model_input_unset
    assert "model_input" not in dumped

    part = ToolResultPart(
        tool_call_id="tc", tool_name="t", result=1, model_input="visible"
    )
    assert part.model_dump()["model_input"] == "visible"

    call = ToolCallPart(tool_call_id="tc", tool_name="t", tool_args="{}")
    # cached_result is None -> excluded by _exclude_if_none
    assert "cached_result" not in call.model_dump()

    msg = Message(role="user", parts=[])
    # replay is False -> excluded by _exclude_if_falsy
    dumped = msg.model_dump()
    assert "replay" not in dumped
    assert Message(role="user", parts=[], replay=True).model_dump()["replay"]


def test_exclude_if_models_are_collectable(tmp_path: pathlib.Path) -> None:
    """A module-level model using the proxy pattern must be collectable.

    With a bare lambda this cycle is uncollectable:
    lambda -> __globals__ -> class -> __pydantic_serializer__ -> lambda.
    """
    sys.path.insert(0, str(tmp_path))
    (tmp_path / "excl_gc_probe.py").write_text(
        "import weakref\n"
        "import pydantic\n"
        "\n"
        "def _exclude_if_none(v: object) -> bool:\n"
        "    return v is None\n"
        "\n"
        "class Probe(pydantic.BaseModel):\n"
        "    x: int | None = pydantic.Field(\n"
        "        default=None, exclude_if=weakref.proxy(_exclude_if_none)\n"
        "    )\n"
    )
    try:
        mod = importlib.import_module("excl_gc_probe")
        assert mod.Probe(x=None).model_dump() == {}
        assert mod.Probe(x=3).model_dump() == {"x": 3}
        ref = weakref.ref(mod.Probe)
        del sys.modules["excl_gc_probe"], mod
        gc.collect()
        gc.collect()
        assert ref() is None, "model class leaked despite weakref.proxy"
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("excl_gc_probe", None)


def test_no_exclude_if_lambdas_in_source() -> None:
    """Fail if the leaking ``exclude_if=lambda`` pattern reappears."""
    offenders = [
        f"{path.relative_to(_AI_SRC)}:{lineno}"
        for path in sorted(_AI_SRC.rglob("*.py"))
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if "exclude_if=lambda" in line
    ]
    assert not offenders, (
        "exclude_if must not be passed a lambda (pydantic-core does not "
        "GC-traverse it; use weakref.proxy of a named function): "
        f"{offenders}"
    )
