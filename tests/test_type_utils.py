from __future__ import annotations

from typing import Any

from ai import type_utils
from ai.types import events


class Base[T, U]:
    pass


class Box[T](Base[list[T], str]):
    pass


class IntBox(Box[int]):
    pass


def test_generic_base_args_from_parameterized_child() -> None:
    assert type_utils.generic_base_args(Box[int], Base) == (list[int], str)


def test_generic_base_args_through_intermediate_base() -> None:
    assert type_utils.generic_base_args(IntBox, Base) == (list[int], str)


class ListAggregator[T](events.Aggregator[T, list[T], str]):
    def feed(self, item: T) -> None:
        pass

    def snapshot(self) -> list[T]:
        return []

    @classmethod
    def to_model_input(cls, snapshot: list[T]) -> str:
        return ""


def test_generic_base_args_from_aggregator() -> None:
    assert type_utils.generic_base_args(
        ListAggregator[int], events.Aggregator
    ) == (
        int,
        list[int],
        str,
    )


def test_bind_typevars_and_replace_typevars() -> None:
    args = type_utils.generic_base_args(ListAggregator, events.Aggregator)
    assert args is not None
    item_type, result_type, _model_input_type = args

    bindings = type_utils.bind_typevars(item_type, int)
    assert type_utils.replace_typevars(result_type, bindings) == list[int]


# Exercise a non-generic class path too.
class PlainAggregator(events.Aggregator[str, dict[str, Any], str]):
    def feed(self, item: str) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        return {}

    @classmethod
    def to_model_input(cls, snapshot: dict[str, Any]) -> str:
        return ""


def test_generic_base_args_from_plain_aggregator() -> None:
    assert type_utils.generic_base_args(PlainAggregator, events.Aggregator) == (
        str,
        dict[str, Any],
        str,
    )
