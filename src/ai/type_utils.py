"""Helpers for runtime type inspection."""

from __future__ import annotations

import functools
import operator
import types
import typing
from typing import Any

_T = typing.TypeVar("_T")
_TYPEVAR_TYPE = type(_T)


def replace_typevars(value: Any, bindings: dict[Any, Any]) -> Any:
    if isinstance(value, _TYPEVAR_TYPE):
        return bindings.get(value, value)

    origin = typing.get_origin(value)
    if origin is None:
        return value

    args = typing.get_args(value)
    if not args:
        return value

    if origin is typing.Annotated:
        return typing.Annotated.__class_getitem__(
            (replace_typevars(args[0], bindings), *args[1:])
        )

    replaced = tuple(replace_typevars(arg, bindings) for arg in args)
    if origin in (typing.Union, types.UnionType):
        return functools.reduce(operator.or_, replaced)

    return origin[replaced]


def bind_typevars(pattern: Any, value: Any) -> dict[Any, Any]:
    """Bind type variables in ``pattern`` to matching positions in ``value``."""
    if isinstance(pattern, _TYPEVAR_TYPE):
        return {pattern: value}

    pattern_origin = typing.get_origin(pattern)
    value_origin = typing.get_origin(value)
    if pattern_origin is None or pattern_origin != value_origin:
        return {}

    bindings: dict[Any, Any] = {}
    for pattern_arg, value_arg in zip(
        typing.get_args(pattern), typing.get_args(value), strict=False
    ):
        bindings.update(bind_typevars(pattern_arg, value_arg))
    return bindings


def resolve_type_alias(value: Any) -> Any:
    """Resolve a PEP 695 type alias, preserving applied type arguments."""
    if isinstance(value, typing.TypeAliasType):
        return value.__value__

    origin = typing.get_origin(value)
    if isinstance(origin, typing.TypeAliasType):
        params = origin.__type_params__
        args = typing.get_args(value)
        return replace_typevars(
            origin.__value__, dict(zip(params, args, strict=False))
        )

    return value


def generic_base_args(child: Any, base: type[Any]) -> tuple[Any, ...] | None:
    """Return ``base`` type arguments for ``child``.

    ``child`` may be a concrete class or a parameterized generic alias. For
    example, given ``class Box[T](Base[list[T]])``,
    ``generic_base_args(Box[int], Base)`` returns ``(list[int],)``.
    """
    return _generic_base_args(child, base, {})


def _generic_base_args(
    child: Any,
    base: type[Any],
    bindings: dict[Any, Any],
) -> tuple[Any, ...] | None:
    child_origin = typing.get_origin(child)
    if child_origin is not None:
        child_args = typing.get_args(child)
        child_params = getattr(child_origin, "__parameters__", ())
        child_bindings = {
            param: replace_typevars(arg, bindings)
            for param, arg in zip(child_params, child_args, strict=False)
        }
        bindings = {**bindings, **child_bindings}
        child = child_origin

    for parent in getattr(child, "__orig_bases__", ()):
        parent_origin = typing.get_origin(parent)
        parent_args = typing.get_args(parent)

        if parent_origin is base:
            return tuple(replace_typevars(arg, bindings) for arg in parent_args)

        if isinstance(parent_origin, type):
            parent_params = getattr(parent_origin, "__parameters__", ())
            parent_bindings = {
                param: replace_typevars(arg, bindings)
                for param, arg in zip(parent_params, parent_args, strict=False)
            }
            result = _generic_base_args(
                parent_origin, base, {**bindings, **parent_bindings}
            )
            if result is not None:
                return result

    return None
