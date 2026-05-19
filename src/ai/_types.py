from collections.abc import Iterator
from typing import Protocol, TypeVar

_T_co = TypeVar("_T_co", covariant=True)


class Collection(Protocol[_T_co]):
    def __contains__(self, value: object, /) -> bool: ...
    def __iter__(self) -> Iterator[_T_co]: ...
    def __len__(self) -> int: ...
