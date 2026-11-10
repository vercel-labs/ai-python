"""What `ai.stream` and `ai.Agent.run` need from a model.

`ai.Model` and its `ai.Provider` have this shape. Anything else with the
same shape goes where a model goes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    import pydantic

    from ... import types
    from . import params as params_


class ProviderLike[M](Protocol):
    """What answers for a model."""

    @property
    def name(self) -> str: ...

    def stream(
        self,
        model: M,
        messages: list[types.messages.Message],
        /,
        *,
        tools: Sequence[types.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
    ) -> AsyncGenerator[types.events.Event]:
        """Answer the WHOLE conversation with ONE assistant message.

        Yield its events, ending with `StreamEnd`.
        """
        ...

    async def generate(
        self,
        model: M,
        messages: list[types.messages.Message],
        /,
        *,
        tools: Sequence[types.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
    ) -> types.messages.Message:
        """Answer without streaming.

        Optional: raise `NotImplementedError`, and
        `ai.experimental_generate` drains `stream` instead.
        """
        ...


@runtime_checkable
class ModelLike(Protocol):
    """An id and the provider that answers for it."""

    @property
    def id(self) -> str: ...

    @property
    def provider(self) -> ProviderLike[Self]: ...
