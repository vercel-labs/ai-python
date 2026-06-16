#!/usr/bin/env python3
"""Run a Python file with model/protocol selection patched in.

Useful for re-running an example against a different model or underlying
wire protocol without editing it.

Usage (from repo root):

    uv run examples/.test_scripts/run-with-patched-model.py <model> <file.py>
    uv run examples/.test_scripts/run-with-patched-model.py \\
        --protocol=responses <file.py>

Example:
    uv run examples/.test_scripts/run-with-patched-model.py \\
        gateway:openai/gpt-5.4-mini \\
        examples/stream.py

"""

import argparse
import runpy
import sys
from collections.abc import Callable
from typing import Any, TypeVar, cast

import ai
from ai import models
from ai.models import core
from ai.models.core import api as _api
from ai.models.core import model as _model
from ai.providers.anthropic import (
    AnthropicCompatibleProvider,
    AnthropicMessagesProtocol,
)
from ai.providers.openai import (
    OpenAIChatCompletionsProtocol,
    OpenAICompatibleProvider,
    OpenAIResponsesProtocol,
)

PROTOCOLS = ("chat", "messages", "responses")

ModelT = TypeVar("ModelT", bound=ai.Model)


def _protocol_factory(
    name: str | None,
) -> Callable[[], ai.ProviderProtocol[Any]] | None:
    if name is None:
        return None

    if name == "chat":
        return OpenAIChatCompletionsProtocol
    if name == "messages":
        return AnthropicMessagesProtocol
    if name == "responses":
        return OpenAIResponsesProtocol

    raise ValueError(f"unsupported protocol: {name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", help="model id, e.g. 'gateway:anthropic/claude-sonnet-4.6'"
    )
    parser.add_argument("--protocol", choices=PROTOCOLS)
    parser.add_argument("args", nargs="+", metavar="ARG")
    args = parser.parse_args()

    if len(args.args) == 1:
        args.file = args.args[0]
    elif len(args.args) == 2 and args.model is None:
        args.model = args.args[0]
        args.file = args.args[1]
    else:
        parser.error("expected <file.py> or legacy <model> <file.py>")

    return args


def main() -> None:
    args = _parse_args()

    protocol_factory = _protocol_factory(args.protocol)

    original_get_model = _model.get_model
    original_stream = _api.stream
    original_generate = _api.generate

    def selected_protocol_for_provider(
        provider: ai.Provider[Any],
    ) -> Callable[[], ai.ProviderProtocol[Any]] | None:
        if args.protocol is None or protocol_factory is None:
            return None
        if args.protocol in ("chat", "responses") and isinstance(
            provider, OpenAICompatibleProvider
        ):
            return protocol_factory
        if args.protocol == "messages" and isinstance(
            provider, AnthropicCompatibleProvider
        ):
            return protocol_factory
        return None

    def selected_protocol_for_model(
        model: ai.Model,
    ) -> Callable[[], ai.ProviderProtocol[Any]] | None:
        return selected_protocol_for_provider(model.provider)

    def with_selected_protocol(model: ModelT) -> ModelT:
        factory = selected_protocol_for_model(model)
        if factory is None:
            return model
        return model.with_protocol(factory)

    class PatchedContext:
        def __init__(self, context: Any) -> None:
            self._context = context
            self._model = with_selected_protocol(context.model)

        @property
        def model(self) -> Any:
            return self._model

        @property
        def messages(self) -> Any:
            return self._context.messages

        @property
        def tools(self) -> Any:
            return self._context.tools

        @property
        def output_type(self) -> Any:
            return self._context.output_type

        @property
        def params(self) -> Any:
            return self._context.params

    def patched_get_model(*_args: Any, **_kwargs: Any) -> ai.Model:
        model_id = args.model or (
            _args[0] if _args else _kwargs.get("model_id")
        )
        model = original_get_model(model_id)
        return with_selected_protocol(model)

    def patched_stream(*call_args: Any, **kwargs: Any) -> Any:
        if call_args:
            call_args = (
                with_selected_protocol(call_args[0]),
                *call_args[1:],
            )
        elif "model" in kwargs and kwargs["model"] is not None:
            kwargs["model"] = with_selected_protocol(kwargs["model"])
        elif kwargs.get("context") is not None:
            kwargs["context"] = PatchedContext(kwargs["context"])
        return original_stream(*call_args, **kwargs)

    async def patched_generate(*call_args: Any, **kwargs: Any) -> Any:
        if call_args:
            call_args = (
                with_selected_protocol(call_args[0]),
                *call_args[1:],
            )
        elif "model" in kwargs and kwargs["model"] is not None:
            kwargs["model"] = with_selected_protocol(kwargs["model"])
        return await original_generate(*call_args, **kwargs)

    class PatchedModel(_model.Model):
        def __init__(self, id: str, **kwargs: Any) -> None:
            super().__init__(id, **kwargs)
            override = selected_protocol_for_provider(self.provider)
            if override is not None:
                self.protocol_factory = override
                self.protocol_args = {}

    cast("Any", ai).get_model = patched_get_model
    cast("Any", models).get_model = patched_get_model
    cast("Any", core).get_model = patched_get_model
    cast("Any", _model).get_model = patched_get_model

    if args.protocol is not None:
        cast("Any", ai).Model = PatchedModel
        cast("Any", models).Model = PatchedModel
        cast("Any", core).Model = PatchedModel
        cast("Any", _model).Model = PatchedModel

        cast("Any", ai).stream = patched_stream
        cast("Any", models).stream = patched_stream
        cast("Any", core).stream = patched_stream
        cast("Any", _api).stream = patched_stream

        cast("Any", ai).generate = patched_generate
        cast("Any", models).generate = patched_generate
        cast("Any", core).generate = patched_generate
        cast("Any", _api).generate = patched_generate

    sys.argv = [args.file]
    runpy.run_path(args.file, run_name="__main__")


if __name__ == "__main__":
    main()
