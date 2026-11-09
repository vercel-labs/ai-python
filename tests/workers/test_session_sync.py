"""Switching from direct input to model use must not replay queued output."""

import contextlib

import pytest

import ai
from ai import workers
from ai.types import events

from . import fakes


async def test_model_use_refuses_unread_direct_output() -> None:
    async with fakes.FakeWorker() as worker:
        await worker.send("one")
        with pytest.raises(workers.WorkerError, match="unread session events"):
            await ai.experimental_generate(worker, [ai.user_message("two")])
        assert worker.fake.turns == ["one"]
        async with contextlib.aclosing(worker.events()) as feed:
            async for event in feed:
                if isinstance(event, events.StreamEnd):
                    break
        answer = await ai.experimental_generate(
            worker, [*worker.messages, ai.user_message("two")]
        )
        assert answer.text == "two"


async def test_sync_replaces_unread_output_with_native_history() -> None:
    async with fakes.FakeWorker() as worker:
        await worker.send("one")
        assert worker._runtime is not None
        assert worker._runtime.pump is not None
        await worker._runtime.pump
        stored = [ai.user_message("one"), ai.assistant_message("native one")]
        worker.fake.stored["s1"] = stored
        await worker.sync()
        assert worker.messages == stored
        answer = await ai.experimental_generate(
            worker, [*worker.messages, ai.user_message("two")]
        )
        assert answer.text == "two"
        assert [message.text for message in worker.messages] == [
            "one",
            "native one",
            "two",
            "two",
        ]
