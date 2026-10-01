"""The picker: every key stands for one command, and stop asks first."""

from __future__ import annotations

import asyncio

import pytest
from afk.rows import Row
from afk.tui import Picker
from rich.text import Text

HERE = [
    Row(
        where="here",
        kind="claude-code",
        session_id="a3f5aaaa",
        title="migrate the billing tests",
        status="active 2m",
    ),
    Row(
        where="here",
        kind="codex",
        session_id="01a0cccc",
        title="audit the auth middleware",
        status="idle 3h",
    ),
]
REMOTE = [
    Row(
        where="remote",
        kind="claude-code",
        session_id="c2d1dddd",
        label="billing",
        status="running",
        sandbox="amber-fox",
        mode="tui",
    ),
    Row(
        where="remote",
        kind="codex",
        session_id="9e7feeee",
        label="tests",
        status="finished",
        sandbox="quiet-owl",
        mode="bg",
    ),
]


async def _remote(delay: float = 0.0) -> list[Row]:
    await asyncio.sleep(delay)
    return REMOTE


async def _press(*keys: str, remote_delay: float = 0.0) -> Picker:
    app = Picker(Text("~/proj"), HERE, _remote(remote_delay))
    async with app.run_test(size=(100, 20)) as pilot:
        await pilot.pause(0.1)
        for key in keys:
            await pilot.press(key)
        await pilot.pause(0.1)
    return app


@pytest.mark.parametrize(
    ("keys", "argv"),
    [
        (["enter"], ["push", "a3f5aaaa"]),
        (["b"], ["push", "a3f5aaaa", "--bg"]),
        (["down", "down", "enter"], ["attach", "c2d1dddd"]),
        (["down", "down", "down", "enter"], ["peek", "9e7feeee"]),
        (["down", "down", "l"], ["pull", "c2d1dddd"]),
        (["down", "down", "f"], ["pull", "c2d1dddd", "--files"]),
        (["down", "down", "s", "y"], ["stop", "c2d1dddd"]),
    ],
)
async def test_keys_choose_a_command(keys: list[str], argv: list[str]) -> None:
    assert (await _press(*keys)).return_value == argv


@pytest.mark.parametrize("answer", ["enter", "n", "escape", "q", "down"])
async def test_stop_is_only_a_yes(answer: str) -> None:
    app = await _press("down", "down", "s", answer)
    assert app.return_value is None
    assert app.confirming is None


@pytest.mark.parametrize("key", ["q", "escape", "ctrl+c"])
async def test_leaving_chooses_nothing(key: str) -> None:
    app = await _press(key)
    assert app.return_value is None
    assert not app.is_running


async def test_ctrl_c_leaves_even_while_asking_to_stop() -> None:
    app = await _press("down", "down", "s", "ctrl+c")
    assert app.return_value is None
    assert not app.is_running


async def test_remote_verbs_do_nothing_on_a_local_row() -> None:
    assert (await _press("s", "l")).return_value is None


async def test_local_rows_are_there_while_sandboxes_answer() -> None:
    app = Picker(Text("~/proj"), HERE, _remote(10))
    async with app.run_test(size=(100, 20)) as pilot:
        await pilot.pause(0.1)
        assert app.remote_rows is None
        await pilot.press("enter")
    assert app.return_value == ["push", "a3f5aaaa"]
