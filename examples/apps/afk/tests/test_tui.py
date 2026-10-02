"""The picker: every key stands for one command, and stop asks first."""

from __future__ import annotations

import asyncio

import pytest
from afk.rows import Row
from afk.tui import Picker
from rich.text import Text
from textual.widgets import Static

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
        (["enter"], [["claude", "--resume", "a3f5aaaa"]]),
        (["down", "enter"], [["codex", "resume", "01a0cccc"]]),
        (["p"], [["afk", "peek", "a3f5aaaa"]]),
        (["u"], [["afk", "push", "a3f5aaaa"]]),
        (["b"], [["afk", "push", "a3f5aaaa", "--bg"]]),
        (["down", "down", "enter"], [["afk", "attach", "c2d1dddd"]]),
        (["down", "down", "down", "enter"], [["afk", "peek", "9e7feeee"]]),
        (["down", "down", "p"], [["afk", "peek", "c2d1dddd"]]),
        (["down", "down", "down", "p"], [["afk", "peek", "9e7feeee"]]),
        (["down", "down", "l"], [["afk", "pull", "c2d1dddd"]]),
        (["down", "down", "f"], [["afk", "pull", "c2d1dddd", "--files"]]),
        (["down", "down", "s", "y"], [["afk", "stop", "c2d1dddd"]]),
    ],
)
async def test_keys_choose_a_command(
    keys: list[str], argv: list[list[str]]
) -> None:
    assert (await _press(*keys)).return_value == argv


@pytest.mark.parametrize(
    ("downs", "keys"),
    [
        (0, "enter resume · p peek · u push · b push --bg · q quit"),
        (
            2,
            "enter attach · p peek · l pull · f pull --files · s stop · q quit",
        ),
        # an unattended agent: Enter already peeks, so p is not shown twice
        (3, "enter peek · l pull · f pull --files · s stop · q quit"),
    ],
)
async def test_the_keys_shown_are_the_rows_own(downs: int, keys: str) -> None:
    app = Picker(Text("~/proj"), HERE, _remote())
    async with app.run_test(size=(100, 20)) as pilot:
        await pilot.pause(0.1)
        for _ in range(downs):
            await pilot.press("down")
        await pilot.pause(0.05)
        assert str(app.query_one("#keys", Static).render()) == keys


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
    assert app.return_value == [["claude", "--resume", "a3f5aaaa"]]


RESUME_77E1 = ["claude", "--resume", "77e1ffff"]

IN_USE = [
    # Open in another terminal, and the harness can say which process.
    Row(
        where="here",
        kind="claude-code",
        session_id="77e1ffff",
        title="refactor the router",
        status="in use 1m",
        running=True,
        pid=4242,
    ),
    # Open somewhere, but no process afk could stop: the SDK lock's
    # pid-less moment, say.
    Row(
        where="here",
        kind="codex",
        session_id="5b0c0000",
        title="write the changelog",
        status="in use 3m",
        running=True,
    ),
]


async def _press_in_use(*keys: str) -> tuple[Picker, str]:
    app = Picker(Text("~/proj"), IN_USE, None)
    async with app.run_test(size=(100, 20)) as pilot:
        await pilot.pause(0.1)
        for key in keys:
            await pilot.press(key)
        await pilot.pause(0.05)
        shown = str(app.query_one("#keys", Static).render())
    return app, shown


@pytest.mark.parametrize(
    ("keys", "argv"),
    [
        # Enter takes over what another terminal has open, once you say y:
        # that process ends first, then the TUI opens here.
        (["enter", "y"], [["afk", "stop", "77e1ffff"], RESUME_77E1]),
        (["enter", "n"], None),
        # No process afk can end: Enter peeks.
        (["down", "enter"], [["afk", "peek", "5b0c0000"]]),
        (["p"], [["afk", "peek", "77e1ffff"]]),
        (["s", "y"], [["afk", "stop", "77e1ffff"]]),
        (["u"], [["afk", "push", "77e1ffff"]]),
    ],
)
async def test_a_conversation_open_elsewhere(
    keys: list[str], argv: list[list[str]] | None
) -> None:
    app, _ = await _press_in_use(*keys)
    assert app.return_value == argv


async def test_stop_here_names_the_process_it_ends() -> None:
    _, shown = await _press_in_use("s")
    assert shown == "stop 77e1 (claude, pid 4242)? y/N"


async def test_take_over_asks_first_and_names_the_process() -> None:
    _, shown = await _press_in_use("enter")
    assert shown == "take over 77e1 from pid 4242? it ends there. y/N"


@pytest.mark.parametrize(
    ("downs", "keys"),
    [
        (
            0,
            "enter take over · p peek · s stop · u push · b push --bg · q quit",
        ),
        # no process afk can name: nothing to stop
        (1, "enter peek · u push · b push --bg · q quit"),
    ],
)
async def test_stop_is_offered_only_with_a_process(
    downs: int, keys: str
) -> None:
    _, shown = await _press_in_use(*(["down"] * downs))
    assert shown == keys


async def test_stop_does_nothing_without_a_process() -> None:
    app, _ = await _press_in_use("down", "s", "y")
    assert app.return_value is None
