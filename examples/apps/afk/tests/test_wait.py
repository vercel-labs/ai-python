"""Knowing what happened without a notification service: the list says what
changed since you last looked, and `afk wait` polls until it is over."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from afk import cli
from afk import state as st
from afk.rows import Row

from ai.harnesses.experimental import Handle
from ai.workspaces.experimental import WorkspaceCoords

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _row(sid: str, status: str) -> Row:
    return Row(
        where="remote", kind="claude-code", session_id=sid, status=status
    )


def test_the_list_says_what_changed_since_you_last_looked() -> None:
    state = st.State()
    assert (
        state.changed([_row("a", "running")]) == []
    ), "first look: nothing new"
    changed = state.changed([_row("a", "finished"), _row("b", "running")])
    assert [(r.session_id, was) for r, was in changed] == [("a", "running")]
    assert state.changed([_row("a", "finished")]) == [], "said once"
    state.changed([_row("a", "unreachable")])
    assert state.seen["a"] == "finished", "could not ask is not a change"


async def test_wait_returns_when_its_conversation_stops_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    handle = Handle(
        kind="claude-code",
        session_id="abcd1234",
        workspace=WorkspaceCoords(provider="sandbox", location="sb"),
    )
    st.save(
        st.State(
            remotes=[
                st.Remote(
                    label="job",
                    origin=str(tmp_path),
                    sandbox="sb",
                    mode="bg",
                    handle=handle,
                    pushed_at=0,
                )
            ]
        )
    )
    looks = iter(["running", "running", "finished"])

    async def remote_rows(*args: Any) -> Any:
        row = _row("abcd1234", next(looks))
        row.label, row.cost = "job", "$0.010"
        return [row], []

    monkeypatch.setattr(cli, "remote_rows", remote_rows)
    code = await cli.waiting(tmp_path, None, "job", every=0)
    out = capsys.readouterr().out
    assert code == 0
    assert "waiting for job" in out
    assert "\a afk: job is finished · $0.010" in out


async def test_wait_with_nothing_running_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def remote_rows(*args: Any) -> Any:
        return [], []

    monkeypatch.setattr(cli, "remote_rows", remote_rows)
    assert await cli.waiting(tmp_path, None, None, every=0) == 1
