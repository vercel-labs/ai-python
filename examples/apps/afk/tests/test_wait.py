"""Knowing what happened without a notification service: the list says what
changed since you last looked, and `afk wait` polls until it is over."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from afk import cli
from afk import state as st
from afk.rows import Row, display_name, titled

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


def test_a_pushed_conversation_is_listed_by_its_title() -> None:
    plain = _row("abcd1234", "running")
    plain.label, plain.title = "abcd", "migrate the billing tests"
    assert display_name(plain) == "migrate the billing tests"
    named = plain.model_copy(update={"label": "billing"})
    assert display_name(named) == "billing · migrate the billing tests"
    untitled = _row("ef567890", "running")
    untitled.label = "ef56"
    assert display_name(untitled) == "ef56", "nothing else to call it by"


def test_one_pushed_before_titles_were_kept_borrows_its_sources() -> None:
    old = _row("remote01", "finished")
    old.from_id = "local001"
    here = Row(
        where="here",
        kind="claude-code",
        session_id="local001",
        status="idle",
        title="add rate limiting",
    )
    assert titled([old], [here])[0].title == "add rate limiting"
