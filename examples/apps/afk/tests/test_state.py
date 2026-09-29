"""afk's state holds only what the machine cannot tell, and round-trips."""

from __future__ import annotations

from afk import state as st

from ai.harnesses.experimental import Handle
from ai.workspaces.experimental import WorkspaceCoords


def _remote(label: str, sandbox: str, origin: str = "/proj") -> st.Remote:
    return st.Remote(
        label=label,
        origin=origin,
        sandbox=sandbox,
        pty=f"afk-{label}",
        mode="tui",
        handle=Handle(
            kind="claude-code",
            session_id=f"{label}-0000-0000",
            workspace=WorkspaceCoords(provider="sandbox", location=sandbox),
        ),
        from_id="a3f5",
        pushed_at=st.now(),
    )


def test_round_trip_and_origin_scoping() -> None:
    state = st.State(
        remotes=[
            _remote("billing", "amber-fox"),
            _remote("other", "blue-owl", origin="/elsewhere"),
        ]
    )
    st.save(state)
    loaded = st.load()
    assert [r.label for r in loaded.for_origin("/proj")] == ["billing"]
    assert loaded.sandboxes("/proj") == ["amber-fox"]
    assert loaded.sandboxes() == ["amber-fox", "blue-owl"]


def test_forgetting_a_sandbox_drops_everything_pushed_there() -> None:
    state = st.State(
        remotes=[
            _remote("a", "amber-fox"),
            _remote("b", "amber-fox"),
            _remote("c", "blue-owl"),
        ]
    )
    state.forget_sandbox("amber-fox")
    assert [r.label for r in state.remotes] == ["c"]


def test_missing_file_is_an_empty_state() -> None:
    assert st.load().remotes == []
