"""What you type becomes one conversation, or afk asks — it never guesses."""

from __future__ import annotations

import time

import pytest
from afk.resolve import AmbiguousError, NoMatchError, match, pick
from afk.rows import Row, Where


def _row(
    sid: str,
    title: str,
    *,
    active: bool = False,
    label: str | None = None,
    where: Where = "here",
) -> Row:
    ts = int(time.time()) - (60 if active else 3 * 3600)
    return Row(
        where=where,
        kind="claude-code",
        session_id=sid,
        title=title,
        updated_at=ts,
        status="",
        label=label,
    )


ROWS = [
    _row("a3f5aaaa-0000", "migrate the billing tests to pytest", active=True),
    _row("b81c1111-0000", "add rate limiting to the webhook"),
    _row(
        "c2d1bbbb-0000",
        "review the auth middleware",
        label="review",
        where="remote",
    ),
]


def test_a_word_from_the_title_or_a_label_is_enough() -> None:
    assert match(ROWS, "webhook").session_id.startswith("b81c")
    assert match(ROWS, "review").label == "review"
    assert match(ROWS, "a3f5").session_id.startswith("a3f5")


def test_no_argument_means_the_one_conversation_clearly_in_use() -> None:
    assert match(ROWS, None).session_id.startswith("a3f5")


def test_two_active_conversations_are_asked_about_not_guessed() -> None:
    rows = [
        *ROWS,
        _row("d9e0cccc-0000", "fix the flaky deploy test", active=True),
    ]
    with pytest.raises(AmbiguousError):
        match(rows, None)


def test_a_word_matching_two_rows_is_ambiguous_and_names_both() -> None:
    with pytest.raises(AmbiguousError) as amb:
        match(ROWS, "the")
    assert len(amb.value.matches) == 3


def test_where_scopes_the_search() -> None:
    with pytest.raises(NoMatchError):
        match(ROWS, "review", where="here")


def test_pick_takes_a_number_and_enter_cancels() -> None:
    assert pick(ROWS, "pick", ask=lambda _: "2") is ROWS[1]
    assert pick(ROWS, "pick", ask=lambda _: "") is None
