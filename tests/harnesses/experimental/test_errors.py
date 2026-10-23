"""An upstream 401 must arrive as something the caller can act on.

This is a pure mapping test on purpose. The behaviour it guards was found
the expensive way — a turn inside a sandbox that failed with
`unexpected status 401 Unauthorized ... api.openai.com` and no indication
that the missing thing was a key in the VM, not a key on the machine
running the SDK.
"""

from __future__ import annotations

import pytest

from ai.harnesses.experimental.errors import (
    TurnFailedError,
    _looks_unauthenticated,
)
from ai.workspaces.experimental.errors import NotAuthenticatedError

CREDENTIALS = [
    # Codex, verbatim, against a sandbox with no provider configured.
    "unexpected status 401 Unauthorized: Missing bearer or basic "
    + "authentication in header, url: https://api.openai.com/v1/responses",
    # Claude, verbatim, against a sandbox with no key.
    "Not logged in · Please run /login",
    "invalid_api_key: incorrect API key provided",
]

NOT_CREDENTIALS = [
    "",
    "the model declined to answer",
    # Authorization, not authentication: sending someone to fix their
    # login when their key simply lacks access wastes their afternoon.
    "403 Forbidden",
    "rate limit exceeded",
    "connection reset by peer",
]


@pytest.mark.parametrize("detail", CREDENTIALS)
def test_credential_failures_are_recognised(detail: str) -> None:
    assert _looks_unauthenticated(detail)


@pytest.mark.parametrize("detail", NOT_CREDENTIALS)
def test_other_failures_are_left_alone(detail: str) -> None:
    assert not _looks_unauthenticated(detail)


def test_codex_raises_not_authenticated_with_a_hint() -> None:
    from ai.harnesses.experimental._adapters.codex import _turn_failure

    error = _turn_failure(CREDENTIALS[0])
    assert isinstance(error, NotAuthenticatedError)
    # The hint has to name the fix, not just the problem.
    assert "OPENAI_API_KEY" in str(error)
    assert "config=" in str(error) or "config={" in str(error)


def test_an_ordinary_failure_is_still_a_turn_failure() -> None:
    from ai.harnesses.experimental._adapters.codex import _turn_failure

    error = _turn_failure("the tool crashed")
    assert isinstance(error, TurnFailedError)
    assert not isinstance(error, NotAuthenticatedError)
