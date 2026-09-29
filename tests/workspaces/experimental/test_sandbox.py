"""An expired platform token must be NAMED as the cause, not surface as 403.

Measured: a token that expired nineteen minutes into a run produced 84
identical setup errors and 5 failures reading `HTTP 403: Not authorized`,
and nothing in twenty minutes of output said "expired".
"""

from __future__ import annotations

import base64
import json
import time

import pytest

from ai.workspaces.experimental._sandbox import (
    _auth_hint,
    _looks_like_auth_failure,
    oidc_token_expiry,
)


def _jwt(exp: float) -> str:
    body = (
        base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode())
        .decode()
        .rstrip("=")
    )
    return f"eyJhbGciOiJub25lIn0.{body}.sig"


def test_expiry_is_read_from_the_token() -> None:
    assert oidc_token_expiry(_jwt(1_800_000_000)) == 1_800_000_000.0


def test_garbage_is_not_a_token() -> None:
    assert oidc_token_expiry("") is None
    assert oidc_token_expiry("not.a.jwt") is None
    assert oidc_token_expiry(None) is None


def test_platform_denials_are_recognised() -> None:
    assert _looks_like_auth_failure(
        Exception("HTTP 403: Not authorized (code=forbidden)")
    )
    assert _looks_like_auth_failure(Exception("HTTP 401 Unauthorized"))
    assert not _looks_like_auth_failure(Exception("HTTP 500: internal"))
    assert not _looks_like_auth_failure(Exception("connection reset"))


def test_the_hint_names_expiry_when_that_is_the_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", _jwt(time.time() - 60))
    hint = _auth_hint()
    assert "expired at" in hint
    assert "vercel env pull" in hint


def test_the_hint_is_generic_when_the_token_is_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", _jwt(time.time() + 3600))
    assert "expired" not in _auth_hint()


def test_the_sdk_credentials_error_is_an_auth_failure() -> None:
    class SandboxCredentialsError(Exception): ...

    assert _looks_like_auth_failure(
        SandboxCredentialsError(
            "OIDC token present but could not determine VERCEL_PROJECT_ID"
        )
    )


def test_the_suite_stops_once_on_an_expired_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard lives in tests/workspaces/experimental/conftest.py.

    It must be tested from inside a test, because conftest's `env_local`
    overrides os.environ from .env.local before any test runs — a fake token
    set in the shell is replaced by the real one before the guard ever sees
    it. Two probes were defeated by exactly that before this test was written.
    """
    from _pytest.outcomes import Exit

    from tests.workspaces.experimental.conftest import sandbox_credentials

    monkeypatch.setenv("VERCEL_OIDC_TOKEN", _jwt(time.time() - 60))
    monkeypatch.delenv("VERCEL_TOKEN", raising=False)
    with pytest.raises(Exit, match=r"expired at .*vercel env pull"):
        sandbox_credentials()


def test_a_live_token_passes_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.workspaces.experimental.conftest import sandbox_credentials

    monkeypatch.setenv("VERCEL_OIDC_TOKEN", _jwt(time.time() + 3600))
    monkeypatch.delenv("VERCEL_TOKEN", raising=False)
    assert sandbox_credentials() == {}
