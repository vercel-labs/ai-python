"""`vercel link` + `vercel env pull` should be enough to get going.

These need no sandbox: the file is read when the workspace is constructed,
so what it did is visible before anything boots.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ai.workspaces.experimental import VercelSandbox

ENV_LOCAL = """# Created by Vercel CLI
VERCEL_OIDC_TOKEN="oidc-abc123"
DATABASE_URL="postgres://example"
FEATURE_FLAG=on
"""


def test_the_project_env_stays_on_your_machine(tmp_path: Path) -> None:
    """Reading .env.local authenticates us.

    It does not put the file's contents on another host — a constructor should
    not ship a secrets file somewhere quietly.
    """
    (tmp_path / ".env.local").write_text(ENV_LOCAL)

    ws = VercelSandbox(env_file=tmp_path / ".env.local")

    assert ws.env == {}


def test_forwarding_the_project_env_is_opt_in(tmp_path: Path) -> None:
    (tmp_path / ".env.local").write_text(ENV_LOCAL)

    ws = VercelSandbox(
        env_file=tmp_path / ".env.local", forward_project_env=True
    )

    assert ws.env["DATABASE_URL"] == "postgres://example"
    assert ws.env["FEATURE_FLAG"] == "on"
    # Even then, the credentials that reach Vercel are not task config.
    assert "VERCEL_OIDC_TOKEN" not in ws.env


def test_credentials_are_made_available_to_the_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env.local").write_text(ENV_LOCAL)
    monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)

    VercelSandbox(env_file=tmp_path / ".env.local")

    assert os.environ["VERCEL_OIDC_TOKEN"] == "oidc-abc123"


def test_the_ambient_environment_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Standard dotenv manners: a variable already set is not replaced."""
    (tmp_path / ".env.local").write_text(ENV_LOCAL)
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "already-set")

    VercelSandbox(env_file=tmp_path / ".env.local")

    assert os.environ["VERCEL_OIDC_TOKEN"] == "already-set"


def test_explicit_env_wins_over_the_file(tmp_path: Path) -> None:
    (tmp_path / ".env.local").write_text(ENV_LOCAL)

    ws = VercelSandbox(
        env_file=tmp_path / ".env.local",
        forward_project_env=True,
        env={"DATABASE_URL": "postgres://override"},
    )

    assert ws.env["DATABASE_URL"] == "postgres://override"
    assert ws.env["FEATURE_FLAG"] == "on"


def test_a_missing_file_is_not_an_error(tmp_path: Path) -> None:
    ws = VercelSandbox(env_file=tmp_path / "nope.local")

    assert ws.env == {}


def test_the_file_is_found_by_walking_up(tmp_path: Path) -> None:
    """`vercel env pull` writes it at the project root; you may well be
    running from a subdirectory."""
    (tmp_path / ".env.local").write_text(ENV_LOCAL)
    nested = tmp_path / "packages" / "api"
    nested.mkdir(parents=True)

    ws = VercelSandbox(project_root=nested, forward_project_env=True)

    assert ws.env["DATABASE_URL"] == "postgres://example"
