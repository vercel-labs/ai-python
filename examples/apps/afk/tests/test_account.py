"""First-run Vercel setup: every decision afk makes, without the network."""

from __future__ import annotations

import email.message
import io
import urllib.error
from typing import Any

import pytest
from afk import account

TEAMS = [
    {"id": "team_a", "slug": "personal"},
    {"id": "team_b", "slug": "acme"},
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "VERCEL_OIDC_TOKEN",
        "VERCEL_TOKEN",
        "VERCEL_TEAM_ID",
        "VERCEL_PROJECT_ID",
        "AFK_TEAM",
    ):
        monkeypatch.delenv(key, raising=False)


def test_one_team_needs_no_question() -> None:
    assert account.choose_team(TEAMS[:1], interactive=False) == TEAMS[0]


def test_no_teams_means_the_personal_scope() -> None:
    assert account.choose_team([], interactive=False) is None


def test_afk_team_decides_without_asking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AFK_TEAM", "acme")
    assert account.choose_team(TEAMS, interactive=True) == TEAMS[1]


def test_an_unknown_afk_team_names_the_real_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AFK_TEAM", "nope")
    with pytest.raises(account.AccountError, match="personal, acme"):
        account.choose_team(TEAMS, interactive=True)


def test_enter_accepts_the_cli_current_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(account, "cli_current_team", lambda: "team_b")
    assert (
        account.choose_team(TEAMS, interactive=True, ask=lambda _: "")
        == (TEAMS[1])
    )


def test_a_number_picks_a_team_and_nonsense_asks_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(account, "cli_current_team", lambda: None)
    answers = iter(["", "9", "x", "1"])
    picked = account.choose_team(
        TEAMS, interactive=True, ask=lambda _: next(answers)
    )
    assert picked == TEAMS[0]


def test_without_a_terminal_the_cli_team_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(account, "cli_current_team", lambda: "team_a")
    assert account.choose_team(TEAMS, interactive=False) == TEAMS[0]


def test_without_a_terminal_or_a_default_afk_says_how_to_choose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(account, "cli_current_team", lambda: None)
    with pytest.raises(account.AccountError, match="AFK_TEAM"):
        account.choose_team(TEAMS, interactive=False)


def test_a_missing_vercel_cli_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "afk.account.oidc_utils.get_vercel_cli_token", lambda: None
    )
    monkeypatch.setattr("afk.account.shutil.which", lambda _: None)
    with pytest.raises(account.AccountError, match="npm i -g vercel"):
        account.cli_token(interactive=True)


def test_not_signed_in_without_a_terminal_says_to_log_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "afk.account.oidc_utils.get_vercel_cli_token", lambda: None
    )
    monkeypatch.setattr("afk.account.shutil.which", lambda _: "/bin/vercel")
    with pytest.raises(account.AccountError, match="vercel login"):
        account.cli_token(interactive=False)


def test_not_signed_in_with_a_terminal_runs_vercel_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = iter([None, "cli-token"])
    ran: list[list[str]] = []
    monkeypatch.setattr(
        "afk.account.oidc_utils.get_vercel_cli_token", lambda: next(tokens)
    )
    monkeypatch.setattr("afk.account.shutil.which", lambda _: "/bin/vercel")
    monkeypatch.setattr(
        "afk.account.subprocess.run", lambda argv, **_: ran.append(argv)
    )
    assert account.cli_token(interactive=True) == "cli-token"
    assert ran == [["vercel", "login"]]


def test_setup_creates_an_afk_project_and_records_only_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, Any]] = []

    def api(
        method: str, path: str, token: str, body: Any = None
    ) -> dict[str, Any]:
        calls.append((method, path, body))
        if path.startswith("/v2/teams"):
            return {"teams": TEAMS[:1]}
        if len([c for c in calls if c[0] == "POST"]) == 1:
            raise urllib.error.HTTPError(
                path, 409, "Conflict", email.message.Message(), io.BytesIO()
            )
        return {"id": "prj_1"}

    monkeypatch.setattr(account, "api", api)
    config = account.setup("cli-token", interactive=False)

    posts = [c for c in calls if c[0] == "POST"]
    assert len(posts) == 2, "a taken name is retried with another"
    assert posts[1][1] == "/v11/projects?teamId=team_a"
    assert config.project_name.startswith("afk-")
    assert config.project_id == "prj_1"
    saved = account.config_path().read_text()
    assert "prj_1" in saved and "token" not in saved.lower()


def test_environment_credentials_win(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "from-env")
    monkeypatch.setattr(
        account, "cli_token", lambda **_: pytest.fail("must not sign in")
    )
    assert account.ensure(interactive=True) == "from-env"


CONFIG = account.Config(
    team_id="team_a",
    team_slug="personal",
    project_id="prj_1",
    project_name="afk-abc123",
)


def test_a_gateway_that_serves_is_recorded_and_not_asked_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(account, "gateway_refusal", lambda _: None)
    config, _ = account._check_gateway("cli", CONFIG, "tok", interactive=False)
    assert config.gateway_ok
    loaded = account.load_config()
    assert loaded is not None and loaded.gateway_ok


def test_a_refusing_gateway_without_a_terminal_says_what_to_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        account, "gateway_refusal", lambda _: "requires a valid credit card"
    )
    with pytest.raises(account.AccountError) as caught:
        account._check_gateway("cli", CONFIG, "tok", interactive=False)
    text = str(caught.value)
    assert "credit card" in text and "AI_GATEWAY_API_KEY" in text
    assert "afk setup" in text


def test_a_refusing_gateway_offers_another_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refusals = iter(["requires a valid credit card", None])
    excluded: list[str | None] = []
    other = CONFIG.model_copy(update={"team_id": "team_b", "team_slug": "acme"})

    def setup(
        login: str, *, interactive: bool, exclude: str | None = None
    ) -> account.Config:
        excluded.append(exclude)
        return other

    monkeypatch.setattr(account, "gateway_refusal", lambda _: next(refusals))
    monkeypatch.setattr(account, "ask_yes", lambda *_, **__: True)
    monkeypatch.setattr(account, "setup", setup)
    monkeypatch.setattr(account, "project_token", lambda *_, **__: "tok2")
    config, token = account._check_gateway(
        "cli", CONFIG, "tok", interactive=True
    )
    assert excluded == ["team_a"], "the refused team is not offered again"
    assert (config.team_slug, token, config.gateway_ok) == (
        "acme",
        "tok2",
        True,
    )
