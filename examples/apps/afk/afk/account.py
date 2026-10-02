"""Vercel credentials for afk, set up the first time a sandbox is needed.

A globally installed CLI cannot rely on a `.env.local` in whatever directory
it runs from. afk signs in through the Vercel CLI instead:

- Credentials already in the environment win (`VERCEL_OIDC_TOKEN`, or
  `VERCEL_TOKEN` with `VERCEL_TEAM_ID` and `VERCEL_PROJECT_ID`).
- Otherwise afk uses your `vercel login`, and a project of its own named
  `afk-<random>`, created on first use and recorded in `~/.afk/config.json`.
  That file holds ids only; every run mints a short-lived project token from
  your login, so nothing afk stores can leak a secret or expire.

The same token reaches the model: AI Gateway accepts it, so
`AI_GATEWAY_API_KEY` is optional.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from vercel.oidc import token as oidc_token
from vercel.oidc import utils as oidc_utils

from . import state as st

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

API = "https://api.vercel.com"
GATEWAY = "https://ai-gateway.vercel.sh"
PROBE_MODEL = "anthropic/claude-haiku-4.5"
INSTALL_HINT = "npm i -g vercel  (or: pnpm add -g vercel)"
TOKEN_MARGIN = 5 * 60
"""Seconds of life a cached project token must have left to be reused."""


class AccountError(Exception):
    """afk could not get Vercel credentials; the message says what to do."""


class Config(BaseModel):
    """The Vercel project afk creates sandboxes in. Ids only, no secrets."""

    team_id: str | None
    team_slug: str | None
    project_id: str
    project_name: str
    gateway_ok: bool = False
    """The team's AI Gateway served a request once; not asked again."""


def config_path() -> Path:
    """`config.json` next to afk's state file: `~/.afk/` by default."""
    return st.state_path().parent / "config.json"


def load_config() -> Config | None:
    try:
        return Config.model_validate_json(config_path().read_text())
    except FileNotFoundError:
        return None


def save_config(config: Config) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2) + "\n")


def ensure(
    *, interactive: bool, fresh: bool = False, for_gateway: bool = False
) -> str | None:
    """Make Vercel credentials available to the sandbox SDK in this process.

    Returns the project token when afk minted one (it doubles as the AI
    Gateway credential), or None when the environment already had
    credentials. `fresh` mints a new token even if a cached one is valid,
    for a push whose sandbox should outlive the cached token.
    `for_gateway` also makes sure the team's AI Gateway will serve that
    token, the first time it is used that way.
    """
    if os.environ.get("VERCEL_OIDC_TOKEN"):
        return os.environ["VERCEL_OIDC_TOKEN"]
    if all(
        os.environ.get(k)
        for k in ("VERCEL_TOKEN", "VERCEL_TEAM_ID", "VERCEL_PROJECT_ID")
    ):
        return None
    login = cli_token(interactive=interactive)
    config = load_config() or setup(login, interactive=interactive)
    try:
        token = project_token(login, config, fresh=fresh)
    except Exception:
        # The recorded project is gone (deleted, or its team left): start
        # over rather than failing every run from now on.
        print(
            f"afk: the Vercel project {config.project_name} is no longer "
            "available; setting up a new one"
        )
        config = setup(login, interactive=interactive)
        token = project_token(login, config, fresh=True)
    if for_gateway and not config.gateway_ok:
        config, token = _check_gateway(
            login, config, token, interactive=interactive
        )
    os.environ["VERCEL_OIDC_TOKEN"] = token
    os.environ["VERCEL_PROJECT_ID"] = config.project_id
    if config.team_id:
        os.environ["VERCEL_TEAM_ID"] = config.team_id
    return token


def cli_token(*, interactive: bool) -> str:
    """The Vercel CLI's login token, signing in first when there is none."""
    token = oidc_utils.get_vercel_cli_token()
    if token:
        return token
    if shutil.which("vercel") is None:
        raise AccountError(
            "afk signs in to Vercel through the Vercel CLI, which is not "
            f"installed. Install it with `{INSTALL_HINT}`, then run afk "
            "again. (Or set VERCEL_TOKEN, VERCEL_TEAM_ID and "
            "VERCEL_PROJECT_ID.)"
        )
    if not interactive:
        raise AccountError(
            "afk needs you signed in to Vercel: run `vercel login`, then run "
            "afk again"
        )
    print("afk: signing you in to Vercel (only needed once)")
    subprocess.run(["vercel", "login"], check=False)
    token = oidc_utils.get_vercel_cli_token()
    if not token:
        raise AccountError(
            "still not signed in to Vercel; run `vercel login` and try again"
        )
    return token


def _check_gateway(
    login: str, config: Config, token: str, *, interactive: bool
) -> tuple[Config, str]:
    """Make sure AI Gateway serves the team; offer another team if not.

    One tiny request, once: a team without a card on file is refused with
    `customer_verification_required`, and a sandbox would only find out
    when its agent could not reach a model.
    """
    while True:
        refusal = gateway_refusal(token)
        if refusal is None:
            config = config.model_copy(update={"gateway_ok": True})
            save_config(config)
            return config, token
        where = config.team_slug or "your account"
        message = (
            f"AI Gateway will not serve {where} yet: {refusal}\n"
            "Add a card there, set AI_GATEWAY_API_KEY, or choose another "
            "team"
        )
        if not interactive:
            raise AccountError(f"{message} with `afk setup`.")
        print(f"afk: {message}.")
        if ask_yes("afk: choose another team now? [Y/n] ", default=True):
            config = setup(login, interactive=True, exclude=config.team_id)
            token = project_token(login, config, fresh=True)
            continue
        raise AccountError("run `afk setup` when you are ready")


def gateway_refusal(token: str) -> str | None:
    """Why AI Gateway refuses this token, or None when it serves it.

    A one-token request: the free endpoints answer even for a team that
    cannot be billed. Network trouble is not a refusal; the push goes on.
    """
    body = {
        "model": PROBE_MODEL,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }
    request = urllib.request.Request(
        f"{GATEWAY}/v1/chat/completions",
        method="POST",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30):
            return None
    except urllib.error.HTTPError as exc:
        if exc.code not in (401, 402, 403):
            return None
        try:
            error = json.load(exc).get("error") or {}
        except ValueError:
            error = {}
        return str(error.get("message") or f"HTTP {exc.code}")
    except OSError:
        return None


def ask_yes(
    prompt: str, *, default: bool, ask: Callable[[str], str] = input
) -> bool:
    answer = ask(prompt).strip().lower()
    return default if not answer else answer in ("y", "yes")


def setup(
    login: str, *, interactive: bool, exclude: str | None = None
) -> Config:
    """Pick a team and create afk's own project in it.

    Runs by itself the first time a sandbox is needed, and again with
    `afk setup` to switch teams. `exclude` leaves out a team that was just
    found unusable.
    """
    teams = [t for t in list_teams(login) if t.get("id") != exclude]
    team = choose_team(teams, interactive=interactive)
    query = f"?teamId={team['id']}" if team else ""
    for _ in range(5):
        name = f"afk-{secrets.token_hex(3)}"
        try:
            project = api(
                "POST", f"/v11/projects{query}", login, {"name": name}
            )
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 409:  # 409: that name is taken; try another
                raise AccountError(
                    f"could not create a Vercel project for afk: HTTP "
                    f"{exc.code} {exc.reason}"
                ) from exc
    else:
        raise AccountError("could not find a free afk-* project name")
    config = Config(
        team_id=team["id"] if team else None,
        team_slug=team["slug"] if team else None,
        project_id=project["id"],
        project_name=name,
    )
    save_config(config)
    where = f" in {team['slug']}" if team else ""
    print(
        f"afk: created the Vercel project {name}{where} for your sandboxes "
        f"(recorded in {_tilde(config_path())})"
    )
    return config


def list_teams(login: str) -> list[dict[str, Any]]:
    try:
        data = api("GET", "/v2/teams?limit=100", login)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise AccountError(
                "your Vercel CLI login has expired; run `vercel login` and "
                "try again"
            ) from exc
        raise
    return list(data.get("teams") or [])


def choose_team(
    teams: list[dict[str, Any]],
    *,
    interactive: bool,
    ask: Callable[[str], str] = input,
) -> dict[str, Any] | None:
    """The team afk's project goes in. Asked once, on first use.

    `AFK_TEAM` (slug or id) decides without asking. One team needs no
    question. Otherwise the Vercel CLI's current team is the default: Enter
    accepts it. Without a terminal to ask on, that default is used; with no
    default either, afk says how to choose.
    """
    if not teams:
        return None  # an account without teams: its personal scope
    wanted = os.environ.get("AFK_TEAM")
    if wanted:
        for team in teams:
            if wanted in (team.get("slug"), team.get("id")):
                return team
        raise AccountError(
            f"AFK_TEAM={wanted} is not one of your Vercel teams: "
            + ", ".join(t["slug"] for t in teams)
        )
    if len(teams) == 1:
        return teams[0]
    current = cli_current_team()
    default = next((t for t in teams if t["id"] == current), None)
    if not interactive:
        if default is not None:
            return default
        raise AccountError(
            "afk needs to know which Vercel team to use; set AFK_TEAM to "
            "one of: " + ", ".join(t["slug"] for t in teams)
        )
    print("afk: which Vercel team should afk's sandboxes belong to?")
    for n, team in enumerate(teams, 1):
        mark = "  (Vercel CLI's current team)" if team is default else ""
        print(f"  {n:>2}. {team['slug']}{mark}")
    prompt = "afk: team number"
    prompt += f" [{teams.index(default) + 1}]: " if default else ": "
    while True:
        answer = ask(prompt).strip()
        if not answer and default is not None:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(teams):
            return teams[int(answer) - 1]
        print(f"afk: enter a number from 1 to {len(teams)}")


def cli_current_team() -> str | None:
    data_dir = oidc_utils.get_vercel_data_dir()
    if not data_dir:
        return None
    try:
        with open(os.path.join(data_dir, "config.json")) as f:
            team = json.load(f).get("currentTeam")
    except (OSError, ValueError):
        return None
    return team if isinstance(team, str) else None


def project_token(login: str, config: Config, *, fresh: bool) -> str:
    """A short-lived token for afk's project, minted from your login.

    Cached by the Vercel SDK in its own data directory, never in the
    current one, and reused while it has life left.
    """
    if not fresh:
        cached = oidc_utils.load_token(config.project_id)
        if cached is not None and seconds_left(cached.token) > TOKEN_MARGIN:
            return str(cached.token)
    minted = oidc_token.fetch_vercel_oidc_token(
        login, config.project_id, config.team_id
    )
    if minted is None:
        raise RuntimeError("the Vercel API returned no project token")
    oidc_utils.save_token(minted, config.project_id)
    return str(minted.token)


def seconds_left(token: str) -> float:
    """How long a project token stays valid."""
    exp = oidc_utils.get_token_payload(token).get("exp")
    return float(exp) - time.time() if isinstance(exp, int | float) else 0.0


def api(
    method: str, path: str, token: str, body: dict[str, Any] | None = None
) -> dict[str, Any]:
    request = urllib.request.Request(
        API + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result: dict[str, Any] = json.load(response)
        return result


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _tilde(path: Path) -> str:
    home = str(path.home())
    text = str(path)
    return "~" + text[len(home) :] if text.startswith(home) else text
