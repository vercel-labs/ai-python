"""The gateway: a neutral record, each adapter's spelling of it, and — on a
sandbox — the credential injected at egress so the VM never holds it.

Three kinds of test. The mapping tests are pure and fast. The live ones run
both harnesses through the gateway on this machine and in a microVM handed
nothing else. The brokering ones are the load-bearing security tests: they
read the VM's environment and process table for the real key, and try to
leave the VM for a host the policy does not allow.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ai.harnesses.experimental._adapters import claude as claude_adapter
from ai.harnesses.experimental._adapters import codex as codex_adapter
from ai.workspaces.experimental import Local, VercelSandbox
from ai.workspaces.experimental._gateway import (
    KEY_VAR,
    PLACEHOLDER,
    Gateway,
    vercel_ai_gateway,
)
from ai.workspaces.experimental._sandbox import INSTALL_HOSTS, egress_policy
from ai.workspaces.experimental.errors import (
    NotAuthenticatedError,
    WorkspaceError,
)
from tests.harnesses.experimental.conftest import SPECS, exact
from tests.workspaces.experimental.conftest import (
    ENV_LOCAL,
    SandboxCredentials,
    sandbox_credentials,
)

# Decided at collection, before `env_local` applies .env.local: read it too.
needs_gateway = pytest.mark.skipif(
    not ENV_LOCAL.get(KEY_VAR, os.environ.get(KEY_VAR)),
    reason=f"needs {KEY_VAR}",
)


# -- the record ----------------------------------------------------------------


def test_the_record_is_a_base_url_and_a_credential_and_nothing_else() -> None:
    gw = vercel_ai_gateway(api_key="k")

    assert set(Gateway.model_fields) == {"base_url", "credential"}
    assert gw.host == "ai-gateway.vercel.sh"
    assert gw.credential == "k"
    assert not gw.is_brokered


def test_brokered_swaps_the_credential_for_the_placeholder_and_nothing_else() -> (  # noqa: E501
    None
):
    gw = vercel_ai_gateway(api_key="k")
    brokered = gw.brokered()

    assert brokered.credential == PLACEHOLDER
    assert brokered.is_brokered
    assert brokered.base_url == gw.base_url
    assert (
        gw.credential == "k"
    ), "brokered() is a copy; the original still has the key"


def test_the_key_can_be_supplied_instead_of_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_VAR, "from-the-environment")

    assert vercel_ai_gateway(api_key="explicit").credential == "explicit"
    assert vercel_ai_gateway().credential == "from-the-environment"


def test_the_base_url_can_be_overridden() -> None:
    gw = vercel_ai_gateway(api_key="k", base_url="https://gw.example.com/")

    assert gw.base_url == "https://gw.example.com"
    assert gw.host == "gw.example.com"


def test_a_missing_key_is_refused_here_rather_than_as_a_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(KEY_VAR, raising=False)

    with pytest.raises(NotAuthenticatedError) as caught:
        vercel_ai_gateway()

    assert KEY_VAR in str(caught.value)
    assert "vercel ai-gateway setup" in str(caught.value)


def test_the_result_is_immutable() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        vercel_ai_gateway(api_key="k").credential = "other"  # type: ignore[misc]


# -- each adapter's spelling ---------------------------------------------------


def test_claude_is_configured_through_the_environment() -> None:
    env = claude_adapter.gateway_env(vercel_ai_gateway(api_key="k"))

    assert (
        env["ANTHROPIC_BASE_URL"] == "https://ai-gateway.vercel.sh/claude-code"
    )
    assert env["ANTHROPIC_AUTH_TOKEN"] == "k"
    # Not decoration: Claude Code checks this first, so a stray value here
    # bypasses the gateway while everything still looks like it worked.
    assert env["ANTHROPIC_API_KEY"] == ""
    assert claude_adapter.gateway_env(None) == {}


def test_codex_config_names_a_key_the_environment_supplies() -> None:
    gw = vercel_ai_gateway(api_key="k")
    config = codex_adapter.gateway_config(gw)

    assert (
        config["model_providers.vercel.base_url"]
        == "https://ai-gateway.vercel.sh/codex/v1"
    )
    assert config["model_providers.vercel.wire_api"] == "responses"
    # The config points at a variable rather than carrying the key, so the
    # two halves are only correct together — and the key is in neither
    # place a repo-controlled process reads config from.
    assert "k" not in " ".join(config.values())
    assert (
        codex_adapter.gateway_env(gw)[config["model_providers.vercel.env_key"]]
        == "k"
    )
    assert (
        codex_adapter.gateway_config(None) == {}
        and codex_adapter.gateway_env(None) == {}
    )


def test_a_brokered_gateway_spells_the_placeholder_in_both_dialects() -> None:
    gw = vercel_ai_gateway(api_key="vck_the_real_key").brokered()

    assert claude_adapter.gateway_env(gw)["ANTHROPIC_AUTH_TOKEN"] == PLACEHOLDER
    assert codex_adapter.gateway_env(gw)[KEY_VAR] == PLACEHOLDER
    spelled = str(claude_adapter.gateway_env(gw)) + str(
        codex_adapter.gateway_env(gw)
    )
    assert "vck_the_real_key" not in spelled


def test_explicit_config_wins_over_the_gateway_key_by_key() -> None:
    adapter = codex_adapter.CodexAdapter(
        gateway=vercel_ai_gateway(api_key="k"),
        config={"model_provider": "mine"},
    )

    assert adapter._config["model_provider"] == "mine"
    # ...without discarding the rest of what the gateway set up.
    assert "model_providers.vercel.base_url" in adapter._config


# -- the workspace's half ------------------------------------------------------


def test_a_workspace_without_a_firewall_never_claims_to_inject() -> None:
    gw = vercel_ai_gateway(api_key="k")
    workspace = Local(".", gateway=gw)

    assert workspace.gateway is gw
    assert not workspace.injects_credentials_for(gw.host)
    assert Local(".").gateway is None


def test_the_egress_policy_injects_at_the_gateway_host_and_nowhere_else() -> (
    None
):
    gw = vercel_ai_gateway(api_key="k")
    policy = egress_policy(gw, (*INSTALL_HOSTS, "example.internal"))

    assert policy.mode == "custom"
    (rule,) = policy.allow[gw.host]
    (transform,) = rule.transform
    assert transform.headers == {"authorization": "Bearer k"}
    for host in (*INSTALL_HOSTS, "example.internal"):
        assert (
            policy.allow[host] == ()
        ), f"{host} must be reachable but never receive the key"
    assert set(policy.allow) == {gw.host, *INSTALL_HOSTS, "example.internal"}


# -- against the real gateway --------------------------------------------------


@pytest.mark.live
@needs_gateway
async def test_both_harnesses_reach_the_gateway_from_the_workspace(
    harness_kind: str, project: Path
) -> None:
    """Local: the gateway is the workspace's, and the harness is given
    nothing — it finds where the model is on the workspace."""
    gw = vercel_ai_gateway()

    async with Local(project, gateway=gw) as workspace:
        async with SPECS[harness_kind](workspace=workspace) as harness:
            result = await harness.run(
                "Read util.py and reply with the function name only."
            )

    exact(result.text, "add")
    assert result.finish_reason == "stop"


@pytest.mark.live
@needs_gateway
async def test_a_gateway_on_the_harness_still_works_and_wins(
    harness_kind: str, project: Path
) -> None:
    gw = vercel_ai_gateway()
    decoy = Gateway(base_url="https://gateway.invalid", credential="not-a-key")

    async with Local(project, gateway=decoy) as workspace:
        async with SPECS[harness_kind](
            workspace=workspace, gateway=gw
        ) as harness:
            result = await harness.run(
                "Read util.py and reply with the function name only."
            )

    exact(result.text, "add")


def _sandbox_creds() -> SandboxCredentials:
    creds = sandbox_credentials()
    if creds is None:
        pytest.skip(
            "needs VERCEL_TOKEN/TEAM_ID/PROJECT_ID or VERCEL_OIDC_TOKEN"
        )
    return creds


EVERY_ENVIRON = (
    "for f in /proc/[0-9]*/environ; do "
    "tr '\\0' '\\n' < \"$f\" 2>/dev/null; done; env"
)


@pytest.mark.live
@pytest.mark.sandbox
@pytest.mark.slow
@pytest.mark.timeout(900)
@needs_gateway
async def test_a_brokered_sandbox_answers_and_never_holds_the_key(
    harness_kind: str,
) -> None:
    """The load-bearing one.

    A VM created with a gateway: the harness inside it reaches a model (so the
    header rewrite works through TLS for both a Node CLI and codex's binary),
    and the real key appears in NO process's environment in the VM — only the
    placeholder does.
    """
    gw = vercel_ai_gateway()

    async with VercelSandbox(
        **_sandbox_creds(), gateway=gw, execution_time_limit=900
    ) as ws:
        assert ws.injects_credentials_for(gw.host)
        await ws.write_text("util.py", "def add(a, b):\n    return a + b\n")
        # No sandbox= for codex: inside a microVM its own sandbox cannot start
        # (the adapter's documented default there is danger-full-access, the
        # VM being the boundary), and a tool that fails reads as a wrong answer.
        async with SPECS[harness_kind](workspace=ws) as harness:
            # A kept session, not run(): run() closes its conversation and with
            # it the CLI process, and the process table is what we grep next.
            session = harness.session()
            result = await session.run(
                "Read util.py and reply with the function name only."
            )
            environs = await ws.exec(["sh", "-c", EVERY_ENVIRON], timeout=60)
            await session.close()

    exact(result.text, "add")
    assert (
        gw.credential not in environs.stdout
    ), "the real key is readable inside the VM"
    # ...and the placeholder IS there, which proves the grep saw the harness's
    # own process rather than an empty table.
    assert (
        PLACEHOLDER in environs.stdout
    ), "the harness process was not handed the placeholder"


@pytest.mark.live
@pytest.mark.sandbox
@pytest.mark.slow
@pytest.mark.timeout(600)
@needs_gateway
async def test_a_brokered_sandbox_cannot_reach_any_other_host() -> None:
    """Deny-all except the gateway and what the install needs.

    A curious agent's `curl` to the provider directly, or anywhere else, is
    refused.
    """
    gw = vercel_ai_gateway()

    async with VercelSandbox(
        **_sandbox_creds(), gateway=gw, execution_time_limit=600
    ) as ws:
        for host in ("api.anthropic.com", "api.openai.com", "example.com"):
            probe = await ws.exec(
                [
                    "curl",
                    "-sS",
                    "-m",
                    "20",
                    "-o",
                    "/dev/null",
                    "-w",
                    "%{http_code}",
                    f"https://{host}/",
                ],
                timeout=60,
            )
            assert probe.exit_code != 0 or probe.stdout.strip() in (
                "000",
                "403",
            ), (
                f"{host} was reachable from a brokered sandbox: "
                f"exit {probe.exit_code}, status {probe.stdout!r}"
            )
        # ...and the allowed hosts are, without any header rewrite on them.
        registry = await ws.exec(
            [
                "curl",
                "-sS",
                "-m",
                "20",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "https://registry.npmjs.org/",
            ],
            timeout=60,
        )
        assert registry.exit_code == 0 and registry.stdout.strip().startswith(
            ("2", "3")
        )


@pytest.mark.live
@pytest.mark.sandbox
@pytest.mark.slow
@pytest.mark.timeout(900)
@needs_gateway
async def test_a_wrong_injected_key_is_named_as_such(harness_kind: str) -> None:
    """A placeholder in the VM plus a bad key in the firewall rule is a 401
    the CLI can do nothing about. The error must point at the key on the
    host and the egress policy — not at a login in the VM."""
    wrong = Gateway(
        base_url=vercel_ai_gateway().base_url, credential="vck_not_a_real_key"
    )

    async with VercelSandbox(
        **_sandbox_creds(), gateway=wrong, execution_time_limit=900
    ) as ws:
        async with SPECS[harness_kind](workspace=ws) as harness:
            with pytest.raises(NotAuthenticatedError) as caught:
                await harness.run("Reply with exactly: READY")

    message = str(caught.value)
    assert "injects the credential" in message and wrong.host in message
    assert KEY_VAR in message


@pytest.mark.live
@pytest.mark.sandbox
@pytest.mark.slow
@pytest.mark.timeout(600)
@needs_gateway
async def test_reconnecting_with_a_gateway_to_a_vm_made_without_one_is_refused() -> (  # noqa: E501
    None
):
    """Egress is fixed at creation.

    Asking a plain VM to broker would put the real key in it; the workspace says
    no instead.
    """
    gw = vercel_ai_gateway()

    async with VercelSandbox(
        **_sandbox_creds(), keep=True, execution_time_limit=600
    ) as plain:
        name = plain.name
        assert name is not None
    try:
        with pytest.raises(
            WorkspaceError, match="without credential injection"
        ):
            async with VercelSandbox(**_sandbox_creds(), name=name, gateway=gw):
                pass
    finally:
        async with VercelSandbox(**_sandbox_creds(), name=name, keep=False):
            pass


@pytest.mark.live
@pytest.mark.sandbox
@pytest.mark.slow
@pytest.mark.timeout(600)
@needs_gateway
async def test_a_brokered_vm_remembers_it_across_a_reconnect() -> None:
    gw = vercel_ai_gateway()

    async with VercelSandbox(
        **_sandbox_creds(), gateway=gw, keep=True, execution_time_limit=600
    ) as ws:
        name = ws.name
        assert name is not None
    try:
        async with VercelSandbox(
            **_sandbox_creds(), name=name, gateway=gw
        ) as again:
            assert again.injects_credentials_for(gw.host)
    finally:
        async with VercelSandbox(**_sandbox_creds(), name=name, keep=False):
            pass


@pytest.mark.live
async def test_the_recipe_still_matches_the_published_one() -> None:
    """A canary, because this recipe is a COPY of someone else's docs.

    It has already drifted once: the endpoint moved from the gateway's
    bare root to /claude-code, and CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY
    appeared. Nothing in the SDK could have noticed. This fails in CI
    instead of in someone's 401.
    """
    import html
    import re
    import urllib.error
    import urllib.request

    url = "https://vercel.com/docs/ai-gateway/coding-agents"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"cannot reach {url}: {exc}")
    page = html.unescape(re.sub(r"<[^>]+>", "", raw))

    gw = vercel_ai_gateway(api_key="k")
    codex_config = codex_adapter.gateway_config(gw)
    expected = [
        claude_adapter.gateway_env(gw)["ANTHROPIC_BASE_URL"],
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY",
        codex_config["model_providers.vercel.base_url"],
        f'wire_api = "{codex_config["model_providers.vercel.wire_api"]}"',
        f'env_key = "{KEY_VAR}"',
    ]
    missing = [value for value in expected if value not in page]

    assert not missing, (
        f"the published recipe no longer mentions {missing} — "
        f"check the adapters' gateway_env/gateway_config and {url}"
    )
