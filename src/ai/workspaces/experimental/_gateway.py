"""Reaching a model through the Vercel AI Gateway, without looking anything up.

A harness authenticates where it RUNS. Your own `claude` login covers a
Local workspace, but a microVM has never seen it — and codex needs a
`model_providers` table it would normally read from ~/.codex/config.toml,
which a fresh VM does not have either.

    from ai.workspaces.experimental import vercel_ai_gateway

    gw = vercel_ai_gateway()                          # AI_GATEWAY_API_KEY

    # the firewall injects the key
    async with VercelSandbox(gateway=gw) as ws:
        async with claude_code(workspace=ws) as agent: ...
        async with codex(workspace=ws) as agent: ...

    # no firewall: the key goes in the env
    async with Local(".", gateway=gw) as ws:
        async with claude_code(workspace=ws) as agent: ...

The record is neutral: a base URL and a credential, nothing about any
harness. The WORKSPACE holds where the model is reached — and, on a
sandbox, injects the credential into requests at egress, so the VM only
ever holds a placeholder. Each ADAPTER spells the record the way its own
CLI wants to be told: claude through its environment, codex through its
`-c` provider overrides. Neither side learns the other's dialect.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict

from . import errors

#: The endpoint each CLI gets a path under. Both are compatibility surfaces
#: meant for that specific harness rather than the gateway's generic `/v1`,
#: so the traffic is attributed correctly and each CLI's model picker is
#: populated. https://vercel.com/docs/ai-gateway/coding-agents
DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh"
KEY_VAR = "AI_GATEWAY_API_KEY"

#: What a CLI is handed in place of the credential when the workspace
#: injects the real one at egress. Deliberately a fixed, recognisable
#: string: it is what a test greps the VM's environment for, and what a
#: 401 hint can name.
PLACEHOLDER = "ai-python-brokered-credential"

SETUP_HINT = (
    f"Set {KEY_VAR}, or pass vercel_ai_gateway(api_key=...). A linked "
    "project gets one from `vercel env pull`; `vercel ai-gateway setup` "
    "creates one and configures your local CLIs too. "
    "https://vercel.com/docs/ai-gateway/coding-agents"
)


class Gateway(BaseModel):
    """Where a harness reaches its model: a base URL and a credential.

    A record, not a protocol, and a neutral one — it names no harness and
    no environment variable. Adapters derive their own configuration from
    it (`gateway_env` in the Claude adapter, `gateway_config` in Codex's);
    workspaces read `host` and `credential` to inject the latter at egress.
    """

    model_config = ConfigDict(frozen=True)

    base_url: str
    credential: str

    @property
    def host(self) -> str:
        return urlparse(self.base_url).hostname or self.base_url

    @property
    def is_brokered(self) -> bool:
        """Whether this copy carries the placeholder rather than the key."""
        return self.credential == PLACEHOLDER

    def brokered(self) -> Gateway:
        """Return this gateway with the credential replaced by `PLACEHOLDER`.

        What an adapter applies on a workspace that injects credentials at
        egress: the CLI sends the placeholder, the firewall overwrites the
        header, and the real key never enters the machine the CLI runs on.
        """
        return self.model_copy(update={"credential": PLACEHOLDER})


def vercel_ai_gateway(
    *, api_key: str | None = None, base_url: str = DEFAULT_BASE_URL
) -> Gateway:
    """Configure both harnesses for the Vercel AI Gateway.

    `api_key` defaults to `AI_GATEWAY_API_KEY` from this process. Pass it
    to use a different key — a per-tenant key, one from a secret manager,
    or one that never touches your environment.

    `base_url` exists so a change at the gateway does not require a release
    of this library.
    """
    key = api_key if api_key is not None else os.environ.get(KEY_VAR, "")
    if not key:
        raise errors.NotAuthenticatedError(
            "vercel-ai-gateway", f"no {KEY_VAR}", SETUP_HINT
        )
    return Gateway(base_url=base_url.rstrip("/"), credential=key)
