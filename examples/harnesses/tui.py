"""Open a harness's own TUI in a sandbox, work in it, detach, come back.

Two ends of one wire: your terminal here
(`ai.workspaces.experimental.tty`), and the pseudo-terminal the TUI runs on
in the workspace (`Harness.tui()` -> `Pty`). Press Ctrl-] to detach: the
TUI keeps running because it has a NAME, and `workspace.attach(name)` picks
it up again — from this process or any other. `tui()` returns a `Tui`: the
conversation's `session_id` and `handle` to save, and the `pty` your
terminal attaches to. Run with a real terminal; the bridge needs one.

    python examples/harnesses/tui.py                  # claude, in a sandbox
    python examples/harnesses/tui.py codex            # codex, in a sandbox
    python examples/harnesses/tui.py codex <sandbox>  # reattach later
    # push a conversation from this directory into it:
    PUSH_FROM=<local session id> python examples/harnesses/tui.py claude

Credentials, from your environment: AI_GATEWAY_API_KEY for the model (the
Vercel AI Gateway, via `vercel_ai_gateway()`), and the sandbox's own —
VERCEL_OIDC_TOKEN, or VERCEL_TOKEN with VERCEL_TEAM_ID and VERCEL_PROJECT_ID.
`vercel env pull .env.local` writes them; the OIDC token is short-lived, and a
403 on sandbox creation usually means it needs pulling again:

    set -a; . ./.env.local; set +a
"""

import asyncio
import os
import sys

from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces
from ai.workspaces.experimental import tty

NAME = "demo-tui"
HARNESSES = {"claude": harnesses.claude_code, "codex": harnesses.codex}


async def main() -> None:
    gw = workspaces.vercel_ai_gateway()
    kind = (
        sys.argv[1]
        if len(sys.argv) > 1 and sys.argv[1] in HARNESSES
        else "claude"
    )
    if len(sys.argv) > 2:
        # Come back: reconnect to the sandbox by name and reattach the TUI.
        async with workspaces.VercelSandbox(name=sys.argv[2]) as workspace:
            print(
                "[*] ptys still running here:",
                [
                    (p.name, "attached" if p.attached else "free")
                    for p in await workspace.ptys()
                ],
            )
            status = await tty.bridge(await workspace.attach(NAME))
            print(
                "[*] detached again"
                if status is None
                else f"[*] the TUI exited with {status}"
            )
        return

    async with workspaces.VercelSandbox(keep=True, gateway=gw) as workspace:
        agent = HARNESSES[kind](workspace=workspace)
        await agent.open()
        history = None
        if os.environ.get("PUSH_FROM"):
            # A conversation from THIS directory, continued in the sandbox TUI:
            # `history` means the same on tui() as on session().
            async with HARNESSES[kind](workspace=workspaces.Local(".")) as here:
                history = await here.history(os.environ["PUSH_FROM"])
        tui = await agent.tui(
            history=history, name=NAME
        )  # in the harness's own TUI
        print(
            "[*] conversation "
            f"{tui.session_id or '(codex mints its id on the first turn)'}; "
            "Ctrl-] detaches and leaves it running",
            flush=True,
        )
        status = await tty.bridge(tui.pty)
        if status is None:
            # Let go of the harness too, leaving it running: its reader stops
            # here instead of holding the workspace's close for its timeouts.
            await agent.detach()
            print(
                f"[*] detached. Come back with:  python {sys.argv[0]} {kind} "
                f"{workspace.name}"
            )
        else:
            await tui.close()
            await agent.close()
            print(f"[*] the TUI exited with {status}; stopping the sandbox")
            async with workspaces.VercelSandbox(
                name=workspace.name, keep=False
            ):
                pass


asyncio.run(main())
