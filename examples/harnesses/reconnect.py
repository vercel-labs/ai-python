"""Push work into a sandbox, close your laptop, come back to it later.

Three ideas, and they are deliberately independent:

* SURVIVAL is the workspace's: `keep=True` leaves the VM — and every harness
  in it — running after you let go. `keep=False` stops it. Release your own
  client cleanly with `agent.detach()`.
* LIVE STATE is the harness's: open a harness INSIDE the workspace and ask
  it what it has (`agent.sessions()`). Each harness finds its own sessions
  with the workspace's primitives; the workspace itself knows nothing about
  harnesses. Two kinds in one VM? Ask each.
* OWNERSHIP is yours: the SDK keeps no registry of "which session is
  which". Save what you need — a `Handle` is exactly that, serializable —
  and match it back on reconnect. This file uses a JSON file as that state.

With no approval hook (as here) the agent runs to completion while you are
away. With a hook it still survives — it just blocks on the next tool call
until a client comes back to answer.
"""

import asyncio
import json
from pathlib import Path

from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces

STATE = Path("reconnect-state.json")  # the app's own bookkeeping


async def push() -> None:
    """Before you close the laptop: start two agents, keep the VM."""
    gw = workspaces.vercel_ai_gateway()
    # The gateway is the workspace's: egress locked to the gateway host, the
    # key injected there, every harness inside handed a placeholder.
    async with workspaces.VercelSandbox(keep=True, gateway=gw) as workspace:
        assert workspace.name is not None  # known once the VM exists
        saved: dict[str, str] = {"sandbox": workspace.name}

        # Two harnesses in one VM, each on its own conversation.
        for label, harness_factory, task in (
            (
                "planner",
                harnesses.claude_code,
                "Create notes.txt with a one-line plan for a TODO app, then "
                "reply: OK",
            ),
            (
                "reviewer",
                harnesses.codex,
                "Remember the codeword BETA. Reply with exactly: OK",
            ),
        ):
            agent = harness_factory(workspace=workspace)
            await agent.open()
            session = agent.session()
            await session.run(task)
            # A Handle is the portable pointer: harness kind + session id +
            # how to reopen the workspace. Save it under YOUR label.
            saved[label] = session.handle.model_dump_json()
            await agent.detach()  # release cleanly; the harness keeps running

        STATE.write_text(json.dumps(saved, indent=2))
        print(
            f"[*] left sandbox {workspace.name} running; remembered "
            f"{len(saved) - 1} sessions"
        )


async def come_back() -> None:
    """Another machine, later: reopen, see what is there, pick up each one."""
    gw = workspaces.vercel_ai_gateway()
    saved = json.loads(STATE.read_text())

    # (a) The simplest path: harness_from(handle) rebuilds the harness the
    #     handle names — workspace included, so the sandbox reopens by name —
    #     and continues its conversation. The gateway is passed again (a
    #     handle never carries a key); the reopened VM remembers that it
    #     injects credentials, so the harness is brokered all the same. One
    #     writer per conversation holds here too: the detached planner still
    #     owns its session, so a plain resume raises SessionBusyError, and
    #     fork=True is how you continue anyway.
    planner = harnesses.Handle.model_validate_json(saved["planner"])
    try:
        async with harnesses.harness_from(planner, gateway=gw) as session:
            result = await session.run(
                "What plan did you write? Reply in one line."
            )
            print("[*] planner picked up:", result.text.strip())
    except harnesses.errors.SessionBusyError as busy:
        print(f"[*] planner is still held ({busy}); forking instead")
        async with harnesses.harness_from(
            planner, gateway=gw, fork=True
        ) as session:
            result = await session.run(
                "What plan did you write? Reply in one line."
            )
            print("[*] planner picked up (forked):", result.text.strip())

    # (b) The listing path: open the workspace by name, instantiate the
    #     harness you want, and ask IT what it has. Match against your state.
    async with workspaces.VercelSandbox(
        name=saved["sandbox"], gateway=gw
    ) as workspace:
        reviewer = harnesses.Handle.model_validate_json(saved["reviewer"])
        agent = harnesses.codex(workspace=workspace)
        await agent.open()
        listed = await agent.sessions()  # codex finds ITS sessions in the VM
        print(f"[*] codex reports {len(listed)} session(s) in this sandbox")
        mine = next(i for i in listed if i.session_id == reviewer.session_id)
        # The harness you detached from is still ALIVE in the VM and holds
        # this conversation: `listed` marks it running, and resume() would
        # raise SessionBusyError — on both harnesses, the same way. From a fresh
        # client, FORK: same history, your own conversation. Reattaching to
        # the live process is the follow-up.
        picked = await agent.fork(mine.session_id)
        result = await picked.run(
            "What is the codeword? Reply with the single word only."
        )
        print("[*] reviewer picked up (forked):", result.text.strip())
        await agent.detach()

    # (c) Done for real: stop the VM. Everything in it ends with it.
    async with workspaces.VercelSandbox(name=saved["sandbox"], keep=False):
        pass
    STATE.unlink()
    print("[*] sandbox stopped")


async def main() -> None:
    await push()
    await come_back()


asyncio.run(main())
