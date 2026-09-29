"""Hand a conversation over: to the other harness, or to another machine.

A conversation is a list of messages — `history()` gives you one, and so does
`result.messages`. To move it, start a new session FROM it. The receiving
harness stores it natively before the first turn: text as it was said, tool
calls as real tool items with their original names, reasoning as a note from
the previous session. Same call whether the destination is the other harness
or a sandbox on another machine.
"""

import asyncio
import os
import tempfile

from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces


async def main() -> None:
    # Your own login when the CLIs have one here; the gateway when a key is
    # around. One record, both workspaces below — on your machine it goes in
    # the environment, in the sandbox it is injected at egress.
    gw = (
        workspaces.vercel_ai_gateway()
        if os.environ.get("AI_GATEWAY_API_KEY")
        else None
    )

    with tempfile.TemporaryDirectory() as tmp:
        async with workspaces.Local(tmp, gateway=gw) as workspace:
            # 1. A conversation with Claude, on this machine. It does real work.
            async with harnesses.claude_code(workspace=workspace) as agent:
                session = agent.session()
                await session.run(
                    "Create notes.txt containing exactly: milk, eggs. "
                    + "Also remember: the deadline is FRIDAY. Reply with "
                    "exactly: OK"
                )
                history = await agent.history(session.session_id)
            calls = sum(
                1 for m in history for p in m.parts if p.kind == "tool_call"
            )
            print(
                f"Claude's conversation: {len(history)} messages, "
                f"{calls} tool call(s)"
            )

            # 2. Codex picks it up — same directory, other harness. It knows
            #    what was said AND what was done, and acts with its own tools.
            async with harnesses.codex(
                workspace=workspace, sandbox="workspace-write"
            ) as agent:
                result = await agent.run(
                    "What is the deadline, and what did we put in notes.txt? "
                    "Then append " + "'bread' to notes.txt. Reply in one line.",
                    history=history,
                )
                print("Codex:", result.text.strip())
                print(
                    "notes.txt:",
                    (await workspace.read_text("notes.txt"))
                    .strip()
                    .replace("\n", " / "),
                )

        # 3. The same history, on another machine entirely. Only the
        #    workspace changes.
        if os.environ.get("VERCEL_OIDC_TOKEN") and gw is not None:
            async with workspaces.VercelSandbox(gateway=gw) as remote:
                async with harnesses.claude_code(workspace=remote) as agent:
                    session = agent.session(history=history)
                    result = await session.run(
                        "What is the deadline? Reply with the day only."
                    )
                    print("Claude, in a sandbox:", result.text.strip())


asyncio.run(main())
