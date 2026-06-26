# multiagent-textual

Multi-agent hook resolution demo with a Textual TUI.  Two sub-agents
run in parallel — each gated by an approval hook — then a third agent
summarises their results.  Hook approvals happen interactively in the
terminal.

The current implementation uses:

- `ai.agent(...)` for each branch and the orchestrator
- `await ai.hook(...)` for branch-specific approvals
- `ai.yield_from(...)` to forward nested agent output into the outer run
- `role="internal"` messages for hook state updates over the WebSocket

## Setup

```bash
uv sync
```

## Running

```bash
# Terminal 1: server
uv run fastapi dev server.py

# Terminal 2: client
uv run python client.py
```

The TUI connects to `ws://localhost:8000/ws` and renders three agent
panels.  When a hook fires, the input bar at the bottom prompts for
`y` / `n`.  Approving one hook does not block the other — both agents
stream concurrently.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AI_GATEWAY_API_KEY` | Vercel AI Gateway API key |
