# durable_agent_workflows

Durable agent demo using AI SDK for Python, Vercel Workflows, FastAPI,
React, AI SDK UI, and AI Elements.

The UI sends full message history to `/api/chat`. The backend starts one
durable workflow turn, waits for the completed turn, and returns one AI SDK UI
message stream. The agent has a single `bash` tool.

## Prerequisites

- Python 3.12+, uv, Node.js, and pnpm for the backend and frontend.
- Vercel CLI installed and signed in (`vercel login`).
- A linked Vercel project (`vercel link` from this app directory).
- `AI_GATEWAY_API_KEY` set in your environment.

## Development

```bash
cd examples/apps/durable_agent_workflows
vercel dev
```

Set `AI_GATEWAY_API_KEY` before running.
