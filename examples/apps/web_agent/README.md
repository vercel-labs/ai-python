# web_agent

Chat demo using the Python Vercel AI SDK with a FastAPI backend and React frontend.
Includes **human-in-the-loop tool approval** — the `talk_to_mothership`
tool is gated behind user confirmation before execution.

## Stack

- **Backend:** FastAPI + AI SDK for Python (Python 3.12)
- **Frontend:** Vite + React + AI SDK UI + AI Elements

## Human-in-the-Loop

`talk_to_mothership` in `backend/agent.py` is declared with
`@ai.tool(require_approval=True)`, so the runtime suspends the run on a
`ToolApproval` hook before executing it. The flow is:

1. LLM emits a call to the gated tool
2. The runtime emits a `HookEvent` with a deferred `HookPart`; the
   backend defers the hook (`event.hook.defer()`) so the turn
   ends and the deferred approval streams to the client
3. The frontend renders Approve / Reject buttons via the
   `<Confirmation>` component (from AI Elements)
4. When the user clicks a button, `addToolApprovalResponse()` patches
   the message and sends a new request with the decision
5. The backend pre-registers the decision via
   `ai.agents.ui.ai_sdk.apply_approvals(...)` and re-runs the turn: the
   resumed run either executes the tool or records a denied tool result

Tool results are appended as separate `role="tool"` messages. The
assistant tool-call message remains immutable.

## Setup

```bash
# Backend
cd backend
uv sync
export AI_GATEWAY_API_KEY=…  # or put it in backend/.env and use `uv run --env-file .env`

# Frontend
cd frontend
pnpm install
```

## Development

```bash
# Terminal 1: Backend
cd backend && uv run fastapi dev main.py

# Terminal 2: Frontend
cd frontend && pnpm dev
```

The frontend dev server proxies `/api` requests to the backend at `localhost:8000`.

## Environment Variables

| Variable             | Description               |
| -------------------- | ------------------------- |
| `AI_GATEWAY_API_KEY` | Vercel AI Gateway API key |

## Storage

The demo backend is stateless. The frontend sends the conversation history
and approval responses on each request.
