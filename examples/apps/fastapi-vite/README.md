# fastapi-chat

Chat demo using the Python Vercel AI SDK with a FastAPI backend and React frontend.
Includes **human-in-the-loop tool approval** — every tool call is gated
behind user confirmation before execution.

## Stack

- **Backend:** FastAPI + AI SDK for Python (Python 3.12)
- **Frontend:** Vite + React + AI SDK UI + AI Elements

## Human-in-the-Loop

The agent graph in `backend/agent.py` uses the function-based hook API
to suspend execution whenever the LLM wants to call a tool. The flow is:

1. LLM emits a tool call
2. Backend calls `await ai.hook(...)` with `payload=ai.ToolApproval`
3. The runtime emits a `HookEvent` containing the `HookPart`
4. The frontend renders Approve / Reject buttons via the
   `<Confirmation>` component (from AI Elements)
5. When the user clicks a button, `addToolApprovalResponse()` patches
   the message and sends a new request with the decision
6. The backend pre-registers the approval via `ai.resolve_hook(...)` on the
   next request, then either executes the tool or returns an error tool-result
   message

Tool results are appended as separate `role="tool"` messages. The
assistant tool-call message remains immutable.

## Setup

```bash
# Backend
cd backend
uv sync
cp .env.example .env  # add your AI_GATEWAY_API_KEY

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
