# durable_agent_temporal

Durable agent demo using AI SDK for Python, Temporal, FastAPI,
React, AI SDK UI, and AI Elements.

The UI sends full message history to `/api/chat`. The backend starts one
durable workflow turn, waits for the completed turn, and returns one AI SDK UI
message stream. The agent has a single `bash` tool. Every LLM call and tool
execution runs as a Temporal activity; Temporal's event history makes the
turn durable across worker restarts.

## Prerequisites

- [Temporal CLI](https://docs.temporal.io/cli) (`brew install temporal`)
- `AI_GATEWAY_API_KEY` environment variable set

## Development

The app is three pieces: a local Temporal server, the Temporal worker that
runs the agent, and the frontend + API served together by `vercel dev`.
Run each in its own terminal, all from `examples/apps/durable_agent_temporal`:

```bash
# 1. Temporal dev server (workflow state, UI at http://localhost:8233)
temporal server start-dev

# 2. Temporal worker — hosts the workflow and its activities,
#    including the LLM calls, so it needs AI_GATEWAY_API_KEY
cd backend && uv run worker

# 3. frontend + API on http://localhost:3000, /api routed to FastAPI
vercel dev
```

`TEMPORAL_ADDRESS` overrides the Temporal server address for both the worker
and the API server (default `localhost:7233`).

## Telemetry

The worker exports spans over OTLP when `OTEL_EXPORTER_OTLP_ENDPOINT` is
set. For a local terminal viewer, run in another terminal:

```bash
cd backend && uv run python -m ai.telemetry.utils.viewer
```

and start the worker with the endpoint set:

```bash
cd backend && OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 uv run worker
```

Each turn prints one trace tree — the agent run, the per-turn LLM calls
(parented across the activity boundary), and tool executions. Any other
OTLP backend (Jaeger, an LLM-aware viewer) works the same way; span
attributes follow the `gen_ai` semantic conventions.
