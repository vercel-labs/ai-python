# durable_agent_workflows

Durable agent demo using AI SDK for Python, Vercel Workflows, FastAPI,
React, AI SDK UI, and AI Elements.

The UI sends full message history to `/api/chat`. The backend starts one
durable workflow turn, waits for the completed turn, and returns one AI SDK UI
message stream. The agent has a single `bash` tool.

## Development

```bash
cd examples/apps/durable_agent_workflows
vercel dev
```

Set `AI_GATEWAY_API_KEY` before running.

## Telemetry

The worker service exports spans over OTLP when
`OTEL_EXPORTER_OTLP_ENDPOINT` is set. For a local terminal viewer, run in
another terminal:

```bash
cd backend && uv run python -m ai.telemetry.utils.viewer
```

and start the app with the endpoint set:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 vercel dev
```

Each turn prints one trace tree — the agent run, the per-turn LLM calls
(parented across the step boundary), and tool executions. Spans that
closed before a workflow suspension are re-emitted on replay with
identical ids and timestamps; backends that upsert by span id show them
once, the terminal viewer shows duplicate rows. Any other OTLP backend
(Jaeger, an LLM-aware viewer) works the same way; span attributes follow
the `gen_ai` semantic conventions.
