# tau-agent

`tau` is a coding-agent demo built on the `ai` library.  This is the
chat-bot baseline — single process, Textual TUI, streaming replies, no
tools yet.  Future iterations will grow real coding capabilities on
top.

## Setup

```bash
uv sync
```

## Running

```bash
uv run python tau.py
```

Type a message, hit enter.  `ctrl+c` to quit.

## Environment

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_GATEWAY_API_KEY` | Vercel AI Gateway API key | — |
| `TAU_MODEL` | Model id passed to `ai.ai_gateway(...)` | `anthropic/claude-sonnet-4.5` |
