# tau-agent

`tau` is a coding-agent demo built on the `ai` library.  Single
process, Textual TUI, streaming replies, pi-style tool surface:

- **`read`** — read files; offset/limit pagination with continuation hints
- **`write`** — create / overwrite a file
- **`edit`** — exact-match str_replace, multiple disjoint edits per call
- **`bash`** — run a shell command in cwd, output truncated to the last 50KB / 2000 lines *(requires approval)*
- **`grep`** — regex search (skips `.git`, `node_modules`, etc.)
- **`find`** — glob match
- **`ls`** — directory listing

Approval-gated tools fire a `ToolApproval` hook; the composer turns
into a `[y/n]` prompt mid-turn.  Unrelated text typed during a
pending approval falls through to the message queue — the hook stays
pending until you give it a y or n.

No workspace jail.  The approval gate is the safety mechanism;
everything else relies on you watching the prompts.

## Setup

```bash
uv sync
```

## Running

```bash
uv run tau
```

Type a message, hit enter.  `ctrl+c` to quit.

## Environment

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_GATEWAY_API_KEY` | Vercel AI Gateway API key | — |
| `TAU_MODEL` | Model id passed to `ai.ai_gateway(...)` | `anthropic/claude-sonnet-4.5` |
