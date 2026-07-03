# coding_agent

Minimal coding-agent TUI built with the AI SDK for Python and
[Textual](https://textual.textualize.io/). One file, one tool.

- a single `bash` tool that streams command output into the transcript
- streamed markdown replies (and reasoning, when the model emits it)
- every bash call is gated behind a `ToolApproval` hook: the run
  suspends, a y/n prompt appears above the composer, and the decision
  is resolved in-process with `ai.resolve_hook()`
- messages typed while the agent is busy are queued and run in order;
  Esc interrupts the current turn

> The agent runs arbitrary shell commands in your working directory
> once you approve them. Run it somewhere you don't mind it touching.

## Running

```bash
cd examples/apps/coding_agent
uv run main.py
```

Set `AI_GATEWAY_API_KEY` first. `MODEL_ID` overrides the default model
(`anthropic/claude-sonnet-4.6`).
