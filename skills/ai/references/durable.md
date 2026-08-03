# Durable execution: extra rules

Read `https://ai-python.dev/docs/basics/durable-execution.md` first.
Additional invariants:

- Use the durable model step's result as a complete `Message`. Do not wrap it
  in `ai.Stream`, `ai.events.replay_message_events`, or `ai.util.merge` —
  those exist for fluent dispatch in non-durable code, and streams are
  side effects in a workflow setting.
- If the workflow system needs separate activity dispatch for tools, schedule
  a zero-arg callable that returns `ai.tool_result(...)`. Do not call
  `tool.fn` directly.
- Guard the model step: if `stream.message is None` after draining, raise
  instead of returning a partial result.
- A queue-based side channel can stream tokens to the caller, but that stream
  cannot dispatch tools or affect control flow.
- For telemetry inside workflow bodies (sinks, deterministic time, cross-step
  spans), see `https://ai-python.dev/docs/reference/telemetry.md`.
