# Durable execution: extra rules

Read `https://ai-python.dev/docs/basics/durable-execution.md` first.
Additional invariants:

- Use the durable model step's result as a complete `Message` for control flow.
  Do not call the provider through `ai.stream` in the workflow body.
- Use `ai.Stream.replay_message(message)` only when the run's consumer needs
  synthetic model events. It does not call the provider again.
- If the workflow system needs separate activity dispatch for tools, schedule
  a zero-arg callable that returns `ai.tool_result(...)`. Do not call
  `tool.fn` directly.
- Wrap the workflow entry point with
  `ai.messages.use_random(workflow.random)` so message and part IDs remain
  deterministic during replay.
- Allow retries for idempotent model steps. Disable retries for tool steps with
  non-idempotent side effects, or make those tools idempotent.
- A queue-based side channel can stream tokens to the caller, but that stream
  cannot dispatch tools or affect control flow.
- For telemetry inside workflow bodies (sinks, deterministic time, cross-step
  spans), see `https://ai-python.dev/docs/reference/telemetry.md`.
