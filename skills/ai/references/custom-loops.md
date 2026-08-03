# Custom loop rules

Read `https://ai-python.dev/docs/basics/custom-loops.md` first. Additional
invariants:

- `context.resolve(event.tool_call)` handles argument validation, approval
  gates, and cached replay results. Do not call `tool.fn` directly unless you
  also handle validation, approvals, and cached results yourself.
- `ToolRunner.schedule(...)` also accepts a zero-arg async callable that
  returns `ai.events.ToolCallResult`.
- If you make a result yourself, use `runner.add_result(ai.tool_result(...))`.
- Every tool call must get exactly one tool result.
- Order matters: `context.add(stream.message)`, then
  `context.add(runner.get_tool_message())`. `context.add(...)` skips replay
  messages, so always route history updates through it.
- Yield events from the loop; `Agent.run` hides replay events from callers.
- Keep `ToolRunner` events flowing (e.g. via `ai.util.merge`); otherwise
  partial tool output never reaches the caller.
- For hooks and approvals, let `context.resolve(...)` build the gated call.
