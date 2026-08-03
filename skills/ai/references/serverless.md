# Serverless resume: extra rules

Read `https://ai-python.dev/docs/basics/human-in-the-loop.md` first
("Resume in serverless flows"). Additionally:

- Drain the stream after deferring a hook. This lets sibling tools finish or
  defer and makes `stream.messages` complete.
- Persist typed messages and pending hook IDs. Round-trip messages with:

```python
data = [message.model_dump(mode="json") for message in stream.messages]
messages = [ai.messages.Message.model_validate(item) for item in data]
```

- Pre-register resolutions inside `agent.run(...)` before iteration, or pass
  its registry explicitly.
- Do not ask the model to make the tool call again; replay reuses completed
  sibling results and feeds deferred hooks the pre-registered resolution.
- Use normal `agent.run(...)`; serverless resume usually does not need a
  custom loop. If you do write one, use `context.resolve(...)`, `ToolRunner`,
  and `context.add(...)` so approvals and replay keep working.
- For custom hooks, pre-register with
  `ai.resolve_hook(hook_id, data, payload=PayloadType)` to validate the data.
- For AI SDK UI clients, see [ui.md](ui.md) for message conversion, approval
  responses, and SSE.
