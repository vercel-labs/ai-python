# AI SDK UI: frontend wiring and hook deferral

Read `https://ai-python.dev/docs/basics/ai-sdk-ui.md` first for the backend
adapter. Below are additional frontend and defer-hook details.

## Frontend

```tsx
const chat = useChat({
  transport: new DefaultChatTransport({ api: "/api/chat" }),
  sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses,
});
```

Use `chat.sendMessage(...)` to send user input. Use
`chat.addToolApprovalResponse(...)` from approval buttons.

## Deferring hooks in the SSE stream

Serverless UI backends must defer pending hooks before the response ends.
Wrap the event stream before `to_sse`:

```python
async def body():
    async with agent.run(model, messages) as stream:
        ai.ui.ai_sdk.apply_approvals(approvals)

        async def events():
            async for event in stream:
                if (
                    isinstance(event, ai.events.HookEvent)
                    and event.hook.status == "pending"
                ):
                    ai.defer_hook(event.hook)
                yield event

        async for chunk in ai.ui.ai_sdk.to_sse(events()):
            yield chunk

return StreamingResponse(
    body(),
    headers=ai.ui.ai_sdk.UI_MESSAGE_STREAM_HEADERS,
)
```

`apply_approvals` pre-registers with the current hook registry, so call it
inside the `agent.run(...)` block (or pass `registry=`).

## Responsibility split

The adapter handles `UIMessage` parsing, message IDs, tool state, approvals,
subagent `MessageBundle` values, and AI SDK UI stream events. You handle the
HTTP route, auth, storage, session lookup, frontend rendering, and when to
defer hooks.

The outbound adapter emits step boundaries between model turns. Pass agent
events to `to_sse` or `to_stream`; do not synthesize step events separately.
