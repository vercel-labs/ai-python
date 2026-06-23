---
name: ai_sdk_ui_adapter
description: Use when connecting Python ai SDK agents to AI SDK UI useChat clients, UIMessage history, streamed responses, and approvals.
---

# ai_sdk_ui_adapter

Frontend:

```tsx
const chat = useChat({
  transport: new DefaultChatTransport({ api: "/api/chat" }),
  sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses,
});
```

Use `chat.sendMessage(...)` to send user input. Use
`chat.addToolApprovalResponse(...)` from approval buttons.

Backend request:

```python
class ChatRequest(pydantic.BaseModel):
    messages: list[ai.agents.ui.ai_sdk.UIMessage]

messages, approvals = ai.agents.ui.ai_sdk.to_messages(request.messages)
ai.agents.ui.ai_sdk.apply_approvals(approvals)
```

Backend stream:

```python
async def body():
    async with agent.run(model, messages) as stream:
        async def events():
            async for event in stream:
                if (
                    isinstance(event, ai.events.HookEvent)
                    and event.hook.status == "pending"
                ):
                    ai.abort_pending_hook(event.hook)
                yield event

        async for chunk in ai.agents.ui.ai_sdk.to_sse(events()):
            yield chunk

return StreamingResponse(
    body(),
    headers=ai.agents.ui.ai_sdk.UI_MESSAGE_STREAM_HEADERS,
)
```

The adapter handles `UIMessage` parsing, message IDs, tool state, approvals,
subagent `MessageBundle` values, and AI SDK UI stream events.

You handle the HTTP route, auth, storage, session lookup, frontend rendering,
and when to abort pending hooks.

For saved UI history, use:

```python
ui_messages = ai.agents.ui.ai_sdk.to_ui_messages(messages)
```
