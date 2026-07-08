import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, getToolName, isToolUIPart } from "ai";
import type { UIDataTypes, UIMessage, UIMessagePart, UITools } from "ai";
import { Fragment, useCallback, useMemo } from "react";
import type { ReactNode } from "react";

import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
} from "@/components/ai-elements/prompt-input";
import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";
import { TooltipProvider } from "@/components/ui/tooltip";

// ---------------------------------------------------------------------------
// renderPart -- one message part.
// ---------------------------------------------------------------------------

function renderPart({
  part,
  key,
  role,
}: {
  part: UIMessagePart<UIDataTypes, UITools>;
  key: string;
  role: UIMessage["role"];
}): ReactNode {
  if (isToolUIPart(part)) {
    const toolName = getToolName(part);
    const isComplete = part.state === "output-available";

    return (
      <Tool
        key={key}
        data-testid="tool-card"
        data-tool-name={toolName}
        data-tool-state={part.state}
        defaultOpen={isComplete}
      >
        {part.type === "dynamic-tool" ? (
          <ToolHeader type={part.type} state={part.state} toolName={toolName} />
        ) : (
          <ToolHeader type={part.type} state={part.state} />
        )}
        <ToolContent>
          <ToolInput input={part.input} />
          <ToolOutput output={part.output} errorText={part.errorText} />
        </ToolContent>
      </Tool>
    );
  }

  if (part.type === "text") {
    return (
      <Message key={key} from={role}>
        <MessageContent>
          <MessageResponse>{part.text}</MessageResponse>
        </MessageContent>
      </Message>
    );
  }

  return null;
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/chat",
      }),
    [],
  );

  const { messages, sendMessage, status, stop } = useChat({ transport });

  const isLoading = status === "submitted" || status === "streaming";

  const handleSubmit = useCallback(
    ({ text }: { text: string }) => {
      const prompt = text.trim();
      if (prompt) {
        sendMessage({ text: prompt });
      }
    },
    [sendMessage],
  );

  return (
    <TooltipProvider>
      <div className="flex h-screen flex-col bg-background">
        <header className="border-b px-4 py-3">
          <div className="mx-auto w-full max-w-3xl">
            <h1 className="text-lg font-semibold">durable temporal agent</h1>
          </div>
        </header>

        <Conversation className="flex-1">
          <ConversationContent data-testid="chat-log">
            <div className="mx-auto w-full max-w-3xl space-y-4 px-4 py-4">
              {messages.length === 0 ? (
                <ConversationEmptyState
                  title="Start a conversation"
                  description="Ask the durable agent to inspect the project"
                />
              ) : (
                messages.map((message) => (
                  <Fragment key={message.id}>
                    {message.parts.map((part, partIndex) =>
                      renderPart({
                        part,
                        key: `${message.id}-${partIndex}`,
                        role: message.role,
                      }),
                    )}
                  </Fragment>
                ))
              )}
            </div>
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>

        <div className="border-t px-4 py-3">
          <div className="mx-auto w-full max-w-3xl">
            <PromptInput onSubmit={handleSubmit}>
              <PromptInputTextarea
                placeholder="Ask the agent to inspect or change code..."
                disabled={isLoading}
              />
              <PromptInputFooter>
                <div />
                <PromptInputSubmit status={status} onStop={stop} />
              </PromptInputFooter>
            </PromptInput>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
