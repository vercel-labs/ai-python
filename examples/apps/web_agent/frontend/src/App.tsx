import { useChat } from "@ai-sdk/react";
import {
  DefaultChatTransport,
  lastAssistantMessageIsCompleteWithApprovalResponses,
} from "ai";
import type { ToolUIPart, UIMessage } from "ai";
import { CheckIcon, XIcon } from "lucide-react";
import { Fragment } from "react";

import {
  Confirmation,
  ConfirmationAccepted,
  ConfirmationAction,
  ConfirmationActions,
  ConfirmationRejected,
  ConfirmationRequest,
  ConfirmationTitle,
} from "@/components/ai-elements/confirmation";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputTextarea,
  PromptInputFooter,
  PromptInputSubmit,
} from "@/components/ai-elements/prompt-input";
import {
  renderUIPart,
  Tool,
  ToolHeader,
  ToolContent,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";
import { TooltipProvider } from "@/components/ui/tooltip";

export default function App() {
  const { messages, sendMessage, addToolApprovalResponse, status, stop } =
    useChat({
      transport: new DefaultChatTransport({
        api: "/api/chat",
      }),
      // After the user approves/rejects a tool, automatically send the
      // updated messages back to the backend so it can resume execution.
      sendAutomaticallyWhen:
        lastAssistantMessageIsCompleteWithApprovalResponses,
    });

  const isLoading = status === "submitted" || status === "streaming";

  return (
    <TooltipProvider>
      <div className="flex h-screen flex-col bg-background">
        <header className="border-b px-4 py-3">
          <div className="mx-auto w-full max-w-3xl">
            <h1 className="text-lg font-semibold">AI SDK Python Chat Demo</h1>
            <p className="text-sm text-muted-foreground">
              Powered by AI SDK for Python and FastAPI
            </p>
          </div>
        </header>

        <Conversation className="flex-1">
          <ConversationContent>
            <div className="mx-auto w-full max-w-3xl space-y-4 px-4 py-4">
              {messages.length === 0 ? (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  <p>Send a message to start chatting</p>
                </div>
              ) : (
                messages.map((message) => (
                  <Fragment key={message.id}>
                    {message.parts.map((part, partIndex) => {
                      // Handle tool parts (type starts with "tool-")
                      if (part.type.startsWith("tool-")) {
                        const toolPart = part as ToolUIPart;
                        const isComplete =
                          toolPart.state === "output-available";

                        return (
                          <Tool
                            key={`${message.id}-${partIndex}`}
                            defaultOpen={isComplete}
                          >
                            <ToolHeader
                              type={toolPart.type}
                              state={toolPart.state}
                            />
                            <ToolContent>
                              <ToolInput input={toolPart.input} />

                              {/* Human-in-the-loop approval UI */}
                              <Confirmation
                                approval={toolPart.approval}
                                state={toolPart.state}
                              >
                                <ConfirmationTitle>
                                  <ConfirmationRequest>
                                    Allow this tool to run?
                                  </ConfirmationRequest>
                                  <ConfirmationAccepted>
                                    <CheckIcon className="size-4 text-green-500" />
                                    <span>Approved</span>
                                  </ConfirmationAccepted>
                                  <ConfirmationRejected>
                                    <XIcon className="size-4 text-destructive" />
                                    <span>Rejected</span>
                                  </ConfirmationRejected>
                                </ConfirmationTitle>
                                <ConfirmationActions>
                                  <ConfirmationAction
                                    variant="outline"
                                    onClick={() =>
                                      addToolApprovalResponse({
                                        id: toolPart.approval!.id,
                                        approved: false,
                                      })
                                    }
                                  >
                                    Reject
                                  </ConfirmationAction>
                                  <ConfirmationAction
                                    variant="default"
                                    onClick={() =>
                                      addToolApprovalResponse({
                                        id: toolPart.approval!.id,
                                        approved: true,
                                      })
                                    }
                                  >
                                    Approve
                                  </ConfirmationAction>
                                </ConfirmationActions>
                              </Confirmation>

                              {toolPart.type === "tool-talk_to_mothership" &&
                              toolPart.state === "output-available" ? (
                                <div className="space-y-2">
                                  {(toolPart.output as UIMessage).parts.map(
                                    (p, i) => renderUIPart(p, i),
                                  )}
                                </div>
                              ) : (
                                <ToolOutput
                                  output={toolPart.output}
                                  errorText={toolPart.errorText}
                                />
                              )}
                            </ToolContent>
                          </Tool>
                        );
                      }

                      // Handle text parts
                      if (part.type === "text") {
                        return (
                          <Message
                            key={`${message.id}-${partIndex}`}
                            from={message.role}
                          >
                            <MessageContent>
                              <MessageResponse>{part.text}</MessageResponse>
                            </MessageContent>
                          </Message>
                        );
                      }

                      return null;
                    })}
                  </Fragment>
                ))
              )}
            </div>
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>

        <div className="border-t p-4">
          <div className="mx-auto w-full max-w-3xl">
            <PromptInput
              onSubmit={({ text }) => {
                if (text.trim()) {
                  sendMessage({ text });
                }
              }}
            >
              <PromptInputTextarea
                placeholder="Ask me anything..."
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
