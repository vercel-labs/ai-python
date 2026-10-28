// Loaded into every `pi` the ai-python SDK launches (`pi -e <this file>`).
// It does nothing on its own: AI_PYTHON_PI, a JSON object in the
// environment, says which of its three jobs this process needs.
//
// - "approve": ask the SDK about every tool call. pi has no permission
//   channel of its own; an extension's `tool_call` handler is the gate, and
//   an RPC dialog (`ctx.ui.input`) is how it reaches the SDK. The answer is
//   a JSON decision: allow (optionally with rewritten input) or deny.
// - "mcp": MCP servers, registered for this session only.
// - "gateway": the Vercel AI Gateway, reached with `Authorization: Bearer`
//   rather than the `x-api-key` header pi's provider sends, so a sandbox
//   firewall that rewrites `Authorization` can inject the credential.
//   `baseUrl` moves the provider when the gateway is not the default one.
const APPROVE = "ai-python:approve";
// Registered LAST: its presence in `get_commands` is how the SDK knows every
// part above loaded — above all the approval gate. No marker, no session.
const READY = "ai-python-ready";

export default function (pi) {
  // Unparseable configuration fails the load (and so the session) rather
  // than reading as "no approval asked for".
  const config = JSON.parse(process.env.AI_PYTHON_PI || "{}") || {};

  if (config.approve) {
    pi.on("tool_call", async (event, ctx) => {
      const answer = await ctx.ui.input(
        APPROVE,
        JSON.stringify({
          toolCallId: event.toolCallId,
          parentToolCallId: event.parentToolCallId,
          toolName: event.toolName,
          input: event.input,
        }),
      );
      let decision = null;
      try {
        decision = JSON.parse(answer ?? "");
      } catch {
        decision = null;
      }
      if (decision && decision.decision === "allow") {
        if (decision.input && typeof decision.input === "object") {
          // pi reads the arguments from this object after the handler:
          // rewrite it in place.
          for (const key of Object.keys(event.input)) delete event.input[key];
          Object.assign(event.input, decision.input);
        }
        return undefined;
      }
      return {
        block: true,
        // No answer: the SDK withdrew the question (the turn was stopped).
        reason:
          answer === undefined
            ? "the approval was withdrawn"
            : (decision && decision.reason) || "denied by policy",
        ...(decision && decision.stop ? { terminate: true } : {}),
      };
    });
  }

  for (const [name, server] of Object.entries(config.mcp || {})) {
    pi.registerMcpServer(name, server);
  }

  const gateway = config.gateway;
  if (gateway) {
    if (gateway.baseUrl) {
      pi.registerProvider("vercel-ai-gateway", { baseUrl: gateway.baseUrl });
    }
    pi.on("before_provider_headers", (event, ctx) => {
      if (ctx.model?.provider !== "vercel-ai-gateway") return;
      for (const key of Object.keys(event.headers)) {
        const lower = key.toLowerCase();
        if (lower === "x-api-key" || lower === "authorization") {
          event.headers[key] = null;
        }
      }
      event.headers["authorization"] =
        "Bearer " + (process.env.AI_GATEWAY_API_KEY || "");
    });
  }

  pi.registerCommand(READY, {
    description: "Marks the ai-python SDK's extension as loaded",
    handler: async () => {},
  });
}
