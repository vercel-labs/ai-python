// Loaded into every `pi` the ai-python SDK launches (`pi -e <this file>`).
//
// AI_PYTHON_PI, a JSON object in the environment, configures it:
//
// - "mcp": MCP servers, registered for this session only. The worker's MCP
//   bridge (the `ai` tools) comes this way: pi has no flag for an MCP server,
//   and a registration is not written to any config file.
//
// pi has no permission prompt, and the SDK asks for no approvals,
// so there is no gate here.

// Registered LAST: its presence in `get_commands` is how the SDK knows every
// part above loaded. No marker, no session.
const READY = "ai-python-ready";

export default function (pi) {
  // Unparseable configuration fails the load (and so the session) rather
  // than starting without the bridge.
  const config = JSON.parse(process.env.AI_PYTHON_PI || "{}") || {};

  for (const [name, server] of Object.entries(config.mcp || {})) {
    pi.registerMcpServer(name, server);
  }

  pi.registerCommand(READY, {
    description: "Marks the ai-python SDK's extension as loaded",
    handler: async () => {},
  });
}
