# slack_agent

Minimal Slack bot built with the AI SDK for Python and
[slack-bolt](https://tools.slack.dev/bolt-python/). One file, one tool.

Mention the bot in a channel or a thread and it replies in the thread,
streaming the answer through Slack's native message-streaming API
(`chat.startStream` / `chat.appendStream` / `chat.stopStream`).

It runs locally over [Socket Mode](https://docs.slack.dev/apis/events-api/using-socket-mode/):
the process opens an outbound WebSocket to Slack, so there is no public
URL, tunnel, or hosting — the bot is online while `main.py` runs. It is
also stateless: every mention rebuilds the conversation from the Slack
thread itself (`conversations.replies`), so restarts lose nothing.

## Setup

You need a workspace where you're allowed to install apps. Any
workspace you're in works. If you don't have one, get a free sandbox
workspace first; otherwise skip to "Create the app".

### Get a sandbox workspace (optional)

1. Go to
   [api.slack.com/developer-program](https://api.slack.com/developer-program)
   and click **Join the Developer Program** (free), signing in with
   your Slack account.
2. In the developer program dashboard, open **Developer sandboxes** and
   provision a sandbox. It creates a full workspace just for you.
3. Follow the **Open in Slack** link and sign into the sandbox
   workspace in your browser — you'll get a PIN by email. This sign-in
   matters: the app-creation dialog only offers workspaces your browser
   session is logged into.

### Create the app from the manifest

4. Go to [api.slack.com/apps/new](https://api.slack.com/apps/new) (or
   [api.slack.com/apps](https://api.slack.com/apps) → **Create New
   App**).
5. Choose **From a manifest** (not "From scratch").
6. In the workspace dropdown, pick your workspace → **Next**.
7. The dialog shows a JSON/YAML editor with a template — select the
   **JSON** tab, delete the template, paste the contents of
   `manifest.json` → **Next**.
8. The review screen lists the scopes (`app_mentions:read`,
   `chat:write`, …) and the `app_mention` event → **Create**.

### Get the two tokens

9. You land on **Basic Information**. Scroll to **App-Level Tokens** →
   **Generate Token and Scopes** → name it anything, **Add Scope** →
   `connections:write` → **Generate**. Copy the `xapp-…` token: this is
   `SLACK_APP_TOKEN`.
10. In the left sidebar, **Install App** → **Install to Workspace** →
    **Allow**. Copy the **Bot User OAuth Token** (`xoxb-…`): this is
    `SLACK_BOT_TOKEN`.

## Running

```bash
cd examples/apps/slack_agent
SLACK_BOT_TOKEN=xoxb-… SLACK_APP_TOKEN=xapp-… AI_GATEWAY_API_KEY=… uv run main.py
```

Or put the variables in a `.env` file here (it's gitignored) and let uv
load it — no python-dotenv needed:

```bash
echo 'SLACK_BOT_TOKEN=xoxb-…
SLACK_APP_TOKEN=xapp-…
AI_GATEWAY_API_KEY=…' > .env

uv run --env-file .env main.py
```

`MODEL_ID` overrides the default model (`anthropic/claude-sonnet-4.6`).

Then mention the bot in any channel: `@ai-agent what's the weather in
Tokyo?`. On the first mention Slack offers to invite the bot to the
channel; accept and mention it again. Follow-ups can mention it inside
the reply thread — the whole thread is its conversation history.

## Toward production

The agent core — rebuild messages, `agent.run()`, forward events —
stays the same; what changes is the shell around it:

- **Internal bot:** Socket Mode is fine in production. Run this same
  process on any host that keeps it alive and restarts it on crash.
  Mentions received while the process is down are dropped, not queued.
- **Serverless / multi-workspace:** switch to the HTTP Events API
  (`socket_mode_enabled: false` plus a request URL in the manifest).
  Slack requires a `200` within 3 seconds and retries otherwise, so the
  endpoint must ack immediately and hand the turn to a durable
  runner — see `../durable_agent_workflows` for the pattern: a FastAPI
  route that starts a Vercel Workflows run in which each LLM call and
  tool is a durable step. The reply then goes back through Slack's Web
  API from inside the workflow, not through the HTTP response. Dedupe
  events by `event_id`, and store conversation state keyed by
  `channel:thread_ts` instead of re-reading the thread — distributed
  apps created after May 2025 get `conversations.replies` limited to
  1 request/minute.
- **Token streaming** fights step-based durability (a step is atomic,
  so a retry restarts the streamed message). Production Slack agents
  typically stream within a single respond step, or post throttled
  status updates and one final message per turn.
