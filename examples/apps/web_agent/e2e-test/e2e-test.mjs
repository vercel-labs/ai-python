// End-to-end smoke test for the chat + tool-approval flow.
//
// Drives the Vite frontend in a headless browser, sends a compound
// question that triggers both the (non-gated) `get_weather` tool and
// the (gated) `talk_to_mothership` tool, approves the mothership
// request, and asserts both tool results render alongside a final
// assistant reply.
//
// Use ./run.sh to launch backend + frontend and run this script.
//
// Run from this directory:  node e2e-test.mjs

import { chromium } from "playwright";

const FRONTEND = process.env.FRONTEND_URL ?? "http://localhost:5173/";
const PROMPT = [
  "Two things:",
  "1. What is the current weather in Tokyo?",
  "2. Ask the mothership when the robot uprising begins",
].join("\n");

const browser = await chromium.launch();
const ctx = await browser.newContext();
const page = await ctx.newPage();

const requests = [];
const responses = [];
page.on("request", (req) => {
  if (req.url().includes("/chat")) {
    requests.push({ method: req.method(), url: req.url(), body: req.postData() });
  }
});
page.on("response", async (resp) => {
  if (resp.url().includes("/chat")) {
    try {
      responses.push({ status: resp.status(), body: await resp.text() });
    } catch {}
  }
});
page.on("pageerror", (err) => console.error("[pageerror]", err.message));

await page.goto(FRONTEND);
await page.waitForLoadState("networkidle");

await page.getByPlaceholder("Ask me anything...").fill(PROMPT);
await page.keyboard.press("Enter");

// Tool collapsible appears once the approval-request event arrives.
const toolToggle = page.getByRole("button", { name: /talk_to_mothership/ });
await toolToggle.waitFor({ state: "visible", timeout: 30000 });
await toolToggle.click();

const approveBtn = page.getByRole("button", { name: "Approve" });
await approveBtn.waitFor({ state: "visible", timeout: 10000 });
await approveBtn.click();

// Wait for the tool to reach the "Completed" state — i.e. the sub-agent
// finished streaming and the tool transitioned from approval-responded
// to output-available.  Don't match on specific reply text: the
// mothership model is non-deterministic.
try {
  await toolToggle
    .getByText("Completed", { exact: true })
    .waitFor({ state: "visible", timeout: 30000 });
} catch (e) {
  console.error("\n=== TIMEOUT — dumping diagnostics ===");
  console.error("REQUEST COUNT:", requests.length);
  if (requests.length >= 2) {
    console.error("\n--- Request #2 (after approval) ---");
    console.error(requests[1].body);
  }
  if (responses.length >= 2) {
    console.error("\n--- Response #2 ---");
    console.error(responses[1].body);
  }
  throw e;
}

// After the tool completes, the agent loops back and streams a final
// reply.  Wait for the textarea to re-enable — useChat clears its
// "streaming" status only when the whole turn is done — so the
// transcript captures the post-tool agent text too.
const textarea = page.getByPlaceholder("Ask me anything...");
try {
  await textarea.waitFor({ state: "visible", timeout: 30000 });
  await page
    .locator('textarea[placeholder="Ask me anything..."]:not([disabled])')
    .waitFor({ timeout: 30000 });
} catch (e) {
  console.error("\n=== TIMEOUT waiting for post-tool stream to settle ===");
  console.error(await page.locator("body").innerText());
  throw e;
}

// Walk the conversation container's direct children and label each
// item as user / assistant / tool.  All three (Message + Tool) are
// rendered as siblings under one div in App.tsx, so we anchor on the
// first message bubble's parent.
const transcript = await page.evaluate(() => {
  const sample = document.querySelector(".is-user, .is-assistant");
  const container = sample?.parentElement;
  if (!container) return "(no conversation container)";
  const blocks = [];
  for (const child of container.children) {
    const text = child.innerText.trim();
    let label;
    if (child.classList.contains("is-user")) {
      label = "USER";
    } else if (child.classList.contains("is-assistant")) {
      label = "ASSISTANT";
    } else if (child.querySelector(".lucide-wrench")) {
      label = "TOOL";
    } else {
      label = "OTHER";
    }
    blocks.push(`[${label}]\n${text}`);
  }
  return blocks.join("\n\n");
});
console.log("=== TRANSCRIPT ===");
console.log(transcript);

// Sanity: the tool should have transitioned out of "Awaiting Approval".
if (transcript.includes("Awaiting Approval")) {
  console.error("FAIL: tool still awaiting approval — replay flow is broken");
  console.error("\n=== REQUESTS ===");
  for (const r of requests) console.error(r.method, r.url, r.body?.slice(0, 1000));
  console.error("\n=== RESPONSES ===");
  for (const r of responses) console.error(r.status, r.body?.slice(0, 2000));
  process.exit(1);
}

// Both tools must have run — the non-gated get_weather alongside
// the gated talk_to_mothership we just approved.
if (!transcript.includes("get_weather")) {
  console.error("FAIL: get_weather tool not rendered in transcript");
  process.exit(1);
}
if (!transcript.includes("talk_to_mothership")) {
  console.error("FAIL: talk_to_mothership tool not rendered in transcript");
  process.exit(1);
}

// The agent must produce a final assistant text bubble after the tool.
// `.is-assistant` is the class set by the Message component for
// `from="assistant"` text parts (tool parts don't carry it), so the last
// one is the post-tool reply.
const assistantBubbles = page.locator(".is-assistant");
const bubbleCount = await assistantBubbles.count();
if (bubbleCount === 0) {
  console.error("FAIL: no assistant text bubble rendered");
  process.exit(1);
}
const finalReply = (await assistantBubbles.last().innerText()).trim();
if (finalReply.length < 20) {
  console.error(
    `FAIL: final assistant reply too short (${finalReply.length} chars): ` +
      JSON.stringify(finalReply)
  );
  process.exit(1);
}
// The compound answer should reference both halves of the question.
if (!finalReply.toLowerCase().includes("tokyo")) {
  console.error(
    `FAIL: final reply doesn't mention Tokyo: ${JSON.stringify(finalReply)}`
  );
  process.exit(1);
}

console.log("PASS");
await browser.close();
