// End-to-end smoke test for the chat + tool-approval flow.
//
// Drives the Vite frontend in a headless browser, sends a message that
// triggers the `talk_to_mothership` tool, approves the request, and
// asserts the tool result is rendered.
//
// Prereqs:
//   - backend running on :8000 (cd backend && uv run --frozen --with-editable ~/src/py-ai/ fastapi dev main.py)
//   - frontend running on :5173 (cd frontend && pnpm dev)
//   - install deps:  npm install && npx playwright install chromium
//
// Run from this directory:  node e2e-test.mjs

import { chromium } from "playwright";

const FRONTEND = process.env.FRONTEND_URL ?? "http://localhost:5173/";
const PROMPT = "Ask the mothership when robots will take over";

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

// Wait for the assistant's final reply to render. The Mothership tool
// returns "Soon." so the model usually echoes that back.
try {
  await page.getByText(/Soon\./i).waitFor({ state: "visible", timeout: 30000 });
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

const transcript = await page.locator("body").innerText();
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

console.log("PASS");
await browser.close();
