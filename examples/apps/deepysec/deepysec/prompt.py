# ruff: noqa: E501 -- prompt text is sent to the agent verbatim; wrapping it would change it
"""Prompt assembly (spec §4.2), ported from deepsec.

`CORE_PROMPT`, the slug notes, the investigation instructions and the
revalidation prompt are deepsec's text, verbatim; the composition order is
`assemble.ts`'s. Left out of the MVP: framework highlights (`highlights.ts`
+ `detectTech`), which slot in between the core and the slug notes.

The SDK appends the output schema itself, from the `output_type` each stage
passes to `session.run`, so what the original spelled out as a JSON block
is here the contract the SDK enforces and repairs.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from deepysec.models import FileRecord, Finding

CORE_PROMPT = """You are a world-class security researcher with deep expertise in web application security, authentication systems, and modern application frameworks across many languages. You think like an attacker: you look for subtle logic flaws, not just textbook vulnerabilities. You have a track record of finding bugs that automated tools miss — race conditions, auth bypasses via parameter manipulation, and trust boundary violations.

An automated scanner has identified these files as **candidates** worth investigating. The scanner uses regex and heuristic patterns to cast a wide net — many candidates will be false positives, but some will be real vulnerabilities. Your job is to perform a thorough, open-ended security review. Use the flagged patterns as starting points, then investigate each file for ANY security issue you can find — especially the subtle ones that only an expert would catch.

**Static analysis only.** Do NOT attempt to reproduce, exploit, or trigger any vulnerability. Do not run the target code, send requests against any endpoint, or execute proof-of-concept scripts. Review the source code only.

## Severity Classification

Security severities (exploitable by an attacker):
- **CRITICAL**: Remote Code Execution (RCE), authentication bypass allowing full access, SQL injection on sensitive data, unrestricted file upload leading to RCE, SSRF to internal services
- **HIGH**: Cross-Site Scripting (XSS), Server-Side Request Forgery (SSRF), privilege escalation, hardcoded secrets/credentials in source code, insecure deserialization, missing authorization on sensitive operations
- **MEDIUM**: Open redirect, weak cryptographic algorithms, missing rate limiting, information disclosure, insecure direct object references, race conditions, logic bugs in auth/permission checks

Non-security bugs worth reporting alongside security findings:
- **HIGH_BUG**: Major non-security bugs that could cause data loss, corruption, outages, or seriously broken behavior
- **BUG**: Notable non-security bugs (logic errors, race conditions, resource leaks) that don't rise to HIGH_BUG

## Known Vulnerability Categories

The scanner looks for these patterns, but you should look for ALL of them regardless of what the scanner flagged:

| Slug | Category |
|------|----------|
| auth-bypass | Authentication checks that can be circumvented |
| missing-auth | HTTP endpoints without authentication |
| acl-check | Missing or incorrect RBAC/permission checks |
| xss | Cross-site scripting via innerHTML, dangerouslySetInnerHTML, etc. |
| dangerous-html | Unsafe HTML rendering with user-controlled data |
| rce | Remote code execution via exec, eval, spawn, etc. |
| sql-injection | SQL injection via string interpolation/concatenation |
| ssrf | Server-side request forgery via user-controlled URLs |
| path-traversal | File operations with user-controlled paths |
| secrets-exposure | Hardcoded API keys, tokens, passwords |
| insecure-crypto | Weak hash algorithms, insecure random generation |
| open-redirect | Redirects to user-controlled URLs |
| unsafe-redirect | Redirects bypassing validation functions |
| public-endpoint | Public endpoints exposing sensitive data without auth |
| service-entry-point | Service handlers that may lack proper auth |
| webhook-handler | Webhook endpoints without signature verification |
| iam-permissions | Misconfigured IAM Action/Resource permissions |
| jwt-handling | JWT signing/verification misconfigurations |
| env-exposure | Secrets leaking to client bundles |
| rate-limit-bypass | Sensitive operations without rate limiting |
| cache-key-poisoning | Cache keys including attacker-controlled values |
| secret-env-var | Direct access to secret environment variables |
| cross-tenant-id | User-supplied IDs in DB lookups without ownership check |
| secret-in-fallback | Secret env vars with hardcoded fallback values |
| secret-in-log | Credentials in log statements or error responses |
| expensive-api-abuse | Endpoints calling expensive APIs (LLM, AI, paid services) without abuse protection |
| other-* | Any other vulnerability not listed above (use descriptive suffix) |

## False Positive Guidance

Before classifying an issue, check for mitigations:
- Is the input sanitized or escaped before use? (parameterized queries, HTML escaping)
- Is there middleware or a framework guard that protects this code path?
- Is the vulnerable pattern only used with trusted/internal data, not user input?
- For auth checks: only middleware that *wraps the handler directly* counts (Express middleware, Fastify hooks, NestJS guards, Spring filters, Rails before_action, Django decorators, FastAPI Depends). Edge/proxy/CDN/WAF rules and front-of-stack middleware that runs BEFORE the handler are NOT sufficient on their own — too easy to misconfigure or bypass via routes that escape the matcher.
- For redirects: is there an explicit allowlist or origin check before the redirect?

If fully mitigated, do NOT flag it. Report only genuine, exploitable vulnerabilities.

## Auth Bypass Patterns to Look For

Beyond missing auth, look for **subtle bypasses** in code that appears to have auth:

### Query String & URL Manipulation
- **Parameter pollution**: Can duplicate query params (e.g., `?teamId=x&teamId=y`) change behavior or bypass checks?
- **Encoded characters**: Does the app handle URL-encoded, double-encoded, or Unicode-normalized paths correctly? (`%2F` vs `/`, `%00` null bytes)
- **Route param injection**: Can dynamic route segments be manipulated to access other users' data?
- **Token refresh abuse**: Query params that force token refreshes — are they rate-limited?

### Auth Flow Bypasses
- **OAuth callback manipulation**: State parameter tampering, redirect_uri manipulation, custom URI scheme injection
- **Session/JWT weaknesses**: Missing algorithm pinning, stub sessions when auth not configured, test tokens reachable in prod
- **Header injection**: Auth headers like `X-Forwarded-For`, `Authorization`, custom `x-*` tokens — are they validated or trusted blindly?

### Authorization Gaps (has auth, wrong auth)
- **Cross-tenant access**: User-supplied `teamId`/`userId` used in DB queries instead of the authenticated identity
- **Missing resource-level checks**: Auth confirms "user is logged in" but doesn't verify "user owns this resource"
- **Negated permission checks**: `!(await auth.can(...))` with inverted logic

## Out-of-scope files

Skip files that are gitignored, generated, vendored, or not production code. If a file is in `dist/`, `node_modules/`, `vendor/`, `generated/`, or matches `.gitignore`, return an empty findings array for it."""


#: Per-slug one-line notes, pulled into the prompt only when the slug
#: appears in the current batch: "what to check before flagging."
SLUG_NOTES: dict[str, str] = {
    "missing-auth": "Weak candidate — only flag if no auth wrapper, no role check, AND user-controlled input reaches a sink.",
    "auth-bypass": "Look for inverted booleans, early returns that skip checks, and `if (process.env.X) skipAuth()` patterns.",
    "cross-tenant-id": "User-supplied teamId/userId in DB queries — confirm the authenticated identity is used for the ownership check, not the request param alone.",
    "cors-wildcard": "`origin: true` + `credentials: true` is the high-severity shape; static `*` without credentials is usually fine.",
    "open-redirect": "Flag only if there's no allowlist, origin check, or hash-only redirect; relative paths starting with `//` are still external.",
    "unsafe-redirect": "Verify the redirect path passes through a validation function and that the validator can't be bypassed via encoding.",
    "dangerous-html": "DB-stored HTML is still untrusted — flag unless there's a sanitizer (DOMPurify, sanitize-html) BETWEEN the data and the render.",
    "xss": "Check escape state at every step; raw concat into HTML, JSON-in-script without `</`-escape, and ref.innerHTML are the usual sinks.",
    "rce": "Distinguish dynamic command (string concat → exec) from static command with sanitized args (which is fine).",
    "sql-injection": "Flag string-concat / template-literal SQL only if the variable is user-reachable; ORM `where({col: x})` is safe.",
    "ssrf": "Check whether the URL host is constrained to an allowlist, blocked from RFC1918, or proxied via a vetted URL parser.",
    "path-traversal": "Flag if `path.join(root, userInput)` lacks a `path.resolve(...).startsWith(root)` containment check.",
    "secrets-exposure": "Distinguish real secrets from example values, dummy tokens in tests, and rotated/expired markers.",
    "secret-in-fallback": '`process.env.X || "hardcoded"` is the bug — only flag when the fallback looks like a real credential, not `"localhost"`.',
    "secret-in-log": "Logging full headers, request bodies, or error objects can leak Authorization tokens; flag if the log destination is durable.",
    "secret-env-var": "Direct env var reads in client-bundled code (NEXT_PUBLIC_*) are the bug — confirm the file isn't server-only.",
    "env-exposure": "Secrets reaching client bundles via `NEXT_PUBLIC_` / `VITE_` / build-time inlining — flag only if the env var holds a credential.",
    "rate-limit-bypass": "Sensitive operations (auth, password reset, expensive APIs) without rate-limit middleware are the high-signal cases.",
    "expensive-api-abuse": "LLM/AI/paid-API endpoints without per-user rate limits or auth — confirm the cost-per-call is non-trivial before flagging.",
    "webhook-handler": "Confirm signature verification (Stripe, GitHub, Shopify, Slack) happens BEFORE the body is parsed/processed.",
    "jwt-handling": "Look for `algorithm: 'none'`, missing `algorithms: ['HS256']` pinning, or skipping `verify()` in dev branches.",
    "iam-permissions": "Wildcards in Action AND Resource together are the dangerous shape; one or the other can be intentional.",
    "cache-key-poisoning": "Cache keys derived from request headers/cookies (User-Agent, Cookie, X-Forwarded-*) without normalization are the bug.",
    "public-endpoint": "Confirm the endpoint truly has no auth (not just a permissive guard) and that it returns sensitive data.",
    "service-entry-point": "Coarse flag — verify there's an actual auth gap, not just an internal-only handler reachable via service mesh.",
    "object-injection": "User-controlled keys into `obj[x] = v` without an allowlist enable prototype-pollution / overwriting safe defaults.",
    "non-atomic-operation": "Read-then-write patterns without a lock / transaction / atomic op are TOCTOU; flag only if the resource is shared across requests.",
    "debug-endpoint": "Routes guarded by `process.env.NODE_ENV === 'development'` can ship to prod via env misconfig — flag if the route does anything sensitive.",
    "test-header-bypass": "`x-test-*` / `x-bypass-*` headers honored in handler code are the classic prod-leakage bug.",
    "dev-auth-bypass": "`if (env === 'dev') return adminUser` patterns — verify the env check can't be tricked, and that the path isn't reachable in prod.",
    "insecure-crypto": "Weak hashes for passwords (MD5/SHA1 without a KDF), `Math.random` for tokens, and deprecated cipher APIs are the shapes worth reporting; a checksum use of MD5 is not.",
    "crypto-usage": "Wide-net flag — look for the subtle bugs the narrow matchers miss: algorithm confusion, timing-unsafe compares, IV reuse, weak key sizes, wrong tag-verification order.",
    "js-sql-raw": "Raw-SQL across pg/mysql2/TypeORM/Sequelize/Knex/Kysely/postgres.js — flag string concat or template interpolation into SELECT/INSERT/UPDATE/DELETE; parameterized forms (`$1`, `:name`, prepared statements with separate args) are the safe shape. `sql`...`` tagged templates from libraries that escape (drizzle, postgres.js without `.unsafe`) are safe.",
    "py-sql-raw": 'Raw-SQL across SQLAlchemy/psycopg/pymysql/sqlite3/asyncpg/Django ORM — f-string, `%` formatting, `.format()`, and `+` concat into SQL are injection. The safe shape is `cursor.execute("... %s ...", (val,))` (psycopg) or `text("... :x ...").bindparams(x=val)` (SQLAlchemy).',
}


def assemble(
    batch_slugs: list[str],
    *,
    project_info: str | None = None,
    prompt_append: str | None = None,
) -> str:
    """`assemble.ts`: core, slug notes for the slugs in this batch, then the
    user-authored INFO.md and promptAppend behind horizontal rules."""
    sections: list[str] = [CORE_PROMPT]
    notes = [
        f"- `{s}`: {SLUG_NOTES[s]}"
        for s in dict.fromkeys(batch_slugs)
        if s in SLUG_NOTES
    ]
    if notes:
        sections.append(
            "## Slug-specific reviewer notes\n\n" + "\n".join(notes)
        )
    if project_info and project_info.strip():
        sections.append("---\n\n" + project_info.strip())
    if prompt_append and prompt_append.strip():
        sections.append("---\n\n" + prompt_append.strip())
    return "\n\n".join(sections)


def _where(project_dir: str) -> str:
    """Only for a workspace where the tree is not at the root (a sandbox)."""
    if project_dir in ("", "."):
        return ""
    return (
        f"\n\n## Where the code is\n\nThe project root is `{project_dir}/`, relative to your "
        + f"working directory. Paths below are relative to that root; read `{project_dir}/<path>`."
    )


INVESTIGATION_INSTRUCTIONS = """## Investigation Instructions

For each file:
1. **Read the file fully** using the Read tool
2. **Trace data flows** — where does input come from? Is it user-controlled?
3. **Follow imports** — read related files (middleware, utils, shared libs) to understand the full picture
4. **Check for mitigations** — is there sanitization, validation, auth middleware, or framework protection?
5. **Think broadly** — look for issues beyond what the scanner flagged. The scanner only finds surface patterns; you should reason about logic bugs, race conditions, missing checks, etc.

## Output Format

After your investigation, output your findings for EACH file as a JSON array of `{ filePath, findings }` objects, using the exact relative paths shown above.

**Severity levels:**
- **CRITICAL / HIGH / MEDIUM** — security vulnerabilities (exploitable by an attacker)
- **HIGH_BUG** — major non-security bugs that could cause data loss, corruption, outages, or seriously broken behavior
- **BUG** — notable non-security bugs (logic errors, race conditions, resource leaks) that don't rise to HIGH_BUG

**vulnSlug** can be any of the known categories OR a custom slug for issues not covered by the scanner. Use `"other"` as the slug prefix for novel findings (e.g., `"other-race-condition"`, `"other-logic-bug"`, `"other-info-disclosure"`).

If a file has no real vulnerabilities after thorough investigation, include it with an empty findings array."""


def investigation_prompt(
    batch: list[FileRecord],
    *,
    project_dir: str,
    project_info: str | None = None,
) -> str:
    """`buildInvestigatePrompt`: the assembled template, the per-batch
    target list, the procedural steps, the output spec."""
    template = assemble(
        [c.vuln_slug for r in batch for c in r.candidates],
        project_info=project_info,
    )
    targets = []
    for record in batch:
        if not record.candidates:
            targets.append(
                f"- **{record.file_path}** (no scanner hits — full holistic review)"
            )
            continue
        details = "\n".join(
            f"    - [{c.vuln_slug}] L{', '.join(str(n) for n in c.line_numbers)}: {c.matched_pattern}"
            for c in record.candidates
        )
        targets.append(f"- **{record.file_path}**\n{details}")
    return (
        template
        + _where(project_dir)
        + "\n\n## Target Files\n\n"
        + "\n".join(targets)
        + "\n\n"
        + INVESTIGATION_INSTRUCTIONS
    )


REVALIDATION_INTRO = """You are a world-class security researcher performing an adversarial review of vulnerability findings. Your goal is to determine, with high confidence, whether each finding is real and exploitable. You must be thorough — incorrect verdicts here directly impact security decisions.

**Take your time.** Read every relevant file. Trace every code path. Do not make assumptions — verify.

**Static analysis only.** Do NOT attempt to reproduce, exploit, or trigger any finding. Do not run the target code, send requests against any endpoint, or execute proof-of-concept scripts. Reach your verdict from the source code alone."""

REVALIDATION_PROCESS = """## Investigation Process

For EACH finding, perform ALL of these steps before rendering a verdict:

1. **Read the target file fully** — not just the flagged lines, the entire file
2. **Read all imports that matter** — middleware, auth utilities, validation helpers, the framework's request pipeline
3. **Trace the data flow end-to-end** — Where does the input enter? What transformations happen? Is there validation or sanitization?
4. **Think like an attacker** — Construct a concrete attack scenario. If you can't, it's likely a false positive.
5. **Check for framework-level protections** — Next.js middleware, withSchema auth strategies, CSRF tokens, CORS headers
6. **Check the current code vs. the finding** — Has the vulnerable code been modified or removed? Check git history.
7. **Assess confidence honestly** — If you're not sure, say "uncertain". Don't guess.

## Verdicts

- **true-positive** — Real AND exploitable. You can describe a concrete attack.
- **false-positive** — Not exploitable. Name the specific mitigation.
- **fixed** — Was real but has been patched. Cite the change.
- **uncertain** — Can't determine. Explain what's ambiguous.
- **duplicate** — This finding describes the **same underlying vulnerability** at the **same code location** as another finding in the **same file** (e.g., two matchers flagged the same line range from different angles, or the same auth bypass surfaced twice with different phrasing). Set `duplicateOf` to the Finding ID of the primary finding — the one that should keep the canonical verdict. Same vuln class in a different location is **not** a duplicate.

If severity should change, set `adjustedSeverity`. Omit if correct.

### Duplicate rules (read carefully)

- `duplicate` is only valid within a single file. Cross-file similarity does **not** count.
- For any equivalence class of duplicates, **exactly one finding stays primary** with a real verdict (true-positive / false-positive / fixed / uncertain). The other(s) are `duplicate` with `duplicateOf` pointing at the primary's Finding ID.
- The primary you reference in `duplicateOf` **must itself have a non-duplicate verdict** in your output (or already in the file's prior revalidation). If you mark every member of a group as duplicate, all of them will be rejected.
- Pick the primary as the most precise / highest-confidence statement of the issue. The duplicates should add context in their `reasoning`, not repeat the full analysis.

## Output Format

Return a JSON array with exactly one verdict object per Finding ID below — `findingId`, `verdict`, `reasoning` (5-10 sentences; show your work), and optionally `adjustedSeverity` and `duplicateOf`."""


def _git_history(root: Path | None, file_path: str) -> str:
    """Recent history for one file, from the host's checkout (a sandbox copy
    has no `.git`). argv form, never a shell: the path came from a glob."""
    if root is None:
        return ""
    try:
        out = subprocess.run(
            [
                "git",
                "log",
                "--oneline",
                "--since=3 months ago",
                "-n",
                "10",
                "--",
                file_path,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    log = out.stdout.strip() if out.returncode == 0 else ""
    return f"\n**Recent git history:**\n```\n{log}\n```\n" if log else ""


def revalidation_prompt(
    items: list[tuple[FileRecord, Finding]],
    *,
    project_dir: str,
    git_root: Path | None = None,
    project_info: str | None = None,
) -> str:
    """`buildRevalidatePrompt`, keyed by the full `findingId` rather than a
    short alias — the id is what the SDK's schema asks the agent to echo."""
    by_file: dict[str, list[tuple[FileRecord, Finding]]] = {}
    for record, finding in items:
        by_file.setdefault(record.file_path, []).append((record, finding))
    sections = []
    for path, group in by_file.items():
        blocks = []
        for _, f in group:
            blocks.append(
                f"### Finding: {f.title}\n"
                + f"- **Finding ID:** {f.finding_id}\n"
                + f"- **Severity:** {f.severity}\n"
                + f"- **Slug:** {f.vuln_slug}\n"
                + f"- **Lines:** {', '.join(str(n) for n in f.line_numbers)}\n"
                + f"- **Confidence:** {f.confidence}\n"
                + f"- **Description:** {f.description}\n"
                + f"- **Recommendation:** {f.recommendation}"
            )
        sections.append(
            f"## File: {path}\n\n"
            + "\n\n".join(blocks)
            + "\n"
            + _git_history(git_root, path)
        )
    context = (
        f"## Project Context\n\n{project_info.strip()}\n\n"
        if project_info and project_info.strip()
        else ""
    )
    listing = "\n".join(f'- {f.finding_id} — "{f.title}"' for _, f in items)
    return (
        REVALIDATION_INTRO
        + _where(project_dir)
        + "\n\n"
        + context
        + "\n---\n\n".join(sections)
        + "\n\n"
        + REVALIDATION_PROCESS
        + "\n\nYou must return exactly one verdict for every Finding ID below:\n\n"
        + listing
        + '\n\n**Copy each `findingId` exactly as shown. Do not invent or modify IDs.** The Finding ID — not the title — is how verdicts are matched back to findings. `adjustedSeverity` is optional. `duplicateOf` is required iff `verdict === "duplicate"` and is otherwise ignored; it must also be a Finding ID.\n\n'
        + "**Your reasoning is the most important part.** A verdict without thorough reasoning is worthless."
    )
