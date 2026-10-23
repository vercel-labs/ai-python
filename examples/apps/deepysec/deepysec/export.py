"""Read-only stages (spec §7): `export` writes findings.json and report.md
under `reports/`; `status` prints a snapshot."""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

from deepysec.models import SEVERITY_ORDER, FileRecord, Finding

if TYPE_CHECKING:
    from pathlib import Path

    from deepysec.store import Store


def _verdict(finding: Finding) -> str:
    return (
        finding.revalidation.verdict
        if finding.revalidation
        else "unrevalidated"
    )


def export(store: Store) -> tuple[Path, Path]:
    records = store.records()
    rows: list[tuple[FileRecord, Finding]] = [
        (r, f) for r in records for f in r.findings
    ]
    rows.sort(key=lambda rf: (SEVERITY_ORDER[rf[1].severity], rf[0].file_path))

    flat = [
        {"filePath": r.file_path, **f.model_dump(mode="json")} for r, f in rows
    ]
    store.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = store.reports_dir / "findings.json"
    json_path.write_text(json.dumps(flat, indent=2))

    lines = [f"# deepysec report — {store.project_id}", ""]
    by_severity = Counter(f.severity for _, f in rows)
    by_verdict = Counter(_verdict(f) for _, f in rows)
    lines.append(
        f"{len(records)} files scanned, "
        f"{sum(1 for r in records if r.status == 'analyzed')} analyzed, "
        + f"{len(rows)} findings."
    )
    lines.append("")
    lines.append("| severity | count |  | verdict | count |")
    lines.append("|---|---|---|---|---|")
    severities = sorted(by_severity, key=lambda s: SEVERITY_ORDER[s])
    verdicts = sorted(by_verdict)
    for i in range(max(len(severities), len(verdicts), 1)):
        left = (
            f"{severities[i]} | {by_severity[severities[i]]}"
            if i < len(severities)
            else " | "
        )
        right = (
            f"{verdicts[i]} | {by_verdict[verdicts[i]]}"
            if i < len(verdicts)
            else " | "
        )
        lines.append(f"| {left} |  | {right} |")
    lines.append("")
    for r, f in rows:
        v = f.revalidation
        badge = f" · **{v.verdict}**" if v else ""
        numbers = ", ".join(map(str, f.line_numbers)) or "n/a"
        lines.append(f"## [{f.severity}] {f.title}{badge}")
        lines.append("")
        lines.append(
            f"`{r.file_path}` lines {numbers} · `{f.vuln_slug}` · confidence "
            f"{f.confidence} · `{f.finding_id}`"
        )
        lines.append("")
        lines.append(f.description.strip())
        if f.recommendation:
            lines.append("")
            lines.append(f"**Fix:** {f.recommendation.strip()}")
        if v:
            lines.append("")
            adjusted = (
                f" (severity → {v.adjusted_severity})"
                if v.adjusted_severity
                else ""
            )
            lines.append(
                f"**Revalidation ({v.model}):** {v.verdict}{adjusted} — "
                f"{v.reasoning.strip()}"
            )
        if r.git_info and r.git_info.recent_committers:
            who = ", ".join(
                f"{c.name} <{c.email}>"
                for c in r.git_info.recent_committers[:3]
            )
            lines.append("")
            lines.append(f"**Recent committers:** {who}")
        lines.append("")
    md_path = store.reports_dir / "report.md"
    md_path.write_text("\n".join(lines))
    return json_path, md_path


def status(store: Store) -> str:
    records = store.records()
    statuses = Counter(r.status for r in records)
    findings = [f for r in records for f in r.findings]
    severities = Counter(f.severity for f in findings)
    verdicts = Counter(_verdict(f) for f in findings)
    runs = store.runs()
    cost = sum(r.stats.total_cost_usd for r in runs)
    out = [
        f"project {store.project_id} — {store.root}",
        f"files:    {len(records)} ("
        + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items()))
        + ")",
        f"findings: {len(findings)} ("
        + ", ".join(
            f"{k} {v}"
            for k, v in sorted(
                severities.items(), key=lambda kv: SEVERITY_ORDER[kv[0]]
            )
        )
        + ")",
        "verdicts: "
        + ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items())),
        f"runs:     {len(runs)}, ${cost:.4f} total",
    ]
    locked = [r for r in records if r.status == "processing"]
    if locked:
        out.append(
            "locked:   "
            + ", ".join(
                f"{r.file_path} (by {r.locked_by_run_id})" for r in locked
            )
        )
    return "\n".join(out)
