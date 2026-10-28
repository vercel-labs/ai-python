"""deepsec's data model, preserved in shape (spec §2).

On disk it is camelCase JSON — what vercel-labs/deepsec writes — so a record
written here reads there. In Python it is snake_case Pydantic. The alias
generator does the translation both ways.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

Severity = Literal["CRITICAL", "HIGH", "MEDIUM", "HIGH_BUG", "BUG", "LOW"]
SEVERITY_ORDER: dict[str, int] = {
    "CRITICAL": 0,
    "HIGH": 1,
    "HIGH_BUG": 2,
    "MEDIUM": 3,
    "BUG": 4,
    "LOW": 5,
}
Confidence = Literal["high", "medium", "low"]
FileStatus = Literal["pending", "processing", "analyzed", "error"]
NoiseTier = Literal["precise", "normal", "noisy"]
NOISE_SCORE: dict[str, int] = {"precise": 0, "normal": 1, "noisy": 2}

#: What the agent may say about a finding. `accepted-risk` is deliberately
#: NOT here: it is a manual marker, and leaving it out of the type is what
#: makes it impossible for an agent to emit — the schema the SDK sends
#: refuses it, and a reply carrying it is re-asked.
AgentVerdict = Literal[
    "true-positive", "false-positive", "fixed", "uncertain", "duplicate"
]
RevalidationVerdict = AgentVerdict | Literal["accepted-risk"]


def now() -> str:
    return (
        datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def new_run_id() -> str:
    return (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
    )


class Record(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
        extra="ignore",
    )


# -- scan ---------------------------------------------------------------------


class CandidateMatch(Record):
    vuln_slug: str
    line_numbers: list[int]
    snippet: str
    matched_pattern: str


# -- process ------------------------------------------------------------------


class Revalidation(Record):
    verdict: RevalidationVerdict
    reasoning: str
    adjusted_severity: Severity | None = None
    duplicate_of: str | None = None
    revalidated_at: str
    run_id: str
    model: str


class Finding(Record):
    finding_id: str | None = None
    severity: Severity
    vuln_slug: str
    title: str
    description: str
    line_numbers: list[int] = Field(default_factory=list)
    recommendation: str = ""
    confidence: Confidence = "medium"
    revalidation: Revalidation | None = None
    produced_by_run_id: str | None = None
    """The run that FIRST surfaced it; never updated."""

    @property
    def signature(self) -> tuple[str, str]:
        return (self.vuln_slug, self.title)


class TokenUsage(Record):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class AnalysisEntry(Record):
    run_id: str
    investigated_at: str
    duration_ms: int
    agent_type: str
    model: str
    agent_session_id: str | None = None
    finding_count: int
    phase: Literal["process", "revalidate"] = "process"
    cost_usd: float | None = None
    """This FILE's share of the batch — see `split_evenly`."""
    usage: TokenUsage | None = None


# -- enrich -------------------------------------------------------------------


class Committer(Record):
    name: str
    email: str
    date: str


class GitInfo(Record):
    recent_committers: list[Committer]
    enriched_at: str


# -- the record ---------------------------------------------------------------


class FileRecord(Record):
    file_path: str
    project_id: str
    candidates: list[CandidateMatch] = Field(default_factory=list)
    last_scanned_at: str
    last_scanned_run_id: str
    file_hash: str
    findings: list[Finding] = Field(default_factory=list)
    analysis_history: list[AnalysisEntry] = Field(default_factory=list)
    git_info: GitInfo | None = None
    status: FileStatus = "pending"
    locked_by_run_id: str | None = None
    locked_at: str | None = None

    @property
    def noise_score(self) -> int:
        """Minimum tier across the file's slugs: precise wins.

        Work ordering only.
        """
        # imported here: matchers imports this module
        from deepysec.matchers import tier_of  # noqa: PLC0415

        return min(
            (NOISE_SCORE[tier_of(c.vuln_slug)] for c in self.candidates),
            default=2,
        )


class RunStats(Record):
    files_scanned: int = 0
    candidates_found: int = 0
    files_processed: int = 0
    findings_count: int = 0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration_ms: int = 0
    findings_revalidated: int = 0
    true_positives: int = 0
    false_positives: int = 0
    fixed: int = 0
    uncertain: int = 0
    duplicates: int = 0


class RunMeta(Record):
    run_id: str = Field(default_factory=new_run_id)
    project_id: str
    root_path: str
    created_at: str = Field(default_factory=now)
    completed_at: str | None = None
    type: Literal["scan", "process", "revalidate"]
    phase: Literal["running", "done", "error"] = "running"
    pid: int | None = None
    hostname: str | None = None
    """pid liveness is only meaningful on the same host."""
    agent_type: str | None = None
    model: str | None = None
    stats: RunStats = Field(default_factory=RunStats)


# -- the agent's side of the contract (spec §4.3, §5) --------------------------


class ReportedFinding(Record):
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "HIGH_BUG", "BUG"]
    vuln_slug: str
    title: str
    description: str
    line_numbers: list[int] = Field(default_factory=list)
    recommendation: str = ""
    confidence: Confidence = "medium"


class FileReport(Record):
    """One entry per file in the batch — `findings` empty when it is clean."""

    file_path: str
    findings: list[ReportedFinding] = Field(default_factory=list)


class Verdict(Record):
    finding_id: str
    verdict: AgentVerdict
    reasoning: str
    adjusted_severity: Severity | None = None
    duplicate_of: str | None = None
    """Required with `duplicate`: the findingId of the primary."""


# -- finding identity (spec §2.4) ----------------------------------------------


def finding_id(project_id: str, file_path: str, title: str) -> str:
    """Derived ONLY from immutable identifying data, so the same finding has
    the same id on every load, across runs, models and machines."""
    norm = file_path.replace("\\", "/").removeprefix("./")
    digest = hashlib.sha256(
        f"{project_id}\0{norm}\0{title}".encode()
    ).hexdigest()
    return f"finding_{digest[:16]}"


def ensure_finding_ids(record: FileRecord) -> bool:
    """Backfill ids on findings that predate the field.

    Same-file title collisions are salted with a stable ordinal in array order —
    findings are append-only, so an existing finding's ordinal never changes.
    """
    changed = False
    used = {f.finding_id for f in record.findings if f.finding_id}
    for finding in record.findings:
        if finding.finding_id:
            continue
        candidate = finding_id(
            record.project_id, record.file_path, finding.title
        )
        n = 2
        while candidate in used:
            candidate = finding_id(
                record.project_id, record.file_path, f"{finding.title}\0{n}"
            )
            n += 1
        used.add(candidate)
        finding.finding_id = candidate
        changed = True
    return changed


def split_evenly(total: int, parts: int) -> list[int]:
    """Integers whose sum is exactly `total`, as equal as integers allow."""
    if parts <= 0:
        return []
    base, remainder = divmod(total, parts)
    return [base + (1 if i < remainder else 0) for i in range(parts)]
