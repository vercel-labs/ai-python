"""What a sandbox has cost so far, from what the platform measured.

Vercel bills a sandbox on (https://vercel.com/docs/vercel-sandbox/pricing,
read 2026-10-19):

- Active CPU: hours of CPU actually used. Time waiting on I/O — a model
  call — does not count. Per region.
- Provisioned memory: GB x hours the VM ran, in 1-minute increments with a
  1-minute minimum. Per region.
- Network: data the sandbox SENDS; downloads are free. Per region.
- Creations: $0.60 per million.
- Snapshot storage: $0.08 per GB-month, every region.

These are list prices (Pro and Enterprise usage rates). A Hobby plan's
monthly allowance and a Pro plan's monthly credit are not subtracted: the
figure is what the usage is worth, not what this month's bill will say.

The platform reports a session's CPU, duration and data sent when it STOPS.
For a running one, memory so far is exact — it is wall-clock — and CPU and
data sent are not known yet: the cost reads as "so far, and more".
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from ai.workspaces.experimental import SandboxInfo, SandboxSession

RATES: dict[str, tuple[float, float, float]] = {
    "arn1": (0.160, 0.0266, 0.15),
    "bom1": (0.140, 0.0232, 0.20),
    "cdg1": (0.177, 0.0292, 0.15),
    "cle1": (0.128, 0.0212, 0.15),
    "cpt1": (0.200, 0.0332, 0.28),
    "dub1": (0.168, 0.0278, 0.15),
    "fra1": (0.184, 0.0304, 0.15),
    "gru1": (0.221, 0.0366, 0.22),
    "hkg1": (0.176, 0.0292, 0.16),
    "hnd1": (0.202, 0.0334, 0.16),
    "iad1": (0.128, 0.0212, 0.15),
    "icn1": (0.169, 0.0280, 0.35),
    "kix1": (0.202, 0.0334, 0.16),
    "lhr1": (0.177, 0.0292, 0.15),
    "pdx1": (0.128, 0.0212, 0.15),
    "sfo1": (0.177, 0.0294, 0.15),
    "sin1": (0.160, 0.0266, 0.16),
    "syd1": (0.180, 0.0298, 0.16),
    "yul1": (0.147, 0.0243, 0.15),
}
"""Per region: USD per Active CPU hour, per provisioned GB-hour of memory,
per GB of data sent. From the pricing page's regional table."""
CREATION = 0.60 / 1_000_000
SNAPSHOT_GB_MONTH = 0.08
GB = 1_000_000_000
MONTH_MS = 30 * 24 * 3600 * 1000


class Cost(BaseModel):
    """USD, by what it is for."""

    cpu: float = 0.0
    memory: float = 0.0
    network: float = 0.0
    creation: float = 0.0
    snapshot: float = 0.0
    """Storage of the snapshot it would resume from, so far."""
    snapshot_per_month: float = 0.0
    """What keeping that snapshot costs per month, from now."""
    pending: bool = False
    """A session is still running: its CPU and data sent come when it stops."""
    per_hour: float = 0.0
    """While running: what its memory costs per hour, billed every hour it
    is up, busy or not. CPU adds while it computes."""
    unpriced: str | None = None
    """A region this table has no rates for."""

    @property
    def total(self) -> float:
        return (
            self.cpu
            + self.memory
            + self.network
            + self.creation
            + self.snapshot
        )

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            cpu=self.cpu + other.cpu,
            memory=self.memory + other.memory,
            network=self.network + other.network,
            creation=self.creation + other.creation,
            snapshot=self.snapshot + other.snapshot,
            snapshot_per_month=self.snapshot_per_month
            + other.snapshot_per_month,
            pending=self.pending or other.pending,
            per_hour=self.per_hour + other.per_hour,
            unpriced=self.unpriced or other.unpriced,
        )


def session_cost(
    session: SandboxSession, region: str | None, now_ms: int | None = None
) -> Cost:
    """One session: memory by the minute it ran, CPU and data as measured."""
    rates = RATES.get(session.region or region or "")
    if rates is None:
        return Cost(unpriced=session.region or region or "unknown")
    cpu_rate, memory_rate, network_rate = rates
    running = session.duration_ms is None
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    started = session.started_at if session.started_at is not None else now
    ran_ms = session.duration_ms if not running else max(0, now - started)
    minutes = max(1, math.ceil((ran_ms or 0) / 60_000))
    gb = (session.memory_mb or 0) / 1024
    return Cost(
        memory=gb * (minutes / 60) * memory_rate,
        cpu=(session.active_cpu_ms or 0) / 3_600_000 * cpu_rate,
        network=(session.egress_bytes or 0) / GB * network_rate,
        pending=running,
        per_hour=gb * memory_rate if running else 0.0,
    )


def sandbox_cost(info: SandboxInfo, now_ms: int | None = None) -> Cost:
    """Everything a sandbox has cost: its sessions, its creation, and its
    snapshot's storage so far."""
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    total = Cost(creation=CREATION)
    for session in info.sessions:
        total = total + session_cost(session, info.region, now)
    if info.snapshot is not None and info.snapshot.size_bytes:
        size = info.snapshot.size_bytes / GB
        made = info.snapshot.created_at
        stored = max(0, now - (made if made is not None else now))
        total = total + Cost(
            snapshot=size * (stored / MONTH_MS) * SNAPSHOT_GB_MONTH,
            snapshot_per_month=size * SNAPSHOT_GB_MONTH,
        )
    return total


def money(value: float) -> str:
    """Dollars, with the precision small numbers need."""
    if value < 0.01:
        return f"${value:.4f}"
    if value < 1:
        return f"${value:.3f}"
    return f"${value:.2f}"


def label(cost: Cost) -> str:
    """The list's cost: "$0.012", or "$0.012 ($0.085/h)" while running."""
    if cost.unpriced:
        return f"? ({cost.unpriced})"
    if cost.pending and cost.per_hour:
        return f"{money(cost.total)} ({money(cost.per_hour)}/h)"
    return money(cost.total)


def spent(cost: Cost) -> str:
    """A cost in a sentence: "$0.012", or "$0.012 so far" while running."""
    if cost.unpriced:
        return f"? ({cost.unpriced})"
    return money(cost.total) + (" so far" if cost.pending else "")
