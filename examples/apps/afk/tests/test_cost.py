"""The cost arithmetic, checked against the pricing page's worked examples
(Pro rates, iad1, 10% CPU utilization)."""

from __future__ import annotations

import pytest
from afk import cost

from ai.workspaces.experimental import (
    SandboxInfo,
    SandboxSession,
    SandboxSnapshot,
)

MIN = 60_000


def _session(
    minutes: float, vcpus: int, memory_gb: int, utilization: float = 0.1
) -> SandboxSession:
    ran = int(minutes * MIN)
    return SandboxSession(
        id="s",
        status="stopped",
        region="iad1",
        vcpus=vcpus,
        memory_mb=memory_gb * 1024,
        started_at=0,
        stopped_at=ran,
        duration_ms=ran,
        active_cpu_ms=int(ran * vcpus * utilization),
        egress_bytes=0,
    )


@pytest.mark.parametrize(
    ("minutes", "vcpus", "memory", "cpu", "mem"),
    [
        (2, 1, 2, 0.0004, 0.0014),  # "Quick test"
        (5, 2, 4, 0.0021, 0.0071),  # "AI code validation"
        (30, 4, 8, 0.0256, 0.0848),  # "Build and test"
    ],
)
def test_the_pricing_pages_examples(
    minutes: float, vcpus: int, memory: int, cpu: float, mem: float
) -> None:
    c = cost.session_cost(_session(minutes, vcpus, memory), "iad1")
    assert round(c.cpu, 4) == cpu
    assert round(c.memory, 4) == mem
    assert not c.pending


def test_memory_is_billed_by_the_minute_with_a_minimum_of_one() -> None:
    short = cost.session_cost(_session(0.1, 2, 4), "iad1")
    one = cost.session_cost(_session(1, 2, 4), "iad1")
    assert short.memory == one.memory


def test_only_data_sent_is_billed_at_the_regions_rate() -> None:
    s = _session(1, 2, 4).model_copy(
        update={"egress_bytes": 2 * cost.GB, "ingress_bytes": 50 * cost.GB}
    )
    assert cost.session_cost(s, "iad1").network == pytest.approx(0.30)
    assert cost.session_cost(
        s.model_copy(update={"region": "icn1"}), None
    ).network == pytest.approx(0.70)


def test_a_running_session_counts_its_memory_so_far_and_says_more_comes() -> (
    None
):
    running = SandboxSession(
        id="r", status="running", memory_mb=4096, started_at=0
    )
    c = cost.session_cost(running, "iad1", now_ms=30 * MIN)
    assert c.pending and c.cpu == 0
    assert c.memory == pytest.approx(4 * 0.5 * 0.0212)
    assert c.per_hour == pytest.approx(4 * 0.0212)
    assert cost.label(c) == "$0.042 · $0.085/h"
    assert cost.spent(c) == "$0.042 so far"


def test_a_sandbox_adds_its_sessions_creation_and_snapshot() -> None:
    info = SandboxInfo(
        name="x",
        status="stopped",
        region="iad1",
        sessions=[_session(5, 2, 4), _session(30, 4, 8)],
        snapshot=SandboxSnapshot(id="snap", size_bytes=cost.GB, created_at=0),
    )
    month = 30 * 24 * 60 * MIN
    c = cost.sandbox_cost(info, now_ms=month // 2)
    assert c.snapshot == pytest.approx(0.04)
    assert c.snapshot_per_month == pytest.approx(0.08)
    assert c.total == pytest.approx(
        0.0021 + 0.00707 + 0.0256 + 0.0848 + 0.04 + cost.CREATION, abs=1e-4
    )


def test_a_region_without_rates_is_said_not_guessed() -> None:
    s = _session(5, 2, 4).model_copy(update={"region": "mars1"})
    c = cost.session_cost(s, None)
    assert c.unpriced == "mars1" and c.total == 0
    assert cost.label(c) == "? (mars1)"
