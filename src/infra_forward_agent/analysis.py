"""Temporal exposure is a prioritization signal, never proof of a causal critical path."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .trace import Trace


def union_us(intervals: Iterable[tuple[float, float]]) -> float:
    end = float("-inf")
    total = 0.0
    for start, finish in sorted(intervals):
        if finish > end:
            total += finish - max(start, end)
            end = finish
    return total


def analyze(trace: Trace) -> dict:
    groups = defaultdict(list)
    for item in trace.slices:
        groups[item.rank, item.device].append(item)
    devices = []
    for (rank, device), items in sorted(groups.items()):
        by_kind = {
            kind: [(item.start_us, item.end_us) for item in items if item.kind == kind]
            for kind in ("compute", "communication", "memory")
        }
        compute, comm = union_us(by_kind["compute"]), union_us(by_kind["communication"])
        overlap = max(0.0, compute + comm - union_us(by_kind["compute"] + by_kind["communication"]))
        busy = union_us((item.start_us, item.end_us) for item in items)
        start, end = trace.windows[rank]
        hotspots = defaultdict(lambda: {"calls": 0, "work_us": 0.0})
        for item in items:
            row = hotspots[item.name, item.kind]
            row["calls"] += 1
            row["work_us"] += item.end_us - item.start_us
        devices.append(
            {
                "rank": rank,
                "device": device,
                "forward_us": end - start,
                "busy_us": busy,
                "idle_us": max(0.0, end - start - busy),
                "compute_us": compute,
                "communication_us": comm,
                "mixed_us": union_us((i.start_us, i.end_us) for i in items if i.kind == "mixed"),
                "component_split_known": not any(i.kind in {"mixed", "unknown"} for i in items),
                "memory_us": union_us(by_kind["memory"]),
                "compute_communication_overlap_us": overlap,
                "communication_without_compute_us": max(0.0, comm - overlap),
                "streams": sorted({item.stream for item in items}),
                "hotspots": sorted(
                    [
                        dict(name=name, kind=kind, **values)
                        for (name, kind), values in hotspots.items()
                    ],
                    key=lambda row: row["work_us"],
                    reverse=True,
                ),
            }
        )
    return {
        "schema_version": 1,
        "devices": devices,
        "ignored_events": trace.ignored_events,
        "classification_sources": sorted({item.classification for item in trace.slices}),
        "limitations": [
            "Hotspot work sums can exceed forward wall time because streams overlap.",
            "Temporal overlap is not causal critical-path attribution or an achievable speedup.",
            "Ranks use local windows; cross-host clocks are not assumed synchronized.",
            "Mixed components are unresolved; component times include known classifications only.",
            "Missing ranks/devices and KV capacity cannot be inferred from this trace alone.",
        ],
    }


def propose(analysis: dict) -> list[dict]:
    """Create engineering tasks. A downstream coding agent owns implementation and measurement."""
    tasks = []
    for device in analysis["devices"]:
        common = {"rank": device["rank"], "device": device["device"], "status": "proposed"}
        if device["hotspots"]:
            hot = device["hotspots"][0]
            tasks.append(
                dict(
                    common,
                    track="communication" if hot["kind"] == "communication" else "operator",
                    target=hot["name"],
                    evidence={"work_us": hot["work_us"], "calls": hot["calls"]},
                    hypothesis="Profile dominant work; verify dependencies before changes.",
                )
            )
        if device["communication_without_compute_us"] > 0:
            tasks.append(
                dict(
                    common,
                    track="overlap",
                    target="compute/communication timeline",
                    evidence={
                        "communication_without_compute_us": device[
                            "communication_without_compute_us"
                        ]
                    },
                    hypothesis="Check data/collective ordering, then test stream scheduling.",
                )
            )
        tasks.append(
            dict(
                common,
                track="distributed",
                target="topology and KV capacity",
                status="needs_evidence",
                evidence={},
                hypothesis="Collect per-rank memory, KV layout and TP/EP/PP/DP placement first.",
            )
        )
    return tasks
