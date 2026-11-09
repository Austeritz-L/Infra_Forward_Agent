from __future__ import annotations

from collections import defaultdict

from ..analysis import union_us
from ..trace import Slice, Trace
from .statistics import summarize


def group_key(item: Slice) -> tuple[str, str, str]:
    return item.operator or item.name, item.compute_type, item.signature


def analyze_timeline(trace: Trace) -> dict:
    devices = defaultdict(list)
    for item in trace.slices:
        devices[item.rank, item.device].append(item)
    output = []
    for (rank, device), items in sorted(devices.items()):
        groups = defaultdict(list)
        for item in items:
            groups[group_key(item)].append(item)
        # Sweep group and stream activity, measuring interval unions without a quadratic scan.
        changes = defaultdict(list)
        for item in items:
            changes[item.start_us].append((group_key(item), item.stream, 1))
            changes[item.end_us].append((group_key(item), item.stream, -1))
        active_groups = defaultdict(int)
        active_streams = defaultdict(int)
        exclusive = defaultdict(float)
        concurrent_us = 0.0
        peak_streams = 0
        previous = None
        for time, events in sorted(changes.items()):
            if previous is not None:
                width = time - previous
                present = [key for key, count in active_groups.items() if count > 0]
                streams = sum(count > 0 for count in active_streams.values())
                if len(present) == 1:
                    exclusive[present[0]] += width
                if streams >= 2:
                    concurrent_us += width
                peak_streams = max(peak_streams, streams)
            for key, stream, delta in events:
                active_groups[key] += delta
                active_streams[stream] += delta
            previous = time
        start, end = trace.windows[rank]
        wall = end - start
        work = sum(item.end_us - item.start_us for item in items)
        busy = union_us((item.start_us, item.end_us) for item in items)
        by_kind = {
            kind: [(item.start_us, item.end_us) for item in items if item.kind == kind]
            for kind in ("compute", "communication", "memory")
        }
        compute = union_us(by_kind["compute"])
        communication = union_us(by_kind["communication"])
        overlap = max(
            0.0, compute + communication - union_us(by_kind["compute"] + by_kind["communication"])
        )
        rows = []
        for key, events in groups.items():
            durations = [item.end_us - item.start_us for item in events]
            covered = union_us((item.start_us, item.end_us) for item in events)
            invocations = {item.invocation for item in events if item.invocation is not None}
            rows.append(
                {
                    "operator": key[0],
                    "compute_type": key[1],
                    "signature": key[2],
                    "kind": sorted({item.kind for item in events}),
                    "gpu_event_count": len(events),
                    "operator_invocations": len(invocations)
                    if all(item.invocation for item in events)
                    else None,
                    "grouping": "explicit_operator"
                    if all(item.operator for item in events)
                    else "kernel_name_fallback",
                    "work_us": sum(durations),
                    "work_share": sum(durations) / work,
                    "union_us": covered,
                    "wall_coverage": covered / wall,
                    "exclusive_us": exclusive[key],
                    "exclusive_wall_fraction": exclusive[key] / wall,
                    "gpu_event_duration_us": summarize(durations),
                    "streams": sorted({item.stream for item in events}),
                    "layer_work_us": {
                        layer: sum(
                            item.end_us - item.start_us for item in events if item.layer == layer
                        )
                        for layer in sorted({item.layer for item in events})
                    },
                }
            )
        output.append(
            {
                "rank": rank,
                "device": device,
                "captured_forward_us": wall,
                "gpu_work_sum_us": work,
                "gpu_busy_union_us": busy,
                "gpu_idle_us": max(0.0, wall - busy),
                "compute_union_us": compute,
                "communication_union_us": communication,
                "memory_union_us": union_us(by_kind["memory"]),
                "compute_communication_overlap_us": overlap,
                "communication_without_compute_us": max(0.0, communication - overlap),
                "communication_overlap_fraction": overlap / communication
                if communication
                else None,
                "multi_stream_active_us": concurrent_us,
                "peak_active_streams": peak_streams,
                "stream_busy_us": {
                    stream: union_us(
                        (item.start_us, item.end_us) for item in items if item.stream == stream
                    )
                    for stream in sorted({item.stream for item in items})
                },
                "operators": sorted(rows, key=lambda row: row["exclusive_us"], reverse=True),
            }
        )
    return {
        "devices": output,
        "limitations": [
            "Temporal coverage and exclusivity are not causal critical path or saved time.",
            "GPU event busy union is temporal occupancy, not SM active-cycle utilization.",
            "Shares use GPU work sum; wall coverage can overlap and does not add to 100%.",
            "Without correlation GPU events are kernels, not framework operator invocations.",
            "Collective duration can include waits; it is not pure wire time.",
            "Stream overlap can introduce contention; area alone cannot establish a speedup.",
            "Use local rank windows; do not implicitly align cross-host clocks or add rank times.",
        ],
    }
