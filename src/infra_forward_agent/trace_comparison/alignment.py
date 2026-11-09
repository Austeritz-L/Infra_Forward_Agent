from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace

from ..analysis import union_us
from ..contracts import number
from ..kernel.task import Sources
from ..performance.timeline import analyze_timeline
from ..trace import Slice, Trace, load_chrome


def semantic_key(item: Slice) -> str | None:
    if (
        not item.operator
        or item.signature == "unknown"
        or item.compute_type == "unknown"
        or item.layer == "unknown"
    ):
        return None
    return json.dumps([item.operator, item.compute_type, item.signature, item.layer])


def _matches(item: Slice, selector: dict) -> bool:
    allowed = {"operator", "signature", "compute_type", "layer", "name"}
    if not selector or set(selector) - allowed:
        raise ValueError("Semantic selectors use exact operator/signature/type/layer/name fields")
    return all(getattr(item, key) == value for key, value in selector.items())


def read_capture(
    spec: dict, expected: dict, ranks: list[int], side: str, mappings: list[dict], sources: Sources
) -> dict:
    issues, devices, normalized, all_groups = [], {}, [], set()
    if spec["identity"] != expected:
        issues.append("capture identity differs from fixed kind/environment/revision/patch")
    protocol = spec["protocol"]
    for flag in (
        "complete_window",
        "no_dropped_events",
        "mapping_verified",
        "overhead_checked",
        "representative_selection_frozen",
    ):
        if protocol.get(flag) is not True:
            issues.append("capture protocol missing: " + flag)
    if protocol.get("mode") != "timeline" or protocol.get("profiled") is not True:
        issues.append("only a complete timeline pass is comparable; counter replay is separate")
    sources.path(protocol["source"])
    specs = spec["ranks"]
    ids = [row["rank"] for row in specs]
    if len(ids) != len(set(ids)) or set(ids) - set(ranks) or any(type(r) is not int for r in ids):
        raise ValueError("Capture rank entries must be unique expected ranks")
    for rank in set(ranks) - set(ids):
        issues.append(f"missing rank {rank}")
    for row in specs:
        rank = row["rank"]
        try:
            trace = load_chrome(
                sources.path(row["path"]),
                rank=rank,
                marker=row.get("marker", "forward"),
                window_selector=row.get("selector"),
                selected_rank_only=True,
            )
            if set(trace.windows) != {rank}:
                raise ValueError("Selected rank/window is missing")
            if trace.clipped_events:
                issues.append(f"rank {rank}: clipped GPU events at the selected boundary")
            if trace.unsupported_gpu_events:
                issues.append(f"rank {rank}: unsupported GPU event phases require normalization")
            physical = row["physical_device"]
            if not row.get("clock_id"):
                issues.append(f"rank {rank}: missing timestamp clock provenance")
            observed = {item.device for item in trace.slices}
            if observed != {str(row["device"])}:
                raise ValueError("Rank/device binding differs from the capture manifest")
            # Category-default compute and name-heuristic communication remain unverified.
            items = [
                replace(item, kind="unknown")
                if item.classification != "explicit" and item.kind != "memory"
                else item
                for item in trace.slices
            ]
            fingerprints = [(i.name, i.stream, i.start_us, i.end_us, i.invocation) for i in items]
            if len(set(fingerprints)) != len(fingerprints):
                issues.append(f"rank {rank}: duplicate GPU interval exports")
            selected_step = row.get("selector", {}).get("ifa_step")
            if selected_step is not None and any(
                i.context.get("ifa_step") != selected_step for i in items
            ):
                issues.append(f"rank {rank}: unbound/background step events in the selected window")
            assigned = {}
            grouped = defaultdict(list)
            proof = {}
            for rule in mappings:
                key = "mapped:" + rule["id"]
                if rule.get("verified") is not True or not rule.get("semantic_contract"):
                    raise ValueError("Fusion/rename mappings require a verified semantic contract")
                sources.path(rule["source"])
                proof[key] = rule.get(side + "_absence_verified", False)
                selected = [
                    i
                    for i, item in enumerate(items)
                    if any(_matches(item, selector) for selector in rule[side])
                ]
                if any(i in assigned for i in selected):
                    raise ValueError("A GPU event cannot belong to two semantic mapping groups")
                for i in selected:
                    assigned[i] = key
            unknown = []
            for i, item in enumerate(items):
                key = assigned.get(i) or semantic_key(item)
                if key is None:
                    unknown.append(item.name)
                    key = "unmapped:" + item.name
                grouped[key].append(item)
            if unknown:
                issues.append(f"rank {rank}: unmapped operator/shape/layer events")
            # Canonical grouping computes exclusive exposure after fusion/rename normalization.
            canonical = [
                replace(item, operator=key, signature="semantic-unit", compute_type="semantic")
                for key, group in grouped.items()
                for item in group
            ]
            timeline = analyze_timeline(
                Trace(canonical, trace.windows, trace.ignored_events, trace.clipped_events)
            )["devices"][0]
            if timeline["unknown_kind_union_us"]:
                issues.append(f"rank {rank}: unknown compute/communication classification")
            groups = {}
            start, finish = trace.windows[rank]
            for group in timeline["operators"]:
                key = group["operator"]
                events = grouped[key]
                invocations = defaultdict(list)
                for item in events:
                    invocation = item.context.get("ifa_logical_invocation")
                    if invocation is not None:
                        invocations[str(invocation)].append(item)
                groups[key] = {
                    **group,
                    "semantic_key": key,
                    "invocation_alignment_available": all(
                        item.context.get("ifa_logical_invocation") is not None for item in events
                    ),
                    "logical_invocations": {
                        iid: {
                            "work_us": sum(i.end_us - i.start_us for i in values),
                            "union_us": union_us((i.start_us, i.end_us) for i in values),
                            "gpu_span_us": max(i.end_us for i in values)
                            - min(i.start_us for i in values),
                            "start_relative_us": min(i.start_us for i in values) - start,
                        }
                        for iid, values in invocations.items()
                    },
                }
                all_groups.add(key)
            stream_roles = row.get("stream_roles", {})
            role_intervals = defaultdict(list)
            for item in items:
                role = stream_roles.get(item.stream)
                if role is None:
                    issues.append(f"rank {rank}: unmapped stream role {item.stream}")
                else:
                    role_intervals[role].append((item.start_us, item.end_us))
            role_busy = {role: union_us(intervals) for role, intervals in role_intervals.items()}
            role_overlap = {}
            for a in sorted(role_intervals):
                for b in sorted(role_intervals):
                    if a < b:
                        role_overlap[json.dumps([a, b])] = max(
                            0.0,
                            role_busy[a]
                            + role_busy[b]
                            - union_us(role_intervals[a] + role_intervals[b]),
                        )
            collectives = defaultdict(list)
            for item in items:
                if item.kind != "communication":
                    continue
                context = item.context
                required = (
                    "ifa_collective_id",
                    "ifa_collective_group",
                    "ifa_collective_op",
                    "ifa_payload_bytes",
                    "ifa_collective_dtype",
                )
                if any(context.get(k) is None for k in required):
                    issues.append(f"rank {rank}: collective identity/payload metadata missing")
                    continue
                key = str(context["ifa_collective_id"])
                collectives[key].append(item)
            collective_rows = {}
            for key, events in collectives.items():
                descriptors = [
                    {
                        k: item.context[k]
                        for k in (
                            "ifa_collective_group",
                            "ifa_collective_op",
                            "ifa_payload_bytes",
                            "ifa_collective_dtype",
                        )
                    }
                    for item in events
                ]
                if any(d != descriptors[0] for d in descriptors):
                    raise ValueError(
                        "Collective logical ID was reused for different payloads/groups"
                    )
                payload = number(descriptors[0]["ifa_payload_bytes"], "logical collective payload")
                wires = [item.context.get("ifa_wire_tx_bytes") for item in events]
                wire_known = all(
                    i.context.get("ifa_wire_byte_convention") == "event_tx_bytes" for i in events
                )
                collective_rows[key] = {
                    "descriptor": descriptors[0],
                    "payload_bytes": payload,
                    "work_us": sum(i.end_us - i.start_us for i in events),
                    "union_us": union_us((i.start_us, i.end_us) for i in events),
                    "wire_tx_bytes": sum(number(v, "per-rank TX wire bytes") for v in wires)
                    if wire_known and all(v is not None for v in wires)
                    else None,
                    "wire_time_us": None,
                    "cross_rank_arrival_skew_us": None,
                    "wire_byte_convention": "event_tx_bytes" if wire_known else "unknown",
                }
            devices[str(rank)] = {
                "physical_device": physical,
                "device": str(row["device"]),
                "capture_sha256": sources.manifest[row["path"]],
                "ignored_events": trace.ignored_events,
                "clipped_gpu_events": trace.clipped_events,
                "unsupported_gpu_events": trace.unsupported_gpu_events,
                "window_start_us": start,
                "window_end_us": finish,
                "timeline": timeline,
                "groups": groups,
                "absence_verified": proof,
                "collectives": collective_rows,
                "role_busy_us": role_busy,
                "role_overlap_us": role_overlap,
                "component_split_known": not timeline["mixed_union_us"]
                and not timeline["unknown_kind_union_us"],
            }
            normalized += [
                {
                    "rank": rank,
                    "physical_device": physical,
                    "stream": item.stream,
                    "start_relative_us": item.start_us - start,
                    "end_relative_us": item.end_us - start,
                    "name": item.name,
                    "operator": item.operator,
                    "signature": item.signature,
                    "layer": item.layer,
                    "kind": item.kind,
                    "context": item.context,
                }
                for item in items
            ]
        except (OSError, ValueError, KeyError, TypeError) as error:
            if "two semantic mapping" in str(error):
                raise
            issues.append(f"rank {rank}: invalid/missing capture: {error}")
    return {
        "devices": devices,
        "issues": sorted(set(issues)),
        "normalized_events": normalized,
        "semantic_groups": sorted(all_groups),
    }
