from __future__ import annotations

import statistics
from pathlib import Path

from ..contracts import number, read_json
from ..kernel.task import Sources, prepare_task
from ..kernel.validation import validate_record
from ..kernel.workflow import compare_records
from .alignment import read_capture

DEVICE_FIELDS = (
    "captured_forward_us",
    "gpu_work_sum_us",
    "gpu_busy_union_us",
    "gpu_idle_us",
    "compute_union_us",
    "communication_union_us",
    "memory_union_us",
    "mixed_union_us",
    "compute_communication_overlap_us",
    "communication_without_compute_us",
    "multi_stream_active_us",
    "peak_active_streams",
)
GROUP_FIELDS = (
    "work_us",
    "work_share",
    "union_us",
    "wall_coverage",
    "exclusive_us",
    "exclusive_wall_fraction",
    "gpu_event_count",
)
COMPONENT_FIELDS = {
    "compute_union_us",
    "communication_union_us",
    "compute_communication_overlap_us",
    "communication_without_compute_us",
}


def delta(old: float, new: float, *, share: bool = False) -> dict:
    result = {
        "baseline": old,
        "candidate": new,
        "candidate_minus_baseline": new - old,
        "relative_change": (new / old - 1) if old else None,
    }
    if share:
        result["percentage_point_change"] = (new - old) * 100
    return result


def _pair_devices(before: dict, after: dict, complete: bool) -> dict:
    fields = {
        field: delta(before["timeline"][field], after["timeline"][field]) for field in DEVICE_FIELDS
    }
    if not complete:
        fields = {field: None for field in DEVICE_FIELDS}
    component_ok = before["component_split_known"] and after["component_split_known"] and complete
    if not component_ok:
        for field in COMPONENT_FIELDS:
            fields[field] = None
    rows = []
    for key in sorted(set(before["groups"]) | set(after["groups"])):
        old, new = before["groups"].get(key), after["groups"].get(key)
        absent_before = old is None and before["absence_verified"].get(key) is True
        absent_after = new is None and after["absence_verified"].get(key) is True
        matched = (
            complete
            and not key.startswith("unmapped:")
            and (old is not None and new is not None or absent_before or absent_after)
        )
        changes = None
        if matched:
            changes = {
                f: delta(
                    old[f] if old else 0,
                    new[f] if new else 0,
                    share=f in {"work_share", "wall_coverage", "exclusive_wall_fraction"},
                )
                for f in GROUP_FIELDS
            }
        invocations = None
        if (
            matched
            and old
            and new
            and old["invocation_alignment_available"]
            and new["invocation_alignment_available"]
            and old["logical_invocations"].keys() == new["logical_invocations"].keys()
        ):
            invocations = {
                iid: {
                    f: delta(old["logical_invocations"][iid][f], new["logical_invocations"][iid][f])
                    for f in ("work_us", "union_us", "gpu_span_us", "start_relative_us")
                }
                for iid in old["logical_invocations"]
            }
        rows.append(
            {
                "semantic_key": key,
                "status": "aligned" if matched else "unmatched_or_incomplete",
                "baseline": old,
                "candidate": new,
                "deltas": changes,
                "logical_invocation_deltas": invocations,
                "absence_verified": absent_before or absent_after,
                "is_causal_critical_path": False,
            }
        )
    comm = []
    for key in sorted(before["collectives"].keys() | after["collectives"].keys()):
        old, new = before["collectives"].get(key), after["collectives"].get(key)
        matched = (
            component_ok
            and old is not None
            and new is not None
            and (old["descriptor"] == new["descriptor"])
        )
        comm.append(
            {
                "logical_collective_id": key,
                "baseline": old,
                "candidate": new,
                "status": "aligned" if matched else "unmatched_or_payload_group_changed",
                "deltas": {f: delta(old[f], new[f]) for f in ("work_us", "union_us")}
                if matched
                else None,
                "wire_tx_bytes_delta": delta(old["wire_tx_bytes"], new["wire_tx_bytes"])
                if matched and old["wire_tx_bytes"] is not None and new["wire_tx_bytes"] is not None
                else None,
                "pure_wire_time_us": None,
                "rank_arrival_skew_us": None,
            }
        )
    role_deltas = {
        role: delta(before["role_busy_us"][role], after["role_busy_us"][role])
        for role in before["role_busy_us"].keys() & after["role_busy_us"].keys()
    }
    overlap_deltas = {
        roles: delta(before["role_overlap_us"][roles], after["role_overlap_us"][roles])
        for roles in before["role_overlap_us"].keys() & after["role_overlap_us"].keys()
    }
    if not complete:
        role_deltas = {key: None for key in role_deltas}
        overlap_deltas = {key: None for key in overlap_deltas}
    return {
        "physical_device": before["physical_device"],
        "deltas": fields,
        "component_deltas_complete": component_ok,
        "operators": rows,
        "collectives": comm,
        "stream_role_busy_deltas": role_deltas,
        "stream_role_overlap_deltas": overlap_deltas,
        "unmatched_stream_roles": sorted(
            before["role_busy_us"].keys() ^ after["role_busy_us"].keys()
        ),
        "raw_stream_ids_aligned": False,
        "cross_rank_clock_alignment": "not_assumed",
    }


def _native_link(spec: dict | None, data: dict, sources: Sources) -> dict:
    if not spec:
        return {"status": "not_supplied", "accepted": None, "real_gain_established": False}
    if any(p["baseline"]["window"] != p["candidate"]["window"] for p in data["pairs"]):
        raise ValueError("Native link cannot excuse a mismatched profile workload/window")
    bundle_path = sources.path(spec["kernel_bundle"])
    bundle = read_json(bundle_path)
    native_sources = Sources(bundle_path.parent)
    task, prepared, fixed = prepare_task(native_sources.path(bundle["task"]))
    for relative in fixed.manifest:
        native_sources.path((fixed.root / relative).relative_to(native_sources.root).as_posix())
    if not prepared["ready_for_candidate"]:
        raise ValueError("Linked kernel task lacks deep analysis")
    old = data["sides"]["baseline"]
    new = data["sides"]["candidate"]
    if task["kind"] != data["kind"] or any(
        task["comparison"] != side["comparison"] for side in (old, new)
    ):
        raise ValueError("Native kernel evidence differs from the trace comparison environment")
    physical = {
        r["rank"]: r["device_id"]
        for replica in prepared["deployment"]["budget"]["replicas"]
        for r in replica["ranks"]
    }
    if any(
        {r["rank"]: r["physical_device"] for r in side["placement"]} != physical
        or side["topology"] != prepared["deployment"]["axes"]
        for side in (old, new)
    ):
        raise ValueError("Native evidence and trace placement/topology differ")
    conditions = {c["id"]: c["spec"] for c in task["conditions"]}
    if conditions != {c["id"]: c["spec"] for c in data["conditions"]}:
        raise ValueError("Native evidence and trace comparison use different service matrices")
    native_conditions = {c["id"]: c for c in task["conditions"]}
    if any(
        p[side]["window"]["boundary_id"]
        != native_conditions[p["condition_id"]]["forward_boundary_id"]
        or p[side]["window"]["global_new_tokens"]
        != native_conditions[p["condition_id"]]["global_new_tokens_per_sample"]
        for p in data["pairs"]
        for side in ("baseline", "candidate")
    ):
        raise ValueError("Profile window and native prefill boundary/token work differ")
    candidates = [c for c in bundle["candidates"] if c["id"] == spec["candidate_id"]]
    if (
        len(candidates) != 1
        or candidates[0]["revision"] != new["revision"]
        or (
            bundle["baseline"]["revision"] != old["revision"]
            or old["patch_sha256"] != "baseline"
            or candidates[0]["patch_sha256"] != new["patch_sha256"]
        )
    ):
        raise ValueError("Native candidate parent/revision/patch differs from the trace pair")
    baseline = validate_record(task, prepared, bundle["baseline"], native_sources, baseline=True)
    candidate = validate_record(task, prepared, candidates[0], native_sources)
    decision = compare_records(task, baseline, candidate)
    for relative in native_sources.manifest:
        sources.path((native_sources.root / relative).relative_to(sources.root).as_posix())
    return {
        "status": "revalidated_imported_kernel_evidence",
        **decision,
        "task_digest": prepared["task_digest"],
        "real_gain_established": data["kind"] == "measured" and decision["accepted"],
        "limitations": "External producer attestations; no hardware verification runs here.",
    }


def _migration(
    pairs: list[dict],
    rank: str,
    minimum: int,
    stable_fraction: float,
    abs_floor: float,
    fraction: float,
    native: dict,
) -> dict:
    valid = [
        p["devices"][rank] for p in pairs if p["status"] == "comparable" and rank in p["devices"]
    ]
    if len(valid) != len(pairs) or len(valid) < minimum:
        return {
            "rank": rank,
            "status": "insufficient_matched_repeats",
            "confirmed_bottleneck": False,
            "next_action": "collect all-rank matched timeline repeats",
        }
    if any(row["deltas"] is None for pair in valid for row in pair["operators"]):
        return {
            "rank": rank,
            "status": "incomplete_semantic_alignment",
            "confirmed_bottleneck": False,
            "next_action": "supply shape/fusion/absence mapping; do not discard unmatched work",
        }
    groups = {row["semantic_key"] for pair in valid for row in pair["operators"]}
    available = {}
    for key in sorted(groups):
        rows = [
            next((r for r in pair["operators"] if r["semantic_key"] == key), None) for pair in valid
        ]
        if any(row is None or row["deltas"] is None for row in rows):
            continue
        available[key] = rows
    if not available:
        return {"rank": rank, "status": "no_stable_semantic_groups", "confirmed_bottleneck": False}

    def dominant(side: str) -> list[str | None]:
        leaders = []
        for i in range(len(valid)):
            values = {
                key: available[key][i][side]["exclusive_us"] if available[key][i][side] else 0
                for key in available
            }
            best = max(values.values())
            candidates = [key for key, value in values.items() if best - value <= abs_floor]
            leaders.append(candidates[0] if best > abs_floor and len(candidates) == 1 else None)
        return leaders

    before, after = dominant("baseline"), dominant("candidate")
    if None in before or None in after:
        return {
            "rank": rank,
            "status": "no_distinct_exposure_leader",
            "confirmed_bottleneck": False,
            "next_action": "review tied/fully overlapped hotspots and dependencies; keep all work",
        }
    old = max(sorted(set(before)), key=before.count)
    new = max(sorted(set(after)), key=after.count)
    if min(before.count(old), after.count(new)) / len(valid) < stable_fraction:
        return {
            "rank": rank,
            "status": "unstable_dominance",
            "confirmed_bottleneck": False,
            "next_action": "increase matched capture coverage; inspect routing/chunk variation",
        }
    rows = available[new]

    def med(field: str, side: str) -> float:
        return statistics.median(r["deltas"][field][side] for r in rows)

    work_old, work_new = med("work_us", "baseline"), med("work_us", "candidate")
    exposed_old, exposed_new = med("exclusive_us", "baseline"), med("exclusive_us", "candidate")
    share_old, share_new = med("work_share", "baseline"), med("work_share", "candidate")
    work_up = work_new - work_old > max(abs_floor, work_old * fraction)
    exposure_up = exposed_new - exposed_old > max(abs_floor, exposed_old * fraction)
    status = (
        "same_dominant_unit"
        if old == new
        else "absolute_work_increase_hypothesis"
        if work_up
        else "exposure_increase_with_stable_work"
        if exposure_up
        else "relative_share_only"
        if share_new > share_old
        else "dominance_reordering_only"
    )
    return {
        "rank": rank,
        "status": status,
        "baseline_dominant": old,
        "candidate_dominant": new,
        "new_dominant_work_us": delta(work_old, work_new),
        "new_dominant_exclusive_us": delta(exposed_old, exposed_new),
        "new_dominant_work_share": delta(share_old, share_new, share=True),
        "native_evidence_accepted": native["accepted"],
        "confirmed_bottleneck": False,
        "service_speedup_from_trace": None,
        "next_action": "verify dependencies, rank arrivals and matched resource limits",
    }


def build_comparison(path: Path) -> tuple[dict, Sources]:
    data = read_json(path)
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or data.get("kind") not in {"synthetic", "measured"}
    ):
        raise ValueError("Trace comparison needs schema_version=1 and explicit evidence kind")
    sources = Sources(path.parent, reserved={"trace-comparison.json", "manifest.json"})
    sides = data["sides"]
    if set(sides) != {"baseline", "candidate"}:
        raise ValueError("Declare baseline and candidate identities")
    for side in sides.values():
        if not side.get("revision") or not side.get("patch_sha256") or not side.get("comparison"):
            raise ValueError("Both trace revisions need fixed comparison and patch identities")
        required = {
            "architecture",
            "model",
            "weights_digest",
            "image_digest",
            "runtime_revision",
            "tokenizer_digest",
            "dataset_digest",
            "global_workload",
            "cache_format",
        }
        if set(side["comparison"]) != required or any(not v for v in side["comparison"].values()):
            raise ValueError("Trace comparison requires the complete fixed environment identity")
        placement = side["placement"]
        ranks = [r["rank"] for r in placement]
        devices = [r["physical_device"] for r in placement]
        if (
            not ranks
            or len(set(ranks)) != len(ranks)
            or len(set(devices)) != len(devices)
            or any(type(r) is not int or r < 0 for r in ranks)
        ):
            raise ValueError("Declare unique nonnegative rank / physical device bindings")
    matrix = read_json(sources.path(data["matrix_source"]))
    records = [r for r in matrix if r["scope"] == "service"]
    declared = {r["workload_id"]: r for r in records}
    conditions = {c["id"]: c for c in data["conditions"]}
    if any(not isinstance(cid, str) or not cid for cid in conditions):
        raise ValueError("Condition IDs must be nonempty strings")
    if (
        len(declared) != len(records)
        or len(conditions) != len(data["conditions"])
        or set(declared) != set(conditions)
        or not conditions
    ):
        raise ValueError("Trace conditions must cover the complete declared service matrix")
    for cid, condition in conditions.items():
        if declared[cid]["phase"] != "prefill" or declared[cid]["parameters"] != condition["spec"]:
            raise ValueError("Trace workload differs from the testing matrix")
    policy = data["policy"]
    minimum = policy["min_matched_pairs"]
    if type(minimum) is not int or minimum < 3:
        raise ValueError("Migration hypotheses require >=3 predeclared matched pairs")
    fraction = number(policy["change_fraction"], "change fraction")
    floor = number(policy["absolute_floor_us"], "absolute trace-change floor")
    stability = number(policy["stable_dominance_fraction"], "dominance stability", positive=True)
    if not 0 < stability <= 1 or fraction >= 1:
        raise ValueError("Trace comparison fractions are invalid")
    rules = data.get("semantic_mappings", [])
    if len({r["id"] for r in rules}) != len(rules):
        raise ValueError("Semantic mapping IDs must be unique")
    global_issues = []
    if any(
        isinstance(value, str)
        and value in {"unknown", "TO_BE_SPECIFIED", "REQUIRED", "requires_live_verification"}
        for side in sides.values()
        for value in side["comparison"].values()
    ):
        global_issues.append("environment identity contains unresolved placeholders")
    if sides["baseline"]["comparison"] != sides["candidate"]["comparison"]:
        global_issues.append("model/architecture/image/runtime/workload/format identity differs")
    if (
        sides["baseline"]["placement"] != sides["candidate"]["placement"]
        or sides["baseline"]["topology"] != sides["candidate"]["topology"]
    ):
        global_issues.append(
            "topology or placement changed; rank-local work needs deployment normalization"
        )
    native = {"status": "not_supplied", "accepted": None, "real_gain_established": False}
    try:
        if data.get("native"):
            native = _native_link(data["native"], data, sources)
    except (OSError, ValueError, KeyError, TypeError) as error:
        native = {
            "status": "invalid_or_incomparable_native_link",
            "accepted": None,
            "real_gain_established": False,
            "reason": str(error),
        }
    pairs, seen, capture_windows = [], set(), set()
    for pair in data["pairs"]:
        if (
            not isinstance(pair["id"], str)
            or not pair["id"]
            or pair["id"] in seen
            or pair["condition_id"] not in conditions
        ):
            raise ValueError("Pair IDs must be unique and reference a declared condition")
        seen.add(pair["id"])
        outputs, issues = {}, list(global_issues)
        for side in ("baseline", "candidate"):
            spec = pair[side]
            if spec["window"]["phase"] != "prefill":
                issues.append("non-prefill window")
            for field in (
                "boundary_id",
                "logical_step",
                "request_manifest_digest",
                "chunk_plan_digest",
                "global_new_tokens",
                "actual_cached_tokens",
                "scope",
                "head_boundary",
            ):
                if field not in spec["window"]:
                    raise ValueError("Incomplete semantic window contract: " + field)
                if spec["window"][field] is None or spec["window"][field] == "unknown":
                    issues.append("unknown semantic window metadata: " + field)
            for field in ("global_new_tokens", "actual_cached_tokens"):
                value = spec["window"][field]
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(
                        "Window token counts must be nonnegative integers or explicit null"
                    )
            protocol_fields = (
                "tool",
                "version",
                "options",
                "clock_domain_policy",
                "replay_mode",
                "cache_control",
                "capture_selection",
            )
            if any(not spec["protocol"].get(k) for k in protocol_fields):
                raise ValueError(
                    "Trace protocol needs tool/options/clock/replay/cache/selection provenance"
                )
            if any(spec["protocol"][k] != pair["baseline"]["protocol"][k] for k in protocol_fields):
                issues.append(
                    "profiler/version/options/clock/replay/cache/selection protocol differs"
                )
            placement = {r["rank"]: r for r in sides[side]["placement"]}
            for row in spec["ranks"]:
                if row["rank"] not in placement or any(
                    row.get(k) != placement[row["rank"]][k] for k in ("physical_device", "device")
                ):
                    issues.append(side + ": rank/device differs from frozen physical placement")
            expected = {"kind": data["kind"], **sides[side]}
            try:
                outputs[side] = read_capture(spec, expected, list(placement), side, rules, sources)
            except (OSError, ValueError, KeyError, TypeError) as error:
                if "two semantic mapping" in str(error):
                    raise
                outputs[side] = {
                    "devices": {},
                    "normalized_events": [],
                    "issues": [str(error)],
                    "semantic_groups": [],
                }
            issues += [side + ": " + issue for issue in outputs[side]["issues"]]
            for rank, device in outputs[side]["devices"].items():
                key = (
                    side,
                    rank,
                    device["capture_sha256"],
                    device["window_start_us"],
                    device["window_end_us"],
                )
                if key in capture_windows:
                    issues.append(side + ": capture window reused across matched pair IDs")
                capture_windows.add(key)
        if pair["baseline"]["window"] != pair["candidate"]["window"]:
            issues.append("request/step/chunk/token/cache/head boundary differs")
        incompatible = bool(global_issues) or any(
            "differs" in issue or "non-prefill" in issue for issue in issues
        )
        status = "incomparable" if incompatible else "partial" if issues else "comparable"
        devices = {}
        if not incompatible:
            for rank in (
                outputs["baseline"]["devices"].keys() & outputs["candidate"]["devices"].keys()
            ):
                devices[rank] = _pair_devices(
                    outputs["baseline"]["devices"][rank],
                    outputs["candidate"]["devices"][rank],
                    not issues,
                )
        pairs.append(
            {
                "id": pair["id"],
                "condition_id": pair["condition_id"],
                "status": status,
                "window_contract": pair["baseline"]["window"],
                "issues": sorted(set(issues)),
                "devices": devices,
                "captures": outputs,
                "profile_delta_is_service_gain": False,
            }
        )
    migrations, coverage = [], []
    for cid in conditions:
        selected = [p for p in pairs if p["condition_id"] == cid]
        complete = len(selected) >= minimum and all(p["status"] == "comparable" for p in selected)
        coverage.append(
            {
                "condition_id": cid,
                "declared_pairs": len(selected),
                "complete_matched_repeats": complete,
            }
        )
        for placement in sides["baseline"]["placement"]:
            migrations.append(
                {
                    "condition_id": cid,
                    **_migration(
                        selected,
                        str(placement["rank"]),
                        minimum,
                        stability,
                        floor,
                        fraction,
                        native,
                    ),
                }
            )
    next_tasks = [
        {
            "track": "measure_first",
            "condition_id": p["condition_id"],
            "pair_id": p["id"],
            "status": "needs_evidence",
            "reason": issue,
        }
        for p in pairs
        for issue in p["issues"]
    ]
    if native["accepted"] is None:
        next_tasks.append(
            {
                "track": "native_acceptance",
                "status": "needs_evidence",
                "reason": "trace timing cannot establish precision or forward/service gain",
            }
        )
    next_tasks += [
        {
            "track": "measure_first"
            if "candidate_dominant" not in m
            else "dependency_overlap_review"
            if "exposure" in m["status"] or m["status"] == "relative_share_only"
            else "resource_model_review",
            "condition_id": m["condition_id"],
            "rank": m["rank"],
            "status": "needs_deep_analysis",
            "reason": m["status"],
            "target": m.get("candidate_dominant"),
            "confirmed_bottleneck": False,
        }
        for m in migrations
    ]
    return {
        "schema_version": 1,
        "kind": data["kind"],
        "sides": sides,
        "pairs": pairs,
        "coverage": coverage,
        "native_evidence": native,
        "bottleneck_migration": migrations,
        "next_tasks": next_tasks,
        "status": "comparable"
        if coverage and all(c["complete_matched_repeats"] for c in coverage)
        else "insufficient_or_incomparable_evidence",
        "causal_critical_path": None,
        "service_speedup_from_trace": None,
        "auto_deploy": False,
        "limitations": [
            "Rank-local windows only; no cross-host clock synchronization or critical DAG.",
            "Work sums, interval coverage and exclusive exposure are different quantities.",
            "Shares are relative; a larger share does not prove absolute slowdown.",
            "Fused mixed events do not disclose their compute/communication components.",
            "Imported source/metadata attestations are not hardware authenticity checks.",
        ],
    }, sources
