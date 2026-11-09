"""Offline selection on a fixed physical budget; evidence attestations are not live validation."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from ..contracts import number, read_json
from ..performance.statistics import service_statistics, summarize
from .communication import estimate_communication
from .memory import budget_candidate, integer


def layout_signature(candidate: dict, devices: list[dict], target: dict) -> str:
    payload = {
        "axes": candidate["axes"],
        "replicas": candidate["replicas"],
        "devices": devices,
        "target": target,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def artifact(root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or ".." in Path(relative).parts
        or Path(relative).as_posix() != relative
    ):
        raise ValueError("Use canonical relative artifact paths without parent traversal")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or Path(relative).is_absolute():
        raise ValueError("Deployment sources must be existing files inside the bundle directory")
    if relative in {"deployment.json", "manifest.json"}:
        raise ValueError("Reserved deployment snapshot filename")
    return path


def _geometric(values: list[tuple[float, float]]) -> float:
    return math.exp(
        sum(weight * math.log(value) for value, weight in values)
        / sum(weight for _, weight in values)
    )


def select_deployment(path: Path) -> dict:
    data = read_json(path)
    if data.get("schema_version") != 1 or data.get("kind") not in {"synthetic", "measured"}:
        raise ValueError("Deployment bundle requires schema_version=1 and synthetic/measured kind")
    root = path.parent.resolve()
    comparison = data["comparison"]
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
    if set(comparison) != required or any(not value for value in comparison.values()):
        raise ValueError("Provide the complete fixed comparison identity")
    target = data["target"]
    context = integer(target["context_tokens"], "total context", 2)
    integer(target["output_reserve_tokens"], "output reserve", 1)
    if target["output_reserve_tokens"] >= context:
        raise ValueError("Output reserve must be smaller than total context")
    native_limit = integer(target["model_native_limit_tokens"], "native model context limit", 1)
    devices = {item["id"]: item for item in data["devices"]}
    if not devices or len(devices) != len(data["devices"]):
        raise ValueError("Physical device IDs must be nonempty and unique")
    for device in devices.values():
        integer(device["memory_bytes"], "physical memory", 1)
        if not device.get("source"):
            raise ValueError("Device inventory needs a retained source")
    sources = set()

    def retain(relative: str) -> None:
        artifact(root, relative)
        sources.add(relative)

    def load(relative: str, identity: dict) -> dict:
        retain(relative)
        report = read_json(artifact(root, relative))
        if report.get("identity") != identity:
            raise ValueError("Evidence kind/comparison/candidate/layout/condition identity differs")
        return report

    for device in devices.values():
        retain(device["source"])
    retain(target["model_config_source"])
    policy = data["policy"]
    minimum = integer(policy["min_context_slots"], "minimum full-context slots", 1)
    rounds = policy["round_ids"]
    if (
        not isinstance(rounds, list)
        or len(rounds) < 3
        or len(set(rounds)) != len(rounds)
        or any(not isinstance(item, str) or not item for item in rounds)
    ):
        raise ValueError("Require >=3 explicit paired independent round IDs")
    max_cv = number(policy["max_round_cv"], "round noise", positive=True)
    latency_stat = policy["latency_stat"]
    performance_metric = policy.get("performance_metric", "throughput")
    if performance_metric not in {"throughput", "latency"}:
        raise ValueError("performance_metric must be throughput or latency")
    if latency_stat not in {"p50", "p95", "p99"}:
        raise ValueError("latency_stat must be p50, p95 or p99")
    conditions = {item["id"]: item for item in data["conditions"]}
    if not conditions or len(conditions) != len(data["conditions"]):
        raise ValueError("Comparison conditions must be nonempty and unique")
    if not any(
        item["context_tokens"] == context and item["cache_profile"] == "cold-miss"
        for item in conditions.values()
    ):
        raise ValueError("Include a cold-miss condition at the complete target context")
    for item in conditions.values():
        if integer(item["context_tokens"], "condition context", 2) > context:
            raise ValueError("Condition exceeds the budgeted context")
        integer(item["concurrency"], "global client concurrency", 1)
        if not item.get("forward_boundary_id"):
            raise ValueError("Each condition needs a fixed complete prefill forward boundary")
        if number(item.get("cache_fraction_tolerance", 0.02), "cache tolerance") > 1:
            raise ValueError("cache_fraction_tolerance must be in [0,1]")
        for key in ("weight", "throughput_reference", "ttft_reference_us"):
            number(item[key], key, positive=True)
        slo = item["slo"]
        for key in ("min_new_tokens_per_s", "max_ttft_us", "max_forward_us"):
            number(slo[key], key, positive=True)

    if not isinstance(data["candidates"], list) or not data["candidates"]:
        raise ValueError("Provide at least one explicit candidate")
    rows = []
    identifiers = set()
    for candidate in data["candidates"]:
        name = candidate["id"]
        if not isinstance(name, str) or not name or name in identifiers:
            raise ValueError("Candidate IDs must be nonempty and unique")
        identifiers.add(name)
        axes = candidate["axes"]
        if set(axes) != {"tp", "ep", "pp", "dp", "cp"}:
            raise ValueError("Declare TP/EP/PP/DP/CP; actual rank placement is authoritative")
        for axis, size in axes.items():
            integer(size, axis, 1)
        budget = budget_candidate(candidate, devices, context)
        row = {
            "id": name,
            "axes": axes,
            "budget": budget,
            "reasons": [],
            "conditions": [],
            "communication": estimate_communication(candidate.get("communication", [])),
            "verified_context_slots": None,
            "reported_context_slots": None,
            "eligible": False,
        }
        for phase in row["communication"]:
            retain(phase["source"])
        for replica in candidate["replicas"]:
            for rank in replica["ranks"]:
                retain(rank["layout_source"])
        rows.append(row)
        reasons = row["reasons"]
        if not budget["single_request_fits_a_replica"]:
            reasons.append("analytical budget cannot fit one complete request on any replica")
        identity = {
            "kind": data["kind"],
            "comparison": comparison,
            "candidate_id": name,
            "layout_signature": layout_signature(candidate, data["devices"], target),
        }
        if not candidate.get("qualification"):
            reasons.append("missing exact-context backend/quality/memory qualification")
            continue
        proof = load(candidate["qualification"], identity)
        if set(proof.get("qualified_round_ids", [])) != set(rounds) or set(
            proof.get("qualified_condition_ids", [])
        ) != set(conditions):
            reasons.append("precision qualification must bind every measured condition and round")
        evidence_sources = proof.get("evidence_sources", [])
        if not evidence_sources:
            reasons.append("missing retained qualification raw sources")
        for source in evidence_sources:
            retain(source)
        for flag in (
            "backend_combination_verified",
            "context_quality_passed",
            "precision_before_passed",
            "precision_after_passed",
            "precision_timing_order_verified",
        ):
            if proof.get(flag) is not True:
                reasons.append("qualification missing or failed: " + flag)
        if proof.get("validated_context_tokens") != context:
            reasons.append("quality/precision qualification must cover the exact target context")
        if context > native_limit and proof.get("context_extension_validated") is not True:
            reasons.append("target exceeds native model limit without validated extension")
        if integer(proof.get("runtime_context_limit_tokens", 0), "runtime limit") < context:
            reasons.append("runtime context limit is below target")
        memory = proof.get("memory", {})
        for flag in (
            "all_lifecycle_phases_covered",
            "prefill_peak_covered",
            "runtime_admission_verified",
            "simultaneous_active_contexts_verified",
            "no_eviction_or_recompute",
            "unprofiled",
        ):
            if memory.get(flag) is not True:
                reasons.append("memory qualification missing: " + flag)
        if (
            memory.get("context_tokens") != context
            or memory.get("cache_state") != "independent_full_contexts"
        ):
            reasons.append("memory must validate independent full contexts, not only cache hits")
        slots = memory.get("replica_slots", {})
        expected_replicas = {replica["id"] for replica in budget["replicas"]}
        if set(slots) != expected_replicas:
            reasons.append("memory admission must cover every replica")
        else:
            for replica in budget["replicas"]:
                count = integer(slots[replica["id"]], "validated replica slots")
                if count > replica["estimated_context_slots"]:
                    reasons.append(
                        "measured admission exceeds budget; reconcile cache layout first"
                    )
            row["reported_context_slots"] = sum(integer(value, "slots") for value in slots.values())
            if row["reported_context_slots"] < minimum:
                reasons.append("validated resident capacity below minimum")
            capacity_case = conditions.get(memory.get("capacity_condition_id"))
            if (
                capacity_case is None
                or capacity_case["context_tokens"] != context
                or capacity_case["cache_profile"] != "cold-miss"
                or capacity_case["concurrency"] < row["reported_context_slots"]
            ):
                reasons.append(
                    "capacity needs a full-context cold service SLO at resident occupancy"
                )
        peaks = memory.get("rank_peak_bytes", {})
        ranks = [rank for replica in budget["replicas"] for rank in replica["ranks"]]
        if set(peaks) != {str(rank["rank"]) for rank in ranks}:
            reasons.append("memory peak evidence must cover every rank")
        else:
            for rank in ranks:
                peak = integer(peaks[str(rank["rank"])], "measured rank peak", 1)
                if (
                    peak + rank["safety_reserve_bytes"] > rank["device_bytes"]
                    or peak < rank["non_kv_total_bytes"]
                ):
                    reasons.append("rank peak violates device reserve or fixed allocations")
        observations = candidate.get("measurements", [])
        if {item["condition_id"] for item in observations} != set(conditions):
            reasons.append("service measurements must cover every comparison condition")
        seen = set()
        for observation in observations:
            condition_id = observation["condition_id"]
            if condition_id in seen or condition_id not in conditions:
                raise ValueError("Duplicate or unknown measured condition")
            seen.add(condition_id)
            condition = conditions[condition_id]
            repeat = observation["rounds"]
            if len(repeat) != len(rounds) or {trial["id"] for trial in repeat} != set(rounds):
                reasons.append("missing matched independent rounds: " + condition_id)
                continue
            throughputs, ttfts, forwards, native_services = [], [], [], []
            for trial in repeat:
                trial_identity = {**identity, "condition": condition, "round_id": trial["id"]}
                service = load(trial["service"], trial_identity)
                stats = service_statistics(service)
                if stats["client_observed_peak_inflight"] != condition["concurrency"]:
                    reasons.append("client in-flight concurrency differs: " + condition_id)
                if (
                    service.get("actual_cache_profile_verified") is not True
                    or service.get("actual_concurrency_verified") is not True
                ):
                    reasons.append(
                        "missing runtime cache/concurrency verification: " + condition_id
                    )
                forward = load(trial["forward"], trial_identity)
                if (
                    forward.get("profiled") is not False
                    or forward.get("all_streams_joined") is not True
                    or set(forward.get("rank_coverage", [])) != {r["rank"] for r in ranks}
                    or forward.get("scope") != "coordinated_prefill_forward"
                    or forward.get("boundary_id") != condition["forward_boundary_id"]
                ):
                    raise ValueError(
                        "Forward timing requires a coordinated native all-rank boundary"
                    )
                forward_stats = summarize(forward["samples_us"])
                if (
                    not stats["request_integrity_valid"]
                    or stats["ttft_us"] is None
                    or (
                        stats["timed_offered_window"] is not None
                        and not stats["timed_offered_window"]["window_completed"]
                    )
                ):
                    reasons.append("failed/incomplete service cohort: " + condition_id)
                    continue
                for request in service["requests"]:
                    if (
                        request["input_tokens"] + target["output_reserve_tokens"]
                        != condition["context_tokens"]
                    ):
                        raise ValueError("Actual token lengths differ from the selected condition")
                    if condition["cache_profile"] == "cold-miss" and request["cached_tokens"] != 0:
                        reasons.append("cold-miss counters show reused tokens: " + condition_id)
                expected_hit = {
                    "cold-miss": 0,
                    "partial-50": 0.5,
                    "partial-90": 0.9,
                    "warm-full-prefix": 1,
                }.get(condition["cache_profile"])
                if expected_hit is not None and abs(
                    stats["cached_token_fraction"] - expected_hit
                ) > condition.get("cache_fraction_tolerance", 0.02):
                    reasons.append("actual cached fraction differs from target: " + condition_id)
                for stat in (stats["ttft_us"], forward_stats):
                    if latency_stat != "p50" and not stat[latency_stat + "_sample_eligible"]:
                        reasons.append("insufficient tail samples: " + condition_id)
                throughputs.append(stats["new_prefill_tokens_per_s"])
                ttfts.append(stats["ttft_us"][latency_stat])
                forwards.append(forward_stats[latency_stat])
                native_services.append(stats)
            if len(throughputs) != len(rounds):
                continue
            if min(throughputs) <= 0:
                reasons.append(
                    "no positive new-prefill work; use a separate cache-only study: " + condition_id
                )
                continue
            cv = statistics.stdev(throughputs) / statistics.mean(throughputs)
            if any(
                statistics.stdev(values) / statistics.mean(values) > max_cv
                for values in (throughputs, ttfts, forwards)
            ):
                reasons.append("unstable independent-round throughput or latency: " + condition_id)
            throughput = statistics.median(throughputs)
            ttft, forward_time = max(ttfts), max(forwards)
            slo = condition["slo"]
            if (
                min(throughputs) < slo["min_new_tokens_per_s"]
                or ttft > slo["max_ttft_us"]
                or forward_time > slo["max_forward_us"]
            ):
                reasons.append("per-condition native service/forward SLO failed: " + condition_id)
            row["conditions"].append(
                {
                    "id": condition_id,
                    "throughput": throughput,
                    "throughput_unit": "global_new_prefill_tokens_per_s",
                    "logical_input_tokens_per_s_median": statistics.median(
                        s["logical_input_tokens_per_s"] for s in native_services
                    ),
                    "requests_per_s_median": statistics.median(
                        s["requests_per_s"] for s in native_services
                    ),
                    "cached_token_fraction_range": [
                        min(s["cached_token_fraction"] for s in native_services),
                        max(s["cached_token_fraction"] for s in native_services),
                    ],
                    "worst_round_ttft_us": ttft,
                    "worst_round_forward_us": forward_time,
                    "throughput_round_cv": cv,
                    "throughput_ratio": throughput / condition["throughput_reference"],
                    "latency_ratio": condition["ttft_reference_us"] / ttft,
                    "weight": condition["weight"],
                }
            )
        row["eligible"] = not reasons
        if row["eligible"]:
            row["verified_context_slots"] = row["reported_context_slots"]
            row["qualified_replica_slots"] = dict(memory["replica_slots"])
            row["throughput_score"] = _geometric(
                [(c["throughput_ratio"], c["weight"]) for c in row["conditions"]]
            )
            row["latency_score"] = _geometric(
                [(c["latency_ratio"], c["weight"]) for c in row["conditions"]]
            )
    qualified = [row for row in rows if row["eligible"]]

    def dominates(a: dict, b: dict) -> bool:
        av = (a["verified_context_slots"], a["throughput_score"], a["latency_score"])
        bv = (b["verified_context_slots"], b["throughput_score"], b["latency_score"])
        return all(x >= y for x, y in zip(av, bv)) and any(x > y for x, y in zip(av, bv))

    frontier = [row for row in qualified if not any(dominates(other, row) for other in qualified)]
    weights = policy["balanced_weights"]
    if set(weights) != {"capacity", "throughput", "latency"}:
        raise ValueError("Declare capacity/throughput/latency balanced weights")
    for value in weights.values():
        number(value, "balanced weight", positive=True)
    for row in qualified:
        row["balanced_score"] = _geometric(
            [
                (
                    row["verified_context_slots"]
                    / max(q["verified_context_slots"] for q in qualified),
                    weights["capacity"],
                ),
                (
                    row["throughput_score"] / max(q["throughput_score"] for q in qualified),
                    weights["throughput"],
                ),
                (
                    row["latency_score"] / max(q["latency_score"] for q in qualified),
                    weights["latency"],
                ),
            ]
        )
    rankings = {
        "capacity": sorted(
            frontier,
            key=lambda r: (
                -r["verified_context_slots"],
                -r["throughput_score"],
                -r["latency_score"],
                r["id"],
            ),
        ),
        "performance": sorted(
            frontier,
            key=lambda r: (
                -r[performance_metric + "_score"],
                -r["latency_score"],
                -r["verified_context_slots"],
                r["id"],
            ),
        ),
        "balanced": sorted(frontier, key=lambda r: (-r["balanced_score"], r["id"])),
    }
    return {
        "schema_version": 1,
        "kind": data["kind"],
        "comparison": comparison,
        "target": target,
        "candidates": rows,
        "pareto_frontier": [row["id"] for row in frontier],
        "rankings": {key: [row["id"] for row in value] for key, value in rankings.items()},
        "recommendations": {
            key: value[0]["id"] if value and data["kind"] == "measured" else None
            for key, value in rankings.items()
        },
        "auto_deploy": False,
        "artifact_sources": sorted(sources),
        "limitations": [
            "Offline imported attestations; no hardware validation is performed here.",
            "Synthetic rankings demonstrate policy only and never select a real deployment.",
            "Ranked scores do not establish statistically significant speedup.",
            "No active offload capacity credit; retained prefixes are a separate budget.",
            "Axes may overlap rank groups; never multiply TP/EP/PP/DP/CP blindly.",
        ],
    }
