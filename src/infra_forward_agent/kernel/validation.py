from __future__ import annotations

import hashlib

from ..contracts import number
from ..performance.statistics import service_statistics, summarize
from .task import Sources, check_diagnosis, digest, inspect_performance, performance_identity


def bound_report(sources: Sources, relative: str, identity: dict) -> dict:
    report = sources.read(relative)
    if report.get("identity") != identity:
        raise ValueError("Kernel evidence kind/task/revision/patch identity differs")
    if report.get("errors") != []:
        raise ValueError("Kernel evidence must explicitly report no errors")
    return report


def _precision(report: dict, task: dict) -> None:
    if set(report.get("modes_verified", [])) != set(task["validation_modes"]):
        raise ValueError("Precision did not cover the frozen stream/graph/repeat/edge modes")
    expected = {}
    for condition in task["conditions"]:
        for case in condition["operator_cases"]:
            expected[("operator", condition["id"], case["rank"], case["id"])] = (
                digest(case),
                case["relative_l2_limit"],
            )
        for rank in task["ranks"]:
            expected[("forward", condition["id"], rank, "")] = (
                digest(condition["forward_reference"]),
                condition["forward_relative_l2_limit"],
            )
        expected[("service", condition["id"], None, "")] = (
            digest(condition["forward_reference"]),
            0,
        )
    actual = {}
    for row in report["results"]:
        key = (row["scope"], row["condition_id"], row["rank"], row.get("case_id", ""))
        if key in actual or key not in expected:
            raise ValueError("Duplicate or unexpected precision result")
        actual[key] = row
        contract, limit = expected[key]
        if row.get("contract_digest") != contract:
            raise ValueError("Precision tolerance/reference contract differs")
        if (
            type(row.get("checked_elements")) is not int
            or row["checked_elements"] < 1
            or any(
                type(row.get(field)) is not int or row[field] != 0
                for field in ("bad_elements", "nonfinite_elements")
            )
        ):
            raise ValueError("Precision failed elementwise/finite/output-parity validation")
        if number(row["relative_l2"], "relative L2 error") > limit:
            raise ValueError("Precision exceeded frozen relative L2 tolerance")
        number(row["max_abs_error"], "max absolute error")
    if actual.keys() != expected.keys():
        raise ValueError("Precision must cover all conditions, operator cases and forward ranks")


def validate_record(
    task: dict, prepared: dict, record: dict, sources: Sources, baseline: bool = False
) -> dict:
    if not record.get("revision"):
        raise ValueError("Candidate revision must be concrete")
    patch_hash = "baseline"
    if baseline:
        if record["revision"] != task["parent_revision"]:
            raise ValueError("Baseline revision differs from the analyzed parent")
    else:
        if record.get("parent_revision") != task["parent_revision"]:
            raise ValueError("Candidate must name the immutable analyzed parent")
        patch_hash = hashlib.sha256(sources.path(record["patch"]).read_bytes()).hexdigest()
        if record.get("patch_sha256") != patch_hash:
            raise ValueError("Candidate patch SHA256 differs")
    identity = {
        "kind": task["kind"],
        "task_digest": prepared["task_digest"],
        "revision": record["revision"],
        "patch_sha256": patch_hash,
    }
    compile_report = bound_report(sources, record["compile"], identity)
    if compile_report.get("toolchain") != task["toolchain"]:
        raise ValueError("Compile target/compiler differs from the frozen toolchain")
    for flag in (
        "device_identity_verified",
        "isa_legal",
        "emitted_path_verified",
        "occupancy_launch_verified",
        "stream_graph_lifetime_verified",
    ):
        if compile_report.get(flag) is not True:
            raise ValueError("Missing compile/resource/compatibility proof: " + flag)
    if not compile_report.get("emitted_instructions"):
        raise ValueError("Source-level feature names do not prove the emitted instruction path")
    binary = sources.path(compile_report["binary_source"])
    if hashlib.sha256(binary.read_bytes()).hexdigest() != compile_report["binary_sha256"]:
        raise ValueError("Compiled binary identity differs")
    sources.path(compile_report["resource_source"])
    resources = compile_report["rank_resources"]
    if len(resources) != len(task["ranks"]) or {r["rank"] for r in resources} != set(task["ranks"]):
        raise ValueError("Compiled resource evidence must cover all ranks")
    for row in resources:
        for field in (
            "registers_per_thread",
            "shared_bytes_per_block",
            "threads_per_block",
            "tmem_columns",
        ):
            used = number(row[field], field)
            if used > number(row["limits"][field], field + " limit"):
                raise ValueError("Compiled resources exceed exact-device limit: " + field)
        number(row["local_memory_bytes"], "local memory bytes")
    memory = bound_report(sources, record["memory"], identity)
    for flag in (
        "all_lifecycle_phases_covered",
        "capacity_and_context_preserved",
        "no_eviction_or_recompute",
        "cache_layout_unchanged",
        "long_context_quality_revalidated",
        "full_occupancy_service_slo_revalidated",
    ):
        if memory.get(flag) is not True:
            raise ValueError("Candidate may consume the qualified KV budget: " + flag)
    if memory.get("validated_context_tokens") != prepared["deployment_target"]["context_tokens"]:
        raise ValueError("Context requalification must match the selected deployment target")
    sources.path(memory["source"])
    deployment = prepared["deployment"]
    peaks = memory["ranks"]
    budget_ranks = {
        rank["rank"]: (rank, replica)
        for replica in deployment["budget"]["replicas"]
        for rank in replica["ranks"]
    }
    if len(peaks) != len(budget_ranks) or {r["rank"] for r in peaks} != set(budget_ranks):
        raise ValueError("Memory evidence must cover every physical rank")
    for row in peaks:
        budget, replica = budget_ranks[row["rank"]]
        slots = memory["resident_slots"].get(replica["id"], 0)
        original_slots = deployment["qualified_replica_slots"][replica["id"]]
        if type(slots) is not int or slots < original_slots:
            raise ValueError("Candidate reduced validated resident KV slots")
        reserve = budget["safety_reserve_bytes"]
        demand = slots * budget["per_context_bytes_with_fragmentation"]
        if (
            number(row["peak_bytes"], "rank peak", positive=True) + reserve > budget["device_bytes"]
            or number(row["non_kv_peak_bytes"], "non-KV peak") + demand + reserve
            > budget["device_bytes"]
        ):
            raise ValueError("Candidate workspace/peak violates the preserved KV capacity budget")
    before = bound_report(sources, record["precision_before"], identity)
    after = bound_report(sources, record["precision_after"], identity)
    _precision(before, task)
    _precision(after, task)
    if (
        not before.get("clock_id")
        or before["clock_id"] != after.get("clock_id")
        or number(before["finished_ns"], "before end")
        <= number(before["started_ns"], "before start")
        or number(after["finished_ns"], "after end") <= number(after["started_ns"], "after start")
    ):
        raise ValueError("Precision must have a common coordinator clock and ordered intervals")
    migrations = []
    models_by_condition = dict(prepared["resource_models"])
    if not baseline:
        for condition in task["conditions"]:
            diagnostic = bound_report(
                sources,
                record["diagnoses"][condition["id"]],
                {**identity, "condition_id": condition["id"]},
            )
            reasons = check_diagnosis(
                diagnostic, diagnostic["identity"], condition["operator_cases"], sources
            )
            parent_cases = {
                c["id"]: c for c in prepared["baseline_diagnoses"][condition["id"]]["cases"]
            }
            for case in diagnostic["cases"]:
                parent = parent_cases[case["id"]]
                test = parent["test"]
                if case.get("test") != test:
                    raise ValueError("Post-analysis changed the predeclared hypothesis test")
                old = parent["metrics"][test["metric"]]
                new = case["metrics"][test["metric"]]
                if old["unit"] != new["unit"] or old["definition"] != new["definition"]:
                    raise ValueError("Hypothesis counter/proxy counting convention changed")
                a, b = number(old["value"], "parent counter"), number(new["value"], "new counter")
                effect = (b - a if test["direction"] == "increase" else a - b) / a if a else 0
                if effect < test["min_relative_change"]:
                    raise ValueError(
                        "Hypothesis refuted/inconclusive; deepen analysis before promotion"
                    )
            analysis, missing = inspect_performance(
                task,
                condition,
                record["performance_bundles"][condition["id"]],
                {
                    **performance_identity(task, condition),
                    "revision": record["revision"],
                    "task_digest": prepared["task_digest"],
                    "patch_sha256": patch_hash,
                },
                sources,
            )
            if reasons or missing:
                raise ValueError(
                    "Candidate lacks deep post-analysis: " + "; ".join(reasons + missing)
                )
            migrations.append(
                {
                    "condition": condition["id"],
                    "attack_plan": analysis["attack_plan"],
                    "confirmed_bottleneck": False,
                }
            )
            models_by_condition[condition["id"]] = analysis["resource_models"]
    policy = task["policy"]
    rounds = record["rounds"]
    if len(rounds) != len(policy["round_ids"]) or {r["id"] for r in rounds} != set(
        policy["round_ids"]
    ):
        raise ValueError("Missing matched independent native rounds")
    result = {}
    condition_map = {c["id"]: c for c in task["conditions"]}
    for trial in rounds:
        if len(trial["conditions"]) != len(condition_map) or {
            c["id"] for c in trial["conditions"]
        } != set(condition_map):
            raise ValueError("Native rounds must cover the complete service matrix")
        trial_result = {}
        for row in trial["conditions"]:
            condition = condition_map[row["id"]]
            native = {}
            for scope in ("operator", "forward", "service"):
                report = bound_report(
                    sources,
                    row[scope],
                    {**identity, "condition_id": row["id"], "round_id": trial["id"]},
                )
                if (
                    report.get("profiled") is not False
                    or report.get("debug_instrumentation") is not False
                    or report.get("completed") is not True
                    or report.get("new_jit") is not False
                    or report.get("clock_id") != before["clock_id"]
                ):
                    raise ValueError(
                        "Timing must be complete, warm and free of profiler/debug effects"
                    )
                start = number(report["started_ns"], "native start")
                end = number(report["finished_ns"], "native end")
                if not before["finished_ns"] <= start < end <= after["started_ns"]:
                    raise ValueError(
                        "Required order: precision before -> all native timing -> precision after"
                    )
                native[scope] = report
            operator = native["operator"]
            cases = {c["id"]: c for c in condition["operator_cases"]}
            if (
                operator.get("scope") != "operator_wrapper"
                or operator.get("all_streams_joined") is not True
                or len(operator["cases"]) != len(cases)
                or {c["id"] for c in operator["cases"]} != set(cases)
            ):
                raise ValueError("Operator wrapper timing must cover every captured case")
            weighted, case_times = {}, {}
            for measured in operator["cases"]:
                case = cases[measured["id"]]
                stats = summarize(measured["samples_us"])
                if stats["count"] < 5:
                    raise ValueError("Too few native wrapper samples")
                bounds = [
                    m["physical_lower_bound_us"]
                    for m in models_by_condition[condition["id"]]
                    if m["rank"] == case["rank"]
                    and str(m["device"]) == str(case["device"])
                    and m["signature"] == case["signature"]
                    and m["operator"] == case["operator"]
                    and m.get("physical_lower_bound_us") is not None
                ]
                if bounds and stats["p50"] * 1.05 < max(bounds):
                    raise ValueError(
                        "Native wrapper contradicts the matched physical resource bound"
                    )
                rank = str(case["rank"])
                case_times[case["id"]] = stats["p50"]
                weighted[rank] = weighted.get(rank, 0) + stats["p50"] * case["call_count"]
            forward = native["forward"]
            if (
                forward.get("scope") != "coordinated_prefill_forward"
                or forward.get("boundary_id") != condition["forward_boundary_id"]
                or forward.get("all_streams_joined") is not True
                or set(forward.get("rank_coverage", [])) != set(task["ranks"])
            ):
                raise ValueError("Forward timing boundary/rank coverage differs")
            fw = summarize(forward["samples_us"])
            if fw["count"] < 200:
                raise ValueError("Forward P95 requires at least 200 samples per independent round")
            tokens = number(forward["global_new_tokens_per_sample"], "global new tokens")
            if tokens != condition["global_new_tokens_per_sample"]:
                raise ValueError("Forward token work differs; client concurrency is not GPU batch")
            service = service_statistics(native["service"])
            if any(
                r["sent_us"] * 1000 < native["service"]["started_ns"]
                or r["terminal_us"] * 1000 > native["service"]["finished_ns"]
                for r in native["service"]["requests"]
            ):
                raise ValueError("Service cohort exceeds the declared coordinator timing interval")
            if (
                not service["request_integrity_valid"]
                or not service["ttft_us"]
                or not service["ttft_us"]["p95_sample_eligible"]
                or (
                    service["timed_offered_window"] is not None
                    and not service["timed_offered_window"]["window_completed"]
                )
            ):
                raise ValueError("Service errors/incomplete cohort/insufficient P95 samples")
            if (
                native["service"].get("cache_state_verified") is not True
                or native["service"].get("load_protocol_verified") is not True
            ):
                raise ValueError("Service cache/load state requires runtime evidence")
            spec = condition["spec"]
            peak = spec.get("request_concurrency", condition.get("expected_peak_inflight"))
            if peak is None or service["client_observed_peak_inflight"] != peak:
                raise ValueError("Actual service concurrency differs")
            expected_hit = {"cold-miss": 0, "partial-50": 0.5, "partial-90": 0.9}.get(
                spec["cache_profile"]
            )
            if (
                expected_hit is not None
                and abs(service["cached_token_fraction"] - expected_hit) > 0.02
            ):
                raise ValueError("Actual service cache hit differs")
            reference = {
                r["id"]: r for r in sources.read(condition["request_manifest"])["requests"]
            }
            if {r["id"] for r in native["service"]["requests"]} != set(reference):
                raise ValueError("Native service cohort differs from the frozen request manifest")
            for request in native["service"]["requests"]:
                expected = reference[request["id"]]
                if request["input_tokens"] != expected["input_tokens"]:
                    raise ValueError("Actual service input length differs")
                if (
                    type(request.get("first_token_id")) is not int
                    or request["first_token_id"] != expected["expected_first_token_id"]
                ):
                    raise ValueError("Service first-token parity failed")
                cached = expected.get("expected_cached_tokens")
                if cached is not None and request["cached_tokens"] != cached:
                    raise ValueError("Service owner cache counters differ from the frozen manifest")
                if spec["cache_profile"] == "cold-miss" and request["cached_tokens"]:
                    raise ValueError("Cold-miss request reused cache")
            slo = condition["slo"]
            if (
                fw["p95"] > slo["max_forward_p95_us"]
                or service["ttft_us"]["p95"] > slo["max_ttft_p95_us"]
                or service["new_prefill_tokens_per_s"] < slo["min_new_tokens_per_s"]
            ):
                raise ValueError("Per-condition forward/service SLO failed")
            trial_result[row["id"]] = {
                "operator_weighted_work_us_by_rank": weighted,
                "operator_case_p50_us": case_times,
                "forward_p50_us": fw["p50"],
                "forward_p95_us": fw["p95"],
                "forward_new_tokens_per_s": tokens / fw["p50"] * 1e6,
                "ttft_p50_us": service["ttft_us"]["p50"],
                "ttft_p95_us": service["ttft_us"]["p95"],
                **{
                    k: service[k]
                    for k in (
                        "new_prefill_tokens_per_s",
                        "logical_input_tokens_per_s",
                        "requests_per_s",
                        "cached_token_fraction",
                    )
                },
            }
        result[trial["id"]] = trial_result
    return {"identity": identity, "rounds": result, "post_analysis": migrations}
