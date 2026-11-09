from __future__ import annotations

import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

from ..contracts import number, read_json
from ..store import RunStore, write_json
from ..trace import Trace, load_chrome
from .ceilings import DEFAULT_POLICY, resource_model
from .statistics import service_statistics, summarize
from .timeline import analyze_timeline


def _artifact(root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or Path(relative).name in {"performance.json", "manifest.json"}
    ):
        raise ValueError("Performance artifact paths must be relative, non-reserved paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or escaped performance artifact: {relative}")
    return path


def _same_identity(report: dict, identity: dict) -> None:
    if report.get("identity") != identity:
        raise ValueError("Native measurements must match the complete bundle identity")


def _label(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_report(path: Path) -> dict:
    root = path.parent.resolve()
    data = read_json(path)
    if data.get("schema_version") != 1 or data.get("focus") != "prefill":
        raise ValueError("Performance bundle requires schema_version=1 and focus=prefill")
    if not any(
        data.get(key)
        for key in ("service", "forward", "traces", "operator_resources", "step_resource_budgets")
    ):
        raise ValueError("Performance bundle must contain evidence")
    identity = data["identity"]
    for field in ("kind", "architecture", "environment", "workload_id"):
        if not isinstance(identity.get(field), str) or not identity[field].strip():
            raise ValueError(f"Missing performance identity.{field}")
    if identity["kind"] not in {"synthetic", "measured"}:
        raise ValueError("Performance kind must be synthetic or measured")
    ranks = data["expected_ranks"]
    if (
        not isinstance(ranks, list)
        or not ranks
        or len(set(ranks)) != len(ranks)
        or any(type(rank) is not int or rank < 0 for rank in ranks)
    ):
        raise ValueError("Expected ranks must explicitly identify unique nonnegative ranks")
    sources = set()

    def load(relative: str) -> dict:
        sources.add(relative)
        return read_json(_artifact(root, relative))

    policy = {**DEFAULT_POLICY, **data.get("resource_model_policy", {})}
    result = {
        "schema_version": 1,
        "identity": identity,
        "service": None,
        "forward": None,
        "capture_timelines": [],
        "resource_models": [],
        "priorities": [],
        "step_resource_budgets": [],
        "forward_resource_ceiling": None,
        "resource_model_policy": policy,
        "service_ceiling": "unknown: operator references do not establish a whole-service ceiling",
        "confirmed_bottleneck": False,
    }
    if data.get("service"):
        native = load(data["service"])
        _same_identity(native, identity)
        result["service"] = service_statistics(native)
    if data.get("forward"):
        native = load(data["forward"])
        _same_identity(native, identity)
        if (
            native.get("profiled") is not False
            or native.get("all_streams_joined") is not True
            or set(native.get("rank_coverage", [])) != set(ranks)
            or native.get("scope") != "coordinated_prefill_forward"
            or not native.get("boundary_id")
        ):
            raise ValueError(
                "Forward samples require an unprofiled coordinated all-stream/all-rank boundary"
            )
        statistics = summarize(native["samples_us"])
        new_tokens = number(
            native["global_new_tokens_per_sample"], "global_new_tokens_per_sample", positive=True
        )
        result["forward"] = {
            "wall_us": statistics,
            "boundary_id": native["boundary_id"],
            "new_tokens_per_sample": new_tokens,
            "new_tokens_per_s_at_median": new_tokens / statistics["p50"] * 1e6,
        }
    captures = defaultdict(list)
    for spec in data.get("traces", []):
        captures[spec["capture_id"]].append(spec)
    for capture, specs in captures.items():
        slices, windows, ignored, clipped = [], {}, 0, 0
        for spec in specs:
            sources.add(spec["path"])
            trace = load_chrome(
                _artifact(root, spec["path"]),
                rank=spec["rank"],
                marker=spec.get("marker", "forward"),
            )
            if windows.keys() & trace.windows.keys():
                raise ValueError("Duplicate rank/window within capture")
            slices.extend(trace.slices)
            windows.update(trace.windows)
            ignored += trace.ignored_events
            clipped += trace.clipped_events
        observed = {item.rank for item in slices}
        if observed - set(ranks):
            raise ValueError("Trace contains unexpected ranks")
        timeline = analyze_timeline(Trace(slices, windows, ignored, clipped))
        durations = {str(rank): end - start for rank, (start, end) in windows.items()}
        result["capture_timelines"].append(
            {
                "capture_id": capture,
                "all_ranks_present": observed == set(ranks),
                "missing_ranks": sorted(set(ranks) - observed),
                "clipped_gpu_events": clipped,
                "complete_forward_window": clipped == 0,
                "rank_local_window_us": durations,
                "rank_duration_max_to_min": max(durations.values()) / min(durations.values()),
                **timeline,
            }
        )
    ceilings = {}
    for ceiling in data.get("ceilings", []):
        if ceiling["id"] in ceilings:
            raise ValueError("Duplicate ceiling id")
        sources.add(ceiling["source"])
        _artifact(root, ceiling["source"])
        ceilings[ceiling["id"]] = ceiling
    models = {}
    for observation in data.get("operator_resources", []):
        row = dict(observation)
        key = (
            row["rank"],
            str(row["device"]),
            row["operator"],
            row["compute_type"],
            row["signature"],
        )
        if (
            type(row["rank"]) is not int
            or row["rank"] not in ranks
            or row["signature"] == "unknown"
        ):
            raise ValueError("Operator model needs a selected rank and concrete shape signature")
        if key in models:
            raise ValueError("Duplicate operator resource model; separate signatures/cases")
        native = load(row["native"]["source"])
        _same_identity(native, identity)
        if (
            native.get("scope") != "operator_invocation"
            or native.get("signature") != row["signature"]
            or native.get("rank") != row["rank"]
            or str(native.get("device")) != key[1]
            or native.get("operator") != row["operator"]
            or native.get("compute_type") != row["compute_type"]
        ):
            raise ValueError("Native operator timing rank/device/operator/shape contract differs")
        row["native"] = {**native, "source": row["native"]["source"]}
        for quantity in row.get("resources", []):
            sources.add(quantity["source"])
            _artifact(root, quantity["source"])
        model = resource_model(row, ceilings, identity, policy)
        models[key] = model
        result["resource_models"].append(
            {
                "rank": key[0],
                "device": key[1],
                "operator": key[2],
                "compute_type": key[3],
                "signature": key[4],
                **model,
            }
        )
    budget_ranks = set()
    budget_tokens = set()
    budget_steps = set()
    for budget in data.get("step_resource_budgets", []):
        rank = budget["rank"]
        if type(rank) is not int or rank not in ranks or rank in budget_ranks:
            raise ValueError("Step resource budgets must use unique selected ranks")
        budget_ranks.add(rank)
        native = load(budget["native"]["source"])
        _same_identity(native, identity)
        if (
            native.get("scope") != "prefill_rank_interval"
            or native.get("rank") != rank
            or native.get("signature") != budget["signature"]
            or native.get("all_streams_joined") is not True
            or not budget.get("step_contract")
            or native.get("step_contract") != budget["step_contract"]
        ):
            raise ValueError("Budget needs a matched native per-rank prefill interval")
        tokens = number(budget["global_new_tokens"], "global_new_tokens", positive=True)
        budget_tokens.add(tokens)
        budget_steps.add(budget["step_contract"])
        if result["forward"] and tokens != result["forward"]["new_tokens_per_sample"]:
            raise ValueError("Step budget and coordinated forward token counts differ")
        for quantity in budget["resources"]:
            sources.add(quantity["source"])
            _artifact(root, quantity["source"])
        model = resource_model(
            {**budget, "native": {**native, "source": budget["native"]["source"]}},
            ceilings,
            identity,
            policy,
        )
        result["step_resource_budgets"].append({"rank": rank, "global_new_tokens": tokens, **model})
    if len(budget_tokens) > 1 or len(budget_steps) > 1:
        raise ValueError("Per-rank budgets must describe the same global prefill step token count")
    budgets = result["step_resource_budgets"]
    if budgets and budget_ranks == set(ranks):
        hard_times = [row.get("physical_lower_bound_us") for row in budgets]
        if all(hard_times) and not any(
            row["status"] == "inconsistent_physical_model" for row in budgets
        ):
            bound = max(hard_times)
            result["forward_resource_ceiling"] = {
                "optimistic_physical_tokens_per_s": next(iter(budget_tokens)) / bound * 1e6,
                "resource_lower_bound_us": bound,
                "scope": "one matched prefill step on these physical ranks",
                "achievable_throughput": False,
                "service_throughput_ceiling": False,
                "limitations": "Omitted latency, CPU and dependencies can lower throughput.",
            }
    for capture in result["capture_timelines"]:
        for device in capture["devices"]:
            for operator in device["operators"]:
                key = (
                    device["rank"],
                    device["device"],
                    operator["operator"],
                    operator["compute_type"],
                    operator["signature"],
                )
                model = models.get(key)
                state = model["status"] if model else "missing_resource_evidence"
                gap = model.get("opportunity_fraction") if model else None
                priority = operator["exclusive_us"] * gap if gap is not None else None
                route = (
                    "calibrate_or_validate_evidence"
                    if gap is None
                    else "reduce_work_or_data_movement"
                    if state == "near_reference_limit"
                    else "investigate_kernel_efficiency"
                )
                service = result["service"]
                invalid_cohort = service is not None and (
                    not service["request_integrity_valid"]
                    or (
                        service["timed_offered_window"] is not None
                        and not service["timed_offered_window"]["window_completed"]
                    )
                )
                if (
                    invalid_cohort
                    or not capture["all_ranks_present"]
                    or not capture["complete_forward_window"]
                ):
                    route, priority = "calibrate_or_validate_evidence", None
                result["priorities"].append(
                    {
                        "capture_id": capture["capture_id"],
                        "rank": key[0],
                        "device": key[1],
                        "operator": key[2],
                        "compute_type": key[3],
                        "signature": key[4],
                        "work_share": operator["work_share"],
                        "exclusive_us": operator["exclusive_us"],
                        "coverage_us": operator["union_us"],
                        "all_ranks_present": capture["all_ranks_present"],
                        "complete_forward_window": capture["complete_forward_window"],
                        "resource_model_status": state,
                        "investigation_route": route,
                        "priority_hint_us_not_predicted_gain": priority,
                        "confirmed_bottleneck": False,
                        "service_speedup_prediction": None,
                        "next_action": "verify dependencies and matched whole-forward/service A/B",
                    }
                )
    result["priorities"].sort(key=lambda row: row["exclusive_us"], reverse=True)
    targets = result["priorities"]
    result["attack_plan"] = {
        "measure_first": [
            row
            for row in targets
            if row["investigation_route"] == "calibrate_or_validate_evidence"
            or not row["all_ranks_present"]
            or not row["complete_forward_window"]
        ],
        "kernel_efficiency": sorted(
            [
                row
                for row in targets
                if row["all_ranks_present"]
                and row["complete_forward_window"]
                and row["investigation_route"] == "investigate_kernel_efficiency"
            ],
            key=lambda row: row["priority_hint_us_not_predicted_gain"],
            reverse=True,
        ),
        "work_or_traffic_reduction": [
            row
            for row in targets
            if row["all_ranks_present"]
            and row["complete_forward_window"]
            and row["investigation_route"] == "reduce_work_or_data_movement"
        ],
        "wait_and_overlap": [
            {
                "capture_id": capture["capture_id"],
                "rank": device["rank"],
                "device": device["device"],
                "idle_us": device["gpu_idle_us"],
                "communication_without_compute_us": device["communication_without_compute_us"],
                "next_action": "collect CPU, event and collective-arrival dependencies",
            }
            for capture in result["capture_timelines"]
            for device in capture["devices"]
            if device["gpu_idle_us"] or device["communication_without_compute_us"]
        ],
        "overlapped_hotspots": [
            row for row in targets if row["exclusive_us"] == 0 and row["coverage_us"] > 0
        ],
    }
    result["artifact_sources"] = sorted(sources)
    result["limitations"] = [
        "This is an offline analysis, not runtime acceptance or automatic production optimization.",
        "Large exclusive exposure is an investigation priority, not a proven critical path.",
        "Native timing and profile/counter captures are separate experiments, not one timeline.",
        "No unmeasured architecture rate, bandwidth, dependency or service ceiling is invented.",
        "Amounts and ceiling applicability are producer attestations with retained sources.",
    ]
    return result


def write_report(path: Path, output: Path, sink=None) -> dict:
    store = RunStore(output, sink)
    store.emit("performance_analysis_started")
    try:
        result = build_report(path)
        inputs = output / "input"
        inputs.mkdir()
        shutil.copyfile(path, inputs / "performance.json")
        manifest = {"performance.json": hashlib.sha256(path.read_bytes()).hexdigest()}
        for relative in result["artifact_sources"]:
            source = _artifact(path.parent.resolve(), relative)
            destination = inputs / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            manifest[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
        write_json(inputs / "manifest.json", manifest)
        write_json(output / "performance.analysis.json", result)
        write_json(output / "attack-plan.json", result["attack_plan"])
        lines = [
            "# Prefill performance investigation",
            "",
            f"Evidence kind: **{result['identity']['kind']}**",
            "",
            "Investigation hypotheses only; no hardware or service speedup is established.",
            "",
        ]
        if result["service"]:
            service = result["service"]
            lines += [
                "## Native service cohort",
                "",
                f"Submitted/completed: {service['submitted']}/{service['completed']}; "
                f"request integrity: {service['request_integrity_valid']} "
                "(not accuracy or performance acceptance).",
                "",
                f"Full scoring horizon: {service['cohort_seconds']:.6f} s; "
                f"logical input: {service['logical_input_tokens_per_s']:.2f} token/s; "
                f"new prefill: {service['new_prefill_tokens_per_s']:.2f} token/s.",
                "",
            ]
            if service["ttft_us"]:
                stats = service["ttft_us"]
                lines += [
                    f"TTFT P50/P95: {stats['p50']:.2f}/{stats['p95']:.2f} us; "
                    f"P95 sample eligible: {stats['p95_sample_eligible']}.",
                    "",
                ]
        if result["forward"]:
            forward = result["forward"]
            stats = forward["wall_us"]
            lines += [
                "## Native prefill forward",
                "",
                f"Boundary: {forward['boundary_id']}; P50/P95: "
                f"{stats['p50']:.2f}/{stats['p95']:.2f} us; "
                f"new tokens/sample: {forward['new_tokens_per_sample']}.",
                "",
            ]
        lines += [
            "## Captured timeline (separate profiled experiment)",
            "",
            "| Capture/rank/device | Wall us | Busy us | Comm us | C/N us | Multi-stream us |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for capture in result["capture_timelines"]:
            for device in capture["devices"]:
                lines.append(
                    f"| {_label(capture['capture_id'])}/{device['rank']}/"
                    f"{_label(device['device'])} "
                    f"| {device['captured_forward_us']:.2f} | {device['gpu_busy_union_us']:.2f} "
                    f"| {device['communication_union_us']:.2f} "
                    f"| {device['compute_communication_overlap_us']:.2f} "
                    f"| {device['multi_stream_active_us']:.2f} |"
                )
        lines += [
            "",
            "## Applicable resource references",
            "",
            "| Rank/operator | Native us | Modeled reference us | Dominant resources | State |",
            "| --- | --- | --- | --- | --- |",
        ]
        for model in result["resource_models"]:
            lines.append(
                f"| {model['rank']}/{_label(model['operator'])} | {model['native_us']:.2f} "
                f"| {model.get('model_reference_us')} "
                f"| {', '.join(model.get('dominant_modeled_resources', []))} | {model['status']} |"
            )
        lines += [
            "",
            "| Rank/operator | Resource | Ceiling rate | Basis | Demand | Reference time us |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for model in result["resource_models"]:
            for resource in model["resources"]:
                lines.append(
                    f"| {model['rank']}/{_label(model['operator'])} | {resource['resource']} "
                    f"| {resource['rate']:.6g} {resource['unit']} | {resource['ceiling_basis']} "
                    f"| {resource['amount']:.6g} | {resource['reference_time_us']:.2f} |"
                )
        lines += [
            "",
            "## Investigation priorities",
            "",
            "| Rank/device | Operator | Work share | Exclusive us | Resource status | Route |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in result["priorities"]:
            lines.append(
                f"| {row['rank']}/{_label(row['device'])} | {_label(row['operator'])} "
                f"| {row['work_share']:.1%} "
                f"| {row['exclusive_us']:.2f} | {row['resource_model_status']} "
                f"| {row['investigation_route']} |"
            )
        lines += ["", "Service ceiling: " + result["service_ceiling"], "", *result["limitations"]]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        store.emit(
            "performance_analysis_completed",
            targets=len(result["priorities"]),
            confirmed_bottleneck=False,
        )
        return result
    except Exception as error:
        store.emit(
            "performance_analysis_failed", error_type=type(error).__name__, message=str(error)
        )
        raise
