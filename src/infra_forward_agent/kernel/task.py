from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..architectures import ArchitectureRegistry
from ..contracts import number, read_json
from ..deployment.selection import select_deployment
from ..performance.report import build_report


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Sources:
    def __init__(self, root: Path, reserved: set[str] | None = None):
        self.root = root.resolve()
        self.manifest: dict[str, str] = {}
        self.reserved = (
            reserved if reserved is not None else {"task.json", "evaluation.json", "manifest.json"}
        )

    def path(self, relative: str) -> Path:
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
            or relative in self.reserved
        ):
            raise ValueError("Use non-reserved canonical relative kernel artifact paths")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Missing or escaped kernel artifact: " + relative)
        self.manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read(self, relative: str) -> dict:
        return read_json(self.path(relative))


def performance_identity(task: dict, condition: dict) -> dict:
    return {
        "kind": task["kind"],
        "architecture": task["comparison"]["architecture"],
        "environment": digest(task["comparison"]),
        "workload_id": condition["id"],
    }


def check_diagnosis(report: dict, expected: dict, cases: list[dict], sources: Sources) -> list[str]:
    """Require instrumentation, counters and a falsifiable interpretation."""
    if report.get("identity") != expected:
        raise ValueError("Kernel diagnosis identity differs")
    reasons = []
    for flag in (
        "separate_profile_pass",
        "all_rank_timeline_complete",
        "mapping_verified",
        "replay_effects_documented",
        "instrumentation_overhead_checked",
    ):
        if report.get(flag) is not True:
            reasons.append("deep analysis missing: " + flag)
    for field in ("timeline_source", "instrumentation_source", "counter_source"):
        if report.get(field):
            sources.path(report[field])
        else:
            reasons.append("deep analysis missing: " + field)
    if not report.get("tool") or not report.get("tool_version"):
        reasons.append("deep analysis missing profiler identity")
    rows = report.get("cases", [])
    expected_ids = {case["id"] for case in cases}
    if len(rows) != len(expected_ids) or {row["id"] for row in rows} != expected_ids:
        return reasons + ["deep analysis does not cover every captured operator case"]
    for row in rows:
        metrics = row.get("metrics", {})
        for name in ("executed_work", "actual_memory_bytes", "occupancy", "stall", "spill_bytes"):
            metric = metrics.get(name, {})
            if metric.get("value") is None:
                reasons.append(
                    f"{row['id']}: missing counter/proxy for {name}; unknown is not zero"
                )
            else:
                number(metric["value"], name)
            if not metric.get("unit") or not metric.get("definition"):
                reasons.append(f"{row['id']}: counter units/counting definition missing")
            if metric.get("source"):
                sources.path(metric["source"])
            else:
                reasons.append(f"{row['id']}: counter source missing")
        for field in (
            "observation",
            "hypothesis",
            "predicted_counter_change",
            "falsification",
            "alternative_explanation",
            "single_change",
            "dependency_and_lifetime_review",
        ):
            if not row.get(field):
                reasons.append(f"{row['id']}: analysis missing {field}")
        if row.get("evidence_mode") not in {"hardware_counter", "validated_proxy"}:
            reasons.append(f"{row['id']}: no counter or validated instrumentation proxy")
        test = row.get("test", {})
        if (
            test.get("metric") not in metrics
            or test.get("direction") not in {"increase", "decrease"}
            or not 0 < number(test.get("min_relative_change", 0), "hypothesis effect") < 1
        ):
            reasons.append(f"{row['id']}: missing quantitative hypothesis test")
        if row.get("evidence_mode") == "validated_proxy":
            if not row.get("proxy_validation_source"):
                reasons.append(
                    f"{row['id']}: unavailable counters require a validated proxy experiment"
                )
            else:
                sources.path(row["proxy_validation_source"])
    return reasons


def inspect_performance(
    task: dict, condition: dict, relative: str, expected: dict, sources: Sources
) -> tuple[dict, list[str]]:
    bundle = sources.path(relative)
    if set(read_json(bundle)["expected_ranks"]) != set(task["ranks"]):
        raise ValueError("Profiling bundle must include every deployment rank")
    analysis = build_report(bundle)
    if analysis["identity"] != expected:
        raise ValueError("Performance analysis is not bound to this kernel workload/revision")
    for name in analysis["artifact_sources"]:
        resolved = (bundle.parent / name).resolve()
        sources.path(resolved.relative_to(sources.root).as_posix())
    reasons = []
    if not analysis["capture_timelines"] or any(
        not c["all_ranks_present"] or not c["complete_forward_window"]
        for c in analysis["capture_timelines"]
    ):
        reasons.append(condition["id"] + ": incomplete timeline")
    mapped = {
        (r["rank"], str(r["device"]), r["operator"], r["compute_type"], r["signature"]): r
        for r in analysis["resource_models"]
    }
    for case in condition["operator_cases"]:
        key = (
            case["rank"],
            str(case["device"]),
            case["operator"],
            case["compute_type"],
            case["signature"],
        )
        model = mapped.get(key)
        if (
            not model
            or model.get("model_reference_us") is None
            or model["status"]
            not in {
                "efficiency_gap_hypothesis",
                "near_reference_limit",
                "empirical_reference_exceeded",
            }
        ):
            reasons.append(case["id"] + ": missing/invalid matched resource limit model")
        if not any(
            p["rank"] == case["rank"]
            and p["operator"] == case["operator"]
            and p["signature"] == case["signature"]
            for p in analysis["priorities"]
        ):
            reasons.append(case["id"] + ": operator is not mapped into the service forward")
    return analysis, reasons


def prepare_task(path: Path) -> tuple[dict, dict, Sources]:
    task = read_json(path)
    if task.get("schema_version") != 1 or task.get("kind") not in {"synthetic", "measured"}:
        raise ValueError("Kernel task requires schema_version=1 and explicit evidence kind")
    if not task.get("id") or not task.get("parent_revision"):
        raise ValueError("Kernel task needs an ID and immutable parent revision")
    sources = Sources(path.parent)
    registry = ArchitectureRegistry.builtin()
    for relative in task.get("architecture_profiles", []):
        registry.load(sources.path(relative))
    profile = registry.get(task["comparison"]["architecture"])
    deployment = sources.read(task["deployment_selection"])
    deployment_path = sources.path(task["deployment_bundle"])
    reproduced = select_deployment(deployment_path)
    if reproduced != deployment:
        raise ValueError("Deployment selection differs from its retained original evidence")
    for relative in deployment["artifact_sources"]:
        sources.path((deployment_path.parent / relative).relative_to(sources.root).as_posix())
    if deployment.get("kind") != task["kind"] or deployment.get("comparison") != task["comparison"]:
        raise ValueError("Kernel task and deployment evidence must fix the same kind/environment")
    choices = [row for row in deployment["candidates"] if row["id"] == task["deployment_candidate"]]
    if len(choices) != 1 or choices[0].get("eligible") is not True:
        raise ValueError("Kernel task requires an eligible imported deployment candidate")
    selected = choices[0]
    if not selected.get("verified_context_slots"):
        raise ValueError("Deployment must retain validated resident capacity")
    ranks = {
        rank["rank"] for replica in selected["budget"]["replicas"] for rank in replica["ranks"]
    }
    if (
        set(task["ranks"]) != ranks
        or len(task["ranks"]) != len(ranks)
        or any(type(r) is not int for r in task["ranks"])
    ):
        raise ValueError("Kernel task must cover the fixed deployment's complete rank set")
    matrix = read_json(sources.path(task["matrix_source"]))
    service_rows = [row for row in matrix if row["scope"] == "service"]
    services = {row["workload_id"]: row for row in service_rows}
    if len(services) != len(service_rows):
        raise ValueError("Duplicate service workload IDs in the frozen matrix")
    conditions = task["conditions"]
    if (
        not services
        or len({c["id"] for c in conditions}) != len(conditions)
        or {c["id"] for c in conditions} != set(services)
    ):
        raise ValueError("Kernel conditions must cover the entire declared service matrix")
    policy = task["policy"]
    rounds = policy["round_ids"]
    if len(rounds) < 5 or len(set(rounds)) != len(rounds):
        raise ValueError("Require at least five independent paired rounds")
    for name in (
        "max_cv",
        "max_regression",
        "operator_min_gain",
        "forward_min_gain",
        "service_min_gain",
    ):
        if number(policy[name], name) >= 1:
            raise ValueError("Kernel policy fractions must be below one")
    if policy["service_objective"] not in {"ttft", "new_prefill_throughput"}:
        raise ValueError("service_objective must be ttft or new_prefill_throughput")
    if policy["forward_objective"] not in {"latency", "throughput"}:
        raise ValueError("forward_objective must be latency or throughput")
    if not set(policy["gain_conditions"]) or set(policy["gain_conditions"]) - set(services):
        raise ValueError("Declare nonempty gain conditions within the full service matrix")
    for field in ("preflight", "instrument", "profile", "precision", "benchmark", "post_precision"):
        if not task["commands"].get(field):
            raise ValueError("Missing reviewable task command: " + field)
    reasons, summaries, all_case_ids, diagnoses, resource_models = [], [], set(), {}, {}
    for condition in conditions:
        matrix_row = services[condition["id"]]
        if (
            matrix_row["phase"] != "prefill"
            or condition["spec"] != matrix_row["parameters"]
            or matrix_row["generation"]["max_new_tokens"] != 1
        ):
            raise ValueError("Condition differs from the frozen prefill testing matrix")
        cases = condition["operator_cases"]
        if not cases:
            if not condition.get("not_applicable_source"):
                reasons.append(condition["id"] + ": unbound case; absence requires capture proof")
            else:
                absent = sources.read(condition["not_applicable_source"])
                if (
                    absent.get("identity")
                    != {
                        **performance_identity(task, condition),
                        "revision": task["parent_revision"],
                    }
                    or absent.get("operator_absent_verified") is not True
                    or not absent.get("reason")
                    or not absent.get("capture_source")
                ):
                    reasons.append(condition["id"] + ": invalid not-applicable proof")
                else:
                    sources.path(absent["capture_source"])
        for case in cases:
            if case["id"] in all_case_ids or case["rank"] not in ranks:
                raise ValueError("Operator cases need unique IDs and a selected rank")
            all_case_ids.add(case["id"])
            if not case.get("signature") or case["signature"] == "unknown":
                raise ValueError("Operator case needs a concrete captured signature")
            for field in (
                "math",
                "shape",
                "dtype",
                "layout",
                "quantization",
                "routing",
                "reference",
            ):
                if not case["contract"].get(field):
                    raise ValueError("Incomplete operator semantic contract: " + field)
            for tolerance in ("atol", "rtol"):
                number(case["contract"]["reference"][tolerance], "frozen elementwise tolerance")
            sources.path(case["capture_source"])
            number(case["call_count"], "captured call count", positive=True)
            number(case["relative_l2_limit"], "case-specific L2 limit")
        sources.path(condition["reference_source"])
        for tolerance in ("atol", "rtol"):
            number(condition["forward_reference"][tolerance], "frozen forward tolerance")
        manifest = sources.read(condition["request_manifest"])
        requests = manifest["requests"]
        if not requests or len({r["id"] for r in requests}) != len(requests):
            raise ValueError("Freeze unique request IDs and reference tokens before timing")
        for request in requests:
            if type(request["expected_first_token_id"]) is not int:
                raise ValueError("Reference token must come from the frozen validator manifest")
            number(request["input_tokens"], "reference prompt length", positive=True)
            if (
                "input_tokens" in condition["spec"]
                and request["input_tokens"] != condition["spec"]["input_tokens"]
            ):
                raise ValueError("Request manifest differs from the fixed-length test condition")
        number(condition["forward_relative_l2_limit"], "forward L2 limit")
        analysis, missing = inspect_performance(
            task,
            condition,
            condition["performance_bundle"],
            performance_identity(task, condition),
            sources,
        )
        reasons += missing
        diagnostic = sources.read(condition["diagnosis"])
        diagnoses[condition["id"]] = diagnostic
        resource_models[condition["id"]] = analysis["resource_models"]
        reasons += check_diagnosis(
            diagnostic,
            {**performance_identity(task, condition), "revision": task["parent_revision"]},
            cases,
            sources,
        )
        summaries.append(
            {
                "condition": condition["id"],
                "attack_plan": analysis["attack_plan"],
                "confirmed_bottleneck": False,
            }
        )
    if not all_case_ids:
        reasons.append("No actual captured operator invocation to optimize")
    fingerprint = digest({"task": task, "fixed_sources_sha256": sources.manifest})
    return (
        task,
        {
            "task_digest": fingerprint,
            "architecture": profile.to_dict(),
            "ready_for_candidate": not reasons,
            "missing_analysis": reasons,
            "deployment": selected,
            "deployment_target": deployment["target"],
            "performance": summaries,
            "baseline_diagnoses": diagnoses,
            "resource_models": resource_models,
            "auto_execute": False,
        },
        sources,
    )
