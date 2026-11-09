from __future__ import annotations

import hashlib
import json
import shutil
import statistics
from pathlib import Path

from ..contracts import read_json
from ..store import RunStore, write_json
from .task import Sources, prepare_task
from .validation import validate_record

LOWER = ("forward_p50_us", "forward_p95_us", "ttft_p50_us", "ttft_p95_us")
HIGHER = (
    "forward_new_tokens_per_s",
    "new_prefill_tokens_per_s",
    "logical_input_tokens_per_s",
    "requests_per_s",
)


def _gain(old: float, new: float, lower: bool) -> float:
    if old == 0:
        if new != 0:
            raise ValueError("Zero-work throughput changed; verify the workload accounting")
        return 0
    return 1 - new / old if lower else new / old - 1


def compare_records(task: dict, base: dict, candidate: dict) -> dict:
    policy = task["policy"]
    failures = {scope: [] for scope in ("operator", "forward", "service")}
    metrics = []
    for condition in task["conditions"]:
        cid = condition["id"]
        target = cid in policy["gain_conditions"]
        for field in (*LOWER, *HIGHER):
            old = [base["rounds"][r][cid][field] for r in policy["round_ids"]]
            new = [candidate["rounds"][r][cid][field] for r in policy["round_ids"]]
            scope = "forward" if field.startswith("forward_") else "service"
            for values in (old, new):
                mean = statistics.mean(values)
                if mean and statistics.stdev(values) / mean > policy["max_cv"]:
                    failures[scope].append(cid + ": unstable paired-round " + field)
            gains = [_gain(a, b, field in LOWER) for a, b in zip(old, new)]
            minimum = min(gains)
            if minimum < -policy["max_regression"]:
                failures[scope].append(cid + ": regression in " + field)
            primary_forward = (
                "forward_p50_us"
                if policy["forward_objective"] == "latency"
                else ("forward_new_tokens_per_s")
            )
            primary_service = (
                "ttft_p95_us"
                if policy["service_objective"] == "ttft"
                else ("new_prefill_tokens_per_s")
            )
            primary = primary_forward if scope == "forward" else primary_service
            if target and field == primary and minimum < policy[scope + "_min_gain"]:
                failures[scope].append(cid + ": insufficient repeated " + primary + " gain")
            metrics.append(
                {
                    "condition": cid,
                    "metric": field,
                    "baseline_rounds": old,
                    "candidate_rounds": new,
                    "worst_paired_gain": minimum,
                    "is_statistical_confidence_interval": False,
                }
            )
        weights = "operator_weighted_work_us_by_rank"
        case_field = "operator_case_p50_us"
        for case in condition["operator_cases"]:
            old = [base["rounds"][r][cid][case_field][case["id"]] for r in policy["round_ids"]]
            new = [candidate["rounds"][r][cid][case_field][case["id"]] for r in policy["round_ids"]]
            gain = min(_gain(a, b, True) for a, b in zip(old, new))
            if gain < -policy["max_regression"]:
                failures["operator"].append(cid + ": captured case regressed: " + case["id"])
            metrics.append(
                {
                    "condition": cid,
                    "metric": "operator_case_p50_us",
                    "case": case["id"],
                    "worst_paired_gain": gain,
                }
            )
        ranks = base["rounds"][policy["round_ids"][0]][cid][weights]
        for rank in ranks:
            old = [base["rounds"][r][cid][weights][rank] for r in policy["round_ids"]]
            new = [candidate["rounds"][r][cid][weights][rank] for r in policy["round_ids"]]
            for values in (old, new):
                if statistics.stdev(values) / statistics.mean(values) > policy["max_cv"]:
                    failures["operator"].append(cid + ": unstable wrapper timing")
            gain = min(_gain(a, b, True) for a, b in zip(old, new))
            if gain < (policy["operator_min_gain"] if target else -policy["max_regression"]):
                failures["operator"].append(cid + ": wrapper work gate failed on rank " + rank)
            metrics.append(
                {
                    "condition": cid,
                    "metric": "operator_weighted_work_us",
                    "rank": rank,
                    "worst_paired_gain": gain,
                    "is_forward_wall_gain": False,
                }
            )
    passed = {scope: not reasons for scope, reasons in failures.items()}
    return {
        "accepted": all(passed.values()),
        "gates": passed,
        "reasons": [reason for values in failures.values() for reason in values],
        "metrics": metrics,
    }


def _snapshot(path: Path, name: str, sources: Sources, output: Path) -> None:
    inputs = output / "input"
    inputs.mkdir()
    shutil.copyfile(path, inputs / name)
    manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest()}
    for relative in list(sources.manifest):
        target = inputs / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources.path(relative), target)
        manifest[relative] = sources.manifest[relative]
    write_json(inputs / "manifest.json", manifest)


def write_kernel_task(path: Path, output: Path, sink=None) -> dict:
    store = RunStore(output, sink)
    store.emit("kernel_task_started")
    try:
        task, prepared, sources = prepare_task(path)
        _snapshot(path, "task.json", sources, output)
        packet = {
            "schema_version": 1,
            "task": task,
            "analysis": prepared,
            "kda_revision": "acdd79343c5e9d0c7cb96289f26fd3f1051ce785",
            "status": "ready_for_candidate"
            if prepared["ready_for_candidate"]
            else "analysis_required",
            "runs_kda": False,
            "runs_commands": False,
        }
        write_json(output / "kernel.task.json", packet)
        docs = output / "docs"
        docs.mkdir()
        (docs / "draft.md").write_text(
            "# Kernel analysis draft\n\n"
            f"Task: {task['id']}\n\nFingerprint: {prepared['task_digest']}\n\n"
            f"Analysis state: {packet['status']}\n\n"
            "Keep the imported hypothesis, counters, source mapping and alternatives in input/.\n"
            "No causal speedup is established. Revalidate ISA, resources and dependencies.\n\n"
            "Missing analysis:\n\n"
            + "\n".join("- " + r for r in prepared["missing_analysis"])
            + "\n",
            encoding="utf-8",
        )
        lines = [
            "# Reviewable kernel plan",
            "",
            "These commands are recorded, never executed by export.",
            "",
            "Analysis must be complete before candidate implementation.",
            "",
            "Precision -> native operator/forward/service -> precision; profile separately.",
            "",
        ]
        fence = chr(96) * 3
        for phase, commands in task["commands"].items():
            lines += [
                "## " + phase,
                "",
                fence + "json",
                json.dumps(commands, ensure_ascii=False, indent=2),
                fence,
                "",
            ]
        (docs / "plan.md").write_text("\n".join(lines), encoding="utf-8")
        store.emit("kernel_task_completed", status=packet["status"], auto_execute=False)
        return packet
    except Exception as error:
        store.emit("kernel_task_failed", error_type=type(error).__name__, message=str(error))
        raise


def write_kernel_evaluation(path: Path, output: Path, sink=None) -> dict:
    store = RunStore(output, sink)
    store.emit("kernel_evaluation_started")
    try:
        data = read_json(path)
        if data.get("schema_version") != 1:
            raise ValueError("Kernel evaluation requires schema_version=1")
        sources = Sources(path.parent)
        task, prepared, fixed = prepare_task(sources.path(data["task"]))
        for relative in fixed.manifest:
            sources.path((fixed.root / relative).relative_to(sources.root).as_posix())
        result = {
            "schema_version": 1,
            "kind": task["kind"],
            "task_digest": prepared["task_digest"],
            "ready_for_candidate": prepared["ready_for_candidate"],
            "missing_analysis": prepared["missing_analysis"],
            "candidates": [],
            "recommended_candidate": None,
            "auto_deploy": False,
            "limitations": [
                "Imported evidence and external validator attestations, not live verification.",
                "Synthetic acceptance never recommends a real kernel or certifies a backend.",
                "Worst paired gain is a conservative check, not statistical significance.",
                "Operator weighted work is rank-local and not a causal forward speedup.",
            ],
        }
        if prepared["ready_for_candidate"]:
            baseline = validate_record(task, prepared, data["baseline"], sources, baseline=True)
            result["baseline"] = baseline
            seen = set()
            for candidate in data["candidates"]:
                cid = candidate["id"]
                if not isinstance(cid, str) or not cid or cid in seen:
                    raise ValueError("Candidate IDs must be unique and nonempty")
                seen.add(cid)
                try:
                    validated = validate_record(task, prepared, candidate, sources)
                    decision = {
                        "id": cid,
                        "parent_revision": candidate["parent_revision"],
                        "revision": candidate["revision"],
                        **compare_records(task, baseline, validated),
                        "post_analysis": validated["post_analysis"],
                    }
                except (ValueError, OSError, KeyError, TypeError) as error:
                    decision = {
                        "id": cid,
                        "accepted": False,
                        "gates": {},
                        "parent_revision": candidate.get("parent_revision"),
                        "revision": candidate.get("revision"),
                        "reasons": [str(error)],
                    }
                result["candidates"].append(decision)
                with (output / "candidates.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(decision, ensure_ascii=False, allow_nan=False) + "\n")
                store.emit(
                    "kernel_candidate_evaluated",
                    candidate=cid,
                    accepted=decision["accepted"],
                    reasons=decision["reasons"],
                )
            accepted = [r for r in result["candidates"] if r["accepted"]]
            if accepted and task["kind"] == "measured":
                # Retain the first accepted parent branch; no unsupported best-of-noise ranking.
                result["recommended_candidate"] = accepted[0]["id"]
        _snapshot(path, "evaluation.json", sources, output)
        write_json(output / "kernel.evaluation.json", result)
        lines = [
            "# Kernel / prefill / service acceptance",
            "",
            f"Evidence kind: **{task['kind']}**.",
            "",
            f"Deep analysis ready: {prepared['ready_for_candidate']}; "
            f"recommendation: {result['recommended_candidate']}.",
            "",
            "| Candidate | Operator | Forward | Service | Accepted | Reasons |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in result["candidates"]:
            gates = row["gates"]
            label = str(row["id"]).replace("|", "\\|").replace("\n", " ")
            reasons = "; ".join(row["reasons"]).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {label} | {gates.get('operator')} | {gates.get('forward')} "
                f"| {gates.get('service')} | {row['accepted']} | {reasons} |"
            )
        lines += ["", *prepared["missing_analysis"], "", *result["limitations"]]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        store.emit(
            "kernel_evaluation_completed",
            kind=task["kind"],
            recommended_candidate=result["recommended_candidate"],
            auto_deploy=False,
        )
        return result
    except Exception as error:
        store.emit("kernel_evaluation_failed", error_type=type(error).__name__, message=str(error))
        raise
