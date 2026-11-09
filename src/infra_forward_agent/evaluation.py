"""Precision -> unprofiled forward measurements -> precision -> promotion gate."""

from __future__ import annotations

import statistics

from .contracts import Decision, Evidence, Policy, number, read_json


def validate(evidence: Evidence, policy: Policy) -> dict:
    """Validate imported reports; correctness remains an external producer's attestation."""
    data = evidence.data
    benchmark = read_json(evidence.artifact(data["benchmark"]))
    if benchmark.get("completed") is not True or benchmark.get("profiled") is not False:
        raise ValueError("Benchmark must be complete and collected without profiling")
    if benchmark.get("errors"):
        raise ValueError("Benchmark has errors")
    if type(benchmark.get("warmup_iterations")) is not int or benchmark["warmup_iterations"] < 1:
        raise ValueError("Benchmark requires warmup iterations")
    start = number(benchmark.get("started_ns"), "benchmark.started_ns")
    end = number(benchmark.get("finished_ns"), "benchmark.finished_ns", positive=True)
    if end <= start:
        raise ValueError("Benchmark timing window must be positive")
    if not benchmark.get("clock_id"):
        raise ValueError("Benchmark must identify the clock domain")
    conditions = benchmark.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Benchmark conditions must be nonempty")
    ids = [row["id"] for row in conditions]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate benchmark condition ids")
    ranks = data["environment"]["topology"].get("ranks")
    if not isinstance(ranks, list) or not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("topology.ranks must explicitly list unique ranks")
    if any(type(rank) is not int or rank < 0 for rank in ranks):
        raise ValueError("topology.ranks must contain nonnegative integers")
    validation_contract = None
    for phase in ("before", "after"):
        report = read_json(evidence.artifact(data["precision"][phase]))
        if any(
            report.get(field) is not True
            for field in ("passed", "outputs_finite", "contract_equal")
        ):
            raise ValueError(f"Precision {phase} did not pass the output contract")
        if report.get("clock_id") != benchmark["clock_id"]:
            raise ValueError("Precision and benchmark must share a clock domain")
        if set(report.get("ranks", [])) != set(ranks) or set(report.get("conditions", [])) != set(
            ids
        ):
            raise ValueError(f"Precision {phase} must cover all ranks and conditions")
        if report.get("errors") or not report.get("validation_contract"):
            raise ValueError(f"Precision {phase} must identify a clean validation contract")
        if phase == "before":
            validation_contract = report["validation_contract"]
        elif report["validation_contract"] != validation_contract:
            raise ValueError("Precision before and after must use the same validation contract")
        report_start = number(report.get("started_ns"), f"precision.{phase}.started_ns")
        report_end = number(
            report.get("finished_ns"), f"precision.{phase}.finished_ns", positive=True
        )
        if report_end <= report_start:
            raise ValueError(f"Precision {phase} timing window must be positive")
        if (phase == "before" and report_end > start) or (phase == "after" and report_start < end):
            raise ValueError("Required order: precision before -> benchmark -> precision after")
    result = {}
    for row in conditions:
        if not row.get("spec"):
            raise ValueError(
                "Each condition must describe input/output lengths, batch or concurrency"
            )
        samples = row.get("samples")
        if not isinstance(samples, list) or len(samples) < policy.min_samples:
            raise ValueError(f"{row['id']}: insufficient samples")
        stats = {"spec": row["spec"], "samples": len(samples)}
        for metric in ("forward_ms", "tokens_per_s"):
            values = [number(sample.get(metric), metric, positive=True) for sample in samples]
            cv = statistics.stdev(values) / statistics.mean(values)
            if cv > policy.max_cv:
                raise ValueError(f"{row['id']}: {metric} noise CV={cv:.4f} exceeds policy")
            stats[metric] = statistics.median(values)
            stats[f"{metric}_cv"] = cv
        result[row["id"]] = stats
    return result


def compare(baseline: Evidence, candidate: Evidence, policy: Policy) -> Decision:
    reasons = []
    for field in ("kind", "environment", "workload"):
        if baseline.data[field] != candidate.data[field]:
            reasons.append(f"incomparable {field}")
    if reasons:
        return Decision(False, reasons, {})
    try:
        base = validate(baseline, policy)
        trial = validate(candidate, policy)
        base_contract = read_json(baseline.artifact(baseline.data["precision"]["before"]))
        trial_contract = read_json(candidate.artifact(candidate.data["precision"]["before"]))
        if base_contract["validation_contract"] != trial_contract["validation_contract"]:
            return Decision(False, ["precision validation contract differs"], {})
    except (ValueError, KeyError, TypeError, OSError) as error:
        return Decision(False, [f"invalid evidence: {error}"], {})
    if base.keys() != trial.keys():
        return Decision(False, ["condition matrix differs"], {})
    metrics = {}
    for condition in base:
        old, new = base[condition], trial[condition]
        if old["spec"] != new["spec"]:
            reasons.append(f"{condition}: condition spec differs")
            continue
        latency_gain = 1 - new["forward_ms"] / old["forward_ms"]
        throughput_gain = new["tokens_per_s"] / old["tokens_per_s"] - 1
        metrics[condition] = {
            "latency_gain": latency_gain,
            "throughput_gain": throughput_gain,
            "baseline": old,
            "candidate": new,
        }
        gain = latency_gain if policy.objective == "latency" else throughput_gain
        if gain < policy.min_gain:
            reasons.append(f"{condition}: {policy.objective} gain below {policy.min_gain:.1%}")
        if min(latency_gain, throughput_gain) < -policy.max_regression:
            reasons.append(f"{condition}: forward latency or throughput regressed beyond policy")
    return Decision(
        not reasons, reasons or ["all conditions passed precision and performance gates"], metrics
    )
