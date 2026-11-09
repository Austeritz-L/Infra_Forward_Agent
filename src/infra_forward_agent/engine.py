"""Bounded artifact replay loop. Live execution and LLM calls stay behind adapter contracts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .analysis import analyze, propose
from .contracts import Evidence, Policy, read_json
from .evaluation import compare, validate
from .store import RunStore, write_json
from .trace import Trace, load_chrome


def analyze_evidence(evidence: Evidence) -> dict:
    slices = []
    windows = {}
    ignored = 0
    # Keep rank files separate until normalization; no cross-host clock arithmetic.
    for spec in evidence.data["traces"]:
        trace = load_chrome(
            evidence.artifact(spec["path"]), rank=spec["rank"], marker=spec.get("marker", "forward")
        )
        if windows.keys() & trace.windows.keys():
            raise ValueError("Duplicate rank traces; provide one selected forward per rank")
        slices.extend(trace.slices)
        windows.update(trace.windows)
        ignored += trace.ignored_events
    expected = set(evidence.data["environment"]["topology"]["ranks"])
    observed = {item.rank for item in slices}
    if observed != expected or set(windows) != expected:
        raise ValueError(f"Trace rank coverage differs: expected {expected}, observed {observed}")
    return analyze(Trace(slices, windows, ignored))


def replay(config_path: Path, output: Path, sink=None) -> dict:
    config = read_json(config_path)
    if config.get("schema_version") != 1 or config.get("mode") != "replay":
        raise ValueError("Only schema_version=1, mode=replay is implemented")
    limit = config.get("max_iterations", 3)
    if type(limit) is not int or limit < 1:
        raise ValueError("max_iterations must be a positive integer")
    policy = Policy(**config.get("policy", {}))
    store = RunStore(output, sink)
    store.emit("run_started", mode="replay", policy=asdict(policy))
    try:
        write_json(output / "config.json", config)
        baseline = store.snapshot(
            Evidence.load(config_path.parent / config["baseline"]), "baseline"
        )
        validate(baseline, policy)
        analysis = analyze_evidence(baseline)
        write_json(output / "baseline.analysis.json", analysis)
        write_json(output / "tasks.000.json", propose(analysis))
        store.emit("baseline_analyzed", evidence=baseline.id, kind=baseline.data["kind"])
        initial = baseline.id
        decisions = []
        candidates = config.get("candidates", [])
        for index, path in enumerate(candidates[:limit], start=1):
            store.emit("candidate_import_started", iteration=index)
            candidate = store.snapshot(
                Evidence.load(config_path.parent / path), f"candidate-{index:03d}"
            )
            decision = compare(baseline, candidate, policy)
            # A candidate cannot become the next baseline without a complete analyzable trace.
            if decision.accepted:
                analysis = analyze_evidence(candidate)
                write_json(output / f"iteration.{index:03d}.analysis.json", analysis)
                write_json(output / f"tasks.{index:03d}.json", propose(analysis))
            record = {
                "iteration": index,
                "parent": baseline.id,
                "candidate": candidate.id,
                **asdict(decision),
            }
            decisions.append(record)
            write_json(output / f"decision.{index:03d}.json", record)
            store.emit(
                "candidate_accepted" if decision.accepted else "candidate_rejected", **record
            )
            if decision.accepted:
                baseline = candidate
                store.emit("baseline_advanced", evidence=baseline.id)
        summary = {
            "schema_version": 1,
            "mode": "replay",
            "kind": baseline.data["kind"],
            "initial_baseline": initial,
            "selected": baseline.id,
            "decisions": decisions,
            "stop_reason": "iteration_budget"
            if len(candidates) > limit
            else "candidates_exhausted",
        }
        write_json(output / "summary.json", summary)
        lines = [
            "# Forward optimization replay",
            "",
            f"Evidence kind: **{summary['kind']}**",
            "",
            "Synthetic results validate the workflow only; no hardware speedup is claimed.",
            "",
            f"Initial baseline: `{initial}`; selected evidence: `{baseline.id}`.",
            "",
            "| Iteration | Parent | Candidate | Accepted | Reason |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in decisions:
            reason = "; ".join(row["reasons"]).replace("|", "\\|")
            lines.append(
                f"| {row['iteration']} | {row['parent']} | {row['candidate']} "
                f"| {row['accepted']} | {reason} |"
            )
        lines += [
            "",
            f"Stop: {summary['stop_reason']}.",
            "",
            "See decision JSON for per-condition latency, throughput, CV and relative gains.",
            "See tasks JSON for the next operator / communication / overlap / distributed work.",
        ]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        store.emit("run_completed", selected=baseline.id, stop_reason=summary["stop_reason"])
        return summary
    except Exception as error:
        store.emit("run_failed", error_type=type(error).__name__, message=str(error))
        raise
