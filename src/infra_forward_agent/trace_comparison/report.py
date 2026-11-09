from __future__ import annotations

import hashlib
import json
import shutil
import statistics
from pathlib import Path

from ..store import RunStore, write_json
from .compare import DEVICE_FIELDS, build_comparison


def summarize_pairs(result: dict) -> list[dict]:
    output = []
    for coverage in result["coverage"]:
        cid = coverage["condition_id"]
        if not coverage["complete_matched_repeats"]:
            output.append({"condition_id": cid, "status": "insufficient_evidence", "devices": {}})
            continue
        pairs = [p for p in result["pairs"] if p["condition_id"] == cid]
        devices = {}
        for rank in pairs[0]["devices"]:
            rows = [p["devices"][rank] for p in pairs]
            fields = {}
            for field in DEVICE_FIELDS:
                values = [row["deltas"][field] for row in rows]
                if any(v is None for v in values):
                    fields[field] = None
                    continue
                changes = [v["candidate_minus_baseline"] for v in values]
                fields[field] = {
                    "baseline_median": statistics.median(v["baseline"] for v in values),
                    "candidate_median": statistics.median(v["candidate"] for v in values),
                    "paired_delta_median": statistics.median(changes),
                    "paired_delta_min": min(changes),
                    "paired_delta_max": max(changes),
                    "matched_windows": len(values),
                    "is_native_timing": False,
                    "tail_slo_qualified": False,
                }
            devices[rank] = fields
        output.append(
            {"condition_id": cid, "status": "descriptive_matched_windows", "devices": devices}
        )
    return output


def write_comparison(path: Path, output: Path, sink=None) -> dict:
    store = RunStore(output, sink)
    store.emit("trace_comparison_started")
    try:
        result, sources = build_comparison(path)
        result["paired_summary"] = summarize_pairs(result)
        inputs = output / "input"
        inputs.mkdir()
        shutil.copyfile(path, inputs / "trace-comparison.json")
        manifest = {"trace-comparison.json": hashlib.sha256(path.read_bytes()).hexdigest()}
        for relative in list(sources.manifest):
            destination = inputs / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sources.path(relative), destination)
            manifest[relative] = sources.manifest[relative]
        write_json(inputs / "manifest.json", manifest)
        write_json(output / "trace.comparison.json", result)
        write_json(output / "bottleneck-migration.json", result["bottleneck_migration"])
        write_json(output / "next-tasks.json", result["next_tasks"])
        with (output / "normalized.events.jsonl").open("w", encoding="utf-8") as handle:
            for pair in result["pairs"]:
                for side, capture in pair["captures"].items():
                    for event in capture["normalized_events"]:
                        handle.write(
                            json.dumps(
                                {
                                    "pair_id": pair["id"],
                                    "condition_id": pair["condition_id"],
                                    "side": side,
                                    **event,
                                },
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                store.emit(
                    "trace_pair_compared",
                    pair=pair["id"],
                    status=pair["status"],
                    issues=pair["issues"],
                )
        lines = [
            "# Prefill trace alignment and comparison",
            "",
            f"Evidence kind: **{result['kind']}**; status: **{result['status']}**.",
            "",
            "Profile deltas are observations, not service speedup or causal critical path.",
            "",
            "| Condition / rank | Profile window us before -> after | Busy us before -> after |",
            "| --- | --- | --- |",
        ]
        for summary in result["paired_summary"]:
            for rank, fields in summary["devices"].items():
                wall, busy = fields["captured_forward_us"], fields["gpu_busy_union_us"]
                lines.append(
                    f"| {summary['condition_id']}/{rank} "
                    f"| {wall['baseline_median']:.2f} -> {wall['candidate_median']:.2f} "
                    f"| {busy['baseline_median']:.2f} -> {busy['candidate_median']:.2f} |"
                )

        def shown(change: dict | None) -> str:
            if change is None:
                return "unknown"
            return f"{change['baseline']:.2f} -> {change['candidate']:.2f}"

        lines += [
            "",
            "## Matched operator units (all supplied pairs)",
            "",
            "GPU work is not wrapper latency; share changes are percentage points.",
            "",
            "| Pair/rank | Semantic unit | Work us before -> after | Share delta pp | "
            "Exclusive us before -> after | State |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for pair in result["pairs"]:
            for rank, device in pair["devices"].items():
                for group in device["operators"]:
                    key = group["semantic_key"].replace("|", "\\|").replace("\n", " ")
                    changes = group["deltas"]
                    share = (
                        f"{changes['work_share']['percentage_point_change']:+.2f}"
                        if changes
                        else "unknown"
                    )
                    lines.append(
                        f"| {pair['id']}/{rank} | {key} "
                        f"| {shown(changes['work_us'] if changes else None)} | {share} "
                        f"| {shown(changes['exclusive_us'] if changes else None)} "
                        f"| {group['status']} |"
                    )
        lines += [
            "",
            "## Communication and distinct-stream overlap",
            "",
            "| Pair/rank | Comm union us | C/N intersection us | Comm without compute us | "
            "Multi-stream us |",
            "| --- | --- | --- | --- | --- |",
        ]
        for pair in result["pairs"]:
            for rank, device in pair["devices"].items():
                fields = device["deltas"]
                lines.append(
                    f"| {pair['id']}/{rank} | {shown(fields['communication_union_us'])} "
                    f"| {shown(fields['compute_communication_overlap_us'])} "
                    f"| {shown(fields['communication_without_compute_us'])} "
                    f"| {shown(fields['multi_stream_active_us'])} |"
                )
        lines += [
            "",
            "## Logical stream roles",
            "",
            "| Pair/rank | Role | Busy us before -> after |",
            "| --- | --- | --- |",
        ]
        for pair in result["pairs"]:
            for rank, device in pair["devices"].items():
                for role, change in device["stream_role_busy_deltas"].items():
                    role = role.replace("|", "\\|").replace("\n", " ")
                    lines.append(f"| {pair['id']}/{rank} | {role} | {shown(change)} |")
        lines += ["", "## Alignment and missing evidence", ""]
        for pair in result["pairs"]:
            label = pair["id"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"- {label}: {pair['status']}; {'; '.join(pair['issues']) or 'matched'}")
        lines += [
            "",
            "## Migration hypotheses",
            "",
            "| Condition/rank | State | Next action |",
            "| --- | --- | --- |",
        ]
        for row in result["bottleneck_migration"]:
            lines.append(
                f"| {row['condition_id']}/{row['rank']} | {row['status']} "
                f"| {row.get('next_action', 'collect dependency evidence')} |"
            )
        native = result["native_evidence"]
        lines += [
            "",
            f"Native evidence: {native['status']}; accepted={native['accepted']}; "
            f"real_gain_established={native['real_gain_established']}.",
            "",
            *result["limitations"],
        ]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        store.emit(
            "trace_comparison_completed",
            status=result["status"],
            native_accepted=native["accepted"],
            auto_deploy=False,
        )
        return result
    except Exception as error:
        store.emit("trace_comparison_failed", error_type=type(error).__name__, message=str(error))
        raise
