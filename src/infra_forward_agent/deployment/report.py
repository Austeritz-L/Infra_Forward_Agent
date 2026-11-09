from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from ..store import RunStore, write_json
from .selection import artifact, select_deployment


def write_deployment_report(path: Path, output: Path, sink=None) -> dict:
    store = RunStore(output, sink)
    store.emit("deployment_selection_started")
    try:
        result = select_deployment(path)
        inputs = output / "input"
        inputs.mkdir()
        shutil.copyfile(path, inputs / "deployment.json")
        manifest = {"deployment.json": hashlib.sha256(path.read_bytes()).hexdigest()}
        for relative in result["artifact_sources"]:
            destination = inputs / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact(path.parent.resolve(), relative), destination)
            manifest[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
        write_json(inputs / "manifest.json", manifest)
        write_json(output / "deployment.selection.json", result)
        lines = [
            "# Distributed capacity/performance selection",
            "",
            f"Evidence kind: **{result['kind']}**; automatic deployment: **false**.",
            "",
            f"Total tokens per context (including output reserve): "
            f"{result['target']['context_tokens']:,}.",
            "",
            "Capacity counts independent resident contexts; prefix hits do not shrink context.",
            "",
            "| Candidate | Estimated slots | Validated slots | Eligible | Reasons |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in result["candidates"]:
            label = row["id"].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {label} | {row['budget']['estimated_total_context_slots']} "
                f"| {row['verified_context_slots']} | {row['eligible']} "
                f"| {'; '.join(row['reasons'])} |"
            )
            store.emit(
                "deployment_candidate_evaluated",
                candidate=row["id"],
                eligible=row["eligible"],
                reasons=row["reasons"],
            )
        lines += ["", "Rankings (synthetic inputs are demonstration only):", ""]
        for objective, ranking in result["rankings"].items():
            lines.append(
                f"- {objective}: {', '.join(ranking) or 'inconclusive'}; "
                f"recommendation={result['recommendations'][objective]}"
            )
        lines += ["", *result["limitations"]]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        store.emit(
            "deployment_selection_completed",
            kind=result["kind"],
            recommendations=result["recommendations"],
            auto_deploy=False,
        )
        return result
    except Exception as error:
        store.emit(
            "deployment_selection_failed", error_type=type(error).__name__, message=str(error)
        )
        raise
