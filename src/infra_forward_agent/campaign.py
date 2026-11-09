"""Coordinate five domain plans across registered architectures without launching hardware jobs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .agents.base import PlanningContext, StagePlan
from .agents.deployment import DeploymentAgent
from .agents.kernel import KernelAgent
from .agents.performance import PerformanceAgent
from .agents.testing import TestingAgent
from .agents.traces import TraceAgent
from .architectures import ArchitectureRegistry
from .contracts import read_json
from .scenarios import expand_suite
from .store import RunStore, write_json


def validate_dependencies(stages: list[StagePlan]) -> None:
    seen = set()
    for stage in stages:
        if stage.id in seen or any(parent not in seen for parent in stage.depends_on):
            raise ValueError(f"Duplicate, missing or out-of-order stage dependency: {stage.id}")
        seen.add(stage.id)


def _paths(values: object, root: Path, label: str) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"{label} must be a list of paths")
    paths = [(root / value).resolve() for value in values]
    if any(not path.exists() for path in paths):
        raise ValueError(f"Missing {label} path")
    return [str(path) for path in paths]


def build_plan(config_path: Path) -> dict:
    config_path = config_path.resolve()
    data = read_json(config_path)
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("mode") != "plan":
        raise ValueError("Campaign requires schema_version=1 and mode=plan")
    if not isinstance(data.get("id"), str) or not data["id"].strip():
        raise ValueError("Campaign id must be a nonempty string")
    registry = ArchitectureRegistry.builtin()
    for path in _paths(
        data.get("architecture_profiles", []), config_path.parent, "architecture_profiles"
    ):
        registry.load(Path(path))
    ids = data.get("architectures")
    if not isinstance(ids, list) or not ids or not all(isinstance(item, str) for item in ids):
        raise ValueError("architectures must be a nonempty list of architecture ids")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate campaign architectures")
    profiles = [registry.get(architecture) for architecture in ids]
    suite_path = data.get("scenario_suite")
    if not isinstance(suite_path, str):
        raise ValueError("scenario_suite must point to a scenario suite JSON")
    suite = read_json(config_path.parent / suite_path)
    scenarios = expand_suite(suite, max_cases=data.get("max_scenarios", 1000))
    testing_contract = {
        "suite_schema_version": suite["schema_version"],
        "focus": suite.get("focus"),
        "generation": suite.get("generation", {}),
        "enabled_tiers": suite.get("enabled_tiers", []),
        "length_mixes": suite.get("length_mixes", {}),
        "contracts": suite.get("contracts", {}),
        "runtime_acceptance_implemented": False,
    }
    deployment = data.get("deployment", {})
    if not isinstance(deployment, dict):
        raise ValueError("deployment must be an object")
    deployment = {"objective": "balanced", **deployment}
    if deployment["objective"] not in {"capacity", "performance", "balanced"}:
        raise ValueError("deployment.objective must be capacity, performance or balanced")
    knowledge = _paths(data.get("knowledge_paths", []), config_path.parent, "knowledge")
    traces = data.get("trace_comparison", {})
    if not isinstance(traces, dict) or type(traces.get("enabled", True)) is not bool:
        raise ValueError("trace_comparison must be an object with boolean enabled")
    pairs = traces.get("pairs", {})
    if not isinstance(pairs, dict) or set(pairs) - set(ids):
        raise ValueError("Trace pairs must be keyed by selected architecture id")
    stages = []
    for profile in profiles:
        pair = pairs.get(profile.id, {})
        if not isinstance(pair, dict):
            raise ValueError("Each trace pair must be an object")
        if pair and (not pair.get("baseline") or not pair.get("candidate")):
            raise ValueError("Each trace pair requires both baseline and candidate")
        paths = _paths(
            [pair[key] for key in ("baseline", "candidate")] if pair else [],
            config_path.parent,
            "trace pair",
        )
        if any(not Path(path).is_file() for path in paths):
            raise ValueError("Trace pairs must reference files")
        trace_input = dict(zip(("baseline", "candidate"), paths))
        context = PlanningContext(
            profile,
            scenarios,
            deployment,
            {"enabled": traces.get("enabled", True), **trace_input},
            knowledge,
            testing_contract=testing_contract,
        )
        stages.append(
            StagePlan(
                id=context.task_id("preflight"),
                module="preflight",
                architecture=profile.id,
                depends_on=[],
                scope="live environment and resource qualification",
                required_inputs=profile.verification_required
                + ["campaign model/image/workload identity"],
                expected_outputs=[
                    "verified architecture capability/resource snapshot",
                    "device ownership and environment identity",
                ],
            )
        )
        for agent in (
            TestingAgent(),
            PerformanceAgent(),
            DeploymentAgent(),
            KernelAgent(),
            TraceAgent(),
        ):
            stages.append(agent.plan(context))
    validate_dependencies(stages)
    return {
        "schema_version": 1,
        "kind": "campaign_plan",
        "id": data["id"],
        "execution_ready": False,
        "architectures": [profile.to_dict() for profile in profiles],
        "target": data.get("target", {}),
        "constraints": data.get("constraints", {}),
        "budget": data.get("budget", {}),
        "knowledge_paths": knowledge,
        "testing_contract": testing_contract,
        "scenario_count_per_architecture": len(scenarios),
        "scenario_count_total": len(scenarios) * len(profiles),
        "scenarios": scenarios,
        "stages": [asdict(stage) for stage in stages],
        "limitations": [
            "Planning only: no service, profiler, model API, optimizer or deployment is executed.",
            "Target capabilities and resources require live verification.",
            "Knowledge paths are references, not automatic retrieval or executable policy.",
            "Service, forward and operator evidence have separate measurement boundaries.",
            "Distributed selection and trace comparison algorithms are not implemented yet.",
            "Budgets and allowed-change constraints are recorded, not enforced by the planner.",
        ],
    }


def write_plan(config_path: Path, output: Path, sink=None) -> dict:
    store = RunStore(output, sink)
    store.emit("planning_started")
    try:
        plan = build_plan(config_path)
        write_json(output / "config.json", read_json(config_path))
        write_json(output / "campaign.plan.json", plan)
        write_json(output / "scenario.matrix.json", plan["scenarios"])
        write_json(output / "testing.contract.json", plan["testing_contract"])
        lines = [
            "# Multi-architecture campaign plan",
            "",
            f"Campaign: {plan['id']}",
            "",
            "Planning only; hardware execution and optimization are not implemented.",
            "",
            f"Scenario count across architectures: {plan['scenario_count_total']}",
            "",
            "| Stage | Architecture | State | Depends on |",
            "| --- | --- | --- | --- |",
        ]
        for stage in plan["stages"]:
            lines.append(
                f"| {stage['module']} | {stage['architecture']} | {stage['status']} "
                f"| {', '.join(stage['depends_on'])} |"
            )
            store.emit(
                "stage_planned", id=stage["id"], module=stage["module"], status=stage["status"]
            )
        (output / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        store.emit(
            "planning_completed",
            campaign=plan["id"],
            stages=len(plan["stages"]),
            scenarios=plan["scenario_count_total"],
            execution_ready=False,
        )
        return plan
    except Exception as error:
        store.emit("planning_failed", error_type=type(error).__name__, message=str(error))
        raise
