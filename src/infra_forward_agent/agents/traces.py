from .base import PlanningContext, StagePlan


class TraceAgent:
    def plan(self, context: PlanningContext) -> StagePlan:
        enabled = context.traces.get("enabled", True)
        baseline, candidate = context.traces.get("baseline"), context.traces.get("candidate")
        status = "skipped" if not enabled or not (baseline and candidate) else "needs_adapter"
        return StagePlan(
            id=context.task_id("traces"),
            module="traces",
            architecture=context.architecture.id,
            depends_on=[context.task_id("performance"), context.task_id("kernel")],
            scope="optional matched forward trace pair",
            required_inputs=[
                "baseline/candidate trace pair with matched scenario and phase",
                "all-rank coverage and event-name/correlation mappings",
            ],
            expected_outputs=[
                "per-rank/device forward and operator deltas",
                "communication/stream overlap deltas",
                "bottleneck migration hypotheses and attribution limitations",
            ],
            status=status,
            options={
                "baseline": baseline,
                "candidate": candidate,
                "reason": "disabled"
                if not enabled
                else ("trace pair provided" if baseline and candidate else "no trace pair yet"),
                "causal_critical_path": "requires_dependency_evidence",
            },
        )
