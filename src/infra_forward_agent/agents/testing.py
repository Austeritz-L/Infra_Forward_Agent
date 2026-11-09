from .base import PlanningContext, StagePlan


class TestingAgent:
    def plan(self, context: PlanningContext) -> StagePlan:
        return StagePlan(
            id=context.task_id("testing"),
            module="testing",
            architecture=context.architecture.id,
            depends_on=[context.task_id("preflight")],
            scope="service/forward/operator",
            required_inputs=[
                "scenario matrix",
                "image and weights identity",
                "reference outputs",
                "cache-state preparation",
                "service and forward runners",
            ],
            expected_outputs=[
                "scenario coverage and skip reasons",
                "all-rank correctness reports",
                "unprofiled raw samples",
                "service failures and timeouts",
            ],
            options={
                "scenario_ids": [row["id"] for row in context.scenarios],
                "sequence": ["precision_before", "unprofiled_timing", "precision_after"],
                "service_and_forward_metrics_separate": True,
            },
        )
