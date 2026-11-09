from .base import PlanningContext, StagePlan


class DeploymentAgent:
    def plan(self, context: PlanningContext) -> StagePlan:
        return StagePlan(
            id=context.task_id("deployment"),
            module="deployment",
            architecture=context.architecture.id,
            depends_on=[context.task_id("performance")],
            scope="fixed-resource distributed deployment",
            required_inputs=[
                "physical resource budget and interconnect topology",
                "per-rank weights/KV/state/workspace/graph/peak memory",
                "global workload and routing distribution",
                "measured topology candidates",
            ],
            expected_outputs=[
                "per-replica/rank TP/EP/PP/DP/CP candidate table with effective KV sharding",
                "KV capacity versus measured performance tradeoff",
                "constraint violations, unmeasured options and Pareto candidates",
                "exact 1Mi/2Mi context support, quality and admission qualification",
            ],
            options={
                **context.deployment,
                "contract": context.deployment_contract,
                "offline_selector": "ifa deployment <bundle.json> --output <new-directory>",
                "auto_deploy": False,
                "capacity_without_memory_evidence": "unknown",
                "legacy_replay_topology_change": "rejected_as_incomparable",
            },
        )
