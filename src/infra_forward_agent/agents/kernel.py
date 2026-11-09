from .base import PlanningContext, StagePlan


class KernelAgent:
    def plan(self, context: PlanningContext) -> StagePlan:
        return StagePlan(
            id=context.task_id("kernel"),
            module="kernel",
            architecture=context.architecture.id,
            depends_on=[context.task_id("performance"), context.task_id("deployment")],
            scope="operator/communication/fusion/stream scheduling",
            required_inputs=[
                "hotspot and math/layout/quantization contract",
                "shape/routing matrix",
                "verified ISA/compiler capabilities and resource limits",
                "stream/collective dependency and buffer lifetime",
            ],
            expected_outputs=[
                "isolated candidate patch and parent revision",
                "operator correctness and counter evidence",
                "whole-forward requalification request",
            ],
            options={
                "architecture_profile": context.architecture.to_dict(),
                "knowledge_references": context.knowledge,
                "cross_architecture_transfer_requires_revalidation": True,
                "optimizer_bridge": "planned",
            },
        )
