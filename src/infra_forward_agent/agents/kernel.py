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
                "all-rank instrumentation and source-correlated counters or validated proxies",
                "matched resource limit model and falsifiable quantitative hypothesis",
                "shape/routing matrix",
                "verified ISA/compiler capabilities and resource limits",
                "stream/collective dependency and buffer lifetime",
            ],
            expected_outputs=[
                "isolated candidate patch and parent revision",
                "operator correctness and counter evidence",
                "whole-forward requalification request",
                "full declared service matrix acceptance and preserved resident KV capacity",
            ],
            options={
                "architecture_profile": context.architecture.to_dict(),
                "knowledge_references": context.knowledge,
                "cross_architecture_transfer_requires_revalidation": True,
                "contract": context.kernel_contract,
                "deep_analysis_required": True,
                "optimizer_bridge": "offline KDA-style task export; no automatic KDA invocation",
                "task_export": "ifa kernel-task <contract.json> --output <new-directory>",
                "evidence_gate": "ifa kernel-evaluate <bundle.json> --output <new-directory>",
            },
        )
