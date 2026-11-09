from .base import PlanningContext, StagePlan


class PerformanceAgent:
    def plan(self, context: PlanningContext) -> StagePlan:
        return StagePlan(
            id=context.task_id("performance"),
            module="performance",
            architecture=context.architecture.id,
            depends_on=[context.task_id("testing")],
            scope="service/forward/operator",
            required_inputs=[
                "per-condition raw samples",
                "forward phase and synchronization boundary",
                "optional all-rank traces",
                "optional operator counter reports",
            ],
            expected_outputs=[
                "service TTFT/TPOT and throughput report",
                "prefill/decode forward latency and effective-token throughput",
                "operator duration and work-share report with explicit denominator",
                "communication/stream overlap hypotheses and missing evidence",
            ],
            options={
                "measurement_scopes": ["service", "forward", "operator"],
                "operator_work_share_denominator": "sum of included GPU event durations",
                "kernel_work_sum_is_forward_latency": False,
                "trace_is_optional": True,
            },
        )
