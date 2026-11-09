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
                "matched per-resource peaks, sustained probes and same-shape reference baselines",
                "ISA/dtype/shape/cache/topology counting contracts and raw counter exports",
            ],
            expected_outputs=[
                (
                    "service TTFT and input/new-prefill throughput report"
                    if context.testing_contract.get("focus") == "prefill"
                    else "service TTFT/TPOT and throughput report"
                ),
                (
                    "prefill forward latency and new-token throughput"
                    if context.testing_contract.get("focus") == "prefill"
                    else "prefill/decode forward latency and effective-token throughput"
                ),
                "operator duration and work-share report with explicit denominator",
                "communication/stream overlap hypotheses and missing evidence",
                "resource-limit and efficiency-gap hypotheses with source/unknowns",
                "attack plan separating calibration, kernel efficiency and work/traffic reduction",
            ],
            options={
                "measurement_scopes": ["service", "forward", "operator"],
                "operator_work_share_denominator": "sum of included GPU event durations",
                "kernel_work_sum_is_forward_latency": False,
                "trace_is_optional": True,
                "focus": context.testing_contract.get("focus"),
                "decode_metrics_in_scope": context.testing_contract.get("focus") != "prefill",
                "performance_contract_reference": "performance.contract.json",
                "offline_analyzer": "ifa performance",
                "collector_status": "needs_adapter",
                "ceiling_levels": [
                    "architectural_peak",
                    "sustained_microbenchmark",
                    "matched_reference",
                ],
                "hotspot_is_confirmed_bottleneck": False,
            },
        )
