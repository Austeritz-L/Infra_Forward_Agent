"""Synthetic resource ceilings and overlapping traces. No real device characteristics."""

from pathlib import Path

from ..store import write_json


def make_performance_demo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "kind": "synthetic",
        "architecture": "synthetic-device",
        "environment": "synthetic-only",
        "workload_id": "synthetic-prefill-case",
    }
    write_json(
        root / "ceilings-source.json", {"kind": "synthetic", "note": "No hardware rates measured"}
    )
    write_json(
        root / "quantities-source.json",
        {"kind": "synthetic", "note": "Invented fixture work/traffic only"},
    )
    ceilings = []
    for resource, rate, unit in (("tensor", 1e13, "flop/s"), ("hbm", 1e12, "byte/s")):
        ceilings.append(
            {
                "id": resource,
                "resource": resource,
                "rate": rate,
                "unit": unit,
                "basis": "architectural_peak",
                "source": "ceilings-source.json",
                "compatibility_key": f"synthetic-{resource}-contract",
                **{key: identity[key] for key in ("kind", "architecture", "environment")},
            }
        )

    def quantity(resource: str, amount: float, unit: str) -> dict:
        return {
            "resource": resource,
            "amount": amount,
            "unit": unit,
            "ceiling_id": resource,
            "compatibility_key": f"synthetic-{resource}-contract",
            "amount_basis": "analytic_executed",
            "source": "quantities-source.json",
        }

    traces, observations, budgets = [], [], []
    for rank in (0, 1):
        offset = rank * 10000
        events = [
            {
                "ph": "X",
                "name": "forward",
                "cat": "user_annotation",
                "ts": offset,
                "dur": 1200,
                "args": {"rank": rank},
            }
        ]
        definitions = [
            ("expert_gemm", "grouped_gemm", "gemm-fixture", 0, 600, 0, "compute"),
            ("all_to_all", "collective", "comm-fixture", 400, 500, 1, "communication"),
            ("norm", "normalization", "norm-fixture", 900, 100, 0, "compute"),
            ("copy", "copy", "copy-fixture", 1000, 100, 2, "memory"),
        ]
        for name, compute_type, signature, start, duration, stream, kind in definitions:
            events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": name + "_kernel",
                    "ts": offset + start,
                    "dur": duration,
                    "args": {
                        "rank": rank,
                        "device": str(rank),
                        "stream": stream,
                        "ifa_kind": kind,
                        "ifa_operator": name,
                        "ifa_compute_type": compute_type,
                        "ifa_signature": signature,
                        "ifa_invocation": "step-1:" + name,
                    },
                }
            )
        path = f"rank-{rank}.trace.json"
        write_json(root / path, {"traceEvents": events})
        traces.append({"capture_id": "representative", "path": path, "rank": rank})
        for name, compute_type, signature, elapsed, quantities in (
            (
                "expert_gemm",
                "grouped_gemm",
                "gemm-fixture",
                500,
                [quantity("tensor", 4e9, "flop"), quantity("hbm", 1e8, "byte")],
            ),
            ("norm", "normalization", "norm-fixture", 100, [quantity("hbm", 1e7, "byte")]),
        ):
            native_path = f"native-{rank}-{name}.json"
            write_json(
                root / native_path,
                {
                    "identity": identity,
                    "scope": "operator_invocation",
                    "profiled": False,
                    "rank": rank,
                    "device": str(rank),
                    "operator": name,
                    "compute_type": compute_type,
                    "signature": signature,
                    "samples_us": [elapsed * factor for factor in (0.998, 0.999, 1, 1.001, 1.002)],
                },
            )
            observations.append(
                {
                    "rank": rank,
                    "device": str(rank),
                    "operator": name,
                    "compute_type": compute_type,
                    "signature": signature,
                    "native": {"source": native_path},
                    "resources": quantities,
                }
            )
        budget_native = f"native-rank-{rank}.json"
        write_json(
            root / budget_native,
            {
                "identity": identity,
                "scope": "prefill_rank_interval",
                "rank": rank,
                "signature": "prefill-step",
                "step_contract": "synthetic-matched-step",
                "all_streams_joined": True,
                "profiled": False,
                "samples_us": [1098, 1099, 1100, 1101, 1102],
            },
        )
        budgets.append(
            {
                "rank": rank,
                "signature": "prefill-step",
                "step_contract": "synthetic-matched-step",
                "global_new_tokens": 8192,
                "native": {"source": budget_native},
                "resources": [quantity("tensor", 4e9, "flop"), quantity("hbm", 1.2e8, "byte")],
            }
        )
    write_json(
        root / "forward.json",
        {
            "identity": identity,
            "profiled": False,
            "scope": "coordinated_prefill_forward",
            "boundary_id": "synthetic-all-streams",
            "all_streams_joined": True,
            "rank_coverage": [0, 1],
            "global_new_tokens_per_sample": 8192,
            "samples_us": [1098, 1099, 1100, 1101, 1102],
        },
    )
    requests = []
    for index in range(20):
        start = index * 1000
        requests.append(
            {
                "id": str(index),
                "clock_id": "client-clock",
                "sent_us": start,
                "first_token_us": start + 1100,
                "terminal_us": start + 1200,
                "status": "completed",
                "done": True,
                "errors": [],
                "output_tokens": 1,
                "input_tokens": 8192,
                "cached_tokens": 0,
                "new_tokens": 8192,
            }
        )
    write_json(
        root / "service.json",
        {
            "identity": identity,
            "profiled": False,
            "cohort": "fixed_work_drained",
            "clock_id": "client-clock",
            "requests": requests,
        },
    )
    path = root / "bundle.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "focus": "prefill",
            "identity": identity,
            "expected_ranks": [0, 1],
            "service": "service.json",
            "forward": "forward.json",
            "traces": traces,
            "ceilings": ceilings,
            "operator_resources": observations,
            "step_resource_budgets": budgets,
        },
    )
    return path
