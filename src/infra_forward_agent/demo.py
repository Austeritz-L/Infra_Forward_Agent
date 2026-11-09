"""Deterministic synthetic fixture generator, never a GPU benchmark."""

from pathlib import Path

from .store import write_json


def make_demo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidates = []
    variants = [
        ("baseline", 10.0, 6000, 3000, 1000, True),
        ("kernel-candidate", 8.0, 2500, 4000, 1500, True),
        ("overlap-candidate", 7.2, 2500, 4000, 1200, True),
        ("incorrect-candidate", 6.0, 2000, 3000, 1000, False),
    ]
    for name, latency, expert, comm, attention, correct in variants:
        directory = root / name
        directory.mkdir()
        traces = []
        for rank in (0, 1):
            events = [
                {
                    "ph": "X",
                    "name": "forward",
                    "cat": "user_annotation",
                    "ts": 0,
                    "dur": latency * 1000,
                    "args": {"rank": rank},
                }
            ]
            comm_start = 2000 if name == "overlap-candidate" else expert
            for kernel, start, duration, stream, kind in (
                ("expert_gemm", 0, expert, 0, "compute"),
                ("collective_all_to_all", comm_start, comm, 1, "communication"),
                ("attention_decode", comm_start + comm, attention, 0, "compute"),
            ):
                events.append(
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": kernel,
                        "ts": start,
                        "dur": duration,
                        "pid": rank,
                        "tid": stream,
                        "args": {"rank": rank, "device": rank, "stream": stream, "ifa_kind": kind},
                    }
                )
            path = f"rank-{rank}.trace.json"
            write_json(directory / path, {"traceEvents": events})
            traces.append({"path": path, "rank": rank, "marker": "forward"})
        conditions = []
        for condition, concurrency in (("decode-c1", 1), ("decode-c16", 16)):
            multiplier = 1 if concurrency == 1 else 2
            samples = [
                {
                    "forward_ms": latency * multiplier * factor,
                    "tokens_per_s": (16000 / latency) / multiplier / factor,
                }
                for factor in (0.998, 0.999, 1.0, 1.001, 1.002)
            ]
            conditions.append(
                {
                    "id": condition,
                    "spec": {
                        "input_length": 1024,
                        "output_length": 128,
                        "concurrency": concurrency,
                        "phase": "decode",
                    },
                    "samples": samples,
                }
            )
        write_json(
            directory / "benchmark.json",
            {
                "completed": True,
                "profiled": False,
                "errors": [],
                "warmup_iterations": 3,
                "clock_id": "synthetic-clock",
                "started_ns": 20,
                "finished_ns": 30,
                "conditions": conditions,
            },
        )
        for phase, start, end in (("before", 1, 10), ("after", 40, 50)):
            write_json(
                directory / f"precision-{phase}.json",
                {
                    "passed": correct,
                    "outputs_finite": True,
                    "contract_equal": correct,
                    "clock_id": "synthetic-clock",
                    "started_ns": start,
                    "finished_ns": end,
                    "validation_contract": "synthetic-output-equality-v1",
                    "ranks": [0, 1],
                    "conditions": [row["id"] for row in conditions],
                    "errors": [],
                },
            )
        write_json(
            directory / "evidence.json",
            {
                "schema_version": 1,
                "kind": "synthetic",
                "id": name,
                "revision": f"fixture:{name}",
                "environment": {
                    "image_digest": "synthetic:no-container",
                    "hardware": "synthetic:no-accelerator",
                    "runtime": "synthetic",
                    "profiler": "fixture-v1",
                    "topology": {"ranks": [0, 1], "tp": 1, "ep": 2, "pp": 1, "dp": 1},
                },
                "workload": {
                    "model": "synthetic-moe",
                    "weights_digest": "fixture-v1",
                    "dtype": "bf16",
                    "seed": 42,
                    "input_digest": "fixture-input-v1",
                    "forward_boundary": "one synchronized decode step across all ranks",
                },
                "traces": traces,
                "benchmark": "benchmark.json",
                "precision": {"before": "precision-before.json", "after": "precision-after.json"},
            },
        )
        if name != "baseline":
            candidates.append(f"{name}/evidence.json")
    config = root / "replay.json"
    write_json(
        config,
        {
            "schema_version": 1,
            "mode": "replay",
            "max_iterations": 3,
            "baseline": "baseline/evidence.json",
            "candidates": candidates,
            "policy": {
                "objective": "latency",
                "min_gain": 0.03,
                "max_regression": 0.01,
                "min_samples": 5,
                "max_cv": 0.05,
            },
        },
    )
    return config
