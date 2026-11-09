"""Opt-in CPU correlation ranges; injected callbacks may attach NVTX/MUSA ranges.

These ranges never measure GPU completion, launch kernels or synchronize a device.
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter_ns
from typing import Callable, Iterator


@contextmanager
def operator_range(
    attributes: dict,
    *,
    enabled: bool = False,
    sink: Callable[[dict], None] | None = None,
    push: Callable[[str], None] | None = None,
    pop: Callable[[], None] | None = None,
) -> Iterator[None]:
    if not enabled:
        yield
        return
    required = {
        "rank",
        "device",
        "stream",
        "ifa_request_id",
        "ifa_step",
        "ifa_operator",
        "ifa_compute_type",
        "ifa_signature",
        "ifa_invocation",
        "ifa_layer",
        "ifa_global_new_tokens",
        "ifa_local_tokens",
        "ifa_query_len",
        "ifa_kv_len",
        "ifa_cache_tier",
    }
    if not required <= attributes.keys() or sink is None or bool(push) != bool(pop):
        raise ValueError("Enabled instrumentation needs complete correlation metadata and a sink")
    metadata = dict(attributes)
    if any(metadata[key] is None for key in required):
        raise ValueError(
            "Unknown shape/token fields require capture, not inferred placeholder values"
        )
    label = str(metadata["ifa_operator"]) + ":" + str(metadata["ifa_invocation"])
    if push:
        push(label)
    start = perf_counter_ns()
    completed = False
    try:
        yield
        completed = True
    finally:
        end = perf_counter_ns()
        try:
            if pop:
                pop()
        finally:
            sink(
                {
                    "name": label,
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": start / 1000,
                    "dur": (end - start) / 1000,
                    "args": {
                        **metadata,
                        "completed": completed,
                        "timing_scope": "cpu_launch_range_not_gpu_duration",
                    },
                }
            )
