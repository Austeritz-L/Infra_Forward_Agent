"""Normalize a bounded scenario matrix; no service or workload is launched here."""

from __future__ import annotations

import itertools
import math


def expand_suite(data: dict, *, max_cases: int = 1000) -> list[dict]:
    if isinstance(data, dict) and data.get("schema_version") == 2:
        from .testing import expand_prefill_suite

        return expand_prefill_suite(data, max_cases=max_cases)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Scenario suite must be an object with schema_version=1")
    if type(max_cases) is not int or max_cases < 1:
        raise ValueError("max_cases must be a positive integer")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Scenario suite must contain cases")
    ids = set()
    matrix = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each scenario case must be an object")
        name = case.get("id")
        if not isinstance(name, str) or not name.strip() or name in ids:
            raise ValueError("Scenario case ids must be nonempty and unique")
        ids.add(name)
        if case.get("scope") not in {"service", "forward", "operator"}:
            raise ValueError(f"{name}: scope must be service, forward or operator")
        if case.get("phase") not in {"prefill", "decode", "mixed"}:
            raise ValueError(f"{name}: phase must be prefill, decode or mixed")
        requirements = case.get("requirements", [])
        if not isinstance(requirements, list) or any(
            not isinstance(item, str) or not item.strip() for item in requirements
        ):
            raise ValueError(f"{name}: requirements must be a list of nonempty strings")
        axes = case.get("axes")
        if not isinstance(axes, dict) or not axes:
            raise ValueError(f"{name}: axes must be a nonempty object")
        for key, values in axes.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"{name}: axis {key} must be a nonempty list")
            if key in {
                "input_tokens",
                "output_tokens",
                "context_tokens",
                "global_tokens",
                "concurrency",
                "batch_size",
            }:
                if any(type(value) is not int or value < 1 for value in values):
                    raise ValueError(f"{name}: {key} values must be positive integers")
            if any(isinstance(value, (dict, list)) or value is None for value in values):
                raise ValueError(f"{name}: axis values must be scalar values")
            if any(isinstance(value, float) and not math.isfinite(value) for value in values):
                raise ValueError(f"{name}: axis values must be finite")
        count = math.prod(len(values) for values in axes.values())
        if len(matrix) + count > max_cases:
            raise ValueError("Scenario expansion exceeds max_cases; narrow the suite")
        keys = list(axes)
        for index, values in enumerate(itertools.product(*axes.values()), start=1):
            matrix.append(
                {
                    "id": f"{name}:{index:04d}",
                    "case": name,
                    "scope": case["scope"],
                    "phase": case["phase"],
                    "parameters": dict(zip(keys, values)),
                    "requirements": requirements,
                    "status": "planned",
                }
            )
    return matrix
