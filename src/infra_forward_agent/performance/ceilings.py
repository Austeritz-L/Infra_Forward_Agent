from __future__ import annotations

from ..contracts import number
from .statistics import summarize

RESOURCE_UNITS = {
    "tensor": {"flop/s", "op/s", "inst/s"},
    "vector": {"flop/s", "op/s", "inst/s"},
    "integer": {"op/s", "inst/s"},
    "atomic": {"op/s"},
    "sfu": {"op/s", "inst/s"},
    "hbm": {"byte/s"},
    "l2": {"byte/s"},
    "shared_memory": {"byte/s"},
    "link": {"byte/s"},
}
BASES = {"architectural_peak", "sustained_microbenchmark", "matched_reference"}
DEFAULT_POLICY = {
    "near_reference_fraction": 0.8,
    "consistency_tolerance": 0.05,
    "dominant_resource_fraction": 0.9,
    "max_native_cv": 0.05,
    "min_native_samples": 5,
}


def resource_model(
    observation: dict, ceilings: dict[str, dict], identity: dict, policy: dict | None = None
) -> dict:
    """A per-invocation/step model. Empirical references never become hard physical bounds."""
    policy = {**DEFAULT_POLICY, **(policy or {})}
    for key, value in policy.items():
        if key not in DEFAULT_POLICY:
            raise ValueError(f"Unknown resource model policy: {key}")
        if key == "min_native_samples":
            if type(value) is not int or value < 2:
                raise ValueError("min_native_samples must be an integer >=2")
            continue
        if number(value, key, positive=True) >= 1:
            raise ValueError("Resource model policy fractions must be between zero and one")
    native = observation["native"]
    if native.get("profiled") is not False or not native.get("source"):
        raise ValueError("Resource models require a sourced unprofiled native timing")
    native_statistics = summarize(native["samples_us"])
    native_us = native_statistics["p50"]
    stable = (
        native_statistics["count"] >= policy["min_native_samples"]
        and native_statistics["cv"] is not None
        and native_statistics["cv"] <= policy["max_native_cv"]
    )
    resources = []
    seen = set()
    for quantity in observation.get("resources", []):
        resource = quantity["resource"]
        if resource in seen:
            raise ValueError("Aggregate quantities sharing one resource before modeling")
        seen.add(resource)
        if resource not in RESOURCE_UNITS:
            raise ValueError(f"Unknown modeled resource: {resource}")
        ceiling = ceilings[quantity["ceiling_id"]]
        if ceiling.get("basis") not in BASES or not ceiling.get("source"):
            raise ValueError("Ceiling needs a basis and source")
        for key in ("architecture", "environment", "kind"):
            if ceiling.get(key) != identity[key]:
                raise ValueError(f"Incomparable ceiling {key}")
        if (
            ceiling.get("resource") != resource
            or ceiling.get("unit") not in RESOURCE_UNITS[resource]
            or quantity.get("unit") != ceiling["unit"].removesuffix("/s")
        ):
            raise ValueError("Resource amount and ceiling counting units must match")
        if not quantity.get("compatibility_key") or quantity["compatibility_key"] != ceiling.get(
            "compatibility_key"
        ):
            raise ValueError("ISA/dtype/layout/shape/topology counting contract differs")
        if quantity.get("amount_basis") not in {"counter", "analytic_executed", "analytic_minimum"}:
            raise ValueError("Resource amount must identify its counter or analytical basis")
        if not quantity.get("source"):
            raise ValueError("Resource quantities require source artifacts")
        rate = number(ceiling["rate"], "ceiling.rate", positive=True)
        amount = number(quantity["amount"], "resource.amount")
        cost = number(amount / rate * 1e6, "modeled resource time")
        resources.append(
            {
                "resource": resource,
                "amount": amount,
                "rate": rate,
                "unit": ceiling["unit"],
                "amount_basis": quantity["amount_basis"],
                "reference_time_us": cost,
                "modeled_rate_at_native_time": amount / native_us * 1e6,
                "reference_attainment": cost / native_us,
                "ceiling_basis": ceiling["basis"],
                "ceiling_id": quantity["ceiling_id"],
                "source": quantity["source"],
            }
        )
    if not resources:
        return {
            "status": "missing_resource_evidence",
            "native_us": native_us,
            "opportunity_fraction": None,
            "resources": [],
            "confirmed_bottleneck": False,
        }
    reference_us = max(row["reference_time_us"] for row in resources)
    dominant = [
        row["resource"]
        for row in resources
        if row["reference_time_us"] >= reference_us * policy["dominant_resource_fraction"]
    ]
    hard = [
        row["reference_time_us"]
        for row in resources
        if row["ceiling_basis"] == "architectural_peak"
    ]
    inconsistent = bool(hard and max(hard) > native_us * (1 + policy["consistency_tolerance"]))
    exceeded = reference_us > native_us * (1 + policy["consistency_tolerance"])
    ratio = reference_us / native_us
    status = (
        "inconsistent_physical_model"
        if inconsistent
        else "unstable_or_insufficient_native_timing"
        if not stable
        else "zero_resource_demand"
        if reference_us == 0
        else "empirical_reference_exceeded"
        if exceeded
        else "near_reference_limit"
        if ratio >= policy["near_reference_fraction"]
        else "efficiency_gap_hypothesis"
    )
    return {
        "status": status,
        "native_us": native_us,
        "native_statistics": native_statistics,
        "model_reference_us": reference_us,
        "physical_lower_bound_us": max(hard) if hard else None,
        "dominant_modeled_resources": dominant,
        "resources": resources,
        "opportunity_fraction": None
        if inconsistent or exceeded or not stable or reference_us == 0
        else max(0.0, 1 - ratio),
        "confirmed_bottleneck": False,
        "next_evidence": "matched counters, omitted resources and controlled perturbation",
        "limitations": [
            "Use the max of potentially overlapping resource times, not their sum.",
            "Sustained and matched references are empirical, not physical upper bounds.",
            "Low bandwidth can reflect dependencies, not bandwidth saturation.",
            "Rate ceilings and amounts must match executed dtype/ISA/cache/traffic conventions.",
            "Modeled headroom is not an achievable operator or service speedup.",
        ],
    }
