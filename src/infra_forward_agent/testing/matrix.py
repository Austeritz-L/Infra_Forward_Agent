from __future__ import annotations

import hashlib
import itertools
import json
import math

SCOPES = ("service", "forward", "operator")
POSITIVE_AXES = {"input_tokens", "request_concurrency", "working_set_ratio", "arrival_rate_ratio"}


def validate_prefill_suite(data: dict) -> None:
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 2
        or data.get("focus") != "prefill"
    ):
        raise ValueError("Prefill suite requires schema_version=2 and focus=prefill")
    generation = data.get("generation", {})
    if (
        type(generation.get("max_new_tokens")) is not int
        or generation["max_new_tokens"] != 1
        or type(generation.get("temperature")) not in {int, float}
        or generation["temperature"] != 0
    ):
        raise ValueError("This prefill suite fixes max_new_tokens=1 and temperature=0")
    contracts = data.get("contracts")
    if not isinstance(contracts, dict):
        raise ValueError("Prefill suite requires contracts")
    for key in ("cache_profiles", "load_protocols", "metrics", "acceptance", "capture_contract"):
        if not isinstance(contracts.get(key), dict) or not contracts[key]:
            raise ValueError(f"Missing testing contract: {key}")
    if set(contracts["metrics"]) != set(SCOPES):
        raise ValueError("Metrics must distinguish service, forward and operator scopes")
    if set(("integrity", "cache", *SCOPES)) - contracts["acceptance"].keys():
        raise ValueError("Acceptance must define integrity, cache and all three scope gates")
    for collection in ("cache_profiles", "load_protocols", "metrics"):
        if any(not isinstance(value, dict) for value in contracts[collection].values()):
            raise ValueError(f"{collection} entries must be objects")
    tiers = data.get("enabled_tiers")
    if not isinstance(tiers, list) or not tiers or not all(isinstance(x, str) for x in tiers):
        raise ValueError("enabled_tiers must be a nonempty list")
    families = data.get("workloads")
    if not isinstance(families, list) or not families:
        raise ValueError("Prefill suite must contain workloads")
    ids = set()
    declared_tiers = set()
    for family in families:
        if not isinstance(family, dict):
            raise ValueError("Each workload must be an object")
        name = family.get("id")
        if not isinstance(name, str) or not name.strip() or name in ids:
            raise ValueError("Workload ids must be nonempty and unique")
        ids.add(name)
        if not isinstance(family.get("tier"), str):
            raise ValueError(f"{name}: tier must be specified")
        declared_tiers.add(family["tier"])
        requirements = family.get("requirements", [])
        if not isinstance(requirements, list) or any(
            not isinstance(item, str) for item in requirements
        ):
            raise ValueError(f"{name}: requirements must be a list of strings")
        if family.get("load_protocol") not in contracts["load_protocols"]:
            raise ValueError(f"{name}: unknown load_protocol")
        if family.get("purpose") not in {"qualification", "characterization"}:
            raise ValueError(f"{name}: purpose must be qualification or characterization")
        axes = family.get("axes")
        if not isinstance(axes, dict) or not axes:
            raise ValueError(f"{name}: axes must be nonempty")
        if ("input_tokens" in axes) == ("length_mix" in axes):
            raise ValueError(f"{name}: specify input_tokens or length_mix, exclusively")
        if "cache_profile" not in axes:
            raise ValueError(f"{name}: cache_profile axis is required")
        for key, values in axes.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"{name}: {key} must have values")
            for value in values:
                if (
                    type(value) not in {str, int, float}
                    or isinstance(value, float)
                    and not math.isfinite(value)
                ):
                    raise ValueError(f"{name}: axis values must be finite scalar strings/numbers")
                if key in POSITIVE_AXES and (not isinstance(value, (int, float)) or value <= 0):
                    raise ValueError(f"{name}: {key} must be positive")
                if key in {"input_tokens", "request_concurrency"} and type(value) is not int:
                    raise ValueError(f"{name}: {key} must be an integer")
            if len(set(values)) != len(values):
                raise ValueError(f"{name}: duplicate axis values")
        if set(axes["cache_profile"]) - contracts["cache_profiles"].keys():
            raise ValueError(f"{name}: unknown cache profile")
        if "length_mix" in axes and set(axes["length_mix"]) - data.get("length_mixes", {}).keys():
            raise ValueError(f"{name}: unknown length_mix")
        scopes = family.get("derive_scopes", list(SCOPES))
        if not isinstance(scopes, list) or set(scopes) != set(SCOPES) or len(scopes) != 3:
            raise ValueError(f"{name}: derive_scopes must contain the three aligned scopes")
    if set(tiers) - declared_tiers:
        raise ValueError("enabled_tiers contains an unknown tier")


def expand_prefill_suite(data: dict, *, max_cases: int = 1000) -> list[dict]:
    validate_prefill_suite(data)
    if type(max_cases) is not int or max_cases < 1:
        raise ValueError("max_cases must be a positive integer")
    matrix = []
    for family in data["workloads"]:
        if family["tier"] not in data["enabled_tiers"]:
            continue
        axes = family["axes"]
        count = math.prod(len(values) for values in axes.values()) * len(SCOPES)
        if len(matrix) + count > max_cases:
            raise ValueError("Scenario expansion exceeds max_cases; select fewer tiers or axes")
        for values in itertools.product(*axes.values()):
            parameters = dict(zip(axes, values))
            identity = json.dumps(
                {"family": family["id"], "parameters": parameters}, sort_keys=True
            )
            suffix = hashlib.sha256(identity.encode()).hexdigest()[:12]
            workload = f"{family['id']}:{suffix}"
            for scope in SCOPES:
                matrix.append(
                    {
                        "id": f"{workload}:{scope}",
                        "case": family["id"],
                        "scope": scope,
                        "phase": "prefill",
                        "tier": family["tier"],
                        "purpose": family["purpose"],
                        "workload_id": workload,
                        "parameters": dict(parameters),
                        "generation": dict(data["generation"]),
                        "load_protocol": family["load_protocol"],
                        "formal_performance_eligible": family["tier"] != "smoke"
                        and family["purpose"] == "qualification",
                        "cache_profile": parameters["cache_profile"],
                        "metric_contract": scope,
                        "acceptance_contracts": ["integrity", "cache", scope],
                        "alignment": {
                            "service_scenario_id": f"{workload}:service",
                            "binding_status": "requires_service_capture"
                            if scope != "service"
                            else "planned",
                            "capture_contract": "capture_contract",
                            "batch_size_equals_request_concurrency": False,
                            "planned_id_is_not_a_measured_input_digest": True,
                        },
                        "requirements": family.get("requirements", [])
                        + [
                            "frozen prompt/tokenizer/template/seed and per-request identity",
                            "verified per-tier cache state and complete sample accounting",
                            "service prefill capture required for forward/operator qualification",
                        ],
                        "status": "planned",
                    }
                )
    if not matrix:
        raise ValueError("No workloads selected by enabled_tiers")
    return matrix
