from __future__ import annotations

import math

NON_KV_CATEGORIES = {
    "weights",
    "runtime",
    "graphs",
    "communication_buffers",
    "prefill_workspace_peak",
    "other",
    "retained_prefix",
}


def integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def component_bytes(component: dict, context: int, replica_ranks: int) -> int:
    """Physical local cache pools, not a divisor inferred from TP/EP/CP flags."""
    copies = integer(component["copies"], "pool copies", 1)
    if component["kind"] == "fixed_state":
        if component.get("sequence_shards", 1) != 1:
            raise ValueError(
                "Recurrent/replay state cannot be divided by CP without a separate layout model"
            )
        return copies * integer(component["bytes_per_request"], "fixed state bytes")
    if component["kind"] != "token_cache":
        raise ValueError("Cache component kind must be token_cache or fixed_state")
    shards = integer(component["sequence_shards"], "effective storage shards", 1)
    if shards > replica_ranks:
        raise ValueError("Storage sharding cannot exceed the ranks serving one request")
    if shards > 1 and component.get("storage_sharding_verified") is not True:
        raise ValueError("Do not credit unverified persistent KV sharding")
    retention = component["retention"]
    mode = retention["mode"]
    if mode == "full":
        tokens = context
    elif mode == "window":
        if retention.get("model_semantics_verified") is not True:
            raise ValueError("Sliding-window savings require verified original model semantics")
        tokens = min(context, integer(retention["window_tokens"], "window", 1))
    elif mode == "compressed":
        if retention.get("model_semantics_verified") is not True:
            raise ValueError("Compression requires a verified model/runtime cache layout")
        tokens = math.ceil(context / integer(retention["ratio"], "compression ratio", 1))
        tokens += integer(retention.get("tail_tokens", 0), "compression tail")
    else:
        raise ValueError("Retention mode must be full, window or compressed")
    local_tokens = math.ceil(tokens / shards)
    page = integer(component["page_tokens"], "page_tokens", 1)
    pages = math.ceil(local_tokens / page)
    layout = component["format"]
    if layout["type"] == "kv_heads":
        values = integer(layout["local_kv_heads"], "local KV heads", 1) * (
            integer(layout["k_head_dim"], "K dimension", 1)
            + integer(layout["v_head_dim"], "V dimension", 1)
        )
        entry = math.ceil(values * integer(layout["bits"], "KV bits", 1) / 8)
        entry += integer(layout["scale_bytes_per_entry"], "scale bytes")
        entry += integer(layout["metadata_bytes_per_entry"], "entry metadata bytes")
    elif layout["type"] == "packed":
        entry = integer(layout["bytes_per_entry_including_scales"], "packed entry bytes", 1)
    else:
        raise ValueError("Use kv_heads or explicitly measured packed cache format")
    return (
        copies * pages * (page * entry + integer(component["page_metadata_bytes"], "page metadata"))
    )


def budget_candidate(candidate: dict, devices: dict[str, dict], context: int) -> dict:
    replicas = candidate["replicas"]
    if not isinstance(replicas, list) or not replicas:
        raise ValueError("Candidate must explicitly map request-serving replicas to physical ranks")
    used_devices, used_ranks, replica_ids = set(), set(), set()
    results = []
    for replica in replicas:
        if replica["id"] in replica_ids:
            raise ValueError("Duplicate request-serving replica")
        replica_ids.add(replica["id"])
        ranks = replica["ranks"]
        if not isinstance(ranks, list) or not ranks:
            raise ValueError("Replica must contain rank budgets")
        rows = []
        for rank in ranks:
            device = rank["device_id"]
            identifier = integer(rank["rank"], "rank")
            if device not in devices or device in used_devices or identifier in used_ranks:
                raise ValueError(
                    "Rank budgets must map unique ranks onto unique selected physical devices"
                )
            used_devices.add(device)
            used_ranks.add(identifier)
            fixed = rank["non_kv_bytes"]
            if set(fixed) != NON_KV_CATEGORIES:
                raise ValueError(
                    "Provide every disjoint non-KV budget category; unknown is not zero"
                )
            fixed_total = sum(integer(value, f"non-KV {key}") for key, value in fixed.items())
            reserve = integer(rank["safety_reserve_bytes"], "safety reserve")
            total = integer(devices[device]["memory_bytes"], "device memory", 1)
            available = max(0, total - fixed_total - reserve)
            components = rank["cache_components"]
            if not isinstance(components, list) or not components:
                raise ValueError("Every rank needs an explicit cache/state layout")
            pools = [component["pool_id"] for component in components]
            if len(set(pools)) != len(pools):
                raise ValueError("Do not count the same physical cache pool more than once")
            detail = {
                component["pool_id"]: component_bytes(component, context, len(ranks))
                for component in components
            }
            raw = sum(detail.values())
            if raw <= 0:
                raise ValueError("Resident cache/state demand must be positive")
            fragment = rank["fragmentation_fraction"]
            if (
                type(fragment) not in {int, float}
                or not math.isfinite(fragment)
                or not 0 <= fragment < 1
            ):
                raise ValueError("fragmentation_fraction must be finite in [0,1)")
            request_bytes = math.ceil(raw / (1 - fragment))
            rows.append(
                {
                    "rank": identifier,
                    "device_id": device,
                    "device_bytes": total,
                    "non_kv_bytes": fixed,
                    "non_kv_total_bytes": fixed_total,
                    "safety_reserve_bytes": reserve,
                    "cache_budget_bytes": available,
                    "per_context_bytes_with_fragmentation": request_bytes,
                    "component_bytes_per_context": detail,
                    "estimated_context_slots": available // request_bytes,
                    "static_budget_exceeds_device": fixed_total + reserve > total,
                }
            )
        slots = min(row["estimated_context_slots"] for row in rows)
        pool_limit = replica.get("runtime_pool_logical_tokens")
        if pool_limit is not None:
            slots = min(slots, integer(pool_limit, "normalized replica token pool") // context)
        results.append(
            {
                "id": replica["id"],
                "estimated_context_slots": slots,
                "ranks": rows,
                "runtime_pool_logical_tokens": pool_limit,
            }
        )
    if used_devices != set(devices):
        raise ValueError("Compare candidates on the same complete physical device budget")
    return {
        "context_tokens_including_output_reserve": context,
        "estimated_total_context_slots": sum(
            replica["estimated_context_slots"] for replica in results
        ),
        "single_request_fits_a_replica": any(
            replica["estimated_context_slots"] > 0 for replica in results
        ),
        "replicas": results,
        "host_storage_adds_active_device_capacity": False,
        "capacity_kind": "analytical estimate; exact-context peak/admission validation required",
    }
