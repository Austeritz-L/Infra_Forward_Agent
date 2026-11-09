"""Deliberately invented 2Mi-token toy model; never a vendor/model capacity claim."""

from __future__ import annotations

import math
from pathlib import Path

from ..store import write_json
from .memory import NON_KV_CATEGORIES
from .selection import layout_signature


def make_deployment_demo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    gib = 2**30
    context = 2**21
    (root / "toy-source.txt").write_text(
        "SYNTHETIC: invented hardware, cache, quality and timing. No real model or GPU.\n"
    )
    data = {
        "schema_version": 1,
        "kind": "synthetic",
        "comparison": {
            key: "toy-" + key
            for key in (
                "architecture",
                "model",
                "weights_digest",
                "image_digest",
                "runtime_revision",
                "tokenizer_digest",
                "dataset_digest",
                "global_workload",
                "cache_format",
            )
        },
        "target": {
            "context_tokens": context,
            "output_reserve_tokens": 1,
            "model_native_limit_tokens": context,
            "model_config_source": "toy-source.txt",
        },
        "devices": [
            {"id": f"gpu-{i}", "memory_bytes": 16 * gib, "source": "toy-source.txt"}
            for i in range(4)
        ],
        "policy": {
            "min_context_slots": 1,
            "round_ids": ["pair-1", "pair-2", "pair-3"],
            "max_round_cv": 0.03,
            "latency_stat": "p95",
            "balanced_weights": {"capacity": 0.5, "throughput": 0.3, "latency": 0.2},
        },
        "conditions": [],
        "candidates": [],
    }
    for cache, concurrency in (("cold-miss", 1), ("partial-90", 2)):
        data["conditions"].append(
            {
                "id": cache,
                "context_tokens": context,
                "concurrency": concurrency,
                "cache_profile": cache,
                "cache_fraction_tolerance": 0.02,
                "forward_boundary_id": "toy-request-all-chunks",
                "weight": 1,
                "throughput_reference": 1e6,
                "ttft_reference_us": 100000,
                "slo": {"min_new_tokens_per_s": 1, "max_ttft_us": 200000, "max_forward_us": 200000},
            }
        )
    capacity_case = dict(data["conditions"][0])
    capacity_case.update(id="resident-load", concurrency=12, weight=0.1)
    data["conditions"].append(capacity_case)
    for name, entry, shards, replicas, tested, latency in (
        ("capacity-cp", 1024, 4, 1, 12, 100000),
        ("balanced-tp", 512, 1, 1, 6, 25000),
        ("performance-dp", 2048, 1, 2, 1, 20000),
    ):
        candidate = {
            "id": name,
            "axes": {"tp": 4 // replicas, "ep": 1, "pp": 1, "dp": replicas, "cp": shards},
            "replicas": [],
            "qualification": f"{name}/qualification.json",
            "measurements": [],
        }
        for replica in range(replicas):
            rank_rows = []
            for index in range(4 // replicas):
                rank = replica * (4 // replicas) + index
                non_kv = dict.fromkeys(NON_KV_CATEGORIES, 0)
                non_kv.update(weights=7 * gib, prefill_workspace_peak=gib)
                rank_rows.append(
                    {
                        "rank": rank,
                        "device_id": f"gpu-{rank}",
                        "non_kv_bytes": non_kv,
                        "safety_reserve_bytes": gib,
                        "fragmentation_fraction": 0,
                        "layout_source": "toy-source.txt",
                        "cache_components": [
                            {
                                "pool_id": "all-unique-toy-layers",
                                "kind": "token_cache",
                                "copies": 1,
                                "sequence_shards": shards,
                                "storage_sharding_verified": True,
                                "retention": {"mode": "full"},
                                "page_tokens": 128,
                                "page_metadata_bytes": 0,
                                "format": {
                                    "type": "packed",
                                    "bytes_per_entry_including_scales": entry,
                                },
                            }
                        ],
                    }
                )
            candidate["replicas"].append({"id": f"replica-{replica}", "ranks": rank_rows})
        identity = {
            "kind": "synthetic",
            "comparison": data["comparison"],
            "candidate_id": name,
            "layout_signature": layout_signature(candidate, data["devices"], data["target"]),
        }
        folder = root / name
        folder.mkdir()
        write_json(
            folder / "qualification.json",
            {
                "identity": identity,
                "qualified_round_ids": data["policy"]["round_ids"],
                "qualified_condition_ids": [c["id"] for c in data["conditions"]],
                "evidence_sources": ["toy-source.txt"],
                "backend_combination_verified": True,
                "validated_context_tokens": context,
                "runtime_context_limit_tokens": context,
                "context_quality_passed": True,
                "precision_before_passed": True,
                "precision_after_passed": True,
                "precision_timing_order_verified": True,
                "memory": {
                    "context_tokens": context,
                    "cache_state": "independent_full_contexts",
                    "all_lifecycle_phases_covered": True,
                    "prefill_peak_covered": True,
                    "runtime_admission_verified": True,
                    "simultaneous_active_contexts_verified": True,
                    "no_eviction_or_recompute": True,
                    "capacity_condition_id": "resident-load",
                    "unprofiled": True,
                    "replica_slots": {r["id"]: tested for r in candidate["replicas"]},
                    "rank_peak_bytes": {
                        str(i): 8 * gib + tested * context * entry // shards for i in range(4)
                    },
                },
            },
        )
        for condition in data["conditions"]:
            observations = {"condition_id": condition["id"], "rounds": []}
            for round_id in data["policy"]["round_ids"]:
                trial_identity = {**identity, "condition": condition, "round_id": round_id}
                stem = f"{name}/{condition['id']}-{round_id}"
                requests = []
                resident = min(condition["concurrency"], replicas * tested)
                wave_us = math.ceil(condition["concurrency"] / resident) * latency
                for i in range(200):
                    start = (i // condition["concurrency"]) * wave_us
                    finish = start + ((i % condition["concurrency"]) // resident + 1) * latency
                    logical = context - 1
                    cached = 0 if condition["cache_profile"] == "cold-miss" else int(logical * 0.9)
                    requests.append(
                        {
                            "id": str(i),
                            "clock_id": "toy-monotonic",
                            "sent_us": start,
                            "terminal_us": finish,
                            "first_token_us": finish,
                            "status": "completed",
                            "done": True,
                            "errors": [],
                            "output_tokens": 1,
                            "input_tokens": logical,
                            "cached_tokens": cached,
                            "new_tokens": logical - cached,
                        }
                    )
                write_json(
                    root / (stem + ".service.json"),
                    {
                        "identity": trial_identity,
                        "profiled": False,
                        "cohort": "fixed_work_drained",
                        "clock_id": "toy-monotonic",
                        "requests": requests,
                        "actual_cache_profile_verified": True,
                        "actual_concurrency_verified": True,
                    },
                )
                write_json(
                    root / (stem + ".forward.json"),
                    {
                        "identity": trial_identity,
                        "profiled": False,
                        "all_streams_joined": True,
                        "rank_coverage": list(range(4)),
                        "scope": "coordinated_prefill_forward",
                        "boundary_id": "toy-request-all-chunks",
                        "samples_us": [latency * 0.8] * 200,
                    },
                )
                observations["rounds"].append(
                    {
                        "id": round_id,
                        "service": stem + ".service.json",
                        "forward": stem + ".forward.json",
                    }
                )
            candidate["measurements"].append(observations)
        data["candidates"].append(candidate)
    write_json(root / "bundle.json", data)
    return root / "bundle.json"
