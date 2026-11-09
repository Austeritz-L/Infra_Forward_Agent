"""All numbers, hardware and validator claims are invented CPU fixtures."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..contracts import read_json
from ..deployment.demo import make_deployment_demo
from ..deployment.selection import layout_signature, select_deployment
from ..performance.demo import make_performance_demo
from ..store import write_json
from .task import digest, performance_identity, prepare_task


def _profile(root: Path, relative: str, identity: dict, operator_us: float = 500) -> str:
    folder = root / relative
    bundle = make_performance_demo(folder)
    for path in folder.glob("*.json"):
        data = read_json(path)
        if "identity" in data:
            data["identity"] = identity
        if path.name in {"native-0-expert_gemm.json", "native-1-expert_gemm.json"}:
            data["samples_us"] = [operator_us] * 200
        if path == bundle:
            for ceiling in data["ceilings"]:
                ceiling.update({k: identity[k] for k in ("kind", "architecture", "environment")})
        write_json(path, data)
    return bundle.relative_to(root).as_posix()


def _diagnosis(identity: dict, cases: list[dict], source: str, post: bool = False) -> dict:
    return {
        "identity": identity,
        "errors": [],
        "separate_profile_pass": True,
        "all_rank_timeline_complete": True,
        "mapping_verified": True,
        "replay_effects_documented": True,
        "instrumentation_overhead_checked": True,
        "tool": "synthetic-profiler",
        "tool_version": "toy",
        "timeline_source": source,
        "instrumentation_source": source,
        "counter_source": source,
        "cases": [
            {
                "id": case["id"],
                "evidence_mode": "hardware_counter",
                "metrics": {
                    name: {
                        "value": value,
                        "unit": unit,
                        "definition": "synthetic normalized " + name,
                        "source": source,
                    }
                    for name, value, unit in (
                        ("executed_work", 4e9, "flop"),
                        ("actual_memory_bytes", 1e8, "byte"),
                        ("occupancy", 0.5, "fraction"),
                        ("stall", 0.1 if post else 0.2, "fraction"),
                        ("spill_bytes", 0, "byte"),
                    )
                },
                "observation": "Toy native time exceeds a matched resource reference",
                "hypothesis": "Toy pipeline has avoidable producer/consumer waits",
                "predicted_counter_change": "Less measured wait without extra traffic/spills",
                "falsification": "No wait reduction or forward/service regression",
                "alternative_explanation": "Launch starvation or late rank",
                "single_change": "One pipeline staging change",
                "test": {"metric": "stall", "direction": "decrease", "min_relative_change": 0.1},
                "dependency_and_lifetime_review": "Toy unique ownership and wait-before-reuse",
            }
            for case in cases
        ],
    }


def make_kernel_demo(root: Path, architecture: str = "nvidia-sm90") -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    source = "synthetic-source.txt"
    (root / source).write_text(
        "SYNTHETIC ONLY. No actual ISA, GPU, compiler or model was tested.\n"
    )
    # Reuse the distributed selector rather than claiming capacity from free memory.
    deployment_path = make_deployment_demo(root / "deployment")
    deployment = read_json(deployment_path)
    deployment["comparison"]["architecture"] = architecture
    deployment["devices"] = deployment["devices"][:2]
    candidate = deployment["candidates"][1]
    deployment["candidates"] = [candidate]
    candidate["axes"]["tp"] = 2
    candidate["replicas"][0]["ranks"] = candidate["replicas"][0]["ranks"][:2]
    identity = {
        "kind": "synthetic",
        "comparison": deployment["comparison"],
        "candidate_id": candidate["id"],
        "layout_signature": layout_signature(
            candidate, deployment["devices"], deployment["target"]
        ),
    }
    proof_path = deployment_path.parent / candidate["qualification"]
    proof = read_json(proof_path)
    proof["identity"] = identity
    proof["memory"]["rank_peak_bytes"] = {str(i): 14 * 2**30 for i in (0, 1)}
    write_json(proof_path, proof)
    for observation, condition in zip(candidate["measurements"], deployment["conditions"]):
        for trial in observation["rounds"]:
            for scope in ("service", "forward"):
                path = deployment_path.parent / trial[scope]
                data = read_json(path)
                data["identity"] = {**identity, "condition": condition, "round_id": trial["id"]}
                if scope == "forward":
                    data["rank_coverage"] = [0, 1]
                write_json(path, data)
    write_json(deployment_path, deployment)
    write_json(root / "deployment-selection.json", select_deployment(deployment_path))
    task = {
        "schema_version": 1,
        "id": "toy-hotspot-task",
        "kind": "synthetic",
        "comparison": deployment["comparison"],
        "parent_revision": "toy-parent",
        "ranks": [0, 1],
        "matrix_source": "matrix.json",
        "deployment_selection": "deployment-selection.json",
        "deployment_bundle": "deployment/bundle.json",
        "deployment_candidate": "balanced-tp",
        "toolchain": {"compiler": "toy", "target": architecture, "version": "synthetic"},
        "validation_modes": [
            "default_stream",
            "non_default_stream",
            "graph",
            "repeat",
            "edge_cases",
        ],
        "policy": {
            "round_ids": [f"pair-{i}" for i in range(5)],
            "max_cv": 0.05,
            "max_regression": 0.02,
            "operator_min_gain": 0.05,
            "forward_min_gain": 0.03,
            "service_min_gain": 0.03,
            "forward_objective": "latency",
            "service_objective": "ttft",
            "gain_conditions": ["cold", "prefix"],
        },
        "commands": {
            phase: [["SYNTHETIC-NOT-EXECUTABLE", phase]]
            for phase in (
                "preflight",
                "instrument",
                "profile",
                "precision",
                "benchmark",
                "post_precision",
            )
        },
        "conditions": [],
    }
    matrix = []
    for cid, concurrency, cache in (("cold", 1, "cold-miss"), ("prefix", 16, "partial-90")):
        condition = {
            "id": cid,
            "spec": {
                "input_tokens": 8192,
                "request_concurrency": concurrency,
                "cache_profile": cache,
            },
            "forward_boundary_id": "toy-complete-prefill",
            "global_new_tokens_per_sample": 8192,
            "reference_source": source,
            "forward_reference": {"id": "toy-reference", "atol": 0.001, "rtol": 0.001},
            "forward_relative_l2_limit": 0.001,
            "slo": {"max_forward_p95_us": 2000, "max_ttft_p95_us": 2000, "min_new_tokens_per_s": 1},
            "operator_cases": [],
        }
        matrix.append(
            {
                "scope": "service",
                "phase": "prefill",
                "workload_id": cid,
                "parameters": condition["spec"],
                "generation": {"max_new_tokens": 1},
            }
        )
        condition["request_manifest"] = "requests-" + cid + ".json"
        write_json(
            root / condition["request_manifest"],
            {
                "requests": [
                    {
                        "id": str(i),
                        "input_tokens": 8192,
                        "expected_first_token_id": 42,
                        "expected_cached_tokens": 0 if cache == "cold-miss" else int(8192 * 0.9),
                    }
                    for i in range(200)
                ]
            },
        )
        for rank in (0, 1):
            condition["operator_cases"].append(
                {
                    "id": f"{cid}-gemm-r{rank}",
                    "rank": rank,
                    "device": str(rank),
                    "operator": "expert_gemm",
                    "compute_type": "grouped_gemm",
                    "signature": "gemm-fixture",
                    "call_count": 1,
                    "capture_source": f"analysis/{cid}/rank-{rank}.trace.json",
                    "relative_l2_limit": 0.001,
                    "contract": {
                        "math": "toy grouped matmul with frozen routing",
                        "shape": {"global_new_tokens": 8192, "local_tokens": 4096, "fixture": True},
                        "dtype": "toy-fp32",
                        "layout": "toy-layout",
                        "quantization": "none",
                        "routing": "frozen toy per-rank routing, not client C times L",
                        "reference": {"id": "toy-reference", "atol": 0.001, "rtol": 0.001},
                    },
                }
            )
        condition["performance_bundle"] = _profile(
            root, f"analysis/{cid}", performance_identity(task, condition)
        )
        condition["diagnosis"] = f"baseline-deep-{cid}.json"
        write_json(
            root / condition["diagnosis"],
            _diagnosis(
                {**performance_identity(task, condition), "revision": task["parent_revision"]},
                condition["operator_cases"],
                source,
            ),
        )
        task["conditions"].append(condition)
    write_json(root / "matrix.json", matrix)
    task_path = root / "contract.json"
    write_json(task_path, task)
    _, prepared, _ = prepare_task(task_path)

    def record(
        name: str,
        operator_us: float,
        forward_us: float,
        ttft_us: float,
        baseline: bool = False,
        bad_precision: bool = False,
    ) -> dict:
        folder = root / name
        folder.mkdir()
        row = {
            "id": name,
            "revision": task["parent_revision"] if baseline else "toy-" + name,
            "parent_revision": task["parent_revision"],
            "rounds": [],
        }
        patch_hash = "baseline"
        if not baseline:
            row["patch"] = name + "/candidate.patch"
            (root / row["patch"]).write_text("Synthetic patch for " + name + "\n")
            patch_hash = hashlib.sha256((root / row["patch"]).read_bytes()).hexdigest()
            row["patch_sha256"] = patch_hash
        identity = {
            "kind": "synthetic",
            "task_digest": prepared["task_digest"],
            "revision": row["revision"],
            "patch_sha256": patch_hash,
        }

        def put(label: str, value: dict) -> str:
            relative = f"{name}/{label}.json"
            write_json(root / relative, value)
            return relative

        binary = name + "/binary.txt"
        (root / binary).write_text("Synthetic compiled bytes, not executable\n")
        row["compile"] = put(
            "compile",
            {
                "identity": identity,
                "errors": [],
                "toolchain": task["toolchain"],
                "device_identity_verified": True,
                "isa_legal": True,
                "emitted_path_verified": True,
                "occupancy_launch_verified": True,
                "stream_graph_lifetime_verified": True,
                "emitted_instructions": ["toy.mma"],
                "binary_source": binary,
                "binary_sha256": hashlib.sha256((root / binary).read_bytes()).hexdigest(),
                "resource_source": source,
                "rank_resources": [
                    {
                        "rank": rank,
                        "registers_per_thread": 32,
                        "shared_bytes_per_block": 1024,
                        "threads_per_block": 128,
                        "tmem_columns": 0,
                        "local_memory_bytes": 0,
                        "limits": {
                            "registers_per_thread": 64,
                            "shared_bytes_per_block": 4096,
                            "threads_per_block": 256,
                            "tmem_columns": 0,
                        },
                    }
                    for rank in (0, 1)
                ],
            },
        )
        row["memory"] = put(
            "memory",
            {
                "identity": identity,
                "errors": [],
                "source": source,
                "all_lifecycle_phases_covered": True,
                "capacity_and_context_preserved": True,
                "no_eviction_or_recompute": True,
                "cache_layout_unchanged": True,
                "long_context_quality_revalidated": True,
                "full_occupancy_service_slo_revalidated": True,
                "validated_context_tokens": deployment["target"]["context_tokens"],
                "resident_slots": {"replica-0": 6},
                "ranks": [
                    {"rank": r, "peak_bytes": 14 * 2**30, "non_kv_peak_bytes": 8 * 2**30}
                    for r in (0, 1)
                ],
            },
        )
        for phase, start in (("before", 10), ("after", 10**15)):
            results = []
            for condition in task["conditions"]:
                for case in condition["operator_cases"]:
                    results.append(
                        {
                            "scope": "operator",
                            "condition_id": condition["id"],
                            "rank": case["rank"],
                            "case_id": case["id"],
                            "contract_digest": digest(case),
                        }
                    )
                for rank in task["ranks"]:
                    results.append(
                        {
                            "scope": "forward",
                            "condition_id": condition["id"],
                            "rank": rank,
                            "contract_digest": digest(condition["forward_reference"]),
                        }
                    )
                results.append(
                    {
                        "scope": "service",
                        "condition_id": condition["id"],
                        "rank": None,
                        "contract_digest": digest(condition["forward_reference"]),
                    }
                )
            for result in results:
                result.update(
                    checked_elements=200,
                    bad_elements=0,
                    nonfinite_elements=0,
                    relative_l2=0,
                    max_abs_error=0,
                )
            if bad_precision:
                results[0]["bad_elements"] = 1
            row["precision_" + phase] = put(
                "precision-" + phase,
                {
                    "identity": identity,
                    "errors": [],
                    "clock_id": "coordinator",
                    "started_ns": start,
                    "finished_ns": start + 1,
                    "modes_verified": task["validation_modes"],
                    "results": results,
                },
            )
        if not baseline:
            row["diagnoses"], row["performance_bundles"] = {}, {}
            for condition in task["conditions"]:
                cid = condition["id"]
                row["diagnoses"][cid] = put(
                    "diagnosis-" + cid,
                    _diagnosis(
                        {**identity, "condition_id": cid},
                        condition["operator_cases"],
                        source,
                        post=True,
                    ),
                )
                row["performance_bundles"][cid] = _profile(
                    root,
                    f"{name}/profile-{cid}",
                    {
                        **performance_identity(task, condition),
                        "revision": row["revision"],
                        "task_digest": prepared["task_digest"],
                        "patch_sha256": patch_hash,
                    },
                    operator_us,
                )
        for index, round_id in enumerate(task["policy"]["round_ids"]):
            trial = {"id": round_id, "conditions": []}
            for ci, condition in enumerate(task["conditions"]):
                measured = {"id": condition["id"]}
                for si, scope in enumerate(("operator", "forward", "service")):
                    start = 10**12 + index * 10**12 + ci * 10**11 + si * 10**10
                    report = {
                        "identity": {
                            **identity,
                            "condition_id": condition["id"],
                            "round_id": round_id,
                        },
                        "errors": [],
                        "clock_id": "coordinator",
                        "started_ns": start,
                        "finished_ns": start + 10**9,
                        "profiled": False,
                        "debug_instrumentation": False,
                        "completed": True,
                        "new_jit": False,
                    }
                    if scope == "operator":
                        report.update(
                            scope="operator_wrapper",
                            all_streams_joined=True,
                            cases=[
                                {"id": case["id"], "samples_us": [operator_us] * 200}
                                for case in condition["operator_cases"]
                            ],
                        )
                    elif scope == "forward":
                        report.update(
                            scope="coordinated_prefill_forward",
                            all_streams_joined=True,
                            rank_coverage=task["ranks"],
                            boundary_id=condition["forward_boundary_id"],
                            global_new_tokens_per_sample=condition["global_new_tokens_per_sample"],
                            samples_us=[forward_us] * 200,
                        )
                    else:
                        requests = []
                        spec = condition["spec"]
                        for i in range(200):
                            sent = start / 1000 + (i // spec["request_concurrency"]) * (
                                ttft_us + 40
                            )
                            cached = 0 if spec["cache_profile"] == "cold-miss" else int(8192 * 0.9)
                            requests.append(
                                {
                                    "id": str(i),
                                    "clock_id": "coordinator",
                                    "sent_us": sent,
                                    "first_token_us": sent + ttft_us,
                                    "terminal_us": sent + ttft_us + 40,
                                    "done": True,
                                    "errors": [],
                                    "status": "completed",
                                    "output_tokens": 1,
                                    "input_tokens": 8192,
                                    "cached_tokens": cached,
                                    "new_tokens": 8192 - cached,
                                    "first_token_id": 42,
                                    "reference_first_token_id": 42,
                                }
                            )
                        report.update(
                            cohort="fixed_work_drained",
                            requests=requests,
                            cache_state_verified=True,
                            load_protocol_verified=True,
                        )
                    measured[scope] = put(f"{round_id}-{condition['id']}-{scope}", report)
                trial["conditions"].append(measured)
            row["rounds"].append(trial)
        return row

    evaluation = {
        "schema_version": 1,
        "task": "contract.json",
        "baseline": record("baseline", 500, 1100, 1200, baseline=True),
        "candidates": [
            record("operator-only", 420, 1200, 1300),
            record("full-gain", 450, 950, 1050),
            record("service-regression", 420, 950, 1400),
            record("precision-failure", 430, 850, 950, bad_precision=True),
        ],
    }
    evaluation_path = root / "bundle.json"
    write_json(evaluation_path, evaluation)
    return task_path, evaluation_path
