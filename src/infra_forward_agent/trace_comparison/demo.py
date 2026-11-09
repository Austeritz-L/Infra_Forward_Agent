"""Synthetic matched trace windows linked to synthetic three-level kernel evidence."""

from __future__ import annotations

import copy
from pathlib import Path

from ..contracts import read_json
from ..kernel.demo import make_kernel_demo
from ..kernel.task import digest, prepare_task
from ..store import write_json


def make_trace_demo(root: Path, architecture: str = "nvidia-sm90") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    task_path, kernel_bundle = make_kernel_demo(root / "kernel", architecture)
    task, prepared, _ = prepare_task(task_path)
    kernel = read_json(kernel_bundle)
    candidate = kernel["candidates"][1]
    placement = [
        {"rank": r, "device": str(r), "physical_device": f"gpu-{r}"} for r in task["ranks"]
    ]
    sides = {
        "baseline": {
            "comparison": task["comparison"],
            "revision": task["parent_revision"],
            "patch_sha256": "baseline",
            "placement": placement,
            "topology": prepared["deployment"]["axes"],
        },
        "candidate": {
            "comparison": task["comparison"],
            "revision": candidate["revision"],
            "patch_sha256": candidate["patch_sha256"],
            "placement": placement,
            "topology": prepared["deployment"]["axes"],
        },
    }
    write_json(
        root / "capture-source.json",
        {
            "kind": "synthetic",
            "note": "Invented windows, annotations, selection and overhead checks; no GPU.",
        },
    )
    data = {
        "schema_version": 1,
        "kind": "synthetic",
        "sides": sides,
        "matrix_source": "kernel/matrix.json",
        "conditions": [{"id": c["id"], "spec": c["spec"]} for c in task["conditions"]],
        "native": {"kernel_bundle": "kernel/bundle.json", "candidate_id": candidate["id"]},
        "policy": {
            "min_matched_pairs": 3,
            "stable_dominance_fraction": 1,
            "change_fraction": 0.1,
            "absolute_floor_us": 2,
        },
        "semantic_mappings": [],
        "pairs": [],
    }
    protocol = {
        "tool": "synthetic-timeline",
        "version": "toy",
        "options": "all-GPU-streams",
        "clock_domain_policy": "rank_local",
        "replay_mode": "none",
        "cache_control": "frozen per condition",
        "capture_selection": "predeclared windows",
        "mode": "timeline",
        "profiled": True,
        "complete_window": True,
        "no_dropped_events": True,
        "mapping_verified": True,
        "overhead_checked": True,
        "representative_selection_frozen": True,
        "source": "capture-source.json",
    }
    for condition in task["conditions"]:
        cid = condition["id"]
        for repeat in range(3):
            pair = {"id": f"{cid}-pair-{repeat}", "condition_id": cid}
            window = {
                "phase": "prefill",
                "boundary_id": condition["forward_boundary_id"],
                "logical_step": f"{cid}-logical-{repeat}",
                "request_manifest_digest": digest(
                    read_json(task_path.parent / condition["request_manifest"])
                ),
                "chunk_plan_digest": "toy-frozen-chunks",
                "global_new_tokens": condition["global_new_tokens_per_sample"],
                "actual_cached_tokens": 0 if cid == "cold" else 73720,
                "scope": "coordinated_prefill_step",
                "head_boundary": "included",
            }
            for side in ("baseline", "candidate"):
                capture = {
                    "identity": {"kind": "synthetic", **copy.deepcopy(sides[side])},
                    "window": copy.deepcopy(window),
                    "protocol": dict(protocol),
                    "ranks": [],
                }
                for rank in task["ranks"]:
                    origin = 100000 * rank + 10000 * repeat + (0 if side == "baseline" else 500000)
                    runtime_step = f"{side}-{cid}-{repeat}"
                    wall = 1200 if side == "baseline" else 1000
                    compute_stream, comm_stream = ("0", "1") if side == "baseline" else ("11", "9")
                    copy_stream = "2" if side == "baseline" else "21"
                    events = [
                        {
                            "name": "forward",
                            "ph": "X",
                            "cat": "user_annotation",
                            "ts": origin,
                            "dur": wall,
                            "args": {"rank": rank, "ifa_step": runtime_step},
                        }
                    ]
                    definitions = [
                        (
                            "gemm",
                            "grouped_gemm",
                            "gemm-fixture",
                            0,
                            700 if side == "baseline" else 400,
                            compute_stream,
                            "compute",
                        ),
                        (
                            "dispatch",
                            "collective",
                            "dispatch-fixture",
                            400 if side == "baseline" else 300,
                            550,
                            comm_stream,
                            "communication",
                        ),
                        (
                            "norm",
                            "normalization",
                            "norm-fixture",
                            950 if side == "baseline" else 850,
                            100,
                            compute_stream,
                            "compute",
                        ),
                        (
                            "copy",
                            "copy",
                            "copy-fixture",
                            1050 if side == "baseline" else 950,
                            50,
                            copy_stream,
                            "memory",
                        ),
                    ]
                    for (
                        name,
                        compute_type,
                        signature,
                        offset,
                        duration,
                        stream,
                        kind,
                    ) in definitions:
                        args = {
                            "rank": rank,
                            "device": str(rank),
                            "stream": stream,
                            "ifa_kind": kind,
                            "ifa_operator": name,
                            "ifa_compute_type": compute_type,
                            "ifa_signature": signature,
                            "ifa_layer": "0",
                            "ifa_step": runtime_step,
                            "ifa_invocation": f"{side}-{name}-{repeat}",
                            "ifa_logical_invocation": f"layer-0/chunk-0/{name}",
                        }
                        if kind == "communication":
                            args.update(
                                ifa_collective_id="dispatch-0",
                                ifa_collective_group="ep:0,1",
                                ifa_collective_op="all_to_all",
                                ifa_payload_bytes=1048576,
                                ifa_collective_dtype="bf16",
                                ifa_wire_tx_bytes=524288,
                                ifa_wire_byte_convention="event_tx_bytes",
                            )
                        events.append(
                            {
                                "name": side + "_" + name,
                                "ph": "X",
                                "cat": "kernel",
                                "ts": origin + offset,
                                "dur": duration,
                                "args": args,
                            }
                        )
                    relative = f"{side}-{cid}-{repeat}-r{rank}.trace.json"
                    write_json(root / relative, {"traceEvents": events})
                    capture["ranks"].append(
                        {
                            "rank": rank,
                            "device": str(rank),
                            "physical_device": f"gpu-{rank}",
                            "path": relative,
                            "clock_id": f"local-{rank}-{side}",
                            "selector": {"ifa_step": runtime_step},
                            "stream_roles": {
                                compute_stream: "compute",
                                comm_stream: "communication",
                                copy_stream: "copy",
                            },
                        }
                    )
                pair[side] = capture
            data["pairs"].append(pair)
    path = root / "bundle.json"
    write_json(path, data)
    return path
