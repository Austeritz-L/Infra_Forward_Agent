import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from infra_forward_agent.campaign import build_plan
from infra_forward_agent.contracts import read_json
from infra_forward_agent.store import write_json
from infra_forward_agent.trace_comparison.compare import build_comparison
from infra_forward_agent.trace_comparison.demo import make_trace_demo
from infra_forward_agent.trace_comparison.report import write_comparison

ROOT = Path(__file__).resolve().parents[1]


class TraceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = make_trace_demo(self.root / "fixture")
        self.data = read_json(self.path)

    def save(self):
        write_json(self.path, self.data)

    def compare(self):
        self.save()
        return build_comparison(self.path)[0]

    def edit_trace(self, rank_spec, change):
        path = self.path.parent / rank_spec["path"]
        trace = read_json(path)
        change(trace["traceEvents"])
        write_json(path, trace)

    def test_rank_local_alignment_stream_roles_and_native_acceptance_remain_separate(self):
        result = self.compare()
        self.assertEqual(result["status"], "comparable")
        device = result["pairs"][0]["devices"]["0"]
        self.assertEqual(device["deltas"]["captured_forward_us"]["candidate_minus_baseline"], -200)
        self.assertEqual(device["deltas"]["gpu_busy_union_us"]["baseline"], 1100)
        self.assertEqual(device["deltas"]["gpu_work_sum_us"]["baseline"], 1400)
        self.assertIn("compute", device["stream_role_busy_deltas"])
        overlap = device["stream_role_overlap_deltas"][json.dumps(["communication", "compute"])]
        self.assertEqual(overlap["baseline"], 300)
        self.assertEqual(overlap["candidate"], 100)
        self.assertFalse(device["raw_stream_ids_aligned"])
        self.assertTrue(result["native_evidence"]["accepted"])
        self.assertFalse(result["native_evidence"]["real_gain_established"])
        self.assertIsNone(result["service_speedup_from_trace"])
        events = result["pairs"][0]["captures"]["candidate"]["normalized_events"]
        self.assertEqual(min(e["start_relative_us"] for e in events), 0)
        self.assertEqual(
            {m["status"] for m in result["bottleneck_migration"]},
            {"exposure_increase_with_stable_work"},
        )
        self.assertTrue(all(not m["confirmed_bottleneck"] for m in result["bottleneck_migration"]))

    def test_missing_rank_invalidates_quorum_and_prevents_cherry_picked_summary(self):
        self.data["pairs"][0]["candidate"]["ranks"].pop()
        result = self.compare()
        self.assertEqual(result["pairs"][0]["status"], "partial")
        self.assertIsNone(result["pairs"][0]["devices"]["0"]["deltas"]["captured_forward_us"])
        self.assertFalse(result["coverage"][0]["complete_matched_repeats"])
        self.assertEqual(
            result["bottleneck_migration"][0]["status"], "insufficient_matched_repeats"
        )

    def test_clipped_or_unhandled_gpu_phases_are_incomplete(self):
        spec = self.data["pairs"][0]["candidate"]["ranks"][0]
        self.edit_trace(spec, lambda events: events[-1].update(dur=500))
        result = self.compare()
        self.assertTrue(any("clipped" in issue for issue in result["pairs"][0]["issues"]))
        self.edit_trace(spec, lambda events: events[-1].update(dur=50))
        self.edit_trace(
            spec,
            lambda events: events.append(
                {
                    "name": "unhandled_gpu",
                    "cat": "kernel",
                    "ph": "B",
                    "ts": events[0]["ts"] + 20,
                    "args": {"rank": 0},
                }
            ),
        )
        result = self.compare()
        self.assertTrue(any("unsupported GPU" in issue for issue in result["pairs"][0]["issues"]))

    def test_unknown_classification_is_not_silent_compute_or_zero_communication(self):
        spec = self.data["pairs"][0]["candidate"]["ranks"][0]
        self.edit_trace(spec, lambda events: events[2]["args"].pop("ifa_kind"))
        result = self.compare()
        self.assertEqual(result["pairs"][0]["status"], "partial")
        self.assertIsNone(result["pairs"][0]["devices"]["0"]["deltas"]["communication_union_us"])

    def test_relative_share_only_does_not_become_absolute_slowdown(self):
        self.data.pop("native")
        for pair in self.data["pairs"]:
            for spec in pair["candidate"]["ranks"]:

                def change(events):
                    origin = events[0]["ts"]
                    events[0]["dur"] = 1200
                    for event in events[1:]:
                        offsets = {"gemm": 300, "dispatch": 400, "norm": 950, "copy": 1050}
                        event["ts"] = origin + offsets[event["args"]["ifa_operator"]]

                self.edit_trace(spec, change)
        result = self.compare()
        self.assertEqual(
            {m["status"] for m in result["bottleneck_migration"]}, {"relative_share_only"}
        )
        migration = result["bottleneck_migration"][0]
        self.assertEqual(migration["new_dominant_work_us"]["candidate_minus_baseline"], 0)
        self.assertGreater(migration["new_dominant_work_share"]["percentage_point_change"], 0)
        self.assertIsNone(migration["native_evidence_accepted"])

    def test_fully_overlapped_units_do_not_get_an_arbitrary_dominant_name(self):
        self.data.pop("native")
        for pair in self.data["pairs"]:
            for side in ("baseline", "candidate"):
                for spec in pair[side]["ranks"]:

                    def overlap(events):
                        origin = events[0]["ts"]
                        del events[3:]
                        for event in events[1:]:
                            event.update(ts=origin, dur=900)

                    self.edit_trace(spec, overlap)
        result = self.compare()
        self.assertEqual(result["status"], "comparable")
        self.assertEqual(
            {m["status"] for m in result["bottleneck_migration"]}, {"no_distinct_exposure_leader"}
        )
        self.assertTrue(
            any(t["reason"] == "no_distinct_exposure_leader" for t in result["next_tasks"])
        )

    def test_fusion_uses_one_disjoint_semantic_group_and_mixed_components_are_unknown(self):
        self.data["semantic_mappings"] = [
            {
                "id": "norm-copy-wrapper",
                "verified": True,
                "source": "capture-source.json",
                "semantic_contract": "same complete wrapper outputs and lifecycle",
                "baseline": [{"operator": "norm"}, {"operator": "copy"}],
                "candidate": [{"operator": "fused_norm_copy"}],
            }
        ]
        for pair in self.data["pairs"]:
            for spec in pair["candidate"]["ranks"]:

                def fuse(events):
                    norm = events[3]
                    norm.update(name="fused_variant", dur=120)
                    norm["args"].update(
                        ifa_operator="fused_norm_copy",
                        ifa_compute_type="fusion",
                        ifa_signature="fused-contract",
                        ifa_kind="mixed",
                    )
                    events.pop()

                self.edit_trace(spec, fuse)
        result = self.compare()
        self.assertEqual(result["status"], "comparable")
        device = result["pairs"][0]["devices"]["0"]
        group = next(
            g for g in device["operators"] if g["semantic_key"] == "mapped:norm-copy-wrapper"
        )
        self.assertEqual(group["deltas"]["gpu_event_count"]["baseline"], 2)
        self.assertEqual(group["deltas"]["gpu_event_count"]["candidate"], 1)
        self.assertEqual(group["deltas"]["work_us"]["baseline"], 150)
        self.assertEqual(group["deltas"]["work_us"]["candidate"], 120)
        self.assertIsNone(group["logical_invocation_deltas"])
        self.assertIsNone(device["deltas"]["communication_without_compute_us"])

    def test_overlapping_semantic_rules_are_rejected(self):
        rule = {
            "id": "a",
            "verified": True,
            "source": "capture-source.json",
            "semantic_contract": "toy",
            "baseline": [{"operator": "gemm"}],
            "candidate": [{"operator": "gemm"}],
        }
        self.data["semantic_mappings"] = [rule, {**rule, "id": "b"}]
        with self.assertRaisesRegex(ValueError, "two semantic mapping"):
            self.compare()

    def test_disappeared_operator_is_unmatched_until_absence_is_proven(self):
        for pair in self.data["pairs"]:
            for spec in pair["candidate"]["ranks"]:
                self.edit_trace(spec, lambda events: events.pop())
        result = self.compare()
        groups = result["pairs"][0]["devices"]["0"]["operators"]
        copy_group = next(
            g for g in groups if g["baseline"] and g["baseline"]["kind"] == ["memory"]
        )
        self.assertIsNone(copy_group["deltas"])
        self.assertEqual(
            result["bottleneck_migration"][0]["status"], "incomplete_semantic_alignment"
        )
        self.data["semantic_mappings"] = [
            {
                "id": "eliminated-copy",
                "verified": True,
                "source": "capture-source.json",
                "semantic_contract": "copy eliminated with unchanged output contract",
                "baseline": [{"operator": "copy"}],
                "candidate": [],
                "candidate_absence_verified": True,
            }
        ]
        group = next(
            g
            for g in self.compare()["pairs"][0]["devices"]["0"]["operators"]
            if g["semantic_key"] == "mapped:eliminated-copy"
        )
        self.assertEqual(group["deltas"]["work_us"]["candidate"], 0)
        self.assertTrue(group["absence_verified"])

    def test_shape_change_is_not_matched_by_kernel_ordinal_or_name(self):
        for pair in self.data["pairs"]:
            for spec in pair["candidate"]["ranks"]:
                self.edit_trace(
                    spec, lambda events: events[1]["args"].update(ifa_signature="other-shape")
                )
        result = self.compare()
        rows = result["pairs"][0]["devices"]["0"]["operators"]
        self.assertEqual(sum(g["deltas"] is None for g in rows), 2)
        self.assertEqual(
            result["bottleneck_migration"][0]["status"], "incomplete_semantic_alignment"
        )

    def test_actual_cache_and_chunk_workload_mismatch_suppresses_deltas(self):
        self.data["pairs"][0]["candidate"]["window"]["actual_cached_tokens"] = 1
        result = self.compare()
        self.assertEqual(result["pairs"][0]["status"], "incomparable")
        self.assertEqual(result["pairs"][0]["devices"], {})
        self.assertEqual(result["native_evidence"]["status"], "invalid_or_incomparable_native_link")

    def test_concurrent_background_step_is_not_attributed_to_the_selected_workload(self):
        spec = self.data["pairs"][0]["candidate"]["ranks"][0]
        self.edit_trace(
            spec, lambda events: events[1]["args"].update(ifa_step="other-inflight-step")
        )
        result = self.compare()
        self.assertEqual(result["pairs"][0]["status"], "partial")
        self.assertTrue(any("background step" in i for i in result["pairs"][0]["issues"]))
        self.assertIsNone(result["pairs"][0]["devices"]["0"]["operators"][0]["deltas"])

    def test_topology_changes_require_deployment_normalization(self):
        self.data["sides"]["candidate"]["topology"] = {"tp": 1, "ep": 2, "pp": 1, "dp": 1, "cp": 1}
        for pair in self.data["pairs"]:
            pair["candidate"]["identity"]["topology"] = self.data["sides"]["candidate"]["topology"]
        result = self.compare()
        self.assertTrue(all(p["status"] == "incomparable" for p in result["pairs"]))
        self.assertTrue(all(not p["devices"] for p in result["pairs"]))
        self.assertIsNone(result["native_evidence"]["accepted"])

    def test_profiler_protocol_difference_is_not_a_speedup(self):
        self.data["pairs"][0]["candidate"]["protocol"]["version"] = "different-version"
        result = self.compare()
        self.assertEqual(result["pairs"][0]["status"], "incomparable")

    def test_collective_payload_and_wire_byte_counting_are_explicit(self):
        spec = self.data["pairs"][0]["candidate"]["ranks"][0]
        self.edit_trace(spec, lambda events: events[2]["args"].update(ifa_payload_bytes=2048))
        result = self.compare()
        comm = result["pairs"][0]["devices"]["0"]["collectives"][0]
        self.assertIsNone(comm["deltas"])
        self.assertIsNone(comm["pure_wire_time_us"])
        self.assertIsNone(comm["rank_arrival_skew_us"])
        self.edit_trace(spec, lambda events: events[2]["args"].update(ifa_payload_bytes=1048576))
        self.edit_trace(spec, lambda events: events[2]["args"].pop("ifa_wire_byte_convention"))
        result = self.compare()
        comm = result["pairs"][0]["devices"]["0"]["collectives"][0]
        self.assertIsNotNone(comm["deltas"])
        self.assertIsNone(comm["wire_tx_bytes_delta"])

    def test_reusing_the_same_capture_does_not_create_repeated_evidence(self):
        for side in ("baseline", "candidate"):
            self.data["pairs"][1][side] = copy.deepcopy(self.data["pairs"][0][side])
        result = self.compare()
        self.assertTrue(any("reused" in i for i in result["pairs"][1]["issues"]))
        self.assertFalse(result["coverage"][0]["complete_matched_repeats"])

    def test_explicit_step_selector_handles_multiple_windows_without_time_warp(self):
        spec = self.data["pairs"][0]["candidate"]["ranks"][0]

        def append_window(events):
            extra = copy.deepcopy(events)
            for event in extra:
                event["ts"] += 5000
                event["args"]["ifa_step"] = "another-step"
            events.extend(extra)

        self.edit_trace(spec, append_window)
        result = self.compare()
        self.assertEqual(result["pairs"][0]["status"], "comparable")
        self.assertEqual(len(result["pairs"][0]["captures"]["candidate"]["normalized_events"]), 8)

    def test_no_native_or_invalid_patch_link_never_claims_service_gain(self):
        self.data["native"]["candidate_id"] = "missing-candidate"
        result = self.compare()
        self.assertEqual(result["native_evidence"]["status"], "invalid_or_incomparable_native_link")
        self.assertFalse(result["native_evidence"]["real_gain_established"])
        self.data.pop("native")
        self.assertEqual(self.compare()["native_evidence"]["status"], "not_supplied")

    def test_partial_pair_count_and_unknown_window_metadata_remain_inconclusive(self):
        self.data["pairs"] = self.data["pairs"][:2]
        result = self.compare()
        self.assertTrue(all(not c["complete_matched_repeats"] for c in result["coverage"]))
        self.data["pairs"][0]["baseline"]["window"]["global_new_tokens"] = None
        self.data["pairs"][0]["candidate"]["window"]["global_new_tokens"] = None
        result = self.compare()
        self.assertEqual(result["pairs"][0]["status"], "partial")

    def test_snapshots_and_normalized_events_are_replayable_and_never_overwritten(self):
        self.save()
        output = self.root / "report"
        result = write_comparison(self.path, output)
        replayed = write_comparison(output / "input/trace-comparison.json", self.root / "replay")
        self.assertEqual(result, replayed)
        manifest = read_json(output / "input/manifest.json")
        for relative, value in manifest.items():
            self.assertEqual(
                hashlib.sha256((output / "input" / relative).read_bytes()).hexdigest(), value
            )
        events = [
            json.loads(line)
            for line in (output / "normalized.events.jsonl").read_text().splitlines()
        ]
        self.assertTrue(all(e["start_relative_us"] >= 0 for e in events))
        self.assertTrue((output / "next-tasks.json").is_file())
        with self.assertRaises(FileExistsError):
            write_comparison(self.path, output)

    def test_planner_keeps_optional_skip_and_records_trace_contract(self):
        plan = build_plan(ROOT / "configs/campaigns/multi-architecture.json")
        self.assertTrue(plan["trace_contract"]["alignment"])
        self.assertTrue(
            all(s["status"] == "skipped" for s in plan["stages"] if s["module"] == "traces")
        )

    def test_planner_records_comparison_manifest_without_executing_it(self):
        original = ROOT / "configs/campaigns/multi-architecture.json"
        data = read_json(original)
        for key in (
            "scenario_suite",
            "performance_contract",
            "deployment_contract",
            "kernel_contract",
            "trace_contract",
        ):
            data[key] = str((original.parent / data[key]).resolve())
        data["architectures"] = ["nvidia-sm90"]
        data["knowledge_paths"] = []
        data["trace_comparison"] = {"enabled": True, "bundles": {"nvidia-sm90": str(self.path)}}
        path = self.root / "campaign.json"
        write_json(path, data)
        plan = build_plan(path)
        trace = next(s for s in plan["stages"] if s["module"] == "traces")
        self.assertEqual(trace["status"], "needs_offline_validation")
        self.assertEqual(trace["options"]["bundle"], str(self.path.resolve()))
        self.assertFalse(plan["execution_ready"])
