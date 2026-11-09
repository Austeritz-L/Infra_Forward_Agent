import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from infra_forward_agent.campaign import build_plan
from infra_forward_agent.contracts import read_json
from infra_forward_agent.deployment.communication import estimate_communication
from infra_forward_agent.deployment.demo import make_deployment_demo
from infra_forward_agent.deployment.memory import budget_candidate, component_bytes
from infra_forward_agent.deployment.report import write_deployment_report
from infra_forward_agent.deployment.selection import layout_signature, select_deployment
from infra_forward_agent.store import write_json

ROOT = Path(__file__).resolve().parents[1]


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = make_deployment_demo(self.root / "source")
        self.data = read_json(self.path)

    def save(self):
        write_json(self.path, self.data)

    def proof(self, index=0, **changes):
        path = self.path.parent / self.data["candidates"][index]["qualification"]
        report = read_json(path)
        report.update(changes)
        write_json(path, report)

    def rebind(self):
        """Test-fixture identities only; this is never a real measurement converter."""
        for candidate in self.data["candidates"]:
            identity = {
                "kind": self.data["kind"],
                "comparison": self.data["comparison"],
                "candidate_id": candidate["id"],
                "layout_signature": layout_signature(
                    candidate, self.data["devices"], self.data["target"]
                ),
            }
            path = self.path.parent / candidate["qualification"]
            proof = read_json(path)
            proof["identity"] = identity
            write_json(path, proof)
            for observation, condition in zip(candidate["measurements"], self.data["conditions"]):
                for trial in observation["rounds"]:
                    for key in ("service", "forward"):
                        path = self.path.parent / trial[key]
                        report = read_json(path)
                        report["identity"] = {
                            **identity,
                            "condition": condition,
                            "round_id": trial["id"],
                        }
                        write_json(path, report)
        self.save()

    def budget(self, candidate):
        return budget_candidate(candidate, {d["id"]: d for d in self.data["devices"]}, 2**21)

    def test_three_objectives_have_different_rankings_but_synthetic_never_recommends(self):
        result = select_deployment(self.path)
        self.assertEqual(result["rankings"]["capacity"][0], "capacity-cp")
        self.assertEqual(result["rankings"]["performance"][0], "performance-dp")
        self.assertEqual(result["rankings"]["balanced"][0], "balanced-tp")
        self.assertTrue(all(row["eligible"] for row in result["candidates"]))
        self.assertEqual(set(result["recommendations"].values()), {None})
        self.assertFalse(result["auto_deploy"])

    def test_measured_kind_path_only_after_every_evidence_identity_matches(self):
        self.data["kind"] = "measured"
        self.save()
        with self.assertRaisesRegex(ValueError, "identity differs"):
            select_deployment(self.path)
        self.rebind()  # Only a CPU fixture exercising kind-dependent output.
        result = select_deployment(self.path)
        self.assertEqual(result["recommendations"]["capacity"], "capacity-cp")

    def test_target_2mi_cannot_inherit_1mi_native_support(self):
        self.data["target"]["model_native_limit_tokens"] = 2**20
        self.rebind()
        result = select_deployment(self.path)
        self.assertEqual(result["pareto_frontier"], [])
        self.assertTrue(
            all(
                any("exceeds native" in reason for reason in row["reasons"])
                for row in result["candidates"]
            )
        )
        self.proof(context_extension_validated=True)
        self.assertEqual(select_deployment(self.path)["pareto_frontier"], ["capacity-cp"])

    def test_capacity_requires_quality_precision_backend_rounds_and_all_rank_peaks(self):
        for changes, expected in (
            ({"context_quality_passed": False}, "context_quality"),
            ({"precision_after_passed": False}, "precision_after"),
            ({"backend_combination_verified": False}, "backend_combination"),
            ({"qualified_round_ids": []}, "every measured"),
            ({"runtime_context_limit_tokens": 2**20}, "runtime context"),
        ):
            path = self.path.parent / self.data["candidates"][0]["qualification"]
            original = read_json(path)
            with self.subTest(changes=changes):
                self.proof(**changes)
                row = select_deployment(self.path)["candidates"][0]
                self.assertFalse(row["eligible"])
                self.assertTrue(any(expected in reason for reason in row["reasons"]))
            write_json(path, original)
        original["memory"]["rank_peak_bytes"].pop("3")
        write_json(path, original)
        self.assertFalse(select_deployment(self.path)["candidates"][0]["eligible"])

    def test_largest_analytical_capacity_without_proof_is_not_selected(self):
        self.data["candidates"][0].pop("qualification")
        self.save()
        result = select_deployment(self.path)
        self.assertGreater(result["candidates"][0]["budget"]["estimated_total_context_slots"], 10)
        self.assertFalse(result["candidates"][0]["eligible"])
        self.assertEqual(result["rankings"]["capacity"][0], "balanced-tp")

    def test_memory_slots_without_full_occupancy_service_slo_or_active_residency_are_rejected(self):
        path = self.path.parent / self.data["candidates"][0]["qualification"]
        proof = read_json(path)
        proof["memory"]["capacity_condition_id"] = "cold-miss"  # C1 cannot qualify 12 slots.
        write_json(path, proof)
        row = select_deployment(self.path)["candidates"][0]
        self.assertTrue(any("resident occupancy" in reason for reason in row["reasons"]))
        proof["memory"]["capacity_condition_id"] = "resident-load"
        proof["memory"]["no_eviction_or_recompute"] = False
        write_json(path, proof)
        row = select_deployment(self.path)["candidates"][0]
        self.assertTrue(any("no_eviction_or_recompute" in reason for reason in row["reasons"]))

    def test_one_failed_service_attempt_cannot_hide_behind_success_statistics(self):
        trial = self.data["candidates"][0]["measurements"][1]["rounds"][0]
        path = self.path.parent / trial["service"]
        report = read_json(path)
        report["requests"][0]["done"] = False
        write_json(path, report)
        row = select_deployment(self.path)["candidates"][0]
        self.assertFalse(row["eligible"])
        self.assertTrue(any("failed/incomplete" in reason for reason in row["reasons"]))

    def test_cache_counters_override_an_incorrect_verification_flag(self):
        trial = self.data["candidates"][0]["measurements"][1]["rounds"][0]
        path = self.path.parent / trial["service"]
        report = read_json(path)
        for request in report["requests"]:
            request["cached_tokens"] = request["input_tokens"] // 2
            request["new_tokens"] = request["input_tokens"] - request["cached_tokens"]
        write_json(path, report)
        row = select_deployment(self.path)["candidates"][0]
        self.assertFalse(row["eligible"])
        self.assertTrue(any("cached fraction" in reason for reason in row["reasons"]))

    def test_every_condition_slo_and_tail_sample_eligibility_are_required(self):
        self.data["conditions"][1]["slo"]["max_ttft_us"] = 30000
        self.rebind()
        self.assertFalse(select_deployment(self.path)["candidates"][0]["eligible"])
        trial = self.data["candidates"][1]["measurements"][0]["rounds"][0]
        path = self.path.parent / trial["forward"]
        report = read_json(path)
        report["samples_us"] = report["samples_us"][:10]
        write_json(path, report)
        row = select_deployment(self.path)["candidates"][1]
        self.assertTrue(any("tail samples" in reason for reason in row["reasons"]))

    def test_changed_layout_cannot_reuse_qualification_or_timing(self):
        self.data["candidates"][0]["replicas"][0]["ranks"][0]["safety_reserve_bytes"] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, "identity differs"):
            select_deployment(self.path)

    def test_zero_new_work_and_changed_boundary_cannot_be_scored_as_prefill(self):
        for trial in self.data["candidates"][0]["measurements"][1]["rounds"]:
            path = self.path.parent / trial["service"]
            report = read_json(path)
            for request in report["requests"]:
                request["cached_tokens"] = request["input_tokens"]
                request["new_tokens"] = 0
            write_json(path, report)
        row = select_deployment(self.path)["candidates"][0]
        self.assertTrue(any("no positive new-prefill" in reason for reason in row["reasons"]))
        self.assertIsNone(row["verified_context_slots"])
        trial = self.data["candidates"][1]["measurements"][0]["rounds"][0]
        path = self.path.parent / trial["forward"]
        report = read_json(path)
        report["boundary_id"] = "fastest-chunk-only"
        write_json(path, report)
        with self.assertRaisesRegex(ValueError, "coordinated native"):
            select_deployment(self.path)

    def test_rank_minimum_dp_separation_and_runtime_pool_limit(self):
        candidate = copy.deepcopy(self.data["candidates"][2])
        for replica in candidate["replicas"]:
            for rank in replica["ranks"]:
                rank["cache_components"][0]["format"]["bytes_per_entry_including_scales"] = 8192
        result = self.budget(candidate)
        self.assertFalse(result["single_request_fits_a_replica"])
        self.assertEqual(result["estimated_total_context_slots"], 0)
        candidate = copy.deepcopy(self.data["candidates"][0])
        candidate["replicas"][0]["ranks"][0]["non_kv_bytes"]["weights"] = 15 * 2**30
        self.assertEqual(self.budget(candidate)["estimated_total_context_slots"], 0)
        candidate = copy.deepcopy(self.data["candidates"][0])
        candidate["replicas"][0]["runtime_pool_logical_tokens"] = 2**20
        self.assertEqual(self.budget(candidate)["estimated_total_context_slots"], 0)

    def test_cache_pool_formula_includes_page_scales_metadata_and_local_heads(self):
        component = {
            "kind": "token_cache",
            "copies": 80,
            "sequence_shards": 1,
            "retention": {"mode": "full"},
            "page_tokens": 128,
            "page_metadata_bytes": 64,
            "format": {
                "type": "kv_heads",
                "local_kv_heads": 2,
                "k_head_dim": 128,
                "v_head_dim": 128,
                "bits": 16,
                "scale_bytes_per_entry": 16,
                "metadata_bytes_per_entry": 4,
            },
        }
        self.assertEqual(component_bytes(component, 129, 8), 80 * 2 * (128 * 1044 + 64))
        component["format"]["local_kv_heads"] = 8
        component["format"]["scale_bytes_per_entry"] = 0
        component["format"]["metadata_bytes_per_entry"] = 0
        component["page_metadata_bytes"] = 0
        self.assertEqual(component_bytes(component, 2**21, 8), 640 * 2**30)

    def test_unverified_shards_window_compression_and_recurrent_state_not_divided(self):
        component = self.data["candidates"][0]["replicas"][0]["ranks"][0]["cache_components"][0]
        component["storage_sharding_verified"] = False
        with self.assertRaisesRegex(ValueError, "unverified"):
            component_bytes(component, 2**21, 4)
        with self.assertRaisesRegex(ValueError, "Recurrent"):
            component_bytes(
                {
                    "kind": "fixed_state",
                    "copies": 1,
                    "bytes_per_request": 123,
                    "sequence_shards": 4,
                },
                2**21,
                4,
            )
        component["storage_sharding_verified"] = True
        component["retention"] = {"mode": "window", "window_tokens": 128}
        with self.assertRaisesRegex(ValueError, "verified original"):
            component_bytes(component, 2**21, 4)
        component["retention"] = {"mode": "compressed", "ratio": 4}
        with self.assertRaisesRegex(ValueError, "verified model"):
            component_bytes(component, 2**21, 4)
        component["retention"]["model_semantics_verified"] = True
        self.assertEqual(component_bytes(component, 2**21, 4), 128 * 2**20)

    def test_unknown_non_kv_and_duplicate_physical_devices_cannot_create_capacity(self):
        candidate = copy.deepcopy(self.data["candidates"][0])
        rank = candidate["replicas"][0]["ranks"][0]
        rank["non_kv_bytes"].pop("graphs")
        with self.assertRaisesRegex(ValueError, "every disjoint"):
            self.budget(candidate)
        candidate = copy.deepcopy(self.data["candidates"][0])
        candidate["replicas"][0]["ranks"][1]["device_id"] = "gpu-0"
        with self.assertRaisesRegex(ValueError, "unique"):
            self.budget(candidate)

    def test_fragmentation_and_additional_prefix_reduce_capacity(self):
        candidate = copy.deepcopy(self.data["candidates"][0])
        for rank in candidate["replicas"][0]["ranks"]:
            rank["fragmentation_fraction"] = 0.5
            rank["non_kv_bytes"]["retained_prefix"] = 2**30
        self.assertEqual(self.budget(candidate)["estimated_total_context_slots"], 6)

    def test_communication_estimate_has_no_invented_rate_or_end_to_end_time(self):
        phase = {
            "source": "matched-calibration",
            "byte_convention": "tx_wire_bytes_per_rank",
            "wire_bytes": 1000,
            "messages": 2,
        }
        self.assertIsNone(estimate_communication([phase])[0]["reference_phase_us"])
        phase.update(startup_us=5, bandwidth_bytes_per_s=1000)
        result = estimate_communication([phase])[0]
        self.assertEqual(result["reference_phase_us"], 1000010)
        self.assertFalse(result["is_measured_end_to_end_latency"])

    def test_snapshot_manifest_is_replayable_and_never_overwrites(self):
        output = self.root / "report"
        events = []
        result = write_deployment_report(self.path, output, events.append)
        self.assertEqual(events[-1]["stage"], "deployment_selection_completed")
        replayed = select_deployment(output / "input/deployment.json")
        self.assertEqual(result, replayed)
        manifest = read_json(output / "input/manifest.json")
        for relative, digest in manifest.items():
            self.assertEqual(
                hashlib.sha256((output / "input" / relative).read_bytes()).hexdigest(), digest
            )
        (self.path.parent / "toy-source.txt").write_text("changed upstream")
        self.assertNotEqual((output / "input/toy-source.txt").read_text(), "changed upstream")
        with self.assertRaises(FileExistsError):
            write_deployment_report(self.path, output)

    def test_long_context_tier_is_opt_in_and_keeps_three_scope_alignment(self):
        plan = build_plan(ROOT / "configs/campaigns/long-context-capacity.json")
        self.assertEqual(plan["scenario_count_per_architecture"], 72)
        self.assertEqual(plan["scenario_count_total"], 216)
        self.assertFalse(plan["execution_ready"])
        service = [row for row in plan["scenarios"] if row["scope"] == "service"]
        self.assertEqual(
            {row["parameters"]["input_tokens"] + 1 for row in service}, {2**18, 2**19, 2**20, 2**21}
        )
        self.assertEqual(
            plan["deployment_contract"]["context_targets_including_output_reserve"][-1], 2**21
        )
