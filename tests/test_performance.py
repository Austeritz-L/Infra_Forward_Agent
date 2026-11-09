import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from infra_forward_agent.contracts import read_json
from infra_forward_agent.performance.demo import make_performance_demo
from infra_forward_agent.performance.report import build_report, write_report
from infra_forward_agent.performance.statistics import service_statistics, summarize
from infra_forward_agent.performance.timeline import analyze_timeline
from infra_forward_agent.store import write_json
from infra_forward_agent.trace import Slice, Trace


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = make_performance_demo(self.root / "inputs")

    def edit_bundle(self, data):
        write_json(self.bundle, data)

    def test_statistics_percentiles_and_tail_sample_eligibility(self):
        result = summarize([1, 2, 3, 4, 5])
        self.assertEqual(result["p50"], 3)
        self.assertAlmostEqual(result["p95"], 4.8)
        self.assertFalse(result["p95_sample_eligible"])
        for bad in ([], [0], [-1], [float("nan")], [float("inf")], [True]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                summarize(bad)

    def test_timeline_shares_union_exclusive_and_streams_do_not_conflate(self):
        report = build_report(self.bundle)
        for device in report["capture_timelines"][0]["devices"]:
            self.assertEqual(device["captured_forward_us"], 1200)
            self.assertEqual(device["gpu_work_sum_us"], 1300)
            self.assertEqual(device["gpu_busy_union_us"], 1100)
            self.assertEqual(device["gpu_idle_us"], 100)
            self.assertEqual(device["compute_communication_overlap_us"], 200)
            self.assertEqual(device["communication_without_compute_us"], 300)
            self.assertEqual(device["multi_stream_active_us"], 200)
            self.assertEqual(device["peak_active_streams"], 2)
            expert = next(row for row in device["operators"] if row["operator"] == "expert_gemm")
            self.assertEqual(expert["exclusive_us"], 400)
            self.assertAlmostEqual(expert["work_share"], 600 / 1300)
            self.assertEqual(expert["wall_coverage"], 0.5)
            self.assertEqual(expert["operator_invocations"], 1)

    def test_same_group_overlap_uses_union_and_distinct_streams(self):
        items = [
            Slice("a", 0, "0", "0", 0, 10, "compute", "explicit"),
            Slice("a", 0, "0", "1", 0, 10, "compute", "explicit"),
            Slice("b", 0, "0", "2", 5, 15, "compute", "explicit"),
        ]
        device = analyze_timeline(Trace(items, {0: (0, 20)}, 0))["devices"][0]
        self.assertEqual(device["gpu_busy_union_us"], 15)
        self.assertEqual(device["multi_stream_active_us"], 10)
        self.assertEqual(device["peak_active_streams"], 3)
        self.assertEqual([row["exclusive_us"] for row in device["operators"]], [5, 5])
        self.assertTrue(all(row["operator_invocations"] is None for row in device["operators"]))

    def test_clipped_windows_and_layer_mapping_remain_explicit(self):
        path = self.bundle.parent / "rank-0.trace.json"
        trace = read_json(path)
        trace["traceEvents"][0]["dur"] = 1050
        trace["traceEvents"][1]["args"]["ifa_layer"] = 7
        write_json(path, trace)
        report = build_report(self.bundle)
        capture = report["capture_timelines"][0]
        self.assertEqual(capture["clipped_gpu_events"], 1)
        self.assertFalse(capture["complete_forward_window"])
        self.assertFalse(report["attack_plan"]["kernel_efficiency"])
        expert = next(
            row for row in capture["devices"][0]["operators"] if row["operator"] == "expert_gemm"
        )
        self.assertEqual(expert["layer_work_us"], {"7": 600})

    def test_resource_models_separate_efficiency_from_work_reduction_and_unknowns(self):
        result = build_report(self.bundle)
        attack = result["attack_plan"]
        self.assertEqual({row["operator"] for row in attack["kernel_efficiency"]}, {"norm"})
        self.assertEqual(
            {row["operator"] for row in attack["work_or_traffic_reduction"]}, {"expert_gemm"}
        )
        self.assertIn("all_to_all", {row["operator"] for row in attack["measure_first"]})
        self.assertFalse(result["confirmed_bottleneck"])
        self.assertTrue(
            all(row["service_speedup_prediction"] is None for row in result["priorities"])
        )
        ceiling = result["forward_resource_ceiling"]
        self.assertEqual(ceiling["resource_lower_bound_us"], 400)
        self.assertAlmostEqual(ceiling["optimistic_physical_tokens_per_s"], 8192 / 400 * 1e6)
        self.assertFalse(ceiling["service_throughput_ceiling"])

    def test_ceiling_identity_units_and_duplicate_resources_rejected(self):
        original = read_json(self.bundle)
        for mutation in ("architecture", "kind", "unit", "contract", "duplicate"):
            data = copy.deepcopy(original)
            if mutation in {"architecture", "kind"}:
                data["ceilings"][0][mutation] = "other"
            elif mutation == "unit":
                data["operator_resources"][0]["resources"][0]["unit"] = "byte"
            elif mutation == "contract":
                data["operator_resources"][0]["resources"][0]["compatibility_key"] = "other"
            else:
                data["operator_resources"][0]["resources"].append(
                    data["operator_resources"][0]["resources"][0]
                )
            self.edit_bundle(data)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_report(self.bundle)

    def test_exceeding_empirical_reference_is_not_exceeding_physics(self):
        data = read_json(self.bundle)
        data["ceilings"][0]["basis"] = "matched_reference"
        data["ceilings"][0]["rate"] = 1e9
        self.edit_bundle(data)
        report = build_report(self.bundle)
        expert = next(row for row in report["resource_models"] if row["operator"] == "expert_gemm")
        self.assertEqual(expert["status"], "empirical_reference_exceeded")
        self.assertIsNone(expert["opportunity_fraction"])
        self.assertEqual(expert["physical_lower_bound_us"], 100)

    def test_impossible_physical_model_is_excluded_from_optimization_queue(self):
        data = read_json(self.bundle)
        data["ceilings"][0]["rate"] = 1e9
        self.edit_bundle(data)
        report = build_report(self.bundle)
        expert = next(row for row in report["resource_models"] if row["operator"] == "expert_gemm")
        self.assertEqual(expert["status"], "inconsistent_physical_model")
        self.assertIsNone(report["forward_resource_ceiling"])

    def test_missing_rank_stays_in_measurement_queue(self):
        data = read_json(self.bundle)
        data["traces"] = data["traces"][:1]
        self.edit_bundle(data)
        report = build_report(self.bundle)
        self.assertEqual(report["capture_timelines"][0]["missing_ranks"], [1])
        self.assertFalse(report["attack_plan"]["kernel_efficiency"])
        self.assertTrue(report["attack_plan"]["measure_first"])
        self.assertTrue(
            all(
                row["investigation_route"] == "calibrate_or_validate_evidence"
                for row in report["priorities"]
            )
        )

    def test_step_resource_budgets_cannot_mix_different_global_work(self):
        data = read_json(self.bundle)
        data["step_resource_budgets"][1]["global_new_tokens"] = 4096
        self.edit_bundle(data)
        with self.assertRaises(ValueError):
            build_report(self.bundle)

    def test_native_timing_cannot_be_profiled_or_unmatched(self):
        path = self.bundle.parent / "native-0-norm.json"
        native = read_json(path)
        for field, value in (("profiled", True), ("signature", "different"), ("rank", 1)):
            write_json(path, {**native, field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_report(self.bundle)

    def test_failed_service_attempts_remain_in_time_and_validity_accounting(self):
        report = read_json(self.bundle.parent / "service.json")
        report["requests"][-1].update(status="failed", done=False, terminal_us=120200)
        result = service_statistics(report)
        self.assertFalse(result["request_integrity_valid"])
        self.assertEqual(result["completed"], 19)
        self.assertAlmostEqual(result["cohort_seconds"], 0.1202)
        self.assertAlmostEqual(result["logical_input_tokens_per_s"], 19 * 8192 / 0.1202)
        report["requests"][0]["clock_id"] = "other-worker"
        with self.assertRaises(ValueError):
            service_statistics(report)

    def test_timed_load_preserves_the_full_window_when_requests_finish_early(self):
        data = read_json(self.bundle.parent / "service.json")
        data["cohort"] = "timed_offer_drained"
        data["offered_window"] = {"started_us": 0, "finished_us": 1200000000, "completed": True}
        report = service_statistics(data)
        self.assertEqual(report["cohort_seconds"], 1200)
        self.assertLess(report["raw_first_send_to_last_terminal_seconds"], 1)
        self.assertAlmostEqual(report["logical_input_tokens_per_s"], 20 * 8192 / 1200)
        self.assertTrue(report["timed_offered_window"]["window_completed"])

    def test_zero_work_and_unstable_native_timings_do_not_create_headroom_claims(self):
        data = read_json(self.bundle)
        for item in data["operator_resources"][0]["resources"]:
            item["amount"] = 0
        self.edit_bundle(data)
        model = build_report(self.bundle)["resource_models"][0]
        self.assertEqual(model["status"], "zero_resource_demand")
        self.assertIsNone(model["opportunity_fraction"])
        path = self.bundle.parent / "native-0-norm.json"
        native = read_json(path)
        native["samples_us"] = [1, 100, 100, 100, 100]
        write_json(path, native)
        models = build_report(self.bundle)["resource_models"]
        norm = next(row for row in models if row["operator"] == "norm" and row["rank"] == 0)
        self.assertEqual(norm["status"], "unstable_or_insufficient_native_timing")
        self.assertIsNone(norm["opportunity_fraction"])

    def test_snapshots_hash_sources_and_preserve_prior_runs(self):
        output = self.root / "report"
        write_report(self.bundle, output)
        for path, digest in read_json(output / "input/manifest.json").items():
            self.assertEqual(
                hashlib.sha256((output / "input" / path).read_bytes()).hexdigest(), digest
            )
        self.assertTrue((output / "attack-plan.json").is_file())
        with self.assertRaises(FileExistsError):
            write_report(self.bundle, output)

    def test_cli_synthetic_analysis(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "infra_forward_agent",
                "performance-demo",
                "--output",
                str(self.root / "cli"),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout.splitlines()[-1])["stage"], "performance_analysis_completed"
        )


if __name__ == "__main__":
    unittest.main()
