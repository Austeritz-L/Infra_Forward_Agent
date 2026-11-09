import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from infra_forward_agent.campaign import build_plan
from infra_forward_agent.contracts import read_json
from infra_forward_agent.kernel.demo import make_kernel_demo
from infra_forward_agent.kernel.instrumentation import operator_range
from infra_forward_agent.kernel.task import performance_identity, prepare_task
from infra_forward_agent.kernel.validation import validate_record
from infra_forward_agent.kernel.workflow import write_kernel_evaluation, write_kernel_task
from infra_forward_agent.store import write_json

ROOT = Path(__file__).resolve().parents[1]


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task_path, self.bundle = make_kernel_demo(self.root / "fixture")
        self.task, self.prepared, _ = prepare_task(self.task_path)
        self.data = read_json(self.bundle)

    def edit(self, relative, change):
        path = self.bundle.parent / relative
        report = read_json(path)
        change(report)
        write_json(path, report)

    def validate_good(self):
        _, _, sources = prepare_task(self.task_path)
        return validate_record(self.task, self.prepared, self.data["candidates"][1], sources)

    def test_three_levels_and_precision_are_independent_gates(self):
        result = write_kernel_evaluation(self.bundle, self.root / "result")
        rows = {r["id"]: r for r in result["candidates"]}
        self.assertTrue(rows["operator-only"]["gates"]["operator"])
        self.assertFalse(rows["operator-only"]["gates"]["forward"])
        self.assertTrue(rows["full-gain"]["accepted"])
        self.assertTrue(rows["service-regression"]["gates"]["forward"])
        self.assertFalse(rows["service-regression"]["gates"]["service"])
        self.assertFalse(rows["precision-failure"]["accepted"])
        self.assertIsNone(result["recommended_candidate"])
        self.assertFalse(result["auto_deploy"])

    def test_missing_counter_blocks_implementation_and_evaluation(self):
        diagnosis = self.task["conditions"][0]["diagnosis"]
        self.edit(diagnosis, lambda d: d["cases"][0]["metrics"]["spill_bytes"].update(value=None))
        packet = write_kernel_task(self.task_path, self.root / "packet")
        self.assertEqual(packet["status"], "analysis_required")
        self.assertTrue(
            any("unknown is not zero" in r for r in packet["analysis"]["missing_analysis"])
        )
        result = write_kernel_evaluation(self.bundle, self.root / "blocked")
        self.assertEqual(result["candidates"], [])

    def test_proxy_needs_its_own_validation_and_all_rank_profile(self):
        diagnosis = self.task["conditions"][0]["diagnosis"]
        self.edit(diagnosis, lambda d: d["cases"][0].update(evidence_mode="validated_proxy"))
        _, prepared, _ = prepare_task(self.task_path)
        self.assertFalse(prepared["ready_for_candidate"])
        condition = self.task["conditions"][0]
        self.edit(condition["performance_bundle"], lambda d: d.update(expected_ranks=[0]))
        with self.assertRaisesRegex(ValueError, "every deployment rank"):
            prepare_task(self.task_path)

    def test_full_matrix_and_reference_are_frozen_in_task_digest(self):
        original = self.prepared["task_digest"]
        self.edit(
            "requests-cold.json", lambda d: d["requests"][0].update(expected_first_token_id=99)
        )
        _, fresh, _ = prepare_task(self.task_path)
        self.assertNotEqual(original, fresh["task_digest"])
        task = copy.deepcopy(self.task)
        task["conditions"].pop()
        write_json(self.task_path, task)
        with self.assertRaisesRegex(ValueError, "entire declared service matrix"):
            prepare_task(self.task_path)

    def test_first_token_reference_cannot_be_changed_inside_candidate_report(self):
        good = self.data["candidates"][1]
        native = good["rounds"][0]["conditions"][0]["service"]
        self.edit(
            native,
            lambda d: d["requests"][0].update(first_token_id=99, reference_first_token_id=99),
        )
        with self.assertRaisesRegex(ValueError, "first-token parity"):
            self.validate_good()

    def test_hypothesis_refuted_or_post_profile_missing_cannot_promote(self):
        good = self.data["candidates"][1]
        self.edit(
            good["diagnoses"]["cold"],
            lambda d: d["cases"][0]["metrics"]["stall"].update(value=0.25),
        )
        with self.assertRaisesRegex(ValueError, "Hypothesis refuted"):
            self.validate_good()
        self.edit(
            good["diagnoses"]["cold"], lambda d: d["cases"][0]["metrics"]["stall"].update(value=0.1)
        )
        good["performance_bundles"].pop("cold")
        with self.assertRaises(KeyError):
            self.validate_good()

    def test_native_profiler_debug_flags_and_precision_order_are_enforced(self):
        path = self.data["candidates"][1]["rounds"][0]["conditions"][0]["forward"]
        self.edit(path, lambda d: d.update(debug_instrumentation=True))
        with self.assertRaisesRegex(ValueError, "profiler/debug"):
            self.validate_good()
        self.edit(path, lambda d: d.update(debug_instrumentation=False, started_ns=0))
        with self.assertRaisesRegex(ValueError, "precision before"):
            self.validate_good()

    def test_memory_peak_cannot_consume_qualified_long_context_capacity(self):
        memory = self.data["candidates"][1]["memory"]
        self.edit(memory, lambda d: d["ranks"][0].update(non_kv_peak_bytes=10 * 2**30))
        with self.assertRaisesRegex(ValueError, "KV capacity budget"):
            self.validate_good()

    def test_emitted_binary_and_actual_resource_limits_are_required(self):
        path = self.data["candidates"][1]["compile"]
        self.edit(path, lambda d: d["rank_resources"][0].update(tmem_columns=1))
        with self.assertRaisesRegex(ValueError, "exact-device limit"):
            self.validate_good()
        self.edit(path, lambda d: d["rank_resources"][0].update(tmem_columns=0))
        self.edit(path, lambda d: d.update(binary_sha256="another-binary"))
        with self.assertRaisesRegex(ValueError, "binary identity"):
            self.validate_good()

    def test_task_parent_and_patch_are_bound_to_every_candidate_report(self):
        good = self.data["candidates"][1]
        path = self.bundle.parent / good["patch"]
        path.write_text("changed patch")
        with self.assertRaisesRegex(ValueError, "patch SHA256"):
            self.validate_good()
        good["patch_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.validate_good()

    def test_missing_precision_rank_and_post_timing_precision_fail(self):
        path = self.data["candidates"][1]["precision_after"]
        self.edit(path, lambda d: d["results"].pop())
        with self.assertRaisesRegex(ValueError, "all conditions"):
            self.validate_good()

    def test_failed_sse_and_tail_sample_shortcuts_are_rejected(self):
        row = self.data["candidates"][1]["rounds"][0]["conditions"][0]
        self.edit(row["service"], lambda d: d["requests"][0].update(done=False))
        with self.assertRaisesRegex(ValueError, "incomplete cohort"):
            self.validate_good()
        self.edit(row["service"], lambda d: d["requests"][0].update(done=True))
        self.edit(row["forward"], lambda d: d.update(samples_us=[950] * 10))
        with self.assertRaisesRegex(ValueError, "200 samples"):
            self.validate_good()

    def test_zero_work_or_inconsistent_reference_cannot_create_an_optimization_task(self):
        condition = self.task["conditions"][0]
        bundle = condition["performance_bundle"]
        folder = (self.bundle.parent / bundle).parent
        native = folder / "native-0-expert_gemm.json"
        report = read_json(native)
        report["samples_us"] = [1] * 200  # Below the matched 400us physical floor.
        write_json(native, report)
        _, prepared, _ = prepare_task(self.task_path)
        self.assertFalse(prepared["ready_for_candidate"])

    def test_absent_operator_needs_positive_not_applicable_capture_proof(self):
        condition = self.task["conditions"][0]
        condition["operator_cases"] = []
        self.edit(condition["diagnosis"], lambda d: d.update(cases=[]))
        write_json(self.task_path, self.task)
        _, prepared, _ = prepare_task(self.task_path)
        self.assertFalse(prepared["ready_for_candidate"])
        condition["not_applicable_source"] = "absent-cold.json"
        write_json(
            self.bundle.parent / "absent-cold.json",
            {
                "identity": {
                    **performance_identity(self.task, condition),
                    "revision": self.task["parent_revision"],
                },
                "operator_absent_verified": True,
                "reason": "toy cache path skips expert compute",
                "capture_source": "analysis/cold/rank-0.trace.json",
            },
        )
        write_json(self.task_path, self.task)
        _, prepared, _ = prepare_task(self.task_path)
        self.assertTrue(prepared["ready_for_candidate"])

    def test_snapshot_handoff_and_evaluation_are_replayable_and_immutable(self):
        packet = self.root / "packet"
        write_kernel_task(self.task_path, packet)
        _, replayed, _ = prepare_task(packet / "input/task.json")
        self.assertEqual(replayed["task_digest"], self.prepared["task_digest"])
        self.assertTrue((packet / "docs/draft.md").is_file())
        result = write_kernel_evaluation(self.bundle, self.root / "run")
        repeat = write_kernel_evaluation(
            self.root / "run/input/evaluation.json", self.root / "repeat"
        )
        self.assertEqual(result, repeat)
        with self.assertRaises(FileExistsError):
            write_kernel_evaluation(self.bundle, self.root / "run")
        manifest = read_json(self.root / "run/input/manifest.json")
        for name, expected in manifest.items():
            self.assertEqual(
                hashlib.sha256((self.root / "run/input" / name).read_bytes()).hexdigest(), expected
            )

    def test_all_three_architectures_use_the_same_gate_without_hardware_claims(self):
        for architecture in ("moore-s5000", "nvidia-sm103"):
            task_path, bundle = make_kernel_demo(self.root / architecture, architecture)
            task, prepared, _ = prepare_task(task_path)
            self.assertEqual(prepared["architecture"]["id"], architecture)
            self.assertEqual(prepared["architecture"]["resources"], {})
            result = write_kernel_evaluation(bundle, self.root / (architecture + "-result"))
            self.assertTrue(result["candidates"][1]["accepted"])
            self.assertIsNone(result["recommended_candidate"])
            self.assertEqual(task["kind"], "synthetic")

    def test_campaign_records_kernel_contract_and_no_execution(self):
        plan = build_plan(ROOT / "configs/campaigns/multi-architecture.json")
        self.assertTrue(plan["kernel_contract"]["mandatory_analysis"])
        stages = [s for s in plan["stages"] if s["module"] == "kernel"]
        self.assertTrue(all(s["options"]["deep_analysis_required"] for s in stages))
        self.assertFalse(plan["execution_ready"])


class InstrumentationTests(unittest.TestCase):
    def test_disabled_instrumentation_does_not_call_callbacks_or_clock(self):
        with operator_range({}, enabled=False, push=lambda _: self.fail(), pop=lambda: self.fail()):
            pass

    def test_correlation_range_preserves_exception_and_balances_callbacks(self):
        attributes = dict.fromkeys(
            (
                "ifa_request_id",
                "ifa_step",
                "ifa_operator",
                "ifa_compute_type",
                "ifa_signature",
                "ifa_invocation",
                "ifa_layer",
                "ifa_cache_tier",
            ),
            "toy",
        )
        attributes.update(
            rank=0,
            device="0",
            stream=1,
            ifa_global_new_tokens=8,
            ifa_local_tokens=4,
            ifa_query_len=8,
            ifa_kv_len=128,
        )
        events, calls = [], []
        with self.assertRaisesRegex(RuntimeError, "body failure"):
            with operator_range(
                attributes,
                enabled=True,
                sink=events.append,
                push=lambda _: calls.append("push"),
                pop=lambda: calls.append("pop"),
            ):
                raise RuntimeError("body failure")
        self.assertEqual(calls, ["push", "pop"])
        self.assertFalse(events[0]["args"]["completed"])
        self.assertEqual(events[0]["args"]["timing_scope"], "cpu_launch_range_not_gpu_duration")
