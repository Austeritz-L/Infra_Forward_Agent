import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from infra_forward_agent.contracts import Evidence, Policy, read_json
from infra_forward_agent.demo import make_demo
from infra_forward_agent.engine import replay
from infra_forward_agent.evaluation import compare
from infra_forward_agent.store import RunStore, write_json


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = make_demo(self.root / "inputs")
        self.base = Evidence.load(self.config.parent / "baseline/evidence.json")
        self.candidate = Evidence.load(self.config.parent / "kernel-candidate/evidence.json")

    def test_closed_loop_reanalyzes_accepted_candidate_and_keeps_rejection(self):
        output = self.root / "run"
        events = []
        summary = replay(self.config, output, events.append)
        self.assertEqual(summary["selected"], "overlap-candidate")
        self.assertEqual([row["accepted"] for row in summary["decisions"]], [True, True, False])
        self.assertEqual(summary["decisions"][1]["parent"], "kernel-candidate")
        self.assertEqual(summary["decisions"][2]["parent"], "overlap-candidate")
        before = read_json(output / "baseline.analysis.json")["devices"][0]["hotspots"][0]
        after = read_json(output / "iteration.001.analysis.json")["devices"][0]["hotspots"][0]
        self.assertEqual(before["name"], "expert_gemm")
        self.assertEqual(after["name"], "collective_all_to_all")
        self.assertEqual(events[-1]["stage"], "run_completed")
        self.assertEqual([row["seq"] for row in events], list(range(1, len(events) + 1)))
        for manifest in output.glob("evidence/*/manifest.json"):
            for name, digest in read_json(manifest).items():
                self.assertEqual(
                    hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest(), digest
                )
        with self.assertRaises(FileExistsError):
            replay(self.config, output)

    def test_reject_incomparable_environment_workload_and_synthetic_kind(self):
        for field, value in (
            ("environment", {"image_digest": "other"}),
            ("workload", {"seed": 99}),
            ("kind", "measured"),
        ):
            with self.subTest(field=field):
                data = copy.deepcopy(self.candidate.data)
                data[field] = value
                result = compare(self.base, Evidence(self.candidate.root, data), Policy())
                self.assertFalse(result.accepted)
                self.assertIn(field, result.reasons[0])

    def test_reject_invalid_measurements_and_noise(self):
        path = self.candidate.artifact("benchmark.json")
        original = read_json(path)
        for value in (0, -1, float("nan"), float("inf"), True, "8.0", 100):
            with self.subTest(value=value):
                data = copy.deepcopy(original)
                data["conditions"][0]["samples"][0]["forward_ms"] = value
                # Permit invalid JSON constants here to exercise the input boundary.
                path.write_text(json.dumps(data))
                self.assertFalse(compare(self.base, self.candidate, Policy()).accepted)

    def test_precision_order_coverage_profiled_and_missing_samples(self):
        pre = self.candidate.artifact("precision-before.json")
        benchmark = self.candidate.artifact("benchmark.json")
        original = read_json(pre)
        for field, value in (
            ("passed", False),
            ("ranks", [0]),
            ("conditions", ["decode-c1"]),
            ("finished_ns", 99),
            ("clock_id", "other"),
        ):
            data = dict(original, **{field: value})
            write_json(pre, data)
            self.assertFalse(compare(self.base, self.candidate, Policy()).accepted)
        write_json(pre, original)
        original_bench = read_json(benchmark)
        for field, value in (("profiled", True), ("completed", False), ("errors", ["timeout"])):
            write_json(benchmark, dict(original_bench, **{field: value}))
            self.assertFalse(compare(self.base, self.candidate, Policy()).accepted)
        data = copy.deepcopy(original_bench)
        data["conditions"][0]["samples"] = data["conditions"][0]["samples"][:1]
        write_json(benchmark, data)
        self.assertFalse(compare(self.base, self.candidate, Policy()).accepted)

    def test_throughput_regression_rejected_even_with_latency_gain(self):
        path = self.candidate.artifact("benchmark.json")
        data = read_json(path)
        for condition in data["conditions"]:
            for sample in condition["samples"]:
                sample["tokens_per_s"] = 1.0
        write_json(path, data)
        self.assertFalse(compare(self.base, self.candidate, Policy()).accepted)

    def test_changed_precision_contract_or_condition_matrix_is_rejected(self):
        for phase in ("before", "after"):
            path = self.candidate.artifact(f"precision-{phase}.json")
            data = read_json(path)
            data["validation_contract"] = "different-or-weaker-contract"
            write_json(path, data)
        decision = compare(self.base, self.candidate, Policy())
        self.assertFalse(decision.accepted)
        self.assertIn("contract differs", decision.reasons[0])
        # Restore the original contract, then change one workload condition.
        for phase in ("before", "after"):
            path = self.candidate.artifact(f"precision-{phase}.json")
            data = read_json(path)
            data["validation_contract"] = "synthetic-output-equality-v1"
            write_json(path, data)
        path = self.candidate.artifact("benchmark.json")
        data = read_json(path)
        data["conditions"][0]["spec"]["input_length"] = 512
        write_json(path, data)
        self.assertFalse(compare(self.base, self.candidate, Policy()).accepted)

    def test_missing_rank_trace_fails_without_promotion_and_is_logged(self):
        trace = self.candidate.artifact("rank-1.trace.json")
        data = read_json(trace)
        data["traceEvents"] = data["traceEvents"][:1]
        write_json(trace, data)
        output = self.root / "failed"
        with self.assertRaises(ValueError):
            replay(self.config, output)
        events = [json.loads(row) for row in (output / "events.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["stage"], "run_failed")
        self.assertNotIn("baseline_advanced", [row["stage"] for row in events])

    def test_iteration_budget(self):
        data = read_json(self.config)
        data["max_iterations"] = 1
        write_json(self.config, data)
        summary = replay(self.config, self.root / "bounded")
        self.assertEqual(summary["stop_reason"], "iteration_budget")
        self.assertEqual(summary["selected"], "kernel-candidate")

    def test_paths_cannot_escape_or_overwrite_evidence(self):
        store = RunStore(self.root / "snapshots")
        for bad in ("../outside", str(self.candidate.artifact("benchmark.json")), "evidence.json"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.candidate.artifact(bad)
        store.snapshot(self.candidate, "valid")

    def test_cli_demo_streams_and_exits(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "infra_forward_agent",
                "demo",
                "--output",
                str(self.root / "cli"),
            ],
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["stage"], "run_completed")


if __name__ == "__main__":
    unittest.main()
