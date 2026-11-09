import gzip
import json
import tempfile
import unittest
from pathlib import Path

from infra_forward_agent.analysis import analyze, union_us
from infra_forward_agent.store import write_json
from infra_forward_agent.trace import load_chrome


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "trace.json"

    def event(self, name, start, duration, device=0, kind="compute", cat="kernel"):
        return {
            "ph": "X",
            "name": name,
            "ts": start,
            "dur": duration,
            "cat": cat,
            "args": {"device": device, "ifa_kind": kind},
        }

    def test_union_handles_overlap_nested_and_touching_intervals(self):
        self.assertEqual(union_us([(0, 5), (1, 2), (5, 8), (10, 11)]), 9)
        self.assertEqual(union_us([]), 0)

    def test_per_device_union_overlap_clipping_and_cpu_filter(self):
        events = [
            self.event("forward", 10, 100, cat="user_annotation"),
            self.event("compute", 0, 70),
            self.event("comm", 50, 40, kind="communication"),
            self.event("memory", 100, 30, kind="memory"),
            self.event("cpu_aten", 10, 100, cat="cpu_op"),
            self.event("device1", 10, 100, device=1),
        ]
        write_json(self.path, {"traceEvents": events, "displayTimeUnit": "ns"})
        devices = analyze(load_chrome(self.path))["devices"]
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0]["forward_us"], 100)
        self.assertEqual(devices[0]["busy_us"], 90)
        self.assertEqual(devices[0]["idle_us"], 10)
        self.assertEqual(devices[0]["compute_communication_overlap_us"], 20)
        self.assertEqual(devices[0]["communication_without_compute_us"], 20)
        self.assertEqual(devices[1]["busy_us"], 100)
        self.assertNotIn("cpu_aten", [row["name"] for row in devices[0]["hotspots"]])

    def test_missing_marker_multiple_steps_and_zero_durations_rejected(self):
        forward = self.event("forward", 0, 100)
        gpu = self.event("gpu", 0, 50)
        for events in ([gpu], [forward, forward, gpu], [forward, self.event("gpu", 0, 0)]):
            with self.subTest(events=events):
                write_json(self.path, events)
                with self.assertRaises(ValueError):
                    load_chrome(self.path)

    def test_name_heuristic_and_gzip(self):
        event = self.event("nccl_allreduce", 0, 50)
        del event["args"]["ifa_kind"]
        path = self.path.with_suffix(".json.gz")
        with gzip.open(path, "wt") as handle:
            json.dump([self.event("forward", 0, 100), event], handle)
        trace = load_chrome(path)
        self.assertEqual(trace.slices[0].kind, "communication")
        self.assertEqual(trace.slices[0].classification, "name_heuristic")


if __name__ == "__main__":
    unittest.main()
