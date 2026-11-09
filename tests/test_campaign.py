import copy
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from infra_forward_agent.agents.base import StagePlan
from infra_forward_agent.architectures import ArchitectureProfile, ArchitectureRegistry
from infra_forward_agent.campaign import build_plan, validate_dependencies, write_plan
from infra_forward_agent.contracts import read_json
from infra_forward_agent.scenarios import expand_suite
from infra_forward_agent.store import write_json


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.suite = {
            "schema_version": 1,
            "cases": [
                {
                    "id": "prefill",
                    "scope": "forward",
                    "phase": "prefill",
                    "axes": {"input_tokens": [1024, 8192], "batch_size": [1, 16]},
                }
            ],
        }
        write_json(self.root / "suite.json", self.suite)
        self.config = {
            "schema_version": 1,
            "mode": "plan",
            "id": "test-campaign",
            "architectures": ["moore-s5000", "nvidia-sm90", "nvidia-sm103"],
            "scenario_suite": "suite.json",
            "deployment": {"objective": "capacity"},
        }
        self.path = self.root / "campaign.json"
        write_json(self.path, self.config)

    def test_five_domains_per_architecture_and_no_claimed_execution(self):
        plan = build_plan(self.path)
        self.assertEqual(plan["scenario_count_per_architecture"], 4)
        self.assertEqual(plan["scenario_count_total"], 12)
        self.assertEqual(len(plan["stages"]), 18)
        self.assertFalse(plan["execution_ready"])
        for architecture in self.config["architectures"]:
            stages = [row for row in plan["stages"] if row["architecture"] == architecture]
            self.assertEqual(
                [row["module"] for row in stages],
                ["preflight", "testing", "performance", "deployment", "kernel", "traces"],
            )
            self.assertEqual(stages[-1]["status"], "skipped")
            self.assertEqual(stages[3]["options"]["objective"], "capacity")
            self.assertFalse(stages[3]["options"]["auto_deploy"])

    def test_new_architecture_profile_needs_no_coordinator_code_change(self):
        registry = ArchitectureRegistry.builtin()
        custom = replace(registry.get("moore-s5000"), id="future-device", vendor="future")
        write_json(self.root / "custom.json", custom.to_dict())
        self.config["architecture_profiles"] = ["custom.json"]
        self.config["architectures"] = ["future-device"]
        write_json(self.path, self.config)
        plan = build_plan(self.path)
        self.assertEqual(plan["architectures"][0]["id"], "future-device")
        self.assertEqual(len(plan["stages"]), 6)
        with self.assertRaises(ValueError):
            registry.register(registry.get("moore-s5000"))

    def test_unknown_duplicate_architecture_and_bad_objective_rejected(self):
        for key, value in (
            ("architectures", ["unknown"]),
            ("architectures", ["moore-s5000", "moore-s5000"]),
            ("deployment", {"objective": "guess"}),
        ):
            with self.subTest(key=key, value=value):
                data = dict(self.config, **{key: value})
                write_json(self.path, data)
                with self.assertRaises(ValueError):
                    build_plan(self.path)

    def test_trace_pairs_are_optional_and_isolated_by_architecture(self):
        (self.root / "before.json").write_text("{}")
        (self.root / "after.json").write_text("{}")
        self.config["trace_comparison"] = {
            "enabled": True,
            "pairs": {"nvidia-sm90": {"baseline": "before.json", "candidate": "after.json"}},
        }
        write_json(self.path, self.config)
        stages = {
            row["architecture"]: row
            for row in build_plan(self.path)["stages"]
            if row["module"] == "traces"
        }
        self.assertEqual(stages["nvidia-sm90"]["status"], "needs_adapter")
        self.assertEqual(stages["moore-s5000"]["status"], "skipped")
        self.config["trace_comparison"]["pairs"]["nvidia-sm90"].pop("candidate")
        write_json(self.path, self.config)
        with self.assertRaises(ValueError):
            build_plan(self.path)

    def test_suite_rejects_invalid_axes_duplicate_cases_and_expansion_overflow(self):
        with self.assertRaises(ValueError):
            expand_suite(self.suite, max_cases=3)
        bad = copy.deepcopy(self.suite)
        bad["cases"].append(bad["cases"][0])
        with self.assertRaises(ValueError):
            expand_suite(bad)
        for values in ([], [0], [True], ["1024"]):
            bad = copy.deepcopy(self.suite)
            bad["cases"][0]["axes"]["input_tokens"] = values
            with self.assertRaises(ValueError):
                expand_suite(bad)

    def test_dependencies_reject_cycles_missing_parents_and_duplicate_ids(self):
        stage = StagePlan("a", "testing", "device", [], "forward", [], [])
        for stages in (
            [replace(stage, depends_on=["missing"])],
            [replace(stage, depends_on=["a"])],
            [stage, stage],
        ):
            with self.assertRaises(ValueError):
                validate_dependencies(stages)

    def test_malformed_user_cases_and_requirements_fail_with_validation_error(self):
        with self.assertRaises(ValueError):
            expand_suite({"schema_version": 1, "cases": ["not-an-object"]})
        bad = copy.deepcopy(self.suite)
        bad["cases"][0]["requirements"] = "not-a-list"
        with self.assertRaises(ValueError):
            expand_suite(bad)

    def test_plan_writes_events_and_preserves_previous_output(self):
        output = self.root / "output"
        events = []
        write_plan(self.path, output, events.append)
        self.assertEqual(events[-1]["stage"], "planning_completed")
        self.assertFalse(read_json(output / "campaign.plan.json")["execution_ready"])
        self.assertTrue((output / "scenario.matrix.json").is_file())
        with self.assertRaises(FileExistsError):
            write_plan(self.path, output)

    def test_builtin_profiles_are_packaged_and_unqualified(self):
        profiles = ArchitectureRegistry.builtin().all()
        self.assertEqual(len(profiles), 3)
        for profile in profiles:
            self.assertEqual(profile.status, "planned")
            self.assertEqual(profile.resources, {})
            self.assertTrue(profile.verification_required)
        with self.assertRaises(ValueError):
            ArchitectureProfile.from_dict({"schema_version": 1})

    def test_cli_plans_with_streaming_events(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "infra_forward_agent",
                "plan",
                str(self.path),
                "--output",
                str(self.root / "cli"),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["stage"], "planning_completed")


if __name__ == "__main__":
    unittest.main()
