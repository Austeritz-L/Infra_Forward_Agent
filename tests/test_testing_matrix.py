import copy
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from infra_forward_agent.campaign import build_plan
from infra_forward_agent.contracts import read_json
from infra_forward_agent.scenarios import expand_suite

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs/scenarios/prefill-service.json"


class PrefillMatrixTests(unittest.TestCase):
    def setUp(self):
        self.suite = read_json(SUITE)

    def test_default_matrix_pairs_each_workload_across_three_prefill_scopes(self):
        rows = expand_suite(self.suite)
        self.assertEqual(len(rows), 132)
        groups = defaultdict(list)
        for row in rows:
            self.assertEqual(row["phase"], "prefill")
            self.assertEqual(row["generation"]["max_new_tokens"], 1)
            self.assertNotIn("batch_size", row["parameters"])
            groups[row["workload_id"]].append(row)
        self.assertEqual(len(groups), 44)
        for rows in groups.values():
            self.assertEqual({row["scope"] for row in rows}, {"service", "forward", "operator"})
            self.assertEqual(len({row["alignment"]["service_scenario_id"] for row in rows}), 1)
            self.assertTrue(all(row["parameters"] == rows[0]["parameters"] for row in rows))
            for row in rows:
                if row["scope"] != "service":
                    self.assertEqual(row["alignment"]["binding_status"], "requires_service_capture")
        self.assertTrue(
            all(
                not row["formal_performance_eligible"]
                for row in expand_suite(self.suite)
                if row["tier"] == "smoke"
            )
        )

    def test_full_matrix_contains_soak_pressure_optional_tiers_without_decode(self):
        self.suite["enabled_tiers"] = ["smoke", "core", "extended", "soak", "stress", "optional"]
        rows = expand_suite(self.suite)
        self.assertEqual(len(rows), 384)
        service = [row for row in rows if row["scope"] == "service"]
        self.assertEqual(len({row["case"] for row in service}), 15)
        self.assertIn("S07-locust-soak", {row["case"] for row in service})
        self.assertIn("storage-restore", {row["cache_profile"] for row in service})
        self.assertTrue(all(row["phase"] == "prefill" for row in rows))

    def test_workload_ids_are_stable_under_axes_and_tier_order_changes(self):
        original = {row["id"] for row in expand_suite(self.suite)}
        self.suite["enabled_tiers"].reverse()
        for family in self.suite["workloads"]:
            family["axes"] = dict(reversed(list(family["axes"].items())))
        self.assertEqual(original, {row["id"] for row in expand_suite(self.suite)})

    def test_invalid_contract_cache_generation_and_tiers_rejected(self):
        for mutation in ("cache", "decode", "boolean", "tier", "metrics", "requirements"):
            data = copy.deepcopy(self.suite)
            if mutation == "cache":
                data["workloads"][0]["axes"]["cache_profile"] = ["guessed-hit"]
            elif mutation == "decode":
                data["generation"]["max_new_tokens"] = 128
            elif mutation == "boolean":
                data["generation"]["max_new_tokens"] = True
            elif mutation == "tier":
                data["enabled_tiers"] = ["unknown"]
            elif mutation == "metrics":
                data["contracts"]["metrics"].pop("operator")
            else:
                data["workloads"][0]["requirements"] = "not-a-list"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                expand_suite(data)

    def test_expansion_limit_counts_all_three_scopes(self):
        with self.assertRaises(ValueError):
            expand_suite(self.suite, max_cases=131)
        self.assertEqual(len(expand_suite(self.suite, max_cases=132)), 132)

    def test_campaign_carries_cache_metrics_acceptance_and_capture_contract(self):
        plan = build_plan(ROOT / "configs/campaigns/multi-architecture.json")
        self.assertEqual(plan["scenario_count_total"], 396)
        contract = plan["testing_contract"]
        self.assertEqual(contract["focus"], "prefill")
        self.assertFalse(contract["runtime_acceptance_implemented"])
        self.assertIn("capture_contract", contract["contracts"])
        self.assertIn("TPOT", contract["contracts"]["metrics"]["service"]["not_in_scope"])
        self.assertEqual(
            Counter(row["scope"] for row in plan["scenarios"]),
            {"service": 44, "forward": 44, "operator": 44},
        )
        for stage in plan["stages"]:
            if stage["module"] == "performance":
                self.assertFalse(stage["options"]["decode_metrics_in_scope"])
                self.assertFalse(any("TPOT" in output for output in stage["expected_outputs"]))


if __name__ == "__main__":
    unittest.main()
