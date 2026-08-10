from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[3]
PROJECT = "train.0040_dragapult_0809_action_boundary_rl"
PROBE = ROOT / ".tmp/official_cpu_search_same_perspective/probe_output.jsonl"


def _rows() -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for line in PROBE.read_text(encoding="utf-8").splitlines():
        if line.startswith("PHASE3_JSON\t"):
            _, case, kind, payload = line.split("\t", 3)
            rows.setdefault(case, {})[kind] = json.loads(payload)
    return rows


def _search_observation(value: object) -> dict:
    assert isinstance(value, dict)
    return value["state"]["observation"]


class ValueSearchV0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not PROBE.is_file():
            raise unittest.SkipTest("Phase 3 official CPU probe output is unavailable")
        cls.rows = _rows()
        cls.eligibility = importlib.import_module(f"{PROJECT}.value_search_v0.eligibility")
        cls.online = importlib.import_module(
            f"{PROJECT}.semantic_policy.deployment.online_runtime"
        )
        cls.config = importlib.import_module(f"{PROJECT}.semantic_policy.model.config")

    def test_exact_phase3_handler_shapes_are_grouped(self) -> None:
        expected = {
            "BENCH_PLACEMENT": "basic_placement",
            "ENERGY_ATTACH_TARGET": "energy_target",
            "RETREAT_ENERGY_PAYMENT": "retreat_payment",
            "RETREAT_SWITCH_TARGET": "retreat_switch_target",
            "BOSS_OPPONENT_TARGET": "opponent_target",
            "TRAINER_EFFECT_TARGET": "attached_resource_destination",
        }
        basic_ids = {947, 992, 998}
        for case, family in expected.items():
            with self.subTest(case=case):
                observation = self.rows[case]["ROOT_OBSERVATION"]
                group, reason = self.eligibility.policy_conditioned_group(
                    observation, [0], is_basic_card=basic_ids.__contains__
                )
                self.assertEqual(reason, "ELIGIBLE_SHAPE")
                self.assertIsNotNone(group)
                self.assertEqual(group.family.value, family)
                self.assertGreaterEqual(len(group.selections), 2)
                self.assertIn((0,), group.selections)

    def test_policy_conditioning_never_crosses_main_families(self) -> None:
        observation = self.rows["ENERGY_ATTACH_TARGET"]["ROOT_OBSERVATION"]
        group, _ = self.eligibility.policy_conditioned_group(
            observation, [0], is_basic_card=lambda _card_id: True
        )
        self.assertEqual(group.selections, ((0,), (1,), (2,)))
        self.assertNotIn((3,), group.selections)  # Play Basic
        self.assertNotIn((4,), group.selections)  # End

    def test_value_winner_maps_to_official_selection_and_ties_keep_policy(self) -> None:
        runtime = importlib.import_module(f"{PROJECT}.value_search_v0.runtime")
        group = self.eligibility.CandidateGroup(
            family=self.eligibility.Family.ENERGY_TARGET,
            selections=((2,), (5,), (9,)),
            policy_selection=(5,),
        )
        winner, policy_value, best_value = runtime.select_value_winner(
            group, (-0.4, 0.1, 0.8)
        )
        self.assertEqual(winner, [9])
        self.assertEqual(policy_value, 0.1)
        self.assertEqual(best_value, 0.8)
        tied, _, _ = runtime.select_value_winner(group, (0.1, 0.8, 0.8))
        self.assertEqual(tied, [5])

    def test_hidden_draw_counterexample_is_not_whitelisted(self) -> None:
        observation = self.rows["HIDDEN_DRAW_COUNTEREXAMPLE"]["ROOT_OBSERVATION"]
        group, reason = self.eligibility.policy_conditioned_group(
            observation, [3], is_basic_card=lambda _card_id: False
        )
        self.assertIsNone(group)
        self.assertEqual(reason, "HANDLER_SHAPE_NOT_WHITELISTED")

    def test_causal_forks_and_siblings_are_isolated(self) -> None:
        case = self.rows["ENERGY_ATTACH_TARGET"]
        root_observation = case["ROOT_OBSERVATION"]
        actor = root_observation["current"]["yourIndex"]
        encoder = self.online.OnlineCausalEncoder(
            actor, case["REGISTERED_DECK"], self.config.ModelConfig()
        )
        encoder.encode(root_observation)
        root_index = encoder.knowledge._decision_index
        root_events = tuple(encoder.knowledge._events)
        branch_a = encoder.fork()
        branch_b = encoder.fork()
        batch_a = branch_a.encode(_search_observation(case["SEARCH_BRANCH_0"]))
        self.assertEqual(encoder.knowledge._decision_index, root_index)
        self.assertEqual(branch_b.knowledge._decision_index, root_index)
        batch_b = branch_b.encode(_search_observation(case["SEARCH_BRANCH_1"]))
        self.assertEqual(encoder.knowledge._decision_index, root_index)
        self.assertEqual(tuple(encoder.knowledge._events), root_events)
        self.assertEqual(branch_a.knowledge._decision_index, root_index + 1)
        self.assertEqual(branch_b.knowledge._decision_index, root_index + 1)
        self.assertTrue(any(not torch.equal(batch_a[key], batch_b[key]) for key in batch_a))

    def test_phase3_search_and_equivalent_live_features_remain_exact(self) -> None:
        for case_name in (
            "BENCH_PLACEMENT",
            "ENERGY_ATTACH_TARGET",
            "RETREAT_ENERGY_PAYMENT",
            "RETREAT_SWITCH_TARGET",
            "BOSS_OPPONENT_TARGET",
            "TRAINER_EFFECT_TARGET",
        ):
            with self.subTest(case=case_name):
                case = self.rows[case_name]
                root = case["ROOT_OBSERVATION"]
                encoder = self.online.OnlineCausalEncoder(
                    root["current"]["yourIndex"],
                    case["REGISTERED_DECK"],
                    self.config.ModelConfig(),
                )
                encoder.encode(root)
                live = encoder.fork().encode(case["LIVE_OBSERVATION"])
                search = encoder.fork().encode(
                    _search_observation(case["SEARCH_BRANCH_0"])
                )
                self.assertEqual(set(live), set(search))
                for key in live:
                    self.assertTrue(torch.equal(live[key], search[key]), key)

    def test_update270_value_and_hidden_perturbation_parity(self) -> None:
        package = ROOT / ".tmp/evaluation/0040_u270_value_search_v0/package"
        if not package.is_dir():
            self.skipTest("materialized U270 experiment package is unavailable")
        script = r'''
import json, sys, torch
from pathlib import Path
package, probe = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(package))
import main
from strategy.contracts.batch import DecisionBatch
from strategy.deployment.online_runtime import OnlineCausalEncoder
rows = {}
for line in probe.read_text().splitlines():
    if line.startswith("PHASE3_JSON\t"):
        _, case, kind, payload = line.split("\t", 3)
        rows.setdefault(case, {})[kind] = json.loads(payload)
for case_name in (
    "BENCH_PLACEMENT", "ENERGY_ATTACH_TARGET", "RETREAT_ENERGY_PAYMENT",
    "RETREAT_SWITCH_TARGET", "BOSS_OPPONENT_TARGET", "TRAINER_EFFECT_TARGET",
):
    case = rows[case_name]
    root = case["ROOT_OBSERVATION"]
    encoder = OnlineCausalEncoder(
        root["current"]["yourIndex"], case["REGISTERED_DECK"], main.POLICY.actor.config
    )
    encoder.encode(root)
    observations = (
        case["LIVE_OBSERVATION"],
        case["SEARCH_BRANCH_0"]["state"]["observation"],
        case["PERTURBED_SEARCH_BRANCH"]["state"]["observation"],
    )
    values = []
    for observation in observations:
        batch = DecisionBatch.from_mapping(encoder.fork().encode(observation))
        with torch.inference_mode():
            validated, state, options = main.POLICY.actor.encode(batch)
            values.append(main.POLICY.value_from_encoded(validated, state, options))
    assert torch.equal(values[0], values[1]), case_name
    assert torch.equal(values[1], values[2]), case_name
print("U270_VALUE_PARITY_PASS")
'''
        completed = subprocess.run(
            [sys.executable, "-c", script, str(package), str(PROBE)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("U270_VALUE_PARITY_PASS", completed.stdout)


if __name__ == "__main__":
    unittest.main()
