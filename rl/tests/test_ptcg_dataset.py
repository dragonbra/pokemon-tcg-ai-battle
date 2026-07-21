from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from rl.ptcg.dataset import (
    DATASET_VERSION,
    SUPPORTED_FEATURE_SCHEMA_VERSIONS,
    load_behavior_cloning_dataset,
)
from rl.ptcg.features import PTCGFeatureConfig
from rl.ptcg.build_dagger_dataset import iter_dagger_records
from rl.ptcg.rewards import observation_potential, potential_shaping


class PTCGDatasetTests(unittest.TestCase):
    def test_loader_keeps_known_legacy_feature_schemas_readable(self) -> None:
        self.assertEqual(
            SUPPORTED_FEATURE_SCHEMA_VERSIONS,
            {"ptcg_features_v1", "ptcg_features_v2", "ptcg_features_v3"},
        )

    def test_visible_potential_components_are_stable_and_auditable(self) -> None:
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {
                        "prizeCount": 5,
                        "deckCount": 40,
                        "active": [{"id": 741, "energies": [5]}],
                        "bench": [{"id": 743}],
                    },
                    {"prizeCount": 4, "deckCount": 42, "active": [], "bench": []},
                ],
            }
        }
        potential = observation_potential(observation)
        self.assertAlmostEqual(potential.prize_race, -1 / 6)
        self.assertAlmostEqual(potential.attack_readiness, 0.58)
        self.assertAlmostEqual(potential.library_safety, 0.0)
        self.assertAlmostEqual(
            potential.total,
            potential.prize_race + potential.attack_readiness + potential.library_safety,
        )

    def test_potential_shaping_is_gamma_discounted_transition_difference(self) -> None:
        current = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {"prizeCount": 6, "deckCount": 16, "active": [], "bench": []},
                    {"prizeCount": 6, "deckCount": 60, "active": [], "bench": []},
                ],
            }
        }
        next_observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {"prizeCount": 5, "deckCount": 10, "active": [], "bench": []},
                    {"prizeCount": 6, "deckCount": 60, "active": [], "bench": []},
                ],
            }
        }
        shaping = potential_shaping(current, next_observation, gamma=0.9)
        self.assertAlmostEqual(shaping["prize_race"], 0.15)
        self.assertAlmostEqual(shaping["library_safety"], -0.3)
        self.assertAlmostEqual(shaping["attack_readiness"], 0.0)
        self.assertAlmostEqual(
            shaping["total"],
            sum(
                shaping[name]
                for name in ("prize_race", "attack_readiness", "library_safety")
            ),
        )

    def test_dagger_relabels_only_main_actions_with_a_legal_teacher_target(self) -> None:
        teacher = ModuleType("toy_teacher")
        calls: list[dict] = []

        def agent(observation: dict) -> list[int]:
            calls.append(observation)
            return [] if observation.get("select") is None else [0]

        teacher.agent = agent  # type: ignore[attr-defined]
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [{"active": [], "bench": []}, {"active": [], "bench": []}],
            },
            "select": {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [{"type": 13}, {"type": 14}],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.json"
            path.write_text(
                json.dumps(
                    {
                        "result": {
                            "candidate_physical_index": 0,
                            "winner": 0,
                            "game_id": "dagger-toy",
                        },
                        "trace": [{"step": 1, "observation": observation, "action": [1]}],
                    }
                ),
                encoding="utf-8",
            )
            records = list(iter_dagger_records(path, teacher))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["target"], 0)
        self.assertEqual(records[0]["label_source"], "rule_teacher_dagger")
        self.assertEqual(len(calls), 2)  # reset plus the candidate-owned decision

    def test_local_trace_is_converted_to_a_legal_main_action_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = {
                "current": {
                    "turn": 1,
                    "yourIndex": 0,
                    "firstPlayer": 0,
                    "players": [
                        {"active": [{"id": 741}], "bench": []},
                        {"active": [], "bench": []},
                    ],
                },
                "select": {
                    "type": 0,
                    "context": 0,
                    "minCount": 1,
                    "maxCount": 1,
                    "option": [{"type": 13}, {"type": 14}],
                },
            }
            trace_path = root / "game.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "result": {"game_id": "toy-real-shape"},
                        "trace": [{"step": 3, "observation": observation, "action": [1]}],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "dataset.jsonl"
            from rl.ptcg.dataset import write_behavior_cloning_dataset

            result = write_behavior_cloning_dataset([trace_path], output)
            self.assertEqual(result["dataset_version"], DATASET_VERSION)
            self.assertEqual(result["records"], 1)
            records = load_behavior_cloning_dataset(output)
            self.assertEqual(records[0]["target"], 1)
            self.assertEqual(
                len(records[0]["encoded"]["state_numeric"]),
                PTCGFeatureConfig().state_numeric_dim,
            )

    def test_effect_or_multiselect_records_are_excluded_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = {
                "current": {
                    "turn": 1,
                    "yourIndex": 0,
                    "players": [{"active": [], "bench": []}, {"active": [], "bench": []}],
                },
                "select": {
                    "type": 2,
                    "context": 0,
                    "minCount": 1,
                    "maxCount": 2,
                    "option": [{"type": 1}, {"type": 2}],
                },
            }
            trace_path = root / "game.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "trace": [
                            {"step": 1, "observation": observation, "action": [0, 1]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            from rl.ptcg.dataset import write_behavior_cloning_dataset

            result = write_behavior_cloning_dataset([trace_path], root / "dataset.jsonl")
            self.assertEqual(result["records"], 0)


if __name__ == "__main__":
    unittest.main()
