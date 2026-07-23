from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from train.alakazam_bc_rl.training.dataset import (
    DATASET_VERSION,
    SUPPORTED_FEATURE_SCHEMA_VERSIONS,
    iter_behavior_cloning_records,
    load_behavior_cloning_dataset,
)
from train.alakazam_bc_rl.features import PTCGFeatureConfig, encode_observation, feature_config_for_schema
from train.alakazam_bc_rl.training.build_dagger_dataset import iter_dagger_records
from train.alakazam_bc_rl.training.annotate_transition_returns import annotate_records
from train.alakazam_bc_rl.training.rewards import observation_potential, potential_shaping
from train.alakazam_bc_rl.training.build_mcts_dataset import (
    _aggregate_search_results,
    _action_value_soft_policy,
    _blend_teacher_policy,
    _sample_hidden_deck,
)


class PTCGDatasetTests(unittest.TestCase):
    def test_transition_returns_are_discounted_per_source_game(self) -> None:
        records = [
            {
                "source": "game-a",
                "step": 1,
                "terminal_outcome": 1.0,
                "potential_shaping": {"total": 0.1},
            },
            {
                "source": "game-a",
                "step": 2,
                "terminal_outcome": 1.0,
                "potential_shaping": {"total": 0.0},
            },
            {
                "source": "game-a",
                "step": 3,
                "terminal_outcome": 1.0,
                "potential_shaping": {"total": -0.2},
            },
        ]
        annotated = annotate_records(records, gamma=0.9, shaping_scale=1.0)
        self.assertAlmostEqual(annotated[2]["terminal_reward"], 1.0)
        self.assertAlmostEqual(annotated[2]["transition_return"], 0.8)
        self.assertAlmostEqual(annotated[1]["transition_return"], 0.72)
        self.assertAlmostEqual(annotated[0]["transition_return"], 0.748)
        self.assertEqual(annotated[0]["reward_target_version"], "transition_return_v1")

    def test_dataset_shaping_keeps_same_player_perspective(self) -> None:
        def observation(your_index: int, own_prize: int) -> dict:
            return {
                "current": {
                    "yourIndex": your_index,
                    "players": [
                        {"prizeCount": own_prize, "active": [], "bench": []},
                        {"prizeCount": 6, "active": [], "bench": []},
                    ],
                },
                "select": {
                    "type": 0,
                    "context": 0,
                    "minCount": 1,
                    "maxCount": 1,
                    "option": [{"type": 14}],
                },
            }

        trace = [
            {"step": 1, "observation": observation(0, 6), "action": [0]},
            {"step": 2, "observation": observation(1, 1), "action": [0]},
            {"step": 3, "observation": observation(0, 5), "action": [0]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "perspective.json"
            path.write_text(json.dumps({"trace": trace}), encoding="utf-8")
            records = list(iter_behavior_cloning_records(path, feature_config=PTCGFeatureConfig()))
        self.assertEqual(len(records), 2)
        self.assertAlmostEqual(records[0]["potential_shaping"]["prize_race"], 0.99 / 6.0)

    def test_terminal_outcome_uses_candidate_physical_perspective(self) -> None:
        observation = {
            "current": {
                "yourIndex": 1,
                "players": [{"active": [], "bench": []}, {"active": [], "bench": []}],
            },
            "select": {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [{"type": 14}],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "second-seat-win.json"
            path.write_text(
                json.dumps(
                    {
                        "result": {"winner": 1, "candidate_physical_index": 1},
                        "trace": [{"step": 1, "observation": observation, "action": [0]}],
                    }
                ),
                encoding="utf-8",
            )
            records = list(iter_behavior_cloning_records(path, teacher_player_index=None))
        self.assertEqual(records[0]["terminal_outcome"], 1.0)

    def test_mcts_determinizations_aggregate_visit_counts(self) -> None:
        results = [
            ([0.2, 0.8], [3, 1], [0.75, 0.25], [0.2, 0.8]),
            ([0.6, 0.4], [1, 3], [0.25, 0.75], [0.6, 0.4]),
        ]
        values, visits, policy, root_values = _aggregate_search_results(results, 2)
        self.assertEqual(visits, [4, 4])
        self.assertEqual(policy, [0.5, 0.5])
        self.assertAlmostEqual(values[0], 0.4)
        self.assertAlmostEqual(values[1], 0.6)
        self.assertEqual(root_values, values)

    def test_mcts_hidden_deck_pool_is_identifier_independent(self) -> None:
        import random

        pool = [101, 102, 103, 104]
        sampled = _sample_hidden_deck(pool, 3, random.Random(7))
        self.assertEqual(len(sampled), 3)
        self.assertEqual(len(set(sampled)), 3)
        self.assertTrue(set(sampled).issubset(set(pool)))
        self.assertEqual(_sample_hidden_deck(None, 2, random.Random(7)), [1072, 1072])

    def test_mcts_teacher_policy_anchor_preserves_a_probability_target(self) -> None:
        policy = _blend_teacher_policy([0.25, 0.75], [0], 0.5)
        self.assertAlmostEqual(sum(policy), 1.0)
        self.assertAlmostEqual(policy[0], 0.625)
        self.assertAlmostEqual(policy[1], 0.375)

    def test_action_value_soft_policy_uses_relative_advantage(self) -> None:
        policy = _action_value_soft_policy([0.8, 0.6, None], 0.1)
        self.assertAlmostEqual(sum(policy), 1.0)
        self.assertGreater(policy[0], policy[1])
        self.assertEqual(policy[2], 0.0)
        with self.assertRaises(ValueError):
            _action_value_soft_policy([0.1], 0.0)

    def test_loader_keeps_known_legacy_feature_schemas_readable(self) -> None:
        self.assertEqual(
            SUPPORTED_FEATURE_SCHEMA_VERSIONS,
            {
                "ptcg_features_v1",
                "ptcg_features_v2",
                "ptcg_features_v3",
                "ptcg_features_v4",
                "ptcg_features_v5",
                "ptcg_features_v6",
                "ptcg_features_universal",
            },
        )

    def test_v4_effect_state_adds_explicit_effect_features(self) -> None:
        config = feature_config_for_schema("ptcg_features_v4")
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [{"active": [], "bench": []}, {"active": [], "bench": []}],
            },
            "select": {
                "type": 1,
                "context": 7,
                "minCount": 1,
                "maxCount": 1,
                "option": [{"type": 1}, {"type": 2}],
                "effect": {"id": 123, "serial": 9},
                "contextCard": {"id": 456},
            },
            "rl_effect_step": 2,
        }
        encoded = encode_observation(observation, config)
        self.assertEqual(len(encoded["state_numeric"]), 40)
        self.assertEqual(encoded["state_numeric"][-1], 2 / 64)

    def test_v5_state_adds_explicit_turn_timing_features(self) -> None:
        config = feature_config_for_schema("ptcg_features_v5")
        observation = {
            "current": {
                "yourIndex": 0,
                "turnActionCount": 3,
                "looking": [{"id": 1}],
                "players": [
                    {
                        "active": [{"id": 741, "appearThisTurn": True}],
                        "bench": [{"id": 742, "appearThisTurn": False}],
                    },
                    {
                        "active": [{"id": 305, "appearThisTurn": False}],
                        "bench": [{"id": 66, "appearThisTurn": True}],
                    },
                ],
            },
            "select": {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [{"type": 14}],
            },
        }
        encoded = encode_observation(observation, config)
        self.assertEqual(len(encoded["state_numeric"]), 46)
        self.assertAlmostEqual(encoded["state_numeric"][-6], 3 / 32)
        self.assertAlmostEqual(encoded["state_numeric"][-4], 1 / 6)
        self.assertEqual(encoded["state_numeric"][-2:], [1.0, 0.0])

    def test_v6_resolves_compact_play_and_attack_options(self) -> None:
        config = feature_config_for_schema("ptcg_features_v6")
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {
                        "active": [{"id": 742}],
                        "bench": [],
                        "hand": [{"id": 1079}, {"id": 1182}],
                    },
                    {"active": [{"id": 678}], "bench": [], "hand": []},
                ],
            },
            "select": {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {"type": 7, "index": 0},
                    {"type": 7, "index": 1},
                    {"type": 13, "attackId": 1072},
                ],
            },
        }
        encoded = encode_observation(observation, config)
        self.assertEqual(encoded["action_card_ids"][:3], [1080, 1183, 0])
        self.assertEqual(encoded["action_target_ids"][:3], [0, 0, 743])

    def test_legacy_v5_keeps_compact_action_semantics(self) -> None:
        config = feature_config_for_schema("ptcg_features_v5")
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {"active": [{"id": 742}], "bench": [], "hand": [{"id": 1079}]},
                    {"active": [], "bench": [], "hand": []},
                ],
            },
            "select": {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [{"type": 7, "index": 0}, {"type": 13, "attackId": 1072}],
            },
        }
        encoded = encode_observation(observation, config)
        self.assertEqual(encoded["action_card_ids"][:2], [0, 0])
        self.assertEqual(encoded["action_target_ids"][:2], [0, 0])

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
            from train.alakazam_bc_rl.training.dataset import write_behavior_cloning_dataset

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
            from train.alakazam_bc_rl.training.dataset import write_behavior_cloning_dataset

            result = write_behavior_cloning_dataset([trace_path], root / "dataset.jsonl")
            self.assertEqual(result["records"], 0)


if __name__ == "__main__":
    unittest.main()
