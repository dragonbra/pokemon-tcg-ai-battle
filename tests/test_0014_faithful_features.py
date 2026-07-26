from __future__ import annotations

import copy
import gzip
import importlib
import json
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
RAW_SHARD = (
    ROOT
    / "rl_runs/0013_semantic_goal_policy/dataset/V1_causal_semantic_v1/train-00000.jsonl.gz"
)
model_0010 = importlib.import_module("train.project_0010_alakazam_sota_model.model")
state_0014 = importlib.import_module(
    "train.0014_faithful_board_causal_features.knowledge.state"
)
compiler_0014 = importlib.import_module(
    "train.0014_faithful_board_causal_features.features.compiler"
)
materialized_0014 = importlib.import_module(
    "train.0014_faithful_board_causal_features.training.materialized"
)
model_0014 = importlib.import_module(
    "train.0014_faithful_board_causal_features.model"
)
ac_model_0014 = importlib.import_module(
    "train.0014_faithful_board_causal_features.ac_model"
)
ac_data_0014 = importlib.import_module(
    "train.0014_faithful_board_causal_features.training.ac_data"
)
semantics_0013 = importlib.import_module(
    "train.0013_semantic_goal_policy.features.card_semantics"
)


def _deck(row: dict) -> list[int]:
    cards: list[int] = []
    for card_id, count in row["deck_manifest"]["counts"]:
        cards.extend([card_id] * count)
    return cards


def _first_rows(count: int) -> list[dict]:
    with gzip.open(RAW_SHARD, "rt", encoding="utf-8") as handle:
        return [json.loads(next(handle)) for _ in range(count)]


class FaithfulFeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = semantics_0013.CardSemanticRegistry.from_official_csv(
            ROOT / "data/official/EN_Card_Data.csv"
        )

    def test_real_rows_preserve_exact_0010_codec(self) -> None:
        rows = _first_rows(32)
        first = rows[0]
        actor = first["identity"]["player_index"]
        knowledge = state_0014.CausalKnowledge(actor, _deck(first))
        legacy_codec = model_0010.IDOnlyCodec(model_0010.IDOnlyConfig())
        checked = 0
        for row in rows:
            if row["identity"]["player_index"] != actor:
                break
            snapshot = knowledge.consume(row["actor_observation"], row["event_cursor"])
            compiled = compiler_0014.compile_row(row, snapshot, self.registry)
            expected = legacy_codec.encode(row["actor_observation"], row["ordered_action"])
            self.assertIsNotNone(expected)
            for name in (
                "global_cat",
                "global_num",
                "entity_cat",
                "entity_num",
                "option_cat",
                "action",
                "min_count",
                "max_count",
            ):
                self.assertEqual(compiled[name], expected[name])
            checked += 1
        self.assertGreater(checked, 5)

    def test_hidden_opponent_draw_never_invents_identity(self) -> None:
        row = _first_rows(1)[0]
        actor = row["identity"]["player_index"]
        opponent = 1 - actor
        self.assertTrue(
            any(
                log.get("type") == 5 and log.get("playerIndex") == opponent
                for log in row["actor_observation"]["logs"]
            )
        )
        knowledge = state_0014.CausalKnowledge(actor, _deck(row))
        snapshot = knowledge.consume(row["actor_observation"], row["event_cursor"])
        opponent_hand_count = row["actor_observation"]["current"]["players"][opponent][
            "handCount"
        ]
        self.assertEqual(snapshot.known_opponent_hand, ())
        self.assertEqual(snapshot.unknown_opponent_hand, opponent_hand_count)
        for item in snapshot.self_ledger.values():
            visible = sum(item.visible.values())
            self.assertLessEqual(item.deck.upper, item.initial - visible)
            self.assertLessEqual(item.prize.upper, item.initial - visible)

    def test_full_deck_view_recovers_deck_and_prize_without_future_data(self) -> None:
        with gzip.open(RAW_SHARD, "rt", encoding="utf-8") as handle:
            rows = (json.loads(line) for line in handle)
            target = next(row for row in rows if row["actor_observation"]["select"]["deck"])
        actor = target["identity"]["player_index"]
        knowledge = state_0014.CausalKnowledge(actor, _deck(target))
        snapshot = knowledge.consume(target["actor_observation"])
        self.assertTrue(snapshot.deck_membership_known)
        self.assertTrue(all(item.deck.value is not None for item in snapshot.self_ledger.values()))
        self.assertTrue(all(item.prize.value is not None for item in snapshot.self_ledger.values()))
        own = target["actor_observation"]["current"]["players"][actor]
        self.assertEqual(
            sum(item.prize.value for item in snapshot.self_ledger.values()), len(own["prize"])
        )

    def test_event_actor_is_relative_and_history_crosses_decisions(self) -> None:
        rows = _first_rows(2)
        actor = rows[0]["identity"]["player_index"]
        self.assertEqual(rows[1]["identity"]["player_index"], actor)
        knowledge = state_0014.CausalKnowledge(actor, _deck(rows[0]))
        first = knowledge.consume(rows[0]["actor_observation"], rows[0]["event_cursor"])
        second = knowledge.consume(rows[1]["actor_observation"], rows[1]["event_cursor"])
        self.assertGreater(len(second.recent_events), len(rows[1]["actor_observation"]["logs"]))
        compiled = compiler_0014.compile_row(rows[1], second, self.registry)
        first_actor_code = compiled["event_cat"][0][1]
        event_actor = second.recent_events[0].actor
        expected = 0 if event_actor is None else (1 if event_actor == actor else 2)
        self.assertEqual(first_actor_code, expected)
        self.assertEqual(first.recent_events[0].decision_index, 0)
        self.assertEqual(second.recent_events[-1].decision_index, 1)

    def test_terminal_outcome_does_not_change_model_features(self) -> None:
        row = _first_rows(1)[0]
        actor = row["identity"]["player_index"]
        altered = copy.deepcopy(row)
        altered["terminal_outcome"] = "loss" if row["terminal_outcome"] != "loss" else "win"
        left = state_0014.CausalKnowledge(actor, _deck(row)).consume(row["actor_observation"])
        right = state_0014.CausalKnowledge(actor, _deck(row)).consume(
            altered["actor_observation"]
        )
        left_features = compiler_0014.compile_row(row, left, self.registry)
        right_features = compiler_0014.compile_row(altered, right, self.registry)
        self.assertEqual(left_features, right_features)

    def test_materialized_a0_view_equals_legacy_collate(self) -> None:
        rows = _first_rows(4)
        actor = rows[0]["identity"]["player_index"]
        knowledge = state_0014.CausalKnowledge(actor, _deck(rows[0]))
        compiled = []
        for row in rows:
            self.assertEqual(row["identity"]["player_index"], actor)
            snapshot = knowledge.consume(row["actor_observation"], row["event_cursor"])
            value = compiler_0014.compile_row(row, snapshot, self.registry)
            value["source_payload_sha256"] = row["source_payload_sha256"]
            compiled.append(value)
        tensors = materialized_0014._compact_records(compiled)
        actual = materialized_0014.select_a0_batch(tensors, torch.arange(len(rows)))
        expected = model_0010.collate_id_only(compiled)
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_cursor_mismatch_fails_closed(self) -> None:
        row = _first_rows(1)[0]
        actor = row["identity"]["player_index"]
        cursor = dict(row["event_cursor"])
        cursor["incoming_log_count"] += 1
        with self.assertRaisesRegex(ValueError, "log count mismatch"):
            state_0014.CausalKnowledge(actor, _deck(row)).consume(
                row["actor_observation"], cursor
            )

    def test_a0_model_is_parameter_identical_and_batched_greedy_is_exact(self) -> None:
        torch.manual_seed(14)
        original = model_0010.IDOnlyPointerPolicy(model_0010.IDOnlyConfig()).eval()
        control = model_0014.Faithful0010PointerPolicy(model_0010.IDOnlyConfig()).eval()
        control.load_state_dict(original.state_dict())
        self.assertEqual(model_0014.parameter_count(control), 7_154_562)
        rows = _first_rows(4)
        encoded = [
            model_0010.IDOnlyCodec(model_0010.IDOnlyConfig()).encode(
                row["actor_observation"], row["ordered_action"]
            )
            for row in rows
        ]
        self.assertTrue(all(row is not None for row in encoded))
        batch = model_0010.collate_id_only(encoded)
        with torch.inference_mode():
            expected_logits = original.teacher_logits(batch)
            encoding = control.encode(batch)
            actual_logits = control.teacher_logits_from_encoding(batch, encoding)
            decoded = control.deterministic_action_tensors(batch, encoded=encoding)
        self.assertTrue(torch.equal(actual_logits, expected_logits))
        expected_actions = []
        for index in range(len(rows)):
            single = {name: value[index : index + 1] for name, value in batch.items()}
            expected_actions.append(original.greedy_action(single))
        actual_actions = [
            decoded.sequences[index, : decoded.lengths[index]].tolist()
            for index in range(len(rows))
        ]
        self.assertEqual(actual_actions, expected_actions)
        self.assertTrue(bool(decoded.legal.all()))

    def test_ac_zero_gates_preserve_a0_and_auxiliary_readers_open(self) -> None:
        rows = _first_rows(4)
        actor = rows[0]["identity"]["player_index"]
        knowledge = state_0014.CausalKnowledge(actor, _deck(rows[0]))
        compiled = []
        for row in rows:
            snapshot = knowledge.consume(row["actor_observation"], row["event_cursor"])
            value = compiler_0014.compile_row(row, snapshot, self.registry)
            value["source_payload_sha256"] = row["source_payload_sha256"]
            compiled.append(value)
        tensors = materialized_0014._compact_records(compiled)
        batch = ac_data_0014.select_ac_batch(tensors, torch.arange(len(rows)))

        torch.manual_seed(14)
        base = model_0014.Faithful0010PointerPolicy(model_0010.IDOnlyConfig()).eval()
        ac = ac_model_0014.AllFeatureCausalPolicy(
            ac_model_0014.ACModelConfig(),
            ontology_path=(
                ROOT
                / "rl_runs/0014_faithful_board_causal_features/dataset/"
                "V1_full_feature_superset/card_ontology.json"
            ),
        ).eval()
        ac.load_state_dict(base.state_dict(), strict=False)
        with torch.inference_mode():
            expected = base.teacher_logits(batch)
            actual = ac.teacher_logits(batch)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(ac_model_0014.parameter_count(ac), 9_885_122)

        with torch.no_grad():
            ac.state_aux_gate.fill_(0.1)
            ac.option_aux_gate.fill_(0.1)
        with torch.inference_mode():
            augmented = ac.teacher_logits(batch)
        self.assertTrue(torch.isfinite(augmented[augmented > -1e30]).all())
        self.assertGreater(float((augmented - expected).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
