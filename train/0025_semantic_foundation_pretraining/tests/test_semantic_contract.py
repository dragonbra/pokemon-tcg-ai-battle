from __future__ import annotations

import copy
import gzip
import importlib
import json
import unittest
from pathlib import Path

import torch

_compiler = importlib.import_module("train.0025_semantic_foundation_pretraining.features.compiler")
_prototypes = importlib.import_module("train.0025_semantic_foundation_pretraining.features.prototypes")
_knowledge = importlib.import_module("train.0025_semantic_foundation_pretraining.knowledge.state")
_batching = importlib.import_module("train.0025_semantic_foundation_pretraining.model.batching")
_model = importlib.import_module("train.0025_semantic_foundation_pretraining.model.multi_memory")
actor_payload, compile_row = _compiler.actor_payload, _compiler.compile_row
FieldState, PrototypeIndex = _prototypes.FieldState, _prototypes.PrototypeIndex
CausalKnowledge = _knowledge.CausalKnowledge
collate = _batching.collate
SemanticFoundationPolicy, SemanticModelConfig = _model.SemanticFoundationPolicy, _model.SemanticModelConfig

ROOT = Path(__file__).resolve().parents[3]
PROTOTYPES = ROOT / "train/0025_semantic_foundation_pretraining/assets/official_public_prototypes_v1.json"
RAW = ROOT / "rl_runs/0019_universal_winner_bc/dataset/V1_universal_winner_raw/train-00000.jsonl.gz"


def first_real_record(prototypes: PrototypeIndex) -> dict:
    with gzip.open(RAW, "rt", encoding="utf-8") as handle:
        row = json.loads(next(handle))
    deck = [int(card_id) for card_id, count in row["deck_manifest"]["counts"] for _ in range(int(count))]
    knowledge = CausalKnowledge(row["identity"]["player_index"], deck)
    snapshot = knowledge.consume(row["actor_observation"], row["event_cursor"])
    return compile_row(row, snapshot, prototypes)


def attack_row(attached_energy_card_id: int) -> dict:
    card = lambda card_id, serial: {"id": card_id, "serial": serial, "playerIndex": 0}
    active = {
        **card(96, 1), "hp": 210, "maxHp": 210, "energyCards": [card(attached_energy_card_id, 2)],
        "tools": [], "preEvolution": [], "appearThisTurn": False,
    }
    opponent = {**card(646, 3), "hp": 70, "maxHp": 70, "energyCards": [], "tools": [], "preEvolution": []}
    player = lambda active_value: {
        "active": [active_value], "bench": [], "benchMax": 5, "deckCount": 53,
        "discard": [], "hand": [], "handCount": 7, "prize": [], "asleep": False,
        "burned": False, "confused": False, "paralyzed": False, "poisoned": False,
    }
    observation = {
        "current": {
            "yourIndex": 0, "firstPlayer": 0, "turn": 3, "turnActionCount": 2,
            "players": [player(active), player(opponent)], "stadium": [], "looking": [],
            "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
            "retreated": False, "turnEnd": False,
        },
        "logs": [],
        "select": {
            "type": 6, "context": 35, "contextCard": None, "effect": None, "deck": [],
            "minCount": 1, "maxCount": 1, "remainDamageCounter": 0,
            "remainEnergyCost": 0, "option": [{"type": 13, "attackId": 120}],
        },
    }
    return {
        "identity": {"date": "2026-08-02", "episode_id": 1, "episode_step": 1, "player_index": 0},
        "actor_observation": observation, "legal_options": observation["select"]["option"],
        "ordered_action": [0], "action_termination": "forced_max", "event_cursor": {"actor_decision_index": 0, "incoming_log_count": 0},
        "deck_manifest": {"counts": [[1, 60]]}, "source_id": 999,
    }


class SemanticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.prototypes = PrototypeIndex.load(PROTOTYPES)

    def compile_synthetic(self, energy: int) -> dict:
        row = attack_row(energy)
        knowledge = CausalKnowledge(0, [1] * 60)
        snapshot = knowledge.consume(row["actor_observation"], row["event_cursor"])
        return compile_row(row, snapshot, self.prototypes)

    def test_distinct_attack_ids_retain_distinct_semantics(self) -> None:
        attack_934 = self.prototypes.attacks[934]
        attack_935 = self.prototypes.attacks[935]
        self.assertNotEqual(attack_934["attack_id"], attack_935["attack_id"])
        self.assertNotEqual(attack_934["energy_types"], attack_935["energy_types"])
        self.assertNotEqual(attack_934["base_damage"], attack_935["base_damage"])

    def test_zero_unknown_not_applicable_and_padding_are_distinct(self) -> None:
        self.assertEqual(len({int(item) for item in FieldState}), 4)
        self.assertEqual(self.prototypes.attacks[934]["base_damage"], {"state": 1, "value": 0})
        self.assertEqual(self.prototypes.cards[1198]["hp"]["state"], int(FieldState.NOT_APPLICABLE))
        self.assertNotEqual(int(FieldState.PAD), int(FieldState.NOT_APPLICABLE))

    def test_ogerpon_typed_energy_gap_distinguishes_grass_and_lightning(self) -> None:
        lightning = self.compile_synthetic(4)
        grass = self.compile_synthetic(1)
        self.assertEqual(lightning["semantic_option_cat"][0][1], 120)
        self.assertEqual(lightning["semantic_option_num"][0][4], 0.3)
        self.assertEqual(grass["semantic_option_num"][0][4], 0.2)

    def test_provenance_cannot_enter_actor_payload(self) -> None:
        record = self.compile_synthetic(4)
        record["source_id"] = 999
        record["team_name"] = "must-not-enter-forward"
        payload = actor_payload(record)
        self.assertNotIn("source_id", payload)
        self.assertNotIn("team_name", payload)
        self.assertTrue(payload["ledger_cat"])

    def test_model_forward_is_finite_and_padding_invariant(self) -> None:
        torch.manual_seed(7)
        first = first_real_record(self.prototypes)
        longer = copy.deepcopy(first)
        longer["prototype_card_refs"] = longer["prototype_card_refs"] + [95, 96, 646, 1198]
        longer["event_cat"] = longer["event_cat"] + [[1, 1, 1, 1, 1, 1, 1, 1]] * 3
        longer["event_num"] = longer["event_num"] + [[0.0, 0.0, 0.0, 0.0]] * 3
        model = SemanticFoundationPolicy(SemanticModelConfig(d_model=64, heads=4), self.prototypes).eval()
        with torch.no_grad():
            alone = model(collate([first]))[0]
            padded = model(collate([first, longer]))[0, : alone.numel()]
        self.assertTrue(torch.isfinite(alone).all())
        torch.testing.assert_close(alone, padded, atol=2e-5, rtol=2e-5)

    def test_autoregressive_contract_masks_selected_and_obeys_counts(self) -> None:
        torch.manual_seed(11)
        record = first_real_record(self.prototypes)
        batch = collate([record])
        model = SemanticFoundationPolicy(SemanticModelConfig(d_model=64, heads=4), self.prototypes).eval()
        with torch.no_grad():
            initial = model(batch)
            after = model(batch, torch.tensor([[0]]))
        floor = torch.finfo(initial.dtype).min
        self.assertEqual(initial[0, -1].item(), floor)
        self.assertEqual(after[0, 0].item(), floor)
        self.assertTrue(torch.isfinite(after[0, -1]))

    def test_no_other_numbered_project_imports(self) -> None:
        project = ROOT / "train/0025_semantic_foundation_pretraining"
        for path in project.rglob("*.py"):
            if path == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("train.0019_", text, str(path))
            self.assertNotIn("train.0022_", text, str(path))

    def test_self_contained_raw_build_entrypoints_import(self) -> None:
        catalog = importlib.import_module("train.0025_semantic_foundation_pretraining.data.replay_catalog")
        raw = importlib.import_module("train.0025_semantic_foundation_pretraining.data.raw_dataset")
        self.assertTrue(callable(catalog.scan_archives))
        self.assertTrue(callable(raw.build_trajectory_index))


if __name__ == "__main__":
    unittest.main()
