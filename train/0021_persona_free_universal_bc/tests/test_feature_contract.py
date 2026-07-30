from __future__ import annotations

import copy
from importlib import import_module
import unittest

import torch


BASE_MODEL = import_module("train.0021_persona_free_universal_bc.base_model")
COMPILER = import_module("train.0021_persona_free_universal_bc.features.compiler")
IDOnlyCodec = BASE_MODEL.IDOnlyCodec
IDOnlyConfig = BASE_MODEL.IDOnlyConfig
IDOnlyPointerPolicy = BASE_MODEL.IDOnlyPointerPolicy
collate_id_only = BASE_MODEL.collate_id_only


def _observation(*, actor: int, first_player: int) -> dict:
    return {
        "current": {
            "yourIndex": actor,
            "firstPlayer": first_player,
            "players": [
                {"deckCount": 40, "handCount": 7, "bench": []},
                {"deckCount": 41, "handCount": 6, "bench": []},
            ],
        },
        "select": {
            "type": 1,
            "context": 2,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": 3, "playerIndex": actor, "area": 2, "cardId": 101},
                {"type": 7, "playerIndex": actor, "area": 2, "cardId": 202},
                {"type": 5, "playerIndex": actor, "area": 2, "cardId": 303},
            ],
        },
    }


class FeatureContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = IDOnlyConfig(
            d_model=32,
            heads=4,
            encoder_layers=1,
            ffn_multiplier=2,
            dropout=0.0,
        )
        self.codec = IDOnlyCodec(self.config)

    def test_first_player_feature_is_actor_relative(self) -> None:
        self_first_0 = self.codec.encode(_observation(actor=0, first_player=0), None)
        self_first_1 = self.codec.encode(_observation(actor=1, first_player=1), None)
        opp_first_0 = self.codec.encode(_observation(actor=1, first_player=0), None)
        opp_first_1 = self.codec.encode(_observation(actor=0, first_player=1), None)
        assert self_first_0 and self_first_1 and opp_first_0 and opp_first_1
        self.assertEqual(self_first_0["global_cat"][2], 1)
        self.assertEqual(self_first_1["global_cat"][2], 1)
        self.assertEqual(opp_first_0["global_cat"][2], 2)
        self.assertEqual(opp_first_1["global_cat"][2], 2)

    def test_zone_rows_are_ordered_self_then_opponent(self) -> None:
        players = ({"marker": 0}, {"marker": 1})
        actor_zero = COMPILER._actor_relative_players(players, 0)
        actor_one = COMPILER._actor_relative_players(players, 1)
        self.assertEqual([index for index, _ in actor_zero], [0, 1])
        self.assertEqual([index for index, _ in actor_one], [1, 0])
        self.assertEqual([player["marker"] for _, player in actor_one], [1, 0])

    def test_option_features_exclude_list_position(self) -> None:
        encoded = self.codec.encode(_observation(actor=0, first_player=0), None)
        assert encoded is not None
        self.assertTrue(encoded["option_cat"])
        self.assertTrue(all(len(row) == 11 for row in encoded["option_cat"]))
        self.assertFalse(hasattr(IDOnlyPointerPolicy(self.config), "option_position"))

    def test_option_encoding_is_permutation_equivariant(self) -> None:
        original_observation = _observation(actor=0, first_player=0)
        permuted_observation = copy.deepcopy(original_observation)
        permutation = torch.tensor([2, 0, 1])
        options = permuted_observation["select"]["option"]
        permuted_observation["select"]["option"] = [options[index] for index in permutation]
        original = self.codec.encode(original_observation, None)
        permuted = self.codec.encode(permuted_observation, None)
        assert original is not None and permuted is not None
        model = IDOnlyPointerPolicy(self.config).eval()
        original_batch = collate_id_only([original])
        permuted_batch = collate_id_only([permuted])
        with torch.inference_mode():
            original_options = model.encode(original_batch)[1][0]
            permuted_options = model.encode(permuted_batch)[1][0]
            original_action = model.greedy_action(original_batch)
            permuted_action = model.greedy_action(permuted_batch)
        inverse = torch.argsort(permutation)
        torch.testing.assert_close(original_options, permuted_options[inverse])
        self.assertEqual(len(original_action), 1)
        self.assertEqual(len(permuted_action), 1)
        self.assertEqual(original_action[0], int(permutation[permuted_action[0]]))


if __name__ == "__main__":
    unittest.main()
