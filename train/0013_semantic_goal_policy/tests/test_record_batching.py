from __future__ import annotations

import importlib
import unittest

import torch

batching=importlib.import_module("train.0013_semantic_goal_policy.training.batching")
model_registry=importlib.import_module("train.0013_semantic_goal_policy.model.registry")


def payload(action=(1,0),termination="forced_max"):
    return {"actor_observation":{"current":{"yourIndex":0,"firstPlayer":0,"turn":3,"turnActionCount":2,"players":[{"active":[],"bench":[],"benchMax":5,"deckCount":45,"discard":[],"hand":[],"handCount":0,"prize":[]},{"active":[],"bench":[],"benchMax":5,"deckCount":50,"discard":[],"hand":[],"handCount":4,"prize":[]}],"stadium":[],"looking":[]},"select":{"type":1,"context":0,"effect":0,"minCount":1,"maxCount":2,"option":[{"type":1},{"type":2},{"type":3}]},"logs":[]},"deck_manifest":{"counts":[[1,30],[30,30]],"card_count":60,"sha256":"a"*64},"legal_options":[{"type":1},{"type":2},{"type":3}],"ordered_action":list(action),"action_termination":termination}


class RecordBatchingTest(unittest.TestCase):
    def test_flat_identity_preserves_causal_state_across_decisions(self):
        first = payload()
        first["identity"] = {
            "date": "2026-07-18", "episode_id": 123, "episode_step": 2,
            "player_index": 0, "submission_id_unavailable": True,
        }
        first["actor_observation"]["current"]["players"][0]["deckCount"] = 40
        first["actor_observation"]["current"]["players"][0]["prize"] = [None] * 20
        first["actor_observation"]["select"]["deck"] = [
            *({"id": 1} for _ in range(20)),
            *({"id": 30} for _ in range(20)),
        ]
        second = payload()
        second["identity"] = dict(first["identity"], episode_step=3)
        second["actor_observation"]["current"]["players"][0]["deckCount"] = 39
        second["actor_observation"]["current"]["players"][0]["prize"] = [None] * 20
        second["actor_observation"]["current"]["players"][0]["hand"] = [
            {"id": 1, "serial": 99}
        ]
        second["actor_observation"]["current"]["players"][0]["handCount"] = 1
        second["actor_observation"]["logs"] = [
            {"type": 4, "playerIndex": 0, "cardId": 1, "serial": 99}
        ]
        prepared = list(batching.iter_prepare_record_stream([first, second]))
        first_ledger = prepared[0]["_compiled_features"]["ledger_num"]
        second_ledger = prepared[1]["_compiled_features"]["ledger_num"]
        self.assertEqual(first_ledger[0][2], 20.0)
        self.assertEqual(first_ledger[1][2], 20.0)
        self.assertEqual(second_ledger[0][2], 19.0)
        self.assertEqual(second_ledger[1][2], 20.0)

    def test_collates_records_and_preserves_ordered_targets(self):
        batch=batching.collate_records([payload(),payload((2,),"optional_stop")])
        self.assertEqual(batch["targets"].tolist(),[[1,0],[2,3]])
        self.assertEqual(batch["target_mask"].tolist(),[[True,True],[True,True]])
        self.assertEqual(batch["deck_multiplicity"].sum(1).tolist(),[60.0,60.0])

    def test_teacher_logits_cover_options_plus_stop_and_backward(self):
        batch=batching.collate_records([payload(),payload((2,),"optional_stop")])
        model=model_registry.create_model("M5")
        logits=model.teacher_logits(batch,batch["targets"])
        self.assertEqual(logits.shape,(2,2,4))
        loss=torch.nn.functional.cross_entropy(logits[batch["target_mask"]],batch["targets"][batch["target_mask"]])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_permutation_reorders_all_option_features_targets_and_aligned_logits(self):
        row = payload()
        reference = batching.collate_records([row])
        permuted = batching.collate_records([row], permutations=[(2, 0, 1)])
        self.assertEqual(permuted["option_permutation_old_to_new"].tolist(), [[1, 2, 0]])
        self.assertEqual(permuted["targets"].tolist(), [[2, 1]])
        self.assertTrue(torch.equal(
            reference["options_cat"][:, (2, 0, 1)],
            permuted["options_cat"],
        ))
        self.assertTrue(torch.equal(
            reference["option_semantic"][:, (2, 0, 1)],
            permuted["option_semantic"],
        ))
        model = model_registry.create_model("M5").eval()
        with torch.no_grad():
            reference_logits = model.teacher_logits(reference, reference["targets"])
            permuted_logits = model.teacher_logits(permuted, permuted["targets"])
        old_to_new = permuted["option_permutation_old_to_new"][0]
        aligned = torch.cat((permuted_logits[:, :, old_to_new], permuted_logits[:, :, -1:]), dim=-1)
        self.assertTrue(torch.allclose(reference_logits, aligned, atol=1e-5, rtol=1e-5))


if __name__=="__main__":unittest.main()
