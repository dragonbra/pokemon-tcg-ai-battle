from __future__ import annotations

import importlib
import unittest

import torch

batching=importlib.import_module("train.0013_semantic_goal_policy.training.batching")
model_registry=importlib.import_module("train.0013_semantic_goal_policy.model.registry")


def payload(action=(1,0),termination="forced_max"):
    return {"actor_observation":{"current":{"yourIndex":0,"firstPlayer":0,"turn":3,"turnActionCount":2,"players":[{"active":[],"bench":[],"benchMax":5,"deckCount":45,"discard":[],"hand":[],"handCount":0,"prize":[]},{"active":[],"bench":[],"benchMax":5,"deckCount":50,"discard":[],"hand":[],"handCount":4,"prize":[]}],"stadium":[],"looking":[]},"select":{"type":1,"context":0,"effect":0,"minCount":1,"maxCount":2,"option":[{"type":1},{"type":2},{"type":3}]},"logs":[]},"deck_manifest":{"counts":[[1,30],[30,30]],"card_count":60,"sha256":"a"*64},"legal_options":[{"type":1},{"type":2},{"type":3}],"ordered_action":list(action),"action_termination":termination}


class RecordBatchingTest(unittest.TestCase):
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


if __name__=="__main__":unittest.main()
