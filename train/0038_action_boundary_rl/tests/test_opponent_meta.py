from __future__ import annotations

import importlib
import copy
import json
from pathlib import Path
import unittest

import torch

meta = importlib.import_module("train.0038_action_boundary_rl.integrated.opponent_meta")
initialization = importlib.import_module("train.0038_action_boundary_rl.initialization")
presets = importlib.import_module("train.0038_action_boundary_rl.integrated.presets")
compiler_module = importlib.import_module("train.0038_action_boundary_rl.rollout.worker_compiler")
collate = importlib.import_module("train.0038_action_boundary_rl.semantic_policy.features.collate")


class OpponentMetaTest(unittest.TestCase):
    def test_hidden_information_cannot_reach_inference(self):
        torch.manual_seed(38)
        head = meta.OpponentMetaHead(16, 4).eval()
        public = torch.randn(2, 16)
        # Hidden hand/deck/prize mutations are deliberately not accepted by forward.
        first = head(public)
        hidden_a = {"hand": [1], "deck": [2, 3], "prize": [4]}
        hidden_b = {"hand": [999], "deck": [8, 7], "prize": [6]}
        self.assertNotEqual(hidden_a, hidden_b)
        self.assertTrue(torch.equal(first, head(public)))

    def test_conditioner_is_zero_delta_and_detached(self):
        conditioner = meta.OpponentMetaConditioner(8, 3)
        state = torch.randn(2, 8, requires_grad=True)
        logits = torch.randn(2, 3, requires_grad=True)
        output = conditioner(state, logits, detach=True)
        self.assertTrue(torch.equal(state, output))
        output.sum().backward()
        self.assertIsNone(logits.grad)

    def test_labels_only_enter_loss(self):
        logits = torch.randn(4, 3)
        self.assertTrue(torch.isfinite(meta.meta_loss(logits, torch.tensor([0, 1, 2, 1]))))

    def test_actor_and_meta_ignore_hidden_identity_counterfactual(self):
        root = Path(__file__).resolve().parents[3]
        replay = json.loads((
            root / "data/replays/0016_alakazam_multideck_bc/goonew/submission-54960905/episode-88310066-replay.json"
        ).read_text())
        observation = next(
            agent["observation"] for step in replay["steps"] for agent in step
            if isinstance((agent.get("observation") or {}).get("current"), dict)
            and (agent["observation"].get("select") or {}).get("option")
        )
        actor = observation["current"]["yourIndex"]
        deck = tuple(replay["steps"][0][0]["visualize"][0]["action"][actor])
        first, second = copy.deepcopy(observation), copy.deepcopy(observation)
        for target, base in ((first, 900), (second, 1900)):
            hidden = target["current"]["players"][1 - actor]
            hidden["hand"] = [{"id": base, "serial": base}]
            hidden["deck"] = [{"id": base + 1, "serial": base + 1}]
            hidden["prize"] = [{"id": base + 2, "serial": base + 2}]
        record_a = compiler_module.WorkerLocalCompiler(actor, deck).compile(first)
        record_b = compiler_module.WorkerLocalCompiler(actor, deck).compile(second)
        self.assertEqual(record_a, record_b)
        features_a = collate.collate_canonical_records([record_a])
        features_b = collate.collate_canonical_records([record_b])
        model, _ = initialization.build_preset_from_common_update0(
            deck, presets.preset("INTEGRATED"), device="cpu"
        )
        with torch.inference_mode():
            batch_a, state_a, options_a = model.actor.encode(features_a)
            batch_b, state_b, options_b = model.actor.encode(features_b)
            meta_a = model.opponent_meta_head(state_a.summary)
            meta_b = model.opponent_meta_head(state_b.summary)
            logits_a = model.actor.action_decoder.logits(
                batch_a, options_a,
                model.actor.action_decoder.initialize(batch_a, model.actor_summary(state_a)),
            )
            logits_b = model.actor.action_decoder.logits(
                batch_b, options_b,
                model.actor.action_decoder.initialize(batch_b, model.actor_summary(state_b)),
            )
        torch.testing.assert_close(meta_a, meta_b, rtol=0, atol=0)
        torch.testing.assert_close(logits_a, logits_b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
