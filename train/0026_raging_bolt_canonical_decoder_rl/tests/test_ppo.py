from __future__ import annotations

import importlib
from pathlib import Path
import unittest

import torch


BASE = "train.0026_raging_bolt_canonical_decoder_rl"
HELPERS = importlib.import_module(f"{BASE}.tests.helpers")


class PPOTests(unittest.TestCase):
    def test_one_update_changes_only_decoder_and_value(self) -> None:
        policy = importlib.import_module(f"{BASE}.policy")
        distribution = importlib.import_module(f"{BASE}.policy.action_distribution")
        online = importlib.import_module(f"{BASE}.focal.deployment.canonical_online_runtime")
        league = importlib.import_module(f"{BASE}.league")
        protocol = importlib.import_module(f"{BASE}.rollout.protocol")
        training = importlib.import_module(f"{BASE}.training")
        model, _ = policy.load_actor_critic("cpu")
        focal = next(
            item for item in league.load_frozen_catalog()
            if item.deck_id == "raging_bolt_ex_james_cox_henry_chao_001"
        )
        opponent = next(item for item in league.load_frozen_catalog() if item.deck_id != focal.deck_id)
        encoder = online.OnlineCausalEncoder(0, focal.deck, model.actor.config)
        features = encoder.encode(HELPERS.observation())
        action = distribution.greedy_actions(model, features)[0]
        episodes = []
        for index, reward in enumerate((1.0, -1.0)):
            job = protocol.RolloutJob(
                game_id=f"test-{index}",
                opponent_id=opponent.deck_id,
                focal_first=True,
                seed=index,
                source_policy_update=0,
                focal_deck=focal.deck,
                opponent_deck=opponent.deck,
                runtime_root=Path("evaluation/arena/opponents"),
            )
            decision = protocol.TrajectoryDecision(
                features,
                action.indices,
                action.stopped,
                action.log_prob,
                action.entropy,
                action.value,
                0,
            )
            episode = protocol.EpisodeTrajectory(job, decisions=[decision])
            episode.finish(reward, 3)
            episodes.append(episode)
        batch = training.prepare_episodes(episodes)
        before_representation = model.representation_sha256()
        before_decoder = model.decoder_sha256()
        trainer = training.PPOTrainer(
            model,
            device=torch.device("cpu"),
            config=training.PPOConfig(epochs=1, batch_size=2),
        )
        metrics = trainer.update(batch)
        self.assertTrue(torch.isfinite(torch.tensor(list(metrics.values()))).all())
        self.assertEqual(model.representation_sha256(), before_representation)
        self.assertNotEqual(model.decoder_sha256(), before_decoder)
        self.assertEqual(batch.source_policy_update, 0)
        self.assertEqual(metrics["ppo/decisions"], 2.0)


if __name__ == "__main__":
    unittest.main()
