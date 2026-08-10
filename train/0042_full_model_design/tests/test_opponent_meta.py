from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import unittest

import torch


PROJECT = "train.0042_full_model_design"
initialization = importlib.import_module(f"{PROJECT}.initialization")
presets = importlib.import_module(f"{PROJECT}.integrated.presets")
compiler_module = importlib.import_module(f"{PROJECT}.rollout.worker_compiler")
collate = importlib.import_module(f"{PROJECT}.semantic_policy.features.collate")


class OpponentMetaInformationSafetyTest(unittest.TestCase):
    def test_actor_value_and_meta_ignore_hidden_identity_counterfactual(self) -> None:
        root = Path(__file__).resolve().parents[3]
        replay = json.loads((
            root / "data/replays/0016_alakazam_multideck_bc/goonew/"
            "submission-54960905/episode-88310066-replay.json"
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
            deck, presets.preset("FULL_MODEL"), device="cpu"
        )
        with torch.inference_mode():
            outputs = []
            for features in (features_a, features_b):
                batch, state, options, value, auxiliary, context = (
                    model.encode_with_strategy(features)
                )
                decoder_state = model.actor.action_decoder.initialize(batch, state.summary)
                logits = model.head.logits(batch, options, decoder_state, context)
                outputs.append((value, auxiliary["meta_logits"], logits))
        for left, right in zip(*outputs, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_meta_target_is_not_an_actor_forward_argument(self) -> None:
        model, _ = initialization.build_preset_from_common_update0(
            tuple(map(int, (
                Path(__file__).resolve().parents[1] / "league/decks/007_dragapult_ex/deck.csv"
            ).read_text().splitlines())),
            presets.preset("FULL_MODEL"),
            device="cpu",
        )
        self.assertNotIn("target", model.actor.expected_batch_keys)
        self.assertNotIn("archetype", model.actor.expected_batch_keys)


if __name__ == "__main__":
    unittest.main()
