from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest

import torch


PROJECT = "train.0038_action_boundary_rl"
ROOT = Path(__file__).resolve().parents[3]
initialization = importlib.import_module(f"{PROJECT}.initialization")
policy = importlib.import_module(f"{PROJECT}.policy")
compiler_module = importlib.import_module(f"{PROJECT}.rollout.worker_compiler")
collate = importlib.import_module(f"{PROJECT}.semantic_policy.features.collate")
batching = importlib.import_module(f"{PROJECT}.policy.batching")


class InitializationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        replay = json.loads((
            ROOT / "data/replays/0016_alakazam_multideck_bc/goonew/submission-54960905/episode-88310066-replay.json"
        ).read_text(encoding="utf-8"))
        cls.deck = tuple(replay["steps"][0][0]["visualize"][0]["action"][0])
        observation = next(
            agent["observation"]
            for step in replay["steps"] for agent in step
            if isinstance((agent.get("observation") or {}).get("current"), dict)
            and (agent["observation"]["current"].get("yourIndex") == 0)
            and (agent["observation"].get("select") or {}).get("option")
        )
        compiler = compiler_module.WorkerLocalCompiler(0, cls.deck)
        cls.features = batching.move_batch(
            collate.collate_canonical_records([compiler.compile(observation)]),
            torch.device("cpu"),
        )

    def test_rejects_rl_updated_path_and_core_missing_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "RL-updated"):
            initialization.assert_no_rl_checkpoint_source(
                ROOT / "rl_runs/0037_dragapult_value_initialized_rl/versions/V5_last_option_qv_lora_r4_eval5_50u/checkpoint/update-000032.pt",
                "unused",
            )
        initialization.assert_only_new_missing_keys(["allocation_head.score.0.weight"])
        with self.assertRaisesRegex(ValueError, "core keys"):
            initialization.assert_only_new_missing_keys(["actor.action_decoder.query.weight"])

    def test_zero_delta_lora_preserves_policy_value_and_mask(self) -> None:
        torch.manual_seed(11)
        reference, _ = policy.load_actor_critic(
            initialization.ACTOR_CHECKPOINT, self.deck, "cpu",
            adaptation=policy.AdaptationConfig(lora=False),
        )
        adapted, _ = initialization.build_update0_model(self.deck, device="cpu")
        with torch.inference_mode():
            ref_batch, ref_state, ref_options, ref_value = reference.encode(self.features)
            new_batch, new_state, new_options, new_value = adapted.encode(self.features)
            ref_decoder = reference.actor.action_decoder
            new_decoder = adapted.actor.action_decoder
            ref_logits = ref_decoder.logits(
                ref_batch, ref_options, ref_decoder.initialize(ref_batch, ref_state.summary)
            )
            new_logits = new_decoder.logits(
                new_batch, new_options, new_decoder.initialize(new_batch, new_state.summary)
            )
            ref_action = ref_decoder.greedy(ref_batch, ref_options, ref_state.summary)
            new_action = new_decoder.greedy(new_batch, new_options, new_state.summary)
        torch.testing.assert_close(new_logits, ref_logits, rtol=0, atol=0)
        torch.testing.assert_close(new_value, ref_value, rtol=0, atol=0)
        self.assertTrue(torch.equal(new_batch.option_mask, ref_batch.option_mask))
        self.assertTrue(torch.equal(new_action.sequences, ref_action.sequences))
        self.assertTrue(torch.equal(new_action.lengths, ref_action.lengths))

    def test_manifest_declares_fresh_training_state(self) -> None:
        manifest = initialization.initialization_manifest()
        self.assertIn("no V5", manifest["lora_initialization"])
        self.assertIn("fresh AdamW", manifest["optimizer_initialization"])
        self.assertIn("empty on-policy", manifest["rollout_initialization"])

    def test_auxiliary_presets_preserve_core_rng_and_zero_shot_logits(self) -> None:
        presets = importlib.import_module(f"{PROJECT}.integrated.presets")
        base, _ = initialization.build_update0_model(
            self.deck, integrated_flags=presets.preset("BASE")
        )
        integrated, _ = initialization.build_update0_model(
            self.deck, integrated_flags=presets.preset("INTEGRATED")
        )
        base_lora = {n: p for n, p in base.actor.named_parameters() if n.endswith(("q_a", "v_a"))}
        other_lora = {n: p for n, p in integrated.actor.named_parameters() if n.endswith(("q_a", "v_a"))}
        self.assertEqual(base_lora.keys(), other_lora.keys())
        for name in base_lora:
            torch.testing.assert_close(base_lora[name], other_lora[name], rtol=0, atol=0)
        with torch.inference_mode():
            base_batch, base_state, base_options, _ = base.encode(self.features)
            other_batch, other_state, other_options, _ = integrated.encode(self.features)
            base_logits = base.actor.action_decoder.logits(
                base_batch, base_options,
                base.actor.action_decoder.initialize(base_batch, base.actor_summary(base_state)),
            )
            other_logits = integrated.actor.action_decoder.logits(
                other_batch, other_options,
                integrated.actor.action_decoder.initialize(
                    other_batch, integrated.actor_summary(other_state)
                ),
            )
        torch.testing.assert_close(base_logits, other_logits, rtol=0, atol=0)

    def test_integrated_preset_loads_common_non_ppo_update0_strictly(self) -> None:
        presets = importlib.import_module(f"{PROJECT}.integrated.presets")
        model, _ = initialization.build_preset_from_common_update0(
            self.deck, presets.preset("INTEGRATED"), device="cpu"
        )
        self.assertIsNotNone(model.prize_aux)
        self.assertIsNotNone(model.opponent_meta_head)
        self.assertTrue(all(
            torch.count_nonzero(value).item() == 0
            for name, value in model.actor.named_parameters()
            if name.endswith(("q_b", "v_b"))
        ))


if __name__ == "__main__":
    unittest.main()
