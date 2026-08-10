from __future__ import annotations

import importlib
import unittest


PROJECT = "train.0042_full_model_design"


class RemovedOptionAdaptationTest(unittest.TestCase):
    def test_lora_and_option_layernorm_requests_fail_closed(self) -> None:
        policy = importlib.import_module(f"{PROJECT}.policy")
        runner = importlib.import_module(f"{PROJECT}.training.run_full_semantic")
        for config in (
            policy.AdaptationConfig(lora=True),
            policy.AdaptationConfig(layernorm_tuning=True),
        ):
            with self.assertRaisesRegex(ValueError, "forbids Option"):
                policy.load_actor_critic(
                    runner.SOURCE_CHECKPOINT,
                    runner.focal_deck(),
                    "cpu",
                    adaptation=config,
                )

    def test_canonical_model_has_no_option_parametrization(self) -> None:
        policy = importlib.import_module(f"{PROJECT}.policy")
        runner = importlib.import_module(f"{PROJECT}.training.run_full_semantic")
        model, _ = policy.load_actor_critic(
            runner.SOURCE_CHECKPOINT, runner.focal_deck(), "cpu"
        )
        model.assert_no_option_adaptation()
        self.assertFalse(any(
            "lora" in name.lower() or ".parametrizations." in name
            for name, _ in model.named_parameters()
        ))


if __name__ == "__main__":
    unittest.main()
