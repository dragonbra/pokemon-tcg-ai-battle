"""Train R14 structured scenario-only modality-dropout policy."""
from __future__ import annotations

from .r14_model import R14ModelConfig, R14ScenarioModalityDropoutPolicy
from .training.feature_runner import run_feature_model


def main() -> None:
    run_feature_model(
        model_class=R14ScenarioModalityDropoutPolicy,
        model_config=R14ModelConfig(),
        family="r14_scenario_modality_dropout",
        default_version="V24_r14_scenario_modality_dropout",
        schema_version="0014_r14_training_v1",
        model_contract={
            "r12_evidence": {
                "loss_best_official_win_rate": 0.77,
                "exact_best_official_win_rate": 0.75,
                "finding": "elementwise dropout across scenario and Goal damaged policy strength",
            },
            "scenario_modality_dropout": {
                "state_path": "complete deterministic R2 state calibration",
                "goal_qkv": "complete deterministic R2 Goal evidence",
                "option_initial_scale": 1.0,
                "scenario_dropout_probability": 0.2,
                "mask_shape": "one shared route mask per sample",
                "inference_uses_full_evidence": True,
            },
        },
    )


if __name__ == "__main__":
    main()
