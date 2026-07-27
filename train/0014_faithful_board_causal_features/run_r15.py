"""Train R15 deterministic full-R2 with gradual option scale."""
from __future__ import annotations

from .r15_model import R15DeterministicGradualOptionPolicy, R15ModelConfig
from .training.feature_runner import run_feature_model


def main() -> None:
    run_feature_model(
        model_class=R15DeterministicGradualOptionPolicy,
        model_config=R15ModelConfig(),
        family="r15_deterministic_gradual_option",
        default_version="V25_r15_deterministic_gradual_option",
        schema_version="0014_r15_training_v1",
        model_contract={
            "r13_evidence": {
                "loss_best_official_win_rate": 0.725,
                "exact_best_official_win_rate": 0.765,
                "finding": "lower option prior did not rescue channelwise evidence corruption",
            },
            "deterministic_gradual_option": {
                "state_path": "complete R2 state calibration",
                "option_path": "complete R2 scenario and Goal-QKV evidence",
                "state_initial_scale": 1.0,
                "option_initial_scale": 0.35,
                "evidence_dropout": 0.0,
                "inference_uses_full_evidence": True,
            },
        },
    )


if __name__ == "__main__":
    main()
