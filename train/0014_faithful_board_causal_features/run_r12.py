"""Train R12 full R2 state calibration with option-only evidence dropout."""
from __future__ import annotations

from .r12_model import R12ModelConfig, R12StatePreservingDropoutPolicy
from .training.feature_runner import run_feature_model


def main() -> None:
    run_feature_model(
        model_class=R12StatePreservingDropoutPolicy,
        model_config=R12ModelConfig(),
        family="r12_state_preserving_option_dropout",
        default_version="V22_r12_state_preserving_option_dropout",
        schema_version="0014_r12_training_v1",
        model_contract={
            "r10_evidence": {
                "loss_best_official_win_rate": 0.75,
                "exact_best_official_win_rate": 0.775,
                "finding": "bounding R2 state scale reduced policy strength",
            },
            "state_preserving_option_dropout": {
                "state_path": "complete unbounded R2 state calibration",
                "state_scale_range": [0.0, 2.0],
                "state_initial_scale": 1.0,
                "option_path": "complete R2 option reader and Goal-QKV",
                "option_evidence_dropout": 0.2,
                "inference_uses_full_evidence": True,
            },
        },
    )


if __name__ == "__main__":
    main()
