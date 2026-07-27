"""Train R13 strong-state, gradual-option evidence-dropout policy."""
from __future__ import annotations

from .r13_model import R13GradualOptionScalePolicy, R13ModelConfig
from .training.feature_runner import run_feature_model


def main() -> None:
    run_feature_model(
        model_class=R13GradualOptionScalePolicy,
        model_config=R13ModelConfig(),
        family="r13_gradual_option_scale",
        default_version="V23_r13_gradual_option_scale",
        schema_version="0014_r13_training_v1",
        model_contract={
            "r11_evidence": {
                "loss_best_official_win_rate": 0.715,
                "exact_best_official_win_rate": 0.755,
                "finding": "dropout with a unit option prior remained too aggressive",
            },
            "gradual_option_scale": {
                "state_path": "complete R2 state calibration",
                "state_initial_scale": 1.0,
                "option_path": "complete R2 option reader and Goal-QKV",
                "option_initial_scale": 0.35,
                "option_evidence_dropout": 0.2,
                "inference_uses_full_evidence": True,
            },
        },
    )


if __name__ == "__main__":
    main()
