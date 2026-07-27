"""Train R10 full R2 with bounded global-state calibration."""
from __future__ import annotations

from .r10_model import R10BoundedStateCalibrationPolicy, R10ModelConfig
from .training.feature_runner import run_feature_model


def main() -> None:
    run_feature_model(
        model_class=R10BoundedStateCalibrationPolicy,
        model_config=R10ModelConfig(),
        family="r10_bounded_state_calibration",
        default_version="V20_r10_bounded_state_calibration",
        schema_version="0014_r10_training_v1",
        model_contract={
            "r8_evidence": {
                "loss_best_official_win_rate": 0.76,
                "exact_best_official_win_rate": 0.755,
                "finding": "deleting the low-scale R2 state branch removed real-policy value",
            },
            "bounded_state_calibration": {
                "preserved": "complete R2 option path and state reader",
                "state_scale_range": [0.0, 0.25],
                "state_initial_scale": 0.05,
                "state_initial_scale_source": "R2 measured state scale mean 0.0327",
                "option_scale_range": [0.0, 2.0],
                "option_initial_scale": 1.0,
            },
        },
    )


if __name__ == "__main__":
    main()
