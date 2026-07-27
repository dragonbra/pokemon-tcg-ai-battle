"""Train R8 faithful R2 option-only conditioning."""
from __future__ import annotations

from .r8_model import R8ModelConfig, R8R2OptionOnlyPolicy
from .training.feature_runner import run_feature_model


def main() -> None:
    run_feature_model(
        model_class=R8R2OptionOnlyPolicy,
        model_config=R8ModelConfig(),
        family="r8_r2_option_only",
        default_version="V18_r8_r2_option_only",
        schema_version="0014_r8_training_v1",
        model_contract={
            "r6_evidence": {
                "loss_best_official_win_rate": 0.72,
                "exact_best_official_win_rate": 0.74,
                "finding": "family routing remained a real-policy shortcut",
            },
            "r2_option_only": {
                "preserved": [
                    "two-layer typed scenario bank",
                    "option scenario attention",
                    "independent Goal-QKV",
                    "family summary FiLM",
                    "dynamic option ScaleGate initialized at 1.0",
                ],
                "removed": "R2 global state attention/FiLM/ScaleGate path",
                "removal_evidence": "R2 state scale mean 0.0327 and median 0.00092",
            },
        },
    )


if __name__ == "__main__":
    main()
