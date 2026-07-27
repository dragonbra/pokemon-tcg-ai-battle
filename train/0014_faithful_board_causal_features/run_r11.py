"""Train R11 strong-scale option-query conditioning with evidence dropout."""
from __future__ import annotations

from .r11_model import R11ModelConfig, R11StrongDropoutOptionQueryPolicy
from .training.feature_runner import run_feature_model


def main() -> None:
    run_feature_model(
        model_class=R11StrongDropoutOptionQueryPolicy,
        model_config=R11ModelConfig(),
        family="r11_strong_dropout_option_query",
        default_version="V21_r11_strong_dropout_option_query",
        schema_version="0014_r11_training_v1",
        model_contract={
            "r9_evidence": {
                "loss_best_official_win_rate": 0.765,
                "exact_best_official_win_rate": 0.765,
                "finding": "strong scale without evidence dropout underperformed late R7",
            },
            "strong_dropout_option_query": {
                "scenario_layers": 1,
                "scenario_ffn_multiplier": 2,
                "route_ffn_multiplier": 1,
                "initial_scale": 1.0,
                "evidence_dropout": 0.2,
                "board_trunk_scenario_injection": False,
                "goal_qkv_independent": True,
            },
        },
    )


if __name__ == "__main__":
    main()
