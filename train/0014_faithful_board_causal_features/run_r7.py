"""Train R7 regularized option-query memory derived from R5 evidence."""
from __future__ import annotations

from .r7_model import R7ModelConfig, R7RegularizedOptionQueryPolicy
from .training.feature_runner import run_feature_model


def main() -> None:
    run_feature_model(
        model_class=R7RegularizedOptionQueryPolicy,
        model_config=R7ModelConfig(),
        family="r7_regularized_option_query",
        default_version="V17_r7_regularized_option_query",
        schema_version="0014_r7_training_v1",
        model_contract={
            "r5_evidence": {
                "loss_best_official_win_rate": 0.78,
                "exact_best_official_win_rate": 0.725,
                "finding": "query-only boundary is viable but strong late residual overfits",
            },
            "regularized_option_query": {
                "initial_scale": 0.35,
                "initial_scale_source": "R2 observed legal-option scale mean 0.347",
                "route_ffn_multiplier": 1,
                "evidence_dropout": 0.2,
                "board_trunk_scenario_injection": False,
                "goal_qkv_independent": True,
            },
        },
    )


if __name__ == "__main__":
    main()
