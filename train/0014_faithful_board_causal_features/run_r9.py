"""Train R9 narrow strong option-query conditioning."""
from __future__ import annotations

from .r9_model import R9ModelConfig, R9NarrowStrongOptionQueryPolicy
from .training.feature_runner import run_feature_model


def main() -> None:
    run_feature_model(
        model_class=R9NarrowStrongOptionQueryPolicy,
        model_config=R9ModelConfig(),
        family="r9_narrow_strong_option_query",
        default_version="V19_r9_narrow_strong_option_query",
        schema_version="0014_r9_training_v1",
        model_contract={
            "r7_evidence": {
                "loss_best_official_win_rate": 0.72,
                "exact_best_official_win_rate": 0.79,
                "finding": "low scale plus evidence dropout delayed real-policy competence",
            },
            "narrow_strong_option_query": {
                "scenario_layers": 1,
                "scenario_ffn_multiplier": 2,
                "route_ffn_multiplier": 1,
                "initial_scale": 1.0,
                "evidence_dropout": 0.0,
                "board_trunk_scenario_injection": False,
                "goal_qkv_independent": True,
            },
        },
    )


if __name__ == "__main__":
    main()
