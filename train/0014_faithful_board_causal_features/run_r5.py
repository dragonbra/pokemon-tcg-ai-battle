"""Train R5 option-query scenario memory derived from R3 evidence."""
from __future__ import annotations

from .r5_model import R5ModelConfig, R5OptionQueryScenarioPolicy
from .training.feature_runner import run_feature_model


def main() -> None:
    run_feature_model(
        model_class=R5OptionQueryScenarioPolicy,
        model_config=R5ModelConfig(),
        family="r5_option_query_scenario",
        default_version="V15_r5_option_query_scenario",
        schema_version="0014_r5_training_v1",
        model_contract={
            "r3_evidence": {
                "loss_best_official_win_rate": 0.775,
                "exact_best_official_win_rate": 0.785,
                "r2_exact_best_official_win_rate": 0.795,
                "finding": "early scenario fusion overfit and trailed option-centric R2",
            },
            "option_query_scenario": {
                "board_trunk_scenario_injection": False,
                "typed_memory": ["zone", "ledger", "event", "hand"],
                "goal_qkv_independent": True,
                "formula": "LN(option + 2*sigmoid(gate(option,read))*delta(option,read,goal))",
                "initial_scale": 1.0,
            },
        },
    )


if __name__ == "__main__":
    main()
