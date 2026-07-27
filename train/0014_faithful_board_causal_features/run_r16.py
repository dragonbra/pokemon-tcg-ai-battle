"""Train R16 deterministic R2 with typed rule-contract option tokens."""
from __future__ import annotations

from .r16_model import R16ModelConfig, R16RuleContractPolicy
from .training.feature_runner import run_feature_model


def main() -> None:
    run_feature_model(
        model_class=R16RuleContractPolicy,
        model_config=R16ModelConfig(),
        family="r16_rule_contract_option_reader",
        default_version="V26_r16_rule_contract_option_reader",
        schema_version="0014_r16_training_v1",
        model_contract={
            "r14_evidence": {
                "loss_best_official_win_rate": 0.735,
                "exact_best_official_win_rate": 0.74,
                "finding": "structured scenario-route dropout still harmed exact planning evidence",
            },
            "rule_contract_reader": {
                "preserved": "complete deterministic R2 state, scenario, and Goal-QKV paths",
                "tokens": [
                    "turn_budget",
                    "prize_race",
                    "library_pressure",
                    "board_relay",
                ],
                "visibility": "actor-visible observation only",
                "option_scale_range": [0.0, 2.0],
                "option_initial_scale": 1.0,
                "evidence_dropout": 0.0,
            },
        },
    )


if __name__ == "__main__":
    main()
