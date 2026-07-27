"""Train the independent R3 typed early-scenario-fusion hypothesis."""
from __future__ import annotations

from .r3_model import R3EarlyScenarioFusionPolicy, R3ModelConfig
from .training.feature_runner import run_feature_model


def main() -> None:
    run_feature_model(
        model_class=R3EarlyScenarioFusionPolicy,
        model_config=R3ModelConfig(),
        family="r3_early_scenario_fusion",
        default_version="V13_r3_early_scenario_fusion",
        schema_version="0014_r3_training_v1",
        model_contract={
            "scenario_families": [
                "public_zone_inventory",
                "registered_deck_causal_ledger",
                "causal_event_memory",
                "known_unknown_opponent_hand",
            ],
            "early_fusion": {
                "typed_scenario_tokens_inside_board_transformer": True,
                "entity_and_cls_attention_from_first_board_layer": True,
                "option_attention_reads_board_and_scenario": True,
                "card_capability_gate": "always_open",
            },
            "goal_qkv": {
                "resource_memory": "contextualized_registered_card_tokens",
                "option_goal_residual": "always_open",
            },
        },
    )


if __name__ == "__main__":
    main()
