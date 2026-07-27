"""Train R4 option-centric feature-family routing derived from R2."""
from __future__ import annotations

from .r4_model import R4ModelConfig, R4OptionFamilyRouterPolicy
from .training.feature_runner import run_feature_model


def main() -> None:
    run_feature_model(
        model_class=R4OptionFamilyRouterPolicy,
        model_config=R4ModelConfig(),
        family="r4_option_family_router",
        default_version="V14_r4_option_family_router",
        schema_version="0014_r4_training_v1",
        model_contract={
            "r2_evidence": {
                "state_scale_mean": 0.032662536948919296,
                "state_scale_median": 0.00091552734375,
                "legal_option_scale_mean": 0.3474218547344208,
                "legal_option_scale_median": 0.26171875,
            },
            "option_family_routes": {
                "families": list(R4OptionFamilyRouterPolicy.family_names),
                "formula": "2 * sigmoid(family_mlp(evidence)) * family_projection(evidence)",
                "global_scenario_state_gate": False,
                "initial_scales": {
                    "ledger": 1.0,
                    "goal": 1.0,
                    "card": 1.0,
                    "zone": 0.5,
                    "event": 0.25,
                    "hand": 0.25,
                },
            },
        },
    )


if __name__ == "__main__":
    main()
