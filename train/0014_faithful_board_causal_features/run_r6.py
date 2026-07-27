"""Train R6 sparse family mixture derived from R4 evidence."""
from __future__ import annotations

from .r6_model import R6ModelConfig, R6SparseFamilyMixturePolicy
from .training.feature_runner import run_feature_model


def main() -> None:
    run_feature_model(
        model_class=R6SparseFamilyMixturePolicy,
        model_config=R6ModelConfig(),
        family="r6_sparse_family_mixture",
        default_version="V16_r6_sparse_family_mixture",
        schema_version="0014_r6_training_v1",
        model_contract={
            "r4_evidence": {
                "loss_best_official_win_rate": 0.695,
                "exact_best_official_win_rate": 0.755,
                "finding": "unconditional six-family residual sum caused interference",
            },
            "sparse_family_mixture": {
                "families": list(R6SparseFamilyMixturePolicy.family_names),
                "null_route": True,
                "initial_priors": {
                    "null": 0.50,
                    "ledger": 0.20,
                    "goal": 0.15,
                    "zone": 0.075,
                    "event": 0.0375,
                    "hand": 0.0375,
                },
                "card_capability_path": "local entity/option identity adapter",
                "formula": "LN(option + 2*sigmoid(amplitude)*softmax_mixture(experts))",
            },
        },
    )


if __name__ == "__main__":
    main()
