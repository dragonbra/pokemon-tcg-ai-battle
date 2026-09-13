from __future__ import annotations

import torch

from pokemon_tcg_ai.policy.actor_critic import (
    DEFAULT_0814_ACTOR_CHECKPOINT,
    DEFAULT_0814_VALUE_CHECKPOINT,
)
from pokemon_tcg_ai.training import train
from pokemon_tcg_ai.training.ppo_full_semantic import PPOConfig, PPOTrainer


def test_public_trainer_starts_from_committed_bc_assets() -> None:
    assert DEFAULT_0814_ACTOR_CHECKPOINT.is_file()
    assert DEFAULT_0814_VALUE_CHECKPOINT.is_file()
    assert train.VERSION == "dragapult_policy0814_bc_ppo"
    assert not hasattr(train, "PARENT_CHECKPOINT")
    assert "runs/versions/V" not in DEFAULT_0814_ACTOR_CHECKPOINT.as_posix()
    assert "runs/versions/V" not in DEFAULT_0814_VALUE_CHECKPOINT.as_posix()


def test_public_model_has_the_final_expanded_lora_boundary() -> None:
    model, _, audit = train._model_and_audit()
    assert audit["tensor_count"] == 125
    assert audit["total_trainable_params"] == 5_597_590
    assert audit["module_totals"]["OptionEncoder.final_ffn_lora"] == 40_960
    assert audit["module_totals"]["StateEncoder.board_transformer.final_ffn_lora"] == 40_960
    assert audit["module_totals"]["OptionEncoder.final_layer_norm"] == 640

    trainer = PPOTrainer(
        model,
        device=torch.device("cpu"),
        config=PPOConfig(
            **train.LEARNING_RATE_PROFILE.rates(),
            batch_size=4096,
            forward_microbatch_size=256,
            behavior_probe_batch_size=256,
            offload_reference_after_cache=True,
        ),
    )
    groups = {row["name"]: row for row in trainer.optimizer_group_manifest()}
    assert groups["policy_option_ffn_lora"]["base_learning_rate"] == 3.0e-5
    assert groups["shared_state_ffn_lora"]["base_learning_rate"] == 1.5e-5
    assert groups["option_final_norm"]["base_learning_rate"] == 5.0e-6
