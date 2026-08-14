from __future__ import annotations

from dataclasses import asdict
import importlib

import pytest


profiles = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.training.lr_profiles"
)
common = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.training.run_v1"
)


EXPECTED_STANDARD = {
    "decoder_learning_rate": 5e-6,
    "policy_adapter_learning_rate": 5e-6,
    "allocation_learning_rate": 5e-6,
    "option_lora_learning_rate": 1e-5,
    "meta_actor_residual_learning_rate": 5e-6,
    "value_learning_rate": 2e-5,
    "prize_learning_rate": 2e-5,
}


def test_standard_lr_profile_is_the_exact_project_default() -> None:
    profile = profiles.STANDARD_LR_PROFILE
    assert profile.profile_id == "0044_standard_lr_v1"
    assert profile.base_profile_id == "0044_standard_lr_v1"
    assert profile.scale == 1.0
    assert profile.scope == "project_default"
    assert profile.default_after_run == "0044_standard_lr_v1"
    assert profile.rates() == EXPECTED_STANDARD

    config = common._config()
    assert {
        key: asdict(config)[key] for key in EXPECTED_STANDARD
    } == EXPECTED_STANDARD


def test_half_standard_is_an_explicit_run_override_not_a_deck_default() -> None:
    profile = profiles.scaled_standard_profile(
        0.5, override_id="0044_v19_half_standard_lr_run_override"
    )
    assert profile.profile_id == "0044_v19_half_standard_lr_run_override"
    assert profile.base_profile_id == "0044_standard_lr_v1"
    assert profile.scale == 0.5
    assert profile.scope == "run_only"
    assert profile.default_after_run == "0044_standard_lr_v1"
    assert profile.deck_id is None
    assert profile.rates() == {
        key: value * 0.5 for key, value in EXPECTED_STANDARD.items()
    }

    config = common._config(learning_rate_profile=profile)
    assert {
        key: asdict(config)[key] for key in EXPECTED_STANDARD
    } == profile.rates()


def test_lr_profile_rejects_ambiguous_or_invalid_overrides() -> None:
    with pytest.raises(ValueError, match="positive"):
        profiles.scaled_standard_profile(0.0, override_id="invalid")
    with pytest.raises(ValueError, match="cannot be combined"):
        common._config(
            learning_rate_profile=profiles.STANDARD_LR_PROFILE,
            actor_learning_rate_scale=0.5,
        )
