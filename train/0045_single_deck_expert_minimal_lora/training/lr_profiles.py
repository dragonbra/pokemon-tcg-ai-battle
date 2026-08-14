"""Named learning-rate contracts for 0045 minimal-specialist PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class LearningRateProfile:
    profile_id: str
    base_profile_id: str
    scale: float | None
    scope: str
    default_after_run: str
    decoder_learning_rate: float
    allocation_learning_rate: float
    option_lora_learning_rate: float
    value_learning_rate: float
    prize_learning_rate: float
    deck_id: None = None

    def rates(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in asdict(self).items()
            if key.endswith("_learning_rate")
        }

    def metadata(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "base_profile_id": self.base_profile_id,
            "scale": self.scale,
            "scope": self.scope,
            "default_after_run": self.default_after_run,
            "deck_id": self.deck_id,
            "rates": self.rates(),
        }


LIMIT_FINETUNE_LR_PROFILE = LearningRateProfile(
    profile_id="0045_limit_finetune_lr_v1",
    base_profile_id="0045_limit_finetune_lr_v1",
    scale=1.0,
    scope="project_limit_finetune",
    default_after_run="0045_limit_finetune_lr_v1",
    decoder_learning_rate=5.0e-6,
    allocation_learning_rate=5.0e-6,
    option_lora_learning_rate=1.0e-5,
    value_learning_rate=2.0e-5,
    prize_learning_rate=2.0e-5,
)


EXPERT_COLD_START_LR_PROFILE = LearningRateProfile(
    profile_id="0045_expert_cold_start_lr_v1",
    base_profile_id="0045_expert_cold_start_lr_v1",
    scale=1.0,
    scope="project_default_for_new_expert",
    default_after_run=LIMIT_FINETUNE_LR_PROFILE.profile_id,
    decoder_learning_rate=1.0e-5,
    allocation_learning_rate=1.0e-5,
    option_lora_learning_rate=2.0e-5,
    value_learning_rate=2.0e-5,
    prize_learning_rate=2.0e-5,
)

# Compatibility name used by the common trainer. In 0045, "standard" now
# means the first-run expert cold-start profile; the old V1 rates are retained
# explicitly as LIMIT_FINETUNE_LR_PROFILE.
STANDARD_LR_PROFILE = EXPERT_COLD_START_LR_PROFILE


def scaled_standard_profile(
    scale: float, *, override_id: str,
) -> LearningRateProfile:
    if scale <= 0:
        raise ValueError("learning-rate scale must be positive")
    if not override_id:
        raise ValueError("run-level learning-rate override requires an ID")
    rates = {
        key: value * float(scale)
        for key, value in STANDARD_LR_PROFILE.rates().items()
    }
    return LearningRateProfile(
        profile_id=override_id,
        base_profile_id=STANDARD_LR_PROFILE.profile_id,
        scale=float(scale),
        scope="run_only",
        default_after_run=STANDARD_LR_PROFILE.profile_id,
        **rates,
    )


def resolve_learning_rate_profile(
    *,
    learning_rate_profile: LearningRateProfile | None = None,
    actor_learning_rate_scale: float | None = None,
    value_learning_rate: float | None = None,
    prize_learning_rate: float | None = None,
) -> LearningRateProfile:
    legacy_values = (
        actor_learning_rate_scale,
        value_learning_rate,
        prize_learning_rate,
    )
    if learning_rate_profile is not None:
        if any(value is not None for value in legacy_values):
            raise ValueError(
                "named learning-rate profile cannot be combined with scalar overrides"
            )
        return learning_rate_profile
    if all(value is None for value in legacy_values):
        return STANDARD_LR_PROFILE

    actor_scale = 1.0 if actor_learning_rate_scale is None else float(
        actor_learning_rate_scale
    )
    if actor_scale <= 0:
        raise ValueError("actor learning-rate scale must be positive")
    rates = {
        "decoder_learning_rate": STANDARD_LR_PROFILE.decoder_learning_rate * actor_scale,
        "allocation_learning_rate": STANDARD_LR_PROFILE.allocation_learning_rate * actor_scale,
        "option_lora_learning_rate": STANDARD_LR_PROFILE.option_lora_learning_rate * actor_scale,
        "value_learning_rate": (
            STANDARD_LR_PROFILE.value_learning_rate
            if value_learning_rate is None else float(value_learning_rate)
        ),
        "prize_learning_rate": (
            STANDARD_LR_PROFILE.prize_learning_rate
            if prize_learning_rate is None else float(prize_learning_rate)
        ),
    }
    if min(rates.values()) <= 0:
        raise ValueError("all learning rates must be positive")
    if rates == STANDARD_LR_PROFILE.rates():
        return STANDARD_LR_PROFILE
    return LearningRateProfile(
        profile_id="0045_explicit_lr_v1",
        base_profile_id=STANDARD_LR_PROFILE.profile_id,
        scale=None,
        scope="run_only",
        default_after_run=STANDARD_LR_PROFILE.profile_id,
        **rates,
    )


__all__ = [
    "EXPERT_COLD_START_LR_PROFILE",
    "LIMIT_FINETUNE_LR_PROFILE",
    "LearningRateProfile",
    "STANDARD_LR_PROFILE",
    "resolve_learning_rate_profile",
    "scaled_standard_profile",
]
