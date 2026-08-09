from __future__ import annotations

from .config import IntegratedFlags


PRESETS = {
    "BASE": IntegratedFlags(),
    "PRIZE": IntegratedFlags(
        enable_prize_aux=True, prize_aux_mode="directional",
        prize_aux_actor_weight=0.10,
    ),
    "META": IntegratedFlags(
        enable_opponent_meta=True, enable_opponent_meta_conditioning=True,
    ),
    "PRIZE_META": IntegratedFlags(
        enable_prize_aux=True, prize_aux_mode="directional",
        prize_aux_actor_weight=0.10,
        enable_opponent_meta=True, enable_opponent_meta_conditioning=True,
    ),
    "INTEGRATED": IntegratedFlags(
        enable_prize_aux=True, prize_aux_mode="directional",
        prize_aux_actor_weight=0.10,
        enable_opponent_meta=True, enable_opponent_meta_conditioning=True,
        enable_tempo_metrics=True, enable_tempo_curriculum=True,
        enable_tempo_aux_loss=False,
    ),
}


def preset(name: str) -> IntegratedFlags:
    try:
        value = PRESETS[name]
    except KeyError as error:
        raise ValueError(f"unknown 0038 preset: {name}") from error
    value.validate()
    return value


for _preset in PRESETS.values():
    _preset.validate()

__all__ = ["PRESETS", "preset"]
