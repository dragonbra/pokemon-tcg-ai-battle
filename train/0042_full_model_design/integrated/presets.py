from __future__ import annotations

from .config import IntegratedFlags


PRESETS = {
    "BASE": IntegratedFlags(meta_anchor_coef=0.10),
    "FULL_MODEL": IntegratedFlags(
        meta_anchor_coef=0.10,
        enable_prize_aux=True, prize_aux_mode="directional",
        prize_aux_actor_weight=0.10,
    ),
}


def preset(name: str) -> IntegratedFlags:
    try:
        value = PRESETS[name]
    except KeyError as error:
        raise ValueError(f"unknown 0042 preset: {name}") from error
    value.validate()
    return value


for _preset in PRESETS.values():
    _preset.validate()

__all__ = ["PRESETS", "preset"]
