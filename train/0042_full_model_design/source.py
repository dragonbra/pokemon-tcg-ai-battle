"""Immutable paired-0809 weight provenance for the 0042 starting point."""

from pathlib import Path

PROJECT_ID = "0042_full_model_design"
ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_ROOT = ROOT / "archive/pretrained/0031_friend_0809_gsb_v5_value_v9"
ACTOR_CHECKPOINT = PRETRAINED_ROOT / "model.pt"
VALUE_CHECKPOINT = PRETRAINED_ROOT / "value_head.pt"
ACTOR_SHA256 = "926321955b6f3144b62e65899b5041ca3a47b17f305202ba9dffc0c92aaa7c7f"
VALUE_SHA256 = "f8cb92f45f625e6519b6f103deb6d4ef68323fca93037ebd4c25db9da122a486"

__all__ = [
    "ACTOR_CHECKPOINT", "ACTOR_SHA256", "PRETRAINED_ROOT", "PROJECT_ID",
    "VALUE_CHECKPOINT", "VALUE_SHA256",
]
