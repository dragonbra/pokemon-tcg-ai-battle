"""Immutable weight provenance for the 0038 zero-shot starting point."""

from pathlib import Path

PROJECT_ID = "0038_action_boundary_rl"
ROOT = Path(__file__).resolve().parents[2]
ACTOR_CHECKPOINT = ROOT / "rl_runs/0037_dragapult_value_initialized_rl/source/friend_0806_epoch11/model.pt"
VALUE_CHECKPOINT = ROOT / "rl_runs/0037_dragapult_value_initialized_rl/source/value_v2_epoch0005/value_head.pt"
ACTOR_SHA256 = "0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8"
VALUE_SHA256 = "e88b2f18089911c4ebd810fa3abfba800a103189184b6265b9d38c03a9e36360"

__all__ = [
    "ACTOR_CHECKPOINT", "ACTOR_SHA256", "PROJECT_ID", "VALUE_CHECKPOINT", "VALUE_SHA256",
]
