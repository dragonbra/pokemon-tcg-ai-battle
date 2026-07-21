"""隔离的 cg runtime 加载与兼容性检查。"""

from .loader import assert_cg_compatible, compute_cg_manifest, load_game_api

__all__ = ["assert_cg_compatible", "compute_cg_manifest", "load_game_api"]
