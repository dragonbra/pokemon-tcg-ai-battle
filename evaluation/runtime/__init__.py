"""隔离的 cg runtime 加载与兼容性检查。"""

from typing import TYPE_CHECKING, Any

from .loader import assert_cg_compatible, compute_cg_manifest, load_game_api

if TYPE_CHECKING:
    from .seeded import SeededRuntimeManifest


def __getattr__(name: str) -> Any:
    if name in {"SeededRuntimeManifest", "build_seeded_runtime"}:
        from . import seeded

        return getattr(seeded, name)
    raise AttributeError(name)

__all__ = [
    "SeededRuntimeManifest",
    "assert_cg_compatible",
    "build_seeded_runtime",
    "compute_cg_manifest",
    "load_game_api",
]
