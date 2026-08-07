"""Chronological official-observation adapter for 0031 inference."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..domain.prototypes import PrototypeIndex
from ..features.collate import collate_canonical_records
from ..features.compiler import compile_canonical_row
from ..features.incremental import IncrementalCanonicalCompiler
from ..knowledge.state import CausalKnowledge
from ..model.config import ModelConfig


def _prototype_paths() -> tuple[Path, Path]:
    packaged = Path(__file__).resolve().parents[1] / "assets"
    if (packaged / "official_public_prototypes_v1.json").is_file():
        return (
            packaged / "official_public_prototypes_v1.json",
            packaged / "official_full_engine_prototypes_v2.json",
        )
    source = Path(__file__).resolve().parents[1] / "assets"
    return (
        source / "official_public_prototypes_v1.json",
        source / "official_full_engine_prototypes_v2.json",
    )


@lru_cache(maxsize=1)
def _load_process_prototypes() -> PrototypeIndex:
    """Load immutable prototype facts once for all battle sessions in this process."""
    public_path, engine_path = _prototype_paths()
    return PrototypeIndex.load(public_path, engine_path)


class OnlineCausalEncoder:
    """Compile one ordered engine session with the training-time feature path."""

    def __init__(
        self,
        actor: int,
        registered_deck: Sequence[int],
        config: ModelConfig,
        *,
        incremental: bool = False,
    ) -> None:
        if actor not in (0, 1):
            raise ValueError("actor must be 0 or 1")
        self.deck = tuple(int(card_id) for card_id in registered_deck)
        if len(self.deck) != 60 or any(card_id <= 0 for card_id in self.deck):
            raise ValueError("registered deck must contain exactly 60 positive card IDs")
        self.actor = actor
        self.config = config
        self.prototypes = _load_process_prototypes()
        self.deck_manifest = {"counts": sorted(Counter(self.deck).items())}
        self.knowledge = CausalKnowledge(actor, self.deck)
        self.incremental = bool(incremental)
        self.incremental_compiler = IncrementalCanonicalCompiler(self.prototypes)

    def encode_record(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        current = observation.get("current")
        select = observation.get("select")
        if not isinstance(current, Mapping) or current.get("yourIndex") != self.actor:
            raise ValueError("0031 online actor mismatch")
        if not isinstance(select, Mapping):
            raise ValueError("0031 online observation has no select payload")
        options = select.get("option")
        if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
            raise ValueError("0031 online observation has no legal options")
        minimum = select.get("minCount", 0)
        maximum = select.get("maxCount", len(options))
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or not 0 <= minimum <= maximum <= len(options)
        ):
            raise ValueError("0031 online selection bounds are invalid")
        if len(options) > self.config.max_options or minimum > self.config.max_action_steps:
            raise RuntimeError("observation exceeds 0031 inference bounds")

        row = {
            "actor_observation": dict(observation),
            "ordered_action": list(range(minimum)),
            "action_termination": "online_structural_target",
            "identity": None,
            "split": "online",
            "deck_manifest": self.deck_manifest,
        }
        snapshot = self.knowledge.consume(observation)
        if self.incremental:
            return self.incremental_compiler.compile(row, snapshot)
        return compile_canonical_row(row, snapshot, self.prototypes)

    def encode(self, observation: Mapping[str, Any]):
        return collate_canonical_records([self.encode_record(observation)])


__all__ = ["OnlineCausalEncoder"]
