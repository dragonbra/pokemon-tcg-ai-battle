"""Versioned random-outcome fixtures used only by parity tests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


RANDOM_OUTCOME_SCHEMA_VERSION = "0038_random_outcome_fixture_v1"


def _integers(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{label} must contain integers")
    return tuple(int(item) for item in value)


@dataclass(frozen=True, slots=True)
class RandomOutcomeFixture:
    shuffle_permutation: tuple[int, ...] = ()
    prize_indices: tuple[int, ...] = ()
    coin_results: tuple[int, ...] = ()
    random_target_indices: tuple[int, ...] = ()
    effect_results: tuple[tuple[str, int], ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RandomOutcomeFixture":
        version = payload.get("schema_version")
        if version != RANDOM_OUTCOME_SCHEMA_VERSION:
            raise ValueError(f"unsupported random outcome schema: {version!r}")
        shuffle = _integers(payload.get("shuffle_permutation", ()), "shuffle_permutation")
        if shuffle and sorted(shuffle) != list(range(len(shuffle))):
            raise ValueError("shuffle_permutation must contain every index exactly once")
        prizes = _integers(payload.get("prize_indices", ()), "prize_indices")
        if len(set(prizes)) != len(prizes) or any(index < 0 or index >= len(shuffle) for index in prizes):
            raise ValueError("prize_indices must be unique indices into shuffle_permutation")
        coins = _integers(payload.get("coin_results", ()), "coin_results")
        if any(value not in (0, 1) for value in coins):
            raise ValueError("coin_results must be 0/1")
        targets = _integers(payload.get("random_target_indices", ()), "random_target_indices")
        if any(value < 0 for value in targets):
            raise ValueError("random_target_indices must be nonnegative")
        raw_effects = payload.get("effect_results", {})
        if not isinstance(raw_effects, Mapping):
            raise ValueError("effect_results must be a mapping")
        effects: list[tuple[str, int]] = []
        for key, value in sorted(raw_effects.items()):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"effect_results[{key!r}] must be an integer")
            effects.append((str(key), int(value)))
        return cls(shuffle, prizes, coins, targets, tuple(effects))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": RANDOM_OUTCOME_SCHEMA_VERSION,
            "shuffle_permutation": list(self.shuffle_permutation),
            "prize_indices": list(self.prize_indices),
            "coin_results": list(self.coin_results),
            "random_target_indices": list(self.random_target_indices),
            "effect_results": dict(self.effect_results),
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class OutcomeCursor:
    """Fail-closed consumption ledger for a single test transition."""

    def __init__(self, fixture: RandomOutcomeFixture):
        self.fixture = fixture
        self._coin = 0
        self._target = 0
        self._effects = dict(fixture.effect_results)
        self._used_effects: set[str] = set()

    def coin(self) -> int:
        if self._coin >= len(self.fixture.coin_results):
            raise RuntimeError("random outcome fixture coin underflow")
        value = self.fixture.coin_results[self._coin]
        self._coin += 1
        return value

    def target(self, option_count: int) -> int:
        if self._target >= len(self.fixture.random_target_indices):
            raise RuntimeError("random outcome fixture target underflow")
        value = self.fixture.random_target_indices[self._target]
        self._target += 1
        if value >= option_count:
            raise RuntimeError(f"random target {value} outside option_count={option_count}")
        return value

    def effect(self, name: str) -> int:
        if name not in self._effects or name in self._used_effects:
            raise RuntimeError(f"random effect outcome missing or already consumed: {name}")
        self._used_effects.add(name)
        return self._effects[name]

    def assert_fully_consumed(self) -> None:
        unused = {
            "coin_results": len(self.fixture.coin_results) - self._coin,
            "random_target_indices": len(self.fixture.random_target_indices) - self._target,
            "effect_results": len(self._effects) - len(self._used_effects),
        }
        unused = {key: value for key, value in unused.items() if value}
        if unused:
            raise RuntimeError(f"unused random outcome fixture values: {unused}")


__all__ = ["RANDOM_OUTCOME_SCHEMA_VERSION", "OutcomeCursor", "RandomOutcomeFixture"]
