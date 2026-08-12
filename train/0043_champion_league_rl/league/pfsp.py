"""Auditable Beta(1,1) PFSP state with ten-update frozen curricula."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "0043_pfsp_state_v1"
CURRICULUM_SCHEMA = "0043_pfsp_curriculum_v1"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PFSPConfig:
    refresh_every_updates: int = 10
    minimum_probability: float = 0.001
    # Safe for the initial two-policy pool. Concentration tuning remains an
    # explicit versioned experiment; infeasible per-pool values still fail.
    maximum_probability: float = 1.0

    @classmethod
    def load(cls, path: Path) -> "PFSPConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "0043_league_config_v1":
            raise ValueError("unsupported 0043 league config schema")
        pfsp = payload.get("pfsp")
        if not isinstance(pfsp, dict):
            raise ValueError("league config has no PFSP object")
        config = cls(
            refresh_every_updates=pfsp["refresh_every_updates"],
            minimum_probability=pfsp["minimum_probability"],
            maximum_probability=pfsp["maximum_probability"],
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.refresh_every_updates != 10:
            raise ValueError("0043 V2.0 refresh_every_updates must be exactly 10")
        if not 0 <= self.minimum_probability < 1:
            raise ValueError("minimum_probability must be in [0, 1)")
        if not 0 < self.maximum_probability <= 1:
            raise ValueError("maximum_probability must be in (0, 1]")
        if self.minimum_probability > self.maximum_probability:
            raise ValueError("PFSP floor cannot exceed cap")

    @property
    def config_hash(self) -> str:
        self.validate()
        return _canonical_hash(asdict(self))


@dataclass(slots=True)
class Record:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    last_updated: int = -1

    @property
    def smoothed_win_rate(self) -> float:
        return (self.wins + 1.0) / (self.games + 2.0)

    @property
    def weakness(self) -> float:
        return 1.0 - self.smoothed_win_rate

    def observe(self, result: str, update: int) -> None:
        if result not in {"win", "loss", "draw"}:
            raise ValueError(f"unsupported terminal result: {result}")
        self.games += 1
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        else:
            self.draws += 1
        self.last_updated = update


def _weights(ids: tuple[str, ...], records: Mapping[str, Record], config: PFSPConfig) -> dict[str, float]:
    if not ids:
        raise ValueError("PFSP pool cannot be empty")
    if config.minimum_probability * len(ids) > 1 + 1e-12:
        raise ValueError("PFSP minimum_probability is infeasible for this pool")
    if config.maximum_probability * len(ids) < 1 - 1e-12:
        raise ValueError("PFSP maximum_probability is infeasible for this pool")
    raw = [records.get(item, Record()).weakness for item in ids]
    if any(not math.isfinite(value) or value < 0 for value in raw):
        raise ValueError("PFSP weakness contains invalid values")
    total = sum(raw)
    values = [value / total if total else 1.0 / len(ids) for value in raw]
    # Projection onto the probability simplex with explicit box constraints.
    fixed: dict[int, float] = {}
    remaining = set(range(len(ids)))
    while remaining:
        budget = 1.0 - sum(fixed.values())
        raw_remaining = sum(raw[index] for index in remaining)
        proposed = {
            index: budget * raw[index] / raw_remaining
            if raw_remaining else budget / len(remaining)
            for index in remaining
        }
        violations = {
            index: config.minimum_probability
            for index, value in proposed.items() if value < config.minimum_probability - 1e-15
        }
        violations.update({
            index: config.maximum_probability
            for index, value in proposed.items() if value > config.maximum_probability + 1e-15
        })
        if not violations:
            for index, value in proposed.items():
                fixed[index] = value
            break
        for index, value in violations.items():
            fixed[index] = value
            remaining.remove(index)
    values = [fixed[index] for index in range(len(ids))]
    correction = 1.0 - sum(values)
    values[max(range(len(values)), key=values.__getitem__)] += correction
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("PFSP produced invalid probabilities")
    return dict(zip(ids, values, strict=True))


@dataclass(frozen=True, slots=True)
class Curriculum:
    curriculum_version: str
    created_at_update: int
    deck_weights: dict[str, float]
    policy_weights: dict[str, float]
    stats_snapshot: dict[str, Any]
    config_hash: str
    training_deck_pool_hash: str
    active_policy_pool_hash: str
    schema_version: str = CURRICULUM_SCHEMA

    @property
    def content_hash(self) -> str:
        return _canonical_hash(asdict(self))


@dataclass(slots=True)
class PFSPState:
    deck_records: dict[str, Record] = field(default_factory=dict)
    policy_records: dict[str, Record] = field(default_factory=dict)
    joint_records: dict[str, Record] = field(default_factory=dict)
    curricula: dict[str, Curriculum] = field(default_factory=dict)
    schema_version: str = SCHEMA

    def observe(self, *, deck_id: str, policy_id: str, result: str, update: int) -> None:
        self.deck_records.setdefault(deck_id, Record()).observe(result, update)
        self.policy_records.setdefault(policy_id, Record()).observe(result, update)
        self.joint_records.setdefault(f"{deck_id}|{policy_id}", Record()).observe(result, update)

    def curriculum_for(
        self, update: int, *, deck_ids: Iterable[str], policy_ids: Iterable[str],
        config: PFSPConfig, training_deck_pool_hash: str, active_policy_pool_hash: str,
    ) -> Curriculum:
        if update < 0:
            raise ValueError("update must be non-negative")
        config.validate()
        deck_ids = tuple(deck_ids)
        policy_ids = tuple(policy_ids)
        version_number = update // config.refresh_every_updates
        version = f"C{version_number:03d}"
        existing = self.curricula.get(version)
        if existing is not None:
            expected = (config.config_hash, training_deck_pool_hash, active_policy_pool_hash)
            actual = (existing.config_hash, existing.training_deck_pool_hash, existing.active_policy_pool_hash)
            if actual != expected:
                raise ValueError("curriculum resume identity mismatch")
            return existing
        snapshot = {
            "decks": {key: asdict(self.deck_records.get(key, Record())) for key in deck_ids},
            "policies": {key: asdict(self.policy_records.get(key, Record())) for key in policy_ids},
        }
        curriculum = Curriculum(
            curriculum_version=version,
            created_at_update=version_number * config.refresh_every_updates,
            deck_weights=_weights(deck_ids, self.deck_records, config),
            policy_weights=_weights(policy_ids, self.policy_records, config),
            stats_snapshot=snapshot,
            config_hash=config.config_hash,
            training_deck_pool_hash=training_deck_pool_hash,
            active_policy_pool_hash=active_policy_pool_hash,
        )
        self.curricula[version] = curriculum
        return curriculum

    def save(self, path: Path) -> None:
        payload = {
            "schema_version": self.schema_version,
            "deck_records": {key: asdict(value) for key, value in self.deck_records.items()},
            "policy_records": {key: asdict(value) for key, value in self.policy_records.items()},
            "joint_records": {key: asdict(value) for key, value in self.joint_records.items()},
            "curricula": {key: asdict(value) for key, value in self.curricula.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "PFSPState":
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != SCHEMA:
            raise ValueError("unsupported PFSP state schema")
        return cls(
            deck_records={key: Record(**value) for key, value in payload["deck_records"].items()},
            policy_records={key: Record(**value) for key, value in payload["policy_records"].items()},
            joint_records={key: Record(**value) for key, value in payload["joint_records"].items()},
            curricula={key: Curriculum(**value) for key, value in payload["curricula"].items()},
        )


__all__ = ["Curriculum", "PFSPConfig", "PFSPState", "Record"]
