"""Backend-independent authoritative rule-state records.

This module is intentionally distinct from :mod:`canonical`, whose public
observation representation hides private identities.  Rule-transition parity
must retain all authoritative identities and ordering, including deck and
Prize order, while policy parity must not expose those fields to the actor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


CANONICAL_STATE_SCHEMA_VERSION = "0038_authority_state_v1"

# These fields cannot affect rules.  Everything else presented by an adapter
# is either represented or rejected; adapters may not silently drop fields.
EXCLUDED_FIELDS: dict[str, str] = {
    "backend": "provenance only; compared separately by the audit manifest",
    "capture_timestamp_ns": "wall-clock capture metadata",
    "host_pointer": "process-local allocation address",
    "device_pointer": "process/device-local allocation address",
    "struct_padding": "backend ABI padding with no rule meaning",
    "debug_render_cache": "derived UI/debug cache, never read by rules",
    "kernel_lane": "execution layout only; game identity is episode_id",
}

REQUIRED_SECTIONS = (
    "game",
    "players",
    "cards",
    "zones",
    "pending",
    "effect_stack",
    "continuation_stack",
    "legal_options",
    "event_history",
    "result",
    "reward",
)

REQUIRED_GAME_FIELDS = (
    "episode_id", "turn", "phase", "seat", "first_player",
    "turn_action_count", "effect_action_count", "move_counter",
)
REQUIRED_RESULT_FIELDS = ("terminal", "winner", "game_result", "finish_reason", "error", "error_detail")
REQUIRED_REWARD_FIELDS = ("reward", "prize_delta", "terminal_reward")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported canonical value type: {type(value).__name__}")


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"canonical authority state missing mapping: {key}")
    return value


def _require_fields(section: Mapping[str, Any], names: Sequence[str], label: str) -> None:
    missing = [name for name in names if name not in section]
    if missing:
        raise ValueError(f"canonical authority state missing {label} fields: {missing}")


def _validate_player(player: Mapping[str, Any], index: int) -> None:
    required = (
        "player", "active", "bench", "hand", "deck", "discard", "prizes",
        "energy", "tools", "pre_evolution", "temporary", "turn_flags",
        "once_per_turn", "supporter_used", "retreat_used", "energy_attached",
    )
    _require_fields(player, required, f"players[{index}]")


def _validate_card(card: Mapping[str, Any], index: int) -> None:
    required = (
        "stable_serial", "card_id", "owner", "zone", "zone_index", "hp",
        "damage", "status", "energy", "tools", "pre_evolution", "turn_flags",
        "continual_flags", "ability_used",
    )
    _require_fields(card, required, f"cards[{index}]")


@dataclass(frozen=True, slots=True)
class CanonicalAuthorityState:
    payload: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, snapshot: Mapping[str, Any]) -> "CanonicalAuthorityState":
        if not isinstance(snapshot, Mapping):
            raise TypeError("authoritative snapshot must be a mapping")
        unknown_exclusions = set(snapshot.get("excluded_fields", ())) - set(EXCLUDED_FIELDS)
        if unknown_exclusions:
            raise ValueError(f"unapproved excluded rule fields: {sorted(unknown_exclusions)}")
        missing = [section for section in REQUIRED_SECTIONS if section not in snapshot]
        if missing:
            raise ValueError(f"canonical authority state missing sections: {missing}")
        game = _require_mapping(snapshot, "game")
        result = _require_mapping(snapshot, "result")
        reward = _require_mapping(snapshot, "reward")
        _require_fields(game, REQUIRED_GAME_FIELDS, "game")
        _require_fields(result, REQUIRED_RESULT_FIELDS, "result")
        _require_fields(reward, REQUIRED_REWARD_FIELDS, "reward")
        players = snapshot.get("players")
        if not isinstance(players, Sequence) or isinstance(players, (str, bytes)) or len(players) != 2:
            raise ValueError("canonical authority state requires exactly two players")
        for index, player in enumerate(players):
            if not isinstance(player, Mapping):
                raise ValueError(f"players[{index}] is not a mapping")
            _validate_player(player, index)
        cards = snapshot.get("cards")
        if not isinstance(cards, Sequence) or isinstance(cards, (str, bytes)):
            raise ValueError("canonical authority state cards is not a sequence")
        for index, card in enumerate(cards):
            if not isinstance(card, Mapping):
                raise ValueError(f"cards[{index}] is not a mapping")
            _validate_card(card, index)
        for sequence_key in (
            "effect_stack", "continuation_stack", "legal_options", "event_history",
        ):
            value = snapshot.get(sequence_key)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ValueError(f"canonical authority state {sequence_key} is not a sequence")
        payload = {key: value for key, value in snapshot.items() if key not in EXCLUDED_FIELDS and key != "excluded_fields"}
        payload["schema_version"] = CANONICAL_STATE_SCHEMA_VERSION
        return cls(_plain(payload))

    @classmethod
    def from_official(cls, snapshot: Mapping[str, Any]) -> "CanonicalAuthorityState":
        return cls.from_mapping(_adapt_backend_snapshot(snapshot, expected_backend="official_cpu"))

    @classmethod
    def from_cuda(cls, snapshot: Mapping[str, Any]) -> "CanonicalAuthorityState":
        return cls.from_mapping(_adapt_backend_snapshot(snapshot, expected_backend="cuda"))

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _adapt_backend_snapshot(snapshot: Mapping[str, Any], *, expected_backend: str) -> Mapping[str, Any]:
    actual = snapshot.get("backend")
    if actual not in (None, expected_backend):
        raise ValueError(f"expected backend {expected_backend!r}, got {actual!r}")
    # Exporters emit the same versioned field names; this adapter is an
    # identity boundary by design.  Backend-specific layout must be resolved
    # before entering Python rather than guessed here.
    return snapshot


@dataclass(frozen=True, slots=True)
class FieldDifference:
    path: str
    left: Any
    right: Any


def canonical_state_diff(
    left: CanonicalAuthorityState,
    right: CanonicalAuthorityState,
) -> list[FieldDifference]:
    differences: list[FieldDifference] = []

    def visit(a: Any, b: Any, path: str) -> None:
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in sorted(set(a) | set(b)):
                child = f"{path}.{key}" if path else str(key)
                if key not in a:
                    differences.append(FieldDifference(child, "<missing>", b[key]))
                elif key not in b:
                    differences.append(FieldDifference(child, a[key], "<missing>"))
                else:
                    visit(a[key], b[key], child)
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                differences.append(FieldDifference(f"{path}.length", len(a), len(b)))
            for index, (left_item, right_item) in enumerate(zip(a, b)):
                visit(left_item, right_item, f"{path}[{index}]")
            return
        if a != b:
            differences.append(FieldDifference(path, a, b))

    visit(left.payload, right.payload, "")
    return differences


__all__ = [
    "CANONICAL_STATE_SCHEMA_VERSION",
    "EXCLUDED_FIELDS",
    "CanonicalAuthorityState",
    "FieldDifference",
    "canonical_state_diff",
]
