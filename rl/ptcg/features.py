from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FEATURE_SCHEMA_VERSION = "ptcg_features_v5"
KNOWN_FEATURE_SCHEMA_VERSIONS = frozenset(
    {
        "ptcg_features_v1",
        "ptcg_features_v2",
        "ptcg_features_v3",
        "ptcg_features_v4",
        FEATURE_SCHEMA_VERSION,
    }
)


@dataclass(frozen=True)
class PTCGFeatureConfig:
    """Stable initial feature contract for the candidate policy/value model."""

    state_numeric_dim: int = 46
    state_token_count: int = 40
    candidate_numeric_dim: int = 10
    max_candidates: int = 64
    card_vocab_size: int = 4096
    action_type_vocab_size: int = 32


def feature_config_for_schema(schema_version: str) -> PTCGFeatureConfig:
    """Return the immutable input contract for a known feature schema."""
    if schema_version == "ptcg_features_v1":
        return PTCGFeatureConfig(state_numeric_dim=24, state_token_count=24)
    if schema_version == "ptcg_features_v2":
        return PTCGFeatureConfig(state_numeric_dim=32, state_token_count=40)
    if schema_version == "ptcg_features_v3":
        return PTCGFeatureConfig(state_numeric_dim=36, state_token_count=40)
    if schema_version == "ptcg_features_v4":
        return PTCGFeatureConfig(state_numeric_dim=40, state_token_count=40)
    if schema_version == FEATURE_SCHEMA_VERSION:
        return PTCGFeatureConfig()
    raise ValueError(f"unsupported feature schema version: {schema_version}")


def feature_schema_for_config(config: PTCGFeatureConfig) -> str:
    """Return the version name for an immutable feature shape."""
    shapes = {
        (24, 24): "ptcg_features_v1",
        (32, 40): "ptcg_features_v2",
        (36, 40): "ptcg_features_v3",
        (40, 40): "ptcg_features_v4",
        (46, 40): FEATURE_SCHEMA_VERSION,
    }
    try:
        return shapes[(config.state_numeric_dim, config.state_token_count)]
    except KeyError as exc:
        raise ValueError("unsupported feature configuration") from exc


def _card_id(card: Any) -> int | None:
    if not isinstance(card, dict):
        return None
    value = card.get("id", card.get("cardId"))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    values = player.get(area) or []
    return [value for value in values if isinstance(value, dict)]


def _count_energy(pokemon: dict[str, Any] | None) -> int:
    if not pokemon:
        return 0
    energies = pokemon.get("energies", pokemon.get("energyCards", [])) or []
    return len(energies)


def _hp(pokemon: dict[str, Any] | None) -> float:
    if not pokemon:
        return 0.0
    try:
        return max(0.0, min(400.0, float(pokemon.get("hp", 0)))) / 400.0
    except (TypeError, ValueError):
        return 0.0


def _player_numeric(player: dict[str, Any], opponent: dict[str, Any]) -> list[float]:
    active = _cards(player, "active")
    opponent_active = _cards(opponent, "active")
    own_active = active[0] if active else None
    other_active = opponent_active[0] if opponent_active else None
    return [
        float(player.get("prizeCount", len(player.get("prize") or []))) / 6.0,
        float(opponent.get("prizeCount", len(opponent.get("prize") or []))) / 6.0,
        float(player.get("deckCount", 0)) / 60.0,
        float(opponent.get("deckCount", 0)) / 60.0,
        float(player.get("handCount", len(player.get("hand") or []))) / 20.0,
        float(opponent.get("handCount", len(opponent.get("hand") or []))) / 20.0,
        float(len(_cards(player, "discard"))) / 60.0,
        float(len(_cards(opponent, "discard"))) / 60.0,
        float(len(_cards(player, "bench"))) / 5.0,
        float(len(_cards(opponent, "bench"))) / 5.0,
        _hp(own_active),
        _hp(other_active),
        float(_count_energy(own_active)) / 10.0,
        float(_count_energy(other_active)) / 10.0,
    ]


def _field_tokens(player: dict[str, Any], slots: int) -> list[int]:
    values = [*_cards(player, "active"), *_cards(player, "bench")]
    result = []
    for card in values[:slots]:
        card_id = _card_id(card)
        result.append((card_id + 1) if card_id is not None else 0)
    return result + [0] * max(0, slots - len(result))


def _zone_tokens(player: dict[str, Any], area: str, slots: int) -> list[int]:
    result = []
    for card in _cards(player, area)[:slots]:
        card_id = _card_id(card)
        result.append((card_id + 1) if card_id is not None else 0)
    return result + [0] * max(0, slots - len(result))


def _target_card_id(
    option: dict[str, Any],
    current: dict[str, Any],
    your_index: int,
) -> int | None:
    for key in ("targetCardId", "targetId", "inPlayCardId"):
        if key in option:
            try:
                return int(option[key])
            except (TypeError, ValueError):
                pass
    area = option.get("inPlayArea")
    index = option.get("inPlayIndex")
    if area not in (4, 5) or not isinstance(index, int):
        return None
    players = current.get("players") or []
    player = players[your_index] if 0 <= your_index < len(players) else {}
    zone = "active" if area == 4 else "bench"
    cards = _cards(player, zone)
    return _card_id(cards[index]) if 0 <= index < len(cards) else None


def _option_card_id(option: dict[str, Any]) -> int | None:
    for key in ("cardId", "id"):
        value = option.get(key)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def _normal(value: Any, divisor: float) -> float:
    try:
        return float(value) / divisor
    except (TypeError, ValueError):
        return 0.0


def _log_numeric(current: dict[str, Any], your_index: int) -> list[float]:
    logs = [log for log in current.get("logs") or [] if isinstance(log, dict)]
    opponent_index = 1 - your_index
    last = logs[-1] if logs else {}
    return [
        min(len(logs), 200) / 200.0,
        min(sum(log.get("playerIndex") == your_index for log in logs), 100) / 100.0,
        min(sum(log.get("playerIndex") == opponent_index for log in logs), 100) / 100.0,
        _normal(last.get("type", 0), 32.0),
        _normal(last.get("cardId", last.get("id", 0)), 4096.0),
        _normal(last.get("attackId", 0), 2000.0),
        min(sum("cardId" in log for log in logs), 100) / 100.0,
        min(sum("attackId" in log for log in logs), 100) / 100.0,
    ]


def _history_numeric(observation: dict[str, Any]) -> list[float]:
    history = [item for item in observation.get("rl_history") or [] if isinstance(item, dict)]
    last = history[-1] if history else {}
    return [
        min(len(history), 100) / 100.0,
        _normal(last.get("type", 0), 32.0),
        _normal(last.get("cardId", 0), 4096.0),
        _normal(last.get("attackId", 0), 2000.0),
    ]


def _timing_numeric(
    current: dict[str, Any],
    player: dict[str, Any],
    opponent: dict[str, Any],
) -> list[float]:
    """Expose legal timing cues used by the V6 rule teacher."""
    own_field = [*_cards(player, "active"), *_cards(player, "bench")]
    opponent_field = [*_cards(opponent, "active"), *_cards(opponent, "bench")]
    return [
        _normal(current.get("turnActionCount", 0), 32.0),
        _normal(current.get("looking", 0) if isinstance(current.get("looking"), int) else len(current.get("looking") or []), 16.0),
        _normal(sum(bool(card.get("appearThisTurn", False)) for card in own_field), 6.0),
        _normal(sum(bool(card.get("appearThisTurn", False)) for card in opponent_field), 6.0),
        1.0 if own_field and bool(own_field[0].get("appearThisTurn", False)) else 0.0,
        1.0 if opponent_field and bool(opponent_field[0].get("appearThisTurn", False)) else 0.0,
    ]


def encode_observation(
    observation: dict[str, Any],
    config: PTCGFeatureConfig = PTCGFeatureConfig(),
) -> dict[str, list[Any]]:
    """Convert the official observation into model-ready Python lists.

    This is intentionally a conservative first schema. It only uses fields
    visible in the observation and does not infer hidden opponent cards.
    The returned keys match ``CandidatePolicyValueNet`` tensor arguments after
    batching. A later schema must be versioned instead of silently changing
    these meanings.
    """
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    players = current.get("players") or []
    your_index = int(current.get("yourIndex", 0) or 0)
    player = players[your_index] if 0 <= your_index < len(players) else {}
    opponent_index = 1 - your_index
    opponent = players[opponent_index] if 0 <= opponent_index < len(players) else {}

    if config.state_numeric_dim not in (24, 32, 36, 40, 46):
        raise ValueError(
            "unsupported feature schema state width: "
            f"{config.state_numeric_dim}; expected 24, 32, 36, 40, or 46"
        )
    state_numeric = _player_numeric(player, opponent)
    result = current.get("result")
    state_numeric.extend(
        [
            _normal(current.get("turn", 0), 100.0),
            1.0 if current.get("firstPlayer") == your_index else 0.0,
            1.0 if current.get("supporterPlayed", False) else 0.0,
            1.0 if current.get("energyAttached", False) else 0.0,
            1.0 if current.get("retreated", False) else 0.0,
            _normal(select.get("type", 0), 16.0),
            _normal(select.get("context", 0), 64.0),
            _normal(select.get("minCount", 0), 8.0),
            _normal(select.get("maxCount", 0), 8.0),
            1.0 if isinstance(result, int) and result >= 0 else 0.0,
        ]
    )
    if config.state_numeric_dim >= 32:
        state_numeric.extend(_log_numeric(current, your_index))
    if config.state_numeric_dim >= 36:
        state_numeric.extend(_history_numeric(observation))
    if config.state_numeric_dim >= 40:
        effect = select.get("effect") or {}
        context_card = select.get("contextCard") or {}
        state_numeric.extend(
            [
                _normal(effect.get("id", select.get("effectId", 0)), 4096.0),
                _normal(context_card.get("id", 0), 4096.0),
                _normal(observation.get("rl_effect_step", 0), 32.0),
                _normal(len(select.get("option") or []), 64.0),
            ]
        )
    if config.state_numeric_dim >= 46:
        state_numeric.extend(_timing_numeric(current, player, opponent))
    if len(state_numeric) != config.state_numeric_dim:
        raise ValueError(
            f"state feature schema has {len(state_numeric)} values, "
            f"expected {config.state_numeric_dim}"
        )

    if config.state_token_count == 24:
        state_tokens = [
            *_field_tokens(player, 6),
            *_field_tokens(opponent, 6),
            *_zone_tokens(player, "hand", 6),
            *_zone_tokens(player, "discard", 4),
            *_zone_tokens(opponent, "discard", 2),
        ]
    elif config.state_token_count == 40:
        state_tokens = [
            *_field_tokens(player, 6),
            *_field_tokens(opponent, 6),
            *_zone_tokens(player, "hand", 16),
            *_zone_tokens(player, "discard", 8),
            *_zone_tokens(opponent, "discard", 4),
        ]
    else:
        raise ValueError(
            "unsupported feature schema token width: "
            f"{config.state_token_count}; expected 24 or 40"
        )
    if len(state_tokens) != config.state_token_count:
        raise ValueError(
            f"state token schema has {len(state_tokens)} values, "
            f"expected {config.state_token_count}"
        )
    state_tokens = [min(config.card_vocab_size, token) for token in state_tokens]

    options = select.get("option") or []
    if len(options) > config.max_candidates:
        raise ValueError(
            f"observation has {len(options)} candidates, "
            f"but the schema only supports {config.max_candidates}"
        )
    action_type_ids: list[int] = []
    action_card_ids: list[int] = []
    action_target_ids: list[int] = []
    action_numeric: list[list[float]] = []
    action_mask: list[bool] = []
    for index, option in enumerate(options[: config.max_candidates]):
        option_type = int(option.get("type", 0) or 0)
        card_id = _option_card_id(option)
        target_id = _target_card_id(option, current, your_index)
        action_type_ids.append(min(config.action_type_vocab_size, option_type + 1))
        action_card_ids.append(
            min(config.card_vocab_size, (card_id + 1) if card_id is not None else 0)
        )
        action_target_ids.append(
            min(config.card_vocab_size, (target_id + 1) if target_id is not None else 0)
        )
        action_numeric.append(
            [
                _normal(index, float(max(1, config.max_candidates - 1))),
                _normal(option.get("area", 0), 8.0),
                _normal(option.get("inPlayArea", 0), 8.0),
                _normal(option.get("index", 0), 16.0),
                _normal(option.get("inPlayIndex", 0), 8.0),
                _normal(option.get("attackId", 0), 2000.0),
                _normal(option.get("number", 0), 8.0),
                _normal(select.get("type", 0), 16.0),
                _normal(select.get("context", 0), 64.0),
                _normal(select.get("maxCount", 0), 8.0),
            ]
        )
        action_mask.append(True)

    while len(action_type_ids) < config.max_candidates:
        action_type_ids.append(0)
        action_card_ids.append(0)
        action_target_ids.append(0)
        action_numeric.append([0.0] * config.candidate_numeric_dim)
        action_mask.append(False)

    if not any(action_mask):
        raise ValueError("observation contains no legal action candidates")
    return {
        "state_numeric": state_numeric,
        "state_card_ids": state_tokens,
        "action_type_ids": action_type_ids,
        "action_card_ids": action_card_ids,
        "action_target_ids": action_target_ids,
        "action_numeric": action_numeric,
        "action_mask": action_mask,
    }
