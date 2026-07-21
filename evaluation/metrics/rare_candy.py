from __future__ import annotations

from collections import defaultdict

from .base import AggregateMetric, GameContext, GameMetric
from .powerful_hand import second_own_turn_entries
from .trace_utils import (
    active,
    as_int,
    bench,
    candidate_index,
    cards,
    field_state,
    logs,
    normalized_evidence,
    option_attack_id,
    option_list,
    player_at,
    selected_attack_id,
    selected_options,
    trace_steps,
)


ABRA_CARD_ID = 741
ALAKAZAM_CARD_ID = 743
RARE_CANDY_CARD_ID = 1079

FAILURE_STAGES = (
    "rare_candy_not_played",
    "evolution_not_completed",
    "alakazam_not_active",
    "no_legal_attack",
    "attack_not_declared",
)


def _option_card_id(step: dict, physical_index: int, option: dict) -> int | None:
    for key in ("cardId", "card_id", "id"):
        value = as_int(option.get(key))
        if value is not None:
            return value
    if as_int(option.get("type")) != 7:
        return None
    hand_index = as_int(option.get("index"))
    hand = cards(player_at(step, physical_index), "hand")
    if hand_index is not None and 0 <= hand_index < len(hand):
        return as_int(hand[hand_index].get("id"))
    return None


def _field_cards(step: dict, physical_index: int) -> list[dict]:
    player = player_at(step, physical_index)
    return [card for card in [active(player), *bench(player)] if card]


def _selected_rare_candy(step: dict, physical_index: int) -> bool:
    return any(
        _option_card_id(step, physical_index, option) == RARE_CANDY_CARD_ID
        for option in selected_options(step)
    )


def _is_rare_candy_evolution(
    step: dict,
    physical_index: int,
    had_abra: bool,
    previous_field: dict[int, dict],
    previous_alakazam: int,
) -> tuple[bool, int | None]:
    log_evolution_found = False
    log_serial: int | None = None
    for log in logs(step):
        log_player = as_int(log.get("playerIndex"))
        if log_player is not None and log_player != physical_index:
            continue
        official_evolution = (
            as_int(log.get("type")) == 12
            and as_int(log.get("cardIdTarget")) == ABRA_CARD_ID
            and as_int(log.get("cardId")) == ALAKAZAM_CARD_ID
        )
        legacy_evolution = (
            as_int(log.get("cardIdBefore")) == ABRA_CARD_ID
            and as_int(log.get("cardIdAfter")) == ALAKAZAM_CARD_ID
        )
        if not official_evolution and not legacy_evolution:
            continue
        serial_key = "serial" if official_evolution else "serialAfter"
        serial = as_int(log.get(serial_key))
        previous = previous_field.get(serial) if serial is not None else None
        if previous is not None and as_int(previous.get("id")) == ALAKAZAM_CARD_ID:
            continue
        log_evolution_found = True
        log_serial = serial
        break

    field_cards = _field_cards(step, physical_index)
    alakazam = [
        card for card in field_cards if as_int(card.get("id")) == ALAKAZAM_CARD_ID
    ]
    for card in alakazam:
        serial = as_int(card.get("serial"))
        previous = previous_field.get(serial) if serial is not None else None
        if previous is not None and as_int(previous.get("id")) == ALAKAZAM_CARD_ID:
            continue
        evolved_from_abra = any(
            as_int(pre.get("id")) == ABRA_CARD_ID
            for pre in card.get("preEvolution") or []
            if isinstance(pre, dict)
        )
        if evolved_from_abra or (
            previous is not None and as_int(previous.get("id")) == ABRA_CARD_ID
        ):
            return True, serial

    if log_evolution_found:
        return True, log_serial
    if had_abra and len(alakazam) > previous_alakazam:
        new_serial = next(
            (
                as_int(card.get("serial"))
                for card in alakazam
                if as_int(card.get("serial")) not in previous_field
            ),
            None,
        )
        return True, new_serial
    return False, None


def _is_evolved_alakazam_active(
    step: dict,
    physical_index: int,
    evolved_serial: int | None,
) -> bool:
    active_card = active(player_at(step, physical_index))
    if active_card is None or as_int(active_card.get("id")) != ALAKAZAM_CARD_ID:
        return False
    return evolved_serial is None or as_int(active_card.get("serial")) == evolved_serial


def _source_evidence(
    trace: dict,
    context: GameContext,
    source: tuple[int, dict | None],
    failure_stage: str | None,
) -> tuple:
    evidence = normalized_evidence(
        trace,
        source[1],
        fallback_step=source[0],
        candidate_physical_index=context.candidate_physical_index,
    )
    evidence["failure_stage"] = failure_stage
    return (evidence,)


class RareCandyPlugin:
    metric_id = "rare_candy"

    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        entries = second_own_turn_entries(trace, context)
        physical_index = candidate_index(trace, context.candidate_physical_index)
        rare_position = next(
            (
                position
                for position, (_, step) in enumerate(entries)
                if _selected_rare_candy(step, physical_index)
            ),
            None,
        )
        failure_stage: str | None = None
        source: tuple[int, dict | None]

        if rare_position is None:
            failure_stage = "rare_candy_not_played"
            source = entries[-1] if entries else self._trace_fallback(trace)
        else:
            rare_source = entries[rare_position]
            before_cards = _field_cards(rare_source[1], physical_index)
            previous_field = field_state(rare_source[1], physical_index)
            had_abra = any(as_int(card.get("id")) == ABRA_CARD_ID for card in before_cards)
            previous_alakazam = sum(
                as_int(card.get("id")) == ALAKAZAM_CARD_ID for card in before_cards
            )
            evolution: tuple[int, int | None] | None = None
            for position in range(rare_position + 1, len(entries)):
                found, evolved_serial = _is_rare_candy_evolution(
                    entries[position][1],
                    physical_index,
                    had_abra,
                    previous_field,
                    previous_alakazam,
                )
                if found:
                    evolution = (position, evolved_serial)
                    break
            if evolution is None:
                failure_stage = "evolution_not_completed"
                source = entries[-1]
            else:
                evolved_position, evolved_serial = evolution
                active_position = next(
                    (
                        position
                        for position in range(evolved_position, len(entries))
                        if _is_evolved_alakazam_active(
                            entries[position][1],
                            physical_index,
                            evolved_serial,
                        )
                    ),
                    None,
                )
                if active_position is None:
                    failure_stage = "alakazam_not_active"
                    source = entries[evolved_position]
                else:
                    legal_positions = [
                        position
                        for position in range(active_position, len(entries))
                        if any(
                            option_attack_id(option) is not None
                            for option in option_list(entries[position][1])
                        )
                    ]
                    selected_position = next(
                        (
                            position
                            for position in legal_positions
                            if selected_attack_id(entries[position][1]) is not None
                        ),
                        None,
                    )
                    if selected_position is not None:
                        source = entries[selected_position]
                    elif legal_positions:
                        failure_stage = "attack_not_declared"
                        source = entries[legal_positions[0]]
                    else:
                        failure_stage = "no_legal_attack"
                        source = entries[active_position]

        success = failure_stage is None
        diagnostic = {
            "game_id": context.game_id,
            "opponent": context.opponent_name,
            "failure_stage": failure_stage,
        }
        return GameMetric(
            metric_id=self.metric_id,
            status="success" if success else "failure",
            numerator=int(success),
            denominator=1,
            value="success" if success else failure_stage,
            evidence=_source_evidence(trace, context, source, failure_stage),
            diagnostics=(diagnostic,),
        )

    @staticmethod
    def _trace_fallback(trace: dict) -> tuple[int, dict | None]:
        steps = trace_steps(trace)
        return (len(steps) - 1, steps[-1]) if steps else (0, None)

    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        numerator = sum(result.numerator for result in results)
        denominator = sum(result.denominator for result in results)
        grouped: dict[str, dict[str, object]] = defaultdict(
            lambda: {"numerator": 0, "denominator": 0, "failure_stages": {}}
        )
        for result in results:
            diagnostic = result.diagnostics[0] if result.diagnostics else {}
            opponent = str(diagnostic.get("opponent", "unknown"))
            group = grouped[opponent]
            group["numerator"] = int(group["numerator"]) + result.numerator
            group["denominator"] = int(group["denominator"]) + result.denominator
            stage = diagnostic.get("failure_stage")
            if stage is not None:
                stages = group["failure_stages"]
                if isinstance(stages, dict):
                    stages[str(stage)] = int(stages.get(str(stage), 0)) + 1
        for group in grouped.values():
            attempts = int(group["denominator"])
            group["value"] = int(group["numerator"]) / attempts if attempts else None
        return AggregateMetric(
            metric_id=self.metric_id,
            numerator=numerator,
            denominator=denominator,
            value=numerator / denominator if denominator else None,
            by_opponent=dict(grouped),
        )
