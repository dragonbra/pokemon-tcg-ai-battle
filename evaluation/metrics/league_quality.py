"""Generic trajectory digestion plus deck-category interpretation for League reports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from math import ceil

from .base import AggregateMetric, GameContext, GameMetric
from .league_profiles import COMMON_METRICS, profile_for_deck
from .trace_utils import (
    active,
    as_int,
    bench,
    candidate_index,
    current,
    field_state,
    knockout_is_confirmed,
    lifecycle_status,
    logs,
    normalized_evidence,
    option_attack_id,
    option_list,
    player_at,
    selected_attack_id,
    selected_options,
    trace_steps,
)


MOVE_LOG = 6
DAMAGE_LOG = 16
ABILITY_OPTION_TYPE = 10


def _round(turn: int | None) -> int | None:
    return ceil(turn / 2) if turn is not None and turn > 0 else None


def _prize_count(player: dict) -> int | None:
    prize = player.get("prize")
    if isinstance(prize, list):
        return len(prize)
    return as_int(player.get("prizeCount"))


def _deck_count(player: dict) -> int | None:
    return as_int(player.get("deckCount"))


def _field(player: dict) -> list[dict]:
    return [card for card in [active(player), *bench(player)] if isinstance(card, dict)]


def _energy_count(card: dict) -> int:
    values = card.get("energies") or []
    return len(values) if isinstance(values, list) else 0


def _is_evolved(card: dict) -> bool:
    values = card.get("preEvolution") or []
    return isinstance(values, list) and bool(values)


def _actor(step: dict) -> int | None:
    value = as_int(current(step).get("yourIndex"))
    return value if value in (0, 1) else None


def _turn(step: dict) -> int | None:
    return as_int(current(step).get("turn"))


def _turn_flags(step: dict) -> tuple[bool, bool, bool]:
    state = current(step)
    return (
        bool(state.get("supporterPlayed")),
        bool(state.get("energyAttached")),
        bool(state.get("retreated")),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


class LeagueQualityPlugin:
    metric_id = "league_quality"

    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        lifecycle = lifecycle_status(trace)
        steps = trace_steps(trace)
        candidate = candidate_index(trace, context.candidate_physical_index)
        opponent = 1 - candidate
        profile = profile_for_deck(context.candidate_name)
        setup_card_id = profile.key_card_ids[0]

        turns = {candidate: set(), opponent: set()}
        attack_turns = {candidate: set(), opponent: set()}
        opportunity_turns = {candidate: set(), opponent: set()}
        supporter_turns: set[int] = set()
        energy_turns: set[int] = set()
        retreat_turns: set[int] = set()
        candidate_prizes: list[int] = []
        candidate_decks: list[int] = []
        max_bench = 0
        max_evolved = 0
        max_energy = 0
        first_setup_round: int | None = None
        ability_actions = 0
        damage_events = 0
        self_field_discards = 0
        candidate_ko_turns: list[int] = []
        previous_candidate_field: dict[int, dict] = {}

        for step in steps:
            actor = _actor(step)
            turn = _turn(step)
            if actor not in (candidate, opponent) or turn is None:
                continue
            turns[actor].add(turn)
            if any(option_attack_id(option) is not None for option in option_list(step)):
                opportunity_turns[actor].add(turn)
            if selected_attack_id(step) is not None:
                attack_turns[actor].add(turn)

            if actor == candidate:
                supporter, attached, retreated = _turn_flags(step)
                supporter_turns.update([turn] if supporter else [])
                energy_turns.update([turn] if attached else [])
                retreat_turns.update([turn] if retreated else [])
                ability_actions += sum(
                    as_int(option.get("type")) == ABILITY_OPTION_TYPE
                    for option in selected_options(step)
                )
                own = player_at(step, candidate)
                opponent_state = player_at(step, opponent)
                field = _field(own)
                max_bench = max(max_bench, len(bench(own)))
                max_evolved = max(max_evolved, sum(_is_evolved(card) for card in field))
                max_energy = max(max_energy, sum(_energy_count(card) for card in field))
                prize = _prize_count(own)
                deck = _deck_count(own)
                if prize is not None:
                    candidate_prizes.append(prize)
                if deck is not None:
                    candidate_decks.append(deck)
                if first_setup_round is None and any(
                    as_int(card.get("id")) == setup_card_id for card in field
                ):
                    first_setup_round = _round(turn)
                for log in logs(step):
                    if (
                        as_int(log.get("type")) == DAMAGE_LOG
                        and as_int(log.get("playerIndex")) == opponent
                    ):
                        damage_events += 1
                    if (
                        as_int(log.get("type")) == MOVE_LOG
                        and as_int(log.get("playerIndex")) == candidate
                        and as_int(log.get("fromArea")) in {4, 5}
                        and as_int(log.get("toArea")) == 3
                    ):
                        self_field_discards += 1

            for log in logs(step):
                if (
                    as_int(log.get("type")) == MOVE_LOG
                    and as_int(log.get("playerIndex")) == candidate
                    and as_int(log.get("fromArea")) in {4, 5}
                    and as_int(log.get("toArea")) == 3
                    and knockout_is_confirmed(
                        step, log, candidate, previous_candidate_field
                    )
                ):
                    candidate_ko_turns.append(turn)
            previous_candidate_field = field_state(step, candidate)

        own_turns = sorted(turns[candidate])
        own_attacks = sorted(attack_turns[candidate])
        first_attack_turn = own_attacks[0] if own_attacks else None
        continuity_turns = (
            [turn for turn in own_turns if turn >= first_attack_turn]
            if first_attack_turn is not None
            else []
        )
        continuity_attacks = (
            [turn for turn in own_attacks if turn >= first_attack_turn]
            if first_attack_turn is not None
            else []
        )
        missed_opportunities = len(
            opportunity_turns[candidate] - attack_turns[candidate]
        )
        prizes_taken = (
            max(candidate_prizes) - min(candidate_prizes) if candidate_prizes else 0
        )
        prize_drops = [
            before - after
            for before, after in zip(candidate_prizes, candidate_prizes[1:])
            if before > after
        ]
        multi_prize_turns = sum(drop >= 2 for drop in prize_drops)
        attack_gaps = []
        for ko_turn in candidate_ko_turns:
            next_attack = next((turn for turn in own_attacks if turn > ko_turn), None)
            if next_attack is not None:
                attack_gaps.append(max(0, (_round(next_attack) or 0) - (_round(ko_turn) or 0)))

        pressure_start = first_attack_turn
        pressured_opponent_turns = (
            {turn for turn in turns[opponent] if pressure_start is not None and turn > pressure_start}
        )
        denied_turns = pressured_opponent_turns - attack_turns[opponent]
        payload = {
            "games": 1,
            "error_games": int(lifecycle == "error"),
            "unfinished_games": int(lifecycle == "unfinished"),
            "profile": {
                "id": profile.profile_id,
                "title": profile.title,
                "focus_metrics": list(profile.focus_metrics),
                "interpretation": profile.interpretation,
                "reward_warning": profile.reward_warning,
                "key_card_ids": list(profile.key_card_ids),
            },
            "candidate_turns": len(own_turns),
            "attack_turns": len(own_attacks),
            "attack_opportunity_turns": len(opportunity_turns[candidate]),
            "first_attack_round": _round(first_attack_turn),
            "attack_continuity": _ratio(len(continuity_attacks), len(continuity_turns)),
            "missed_attack_opportunities": missed_opportunities,
            "prizes_taken": prizes_taken,
            "prizes_per_attack": _ratio(prizes_taken, len(own_attacks)),
            "multi_prize_turns": multi_prize_turns,
            "post_ko_attack_gap": (
                sum(attack_gaps) / len(attack_gaps) if attack_gaps else None
            ),
            "post_ko_samples": len(attack_gaps),
            "max_bench": max_bench,
            "max_evolved_pokemon": max_evolved,
            "max_attached_energy": max_energy,
            "deck_cards_consumed": (
                max(candidate_decks) - min(candidate_decks) if candidate_decks else 0
            ),
            "minimum_deck_count": min(candidate_decks) if candidate_decks else None,
            "supporter_turns": len(supporter_turns),
            "energy_attach_turns": len(energy_turns),
            "retreat_turns": len(retreat_turns),
            "supporter_turn_rate": _ratio(len(supporter_turns), len(own_turns)),
            "energy_attach_turn_rate": _ratio(len(energy_turns), len(own_turns)),
            "opponent_pressured_turns": len(pressured_opponent_turns),
            "opponent_denied_attack_turns": len(denied_turns),
            "opponent_attack_denial_rate": _ratio(
                len(denied_turns), len(pressured_opponent_turns)
            ),
            "key_setup_round": first_setup_round,
            "ability_actions": ability_actions,
            "damage_events": damage_events,
            "self_field_discards": self_field_discards,
        }
        source = normalized_evidence(
            trace,
            steps[-1] if steps else None,
            fallback_step=len(steps) - 1 if steps else 0,
            candidate_physical_index=context.candidate_physical_index,
        )
        return GameMetric(
            metric_id=self.metric_id,
            status=(
                "error"
                if lifecycle == "error"
                else "unavailable"
                if lifecycle == "unfinished"
                else "success"
            ),
            numerator=len(own_attacks),
            denominator=len(own_turns),
            value=_ratio(len(own_attacks), len(own_turns)),
            evidence=(source,),
            diagnostics=(
                {
                    "game_id": context.game_id,
                    "opponent": context.opponent_name,
                    "candidate_first": context.candidate_first,
                    "turn_order": "first" if context.candidate_first else "second",
                },
            ),
            payload=payload,
        )

    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        payload = _aggregate_payload(results)
        grouped: dict[str, list[GameMetric]] = defaultdict(list)
        for result in results:
            diagnostic = result.diagnostics[0] if result.diagnostics else {}
            opponent = str(diagnostic.get("opponent", "unknown"))
            grouped[opponent].append(result)
        numerator = sum(result.numerator for result in results)
        denominator = sum(result.denominator for result in results)
        return AggregateMetric(
            metric_id=self.metric_id,
            numerator=numerator,
            denominator=denominator,
            value=_ratio(numerator, denominator),
            by_opponent={
                opponent: _aggregate_payload(values)
                for opponent, values in grouped.items()
            },
            payload=payload,
        )


def _mean(values: Iterable[object]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


def _aggregate_payload(results: Iterable[GameMetric]) -> dict[str, object]:
    values = list(results)
    payloads = [result.payload for result in values]
    profile = next((payload.get("profile") for payload in payloads), {})
    sum_fields = (
        "candidate_turns",
        "attack_turns",
        "attack_opportunity_turns",
        "missed_attack_opportunities",
        "prizes_taken",
        "multi_prize_turns",
        "post_ko_samples",
        "deck_cards_consumed",
        "supporter_turns",
        "energy_attach_turns",
        "retreat_turns",
        "opponent_pressured_turns",
        "opponent_denied_attack_turns",
        "ability_actions",
        "damage_events",
        "self_field_discards",
    )
    mean_fields = (
        "first_attack_round",
        "attack_continuity",
        "prizes_per_attack",
        "post_ko_attack_gap",
        "max_bench",
        "max_evolved_pokemon",
        "max_attached_energy",
        "minimum_deck_count",
        "supporter_turn_rate",
        "energy_attach_turn_rate",
        "opponent_attack_denial_rate",
        "key_setup_round",
    )
    result: dict[str, object] = {
        "games": len(values),
        "error_games": sum(int(payload.get("error_games", 0)) for payload in payloads),
        "unfinished_games": sum(
            int(payload.get("unfinished_games", 0)) for payload in payloads
        ),
        "profile": profile,
        "metric_vocabulary": [
            *COMMON_METRICS,
            "key_setup_round",
            "ability_actions",
            "damage_events",
            "self_field_discards",
        ],
    }
    result.update(
        {
            field: sum(int(payload.get(field, 0) or 0) for payload in payloads)
            for field in sum_fields
        }
    )
    result.update(
        {field: _mean(payload.get(field) for payload in payloads) for field in mean_fields}
    )
    candidate_turns = int(result["candidate_turns"])
    attack_turns = int(result["attack_turns"])
    result["attack_turn_rate"] = _ratio(attack_turns, candidate_turns)
    result["prizes_per_attack"] = _ratio(
        int(result["prizes_taken"]), attack_turns
    )
    result["supporter_turn_rate"] = _ratio(
        int(result["supporter_turns"]), candidate_turns
    )
    result["energy_attach_turn_rate"] = _ratio(
        int(result["energy_attach_turns"]), candidate_turns
    )
    result["opponent_attack_denial_rate"] = _ratio(
        int(result["opponent_denied_attack_turns"]),
        int(result["opponent_pressured_turns"]),
    )
    return result


__all__ = ["LeagueQualityPlugin"]
