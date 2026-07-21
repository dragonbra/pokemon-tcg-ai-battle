from __future__ import annotations

from collections import defaultdict

from .base import AggregateMetric, GameContext, GameMetric
from .powerful_hand import _is_candidate_step
from .trace_utils import (
    active,
    as_int,
    bench,
    candidate_index,
    normalized_evidence,
    option_list,
    player_at,
    selected_options,
    trace_steps,
)


DUDUNSPARCE_CARD_ID = 66
RUN_AWAY_DRAW_OPTION_TYPE = 10


def _option_card_id(option: dict, step: dict, physical_index: int) -> int | None:
    for key in ("cardId", "card_id", "id"):
        card_id = as_int(option.get(key))
        if card_id is not None:
            return card_id
    area = as_int(option.get("area", option.get("inPlayArea")))
    if area != 4:
        return None
    active_card = active(player_at(step, physical_index))
    return as_int(active_card.get("id")) if active_card else None


def _is_run_away_draw(option: dict, step: dict, physical_index: int) -> bool:
    return (
        as_int(option.get("type")) == RUN_AWAY_DRAW_OPTION_TYPE
        and _option_card_id(option, step, physical_index) == DUDUNSPARCE_CARD_ID
    )


class RunAwayDrawPlugin:
    metric_id = "run_away_draw"

    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        physical_index = candidate_index(trace, context.candidate_physical_index)
        evidence: list[dict[str, object]] = []
        opportunities = 0
        for index, step in enumerate(trace_steps(trace)):
            if not _is_candidate_step(step, physical_index):
                continue
            player = player_at(step, physical_index)
            active_card = active(player)
            empty_bench = not bench(player)
            if (
                not active_card
                or as_int(active_card.get("id")) != DUDUNSPARCE_CARD_ID
                or not empty_bench
            ):
                continue
            if any(_is_run_away_draw(option, step, physical_index) for option in option_list(step)):
                opportunities += 1
            if not any(
                _is_run_away_draw(option, step, physical_index)
                for option in selected_options(step)
            ):
                continue
            source = normalized_evidence(
                trace,
                step,
                fallback_step=index,
                candidate_physical_index=context.candidate_physical_index,
            )
            source["empty_bench"] = True
            source["active_card_id"] = DUDUNSPARCE_CARD_ID
            evidence.append(source)

        if not evidence:
            steps = trace_steps(trace)
            evidence.append(
                normalized_evidence(
                    trace,
                    steps[-1] if steps else None,
                    fallback_step=len(steps) - 1 if steps else 0,
                    candidate_physical_index=context.candidate_physical_index,
                )
            )
        count = len(evidence) if evidence[0].get("empty_bench") is True else 0
        diagnostic = {
            "game_id": context.game_id,
            "opponent": context.opponent_name,
            "empty_bench_opportunities": opportunities,
            "empty_bench_run_away_draw_count": count,
        }
        return GameMetric(
            metric_id=self.metric_id,
            status="success",
            numerator=count,
            denominator=1,
            value=count,
            evidence=tuple(evidence),
            diagnostics=(diagnostic,),
        )

    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        numerator = sum(result.numerator for result in results)
        denominator = sum(result.denominator for result in results)
        grouped: dict[str, dict[str, object]] = defaultdict(
            lambda: {"count": 0, "games": 0, "opportunities": 0}
        )
        for result in results:
            diagnostic = result.diagnostics[0] if result.diagnostics else {}
            opponent = str(diagnostic.get("opponent", "unknown"))
            group = grouped[opponent]
            group["count"] = int(group["count"]) + result.numerator
            group["games"] = int(group["games"]) + result.denominator
            group["opportunities"] = (
                int(group["opportunities"])
                + int(diagnostic.get("empty_bench_opportunities", 0))
            )
        for group in grouped.values():
            games = int(group["games"])
            group["value"] = int(group["count"]) / games if games else None
        return AggregateMetric(
            metric_id=self.metric_id,
            numerator=numerator,
            denominator=denominator,
            value=numerator / denominator if denominator else None,
            by_opponent=dict(grouped),
        )
