from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import html

from .base import AggregateMetric, GameContext, GameMetric, MetricPresentation
from .powerful_hand import POWERFUL_HAND_ATTACK_ID, _is_candidate_step
from .trace_utils import (
    active,
    as_int,
    candidate_index,
    current,
    logs,
    normalized_evidence,
    option_attack_id,
    player_at,
    selected_options,
    lifecycle_status,
    trace_steps,
)


RESULT_LOG_TYPE = 15
DAMAGE_LOG_TYPE = 16


def _prize_count(step: dict, physical_index: int) -> int | None:
    player = player_at(step, physical_index)
    prize = player.get("prize")
    if isinstance(prize, list):
        return len(prize)
    return as_int(player.get("prizeCount"))


def _selected_attacks(step: dict) -> list[dict]:
    return [option for option in selected_options(step) if option_attack_id(option) is not None]


def _resolution(
    steps: list[dict], start_index: int, physical_index: int, attack_id: int
) -> tuple[int, list[dict]] | None:
    for index in range(start_index + 1, len(steps)):
        candidate = steps[index]
        matching = [
            log
            for log in logs(candidate)
            if as_int(log.get("type")) == RESULT_LOG_TYPE
            and as_int(log.get("playerIndex")) == physical_index
            and as_int(log.get("attackId")) == attack_id
        ]
        if matching:
            return index, logs(candidate)
        if _selected_attacks(candidate) and _is_candidate_step(candidate, physical_index):
            break
    return None


class AttackQualityPlugin:
    metric_id = "attack_quality"

    def render(
        self, aggregate: AggregateMetric, results: tuple[GameMetric, ...]
    ) -> MetricPresentation:
        payload = aggregate.payload
        non_prize = payload.get("non_prize_attacks", {})
        non_prize_values = non_prize if isinstance(non_prize, dict) else {}
        rows = (
            ("attack submissions", payload.get("attack_submissions", 0)),
            ("resolved", payload.get("resolved_attacks", 0)),
            ("unresolved", payload.get("unresolved_attacks", 0)),
            ("unknown prize", payload.get("unknown_prize_attacks", 0)),
            ("non-prize attacks", f"{non_prize_values.get('numerator', 0)}/{non_prize_values.get('denominator', 0)}"),
        )
        markdown = "\n".join(
            ["## Attack quality", "", *[f"- {label}: {value}" for label, value in rows], ""]
        )
        html_rows = "".join(
            f"<tr><td>{html.escape(str(label))}</td><td>{html.escape(str(value))}</td></tr>"
            for label, value in rows
        )
        html_fragment = (
            '<section class="metric-plugin"><h2>Attack quality</h2>'
            "<table><thead><tr><th>field</th><th>value</th></tr></thead>"
            f"<tbody>{html_rows}</tbody></table></section>"
        )
        return MetricPresentation(self.metric_id, "Attack quality", markdown, html_fragment)

    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        lifecycle = lifecycle_status(trace)
        steps = trace_steps(trace)
        physical_index = candidate_index(trace, context.candidate_physical_index)
        submissions = 0
        resolved = 0
        unresolved = 0
        unknown_prize = 0
        non_prize = 0
        powerful_resolved = 0
        powerful_unknown_prize = 0
        powerful_non_prize = 0
        attacks: list[dict[str, object]] = []
        evidence: list[dict[str, object]] = []

        for index, step in enumerate(steps):
            if not _is_candidate_step(step, physical_index):
                continue
            for option in _selected_attacks(step):
                submissions += 1
                attack_id = option_attack_id(option)
                if attack_id is None:
                    continue
                powerful = attack_id == POWERFUL_HAND_ATTACK_ID
                before_prize = _prize_count(step, physical_index)
                resolved_entry = _resolution(steps, index, physical_index, attack_id)
                attack: dict[str, object] = {
                    "attack_id": attack_id,
                    "powerful_hand": powerful,
                    "resolved": False,
                    "non_prize": None,
                    "prize_delta": None,
                    "damage_observed": False,
                }
                if resolved_entry is None:
                    unresolved += 1
                    attacks.append(attack)
                    continue

                resolution_index, resolution_logs = resolved_entry
                resolved += 1
                after_index = resolution_index + 1
                after_step = steps[after_index] if after_index < len(steps) else None
                after_prize = (
                    _prize_count(after_step, physical_index) if after_step is not None else None
                )
                prize_delta = (
                    before_prize - after_prize
                    if before_prize is not None and after_prize is not None
                    else None
                )
                if prize_delta is None:
                    unknown_prize += 1
                else:
                    no_prize = prize_delta <= 0
                    non_prize += int(no_prize)
                if powerful:
                    powerful_resolved += 1
                    powerful_unknown_prize += int(prize_delta is None)
                    powerful_non_prize += int(prize_delta is not None and prize_delta <= 0)
                attack.update(
                    {
                        "resolved": True,
                        "non_prize": (
                            None if prize_delta is None else prize_delta <= 0
                        ),
                        "prize_delta": prize_delta,
                        "damage_observed": any(
                            as_int(log.get("type")) == DAMAGE_LOG_TYPE
                            and as_int(log.get("playerIndex")) != physical_index
                            for log in resolution_logs
                        ),
                    }
                )
                attacks.append(attack)
                source = normalized_evidence(
                    trace,
                    after_step or steps[resolution_index],
                    fallback_step=after_index if after_step is not None else resolution_index,
                    candidate_physical_index=context.candidate_physical_index,
                )
                source.update(attack)
                evidence.append(source)

        denominator = resolved - unknown_prize
        payload = {
            "games": 1,
            "error_games": int(lifecycle == "error"),
            "unfinished_games": int(lifecycle == "unfinished"),
            "lifecycle_status": lifecycle,
            "attack_submissions": submissions,
            "resolved_attacks": resolved,
            "unresolved_attacks": unresolved,
            "unknown_prize_attacks": unknown_prize,
            "non_prize_attacks": non_prize,
            "powerful_hand": {
                "resolved_attacks": powerful_resolved,
                "unknown_prize_attacks": powerful_unknown_prize,
                "non_prize_attacks": powerful_non_prize,
            },
            "attacks": attacks,
        }
        if not evidence:
            source_step = steps[-1] if steps else None
            evidence.append(
                normalized_evidence(
                    trace,
                    source_step,
                    fallback_step=len(steps) - 1 if steps else 0,
                    candidate_physical_index=context.candidate_physical_index,
                )
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
            numerator=non_prize,
            denominator=denominator,
            value=non_prize / denominator if denominator else None,
            evidence=tuple(evidence),
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
        values = list(results)
        payload = _aggregate_payload(values)
        denominator = int(payload["resolved_attacks"]) - int(payload["unknown_prize_attacks"])
        numerator = int(payload["non_prize_attacks"]["numerator"])
        by_turn_order = {
            turn_order: _aggregate_payload(
                result
                for result in values
                if result.diagnostics
                and isinstance(result.diagnostics[0], dict)
                and result.diagnostics[0].get("turn_order") == turn_order
            )
            for turn_order in ("first", "second")
        }
        payload["by_turn_order"] = by_turn_order
        return AggregateMetric(
            metric_id=self.metric_id,
            numerator=numerator,
            denominator=denominator,
            value=numerator / denominator if denominator else None,
            by_opponent=_by_opponent(values),
            payload=payload,
        )


def _aggregate_payload(results: Iterable[GameMetric]) -> dict[str, object]:
    values = list(results)
    submissions = sum(int(result.payload.get("attack_submissions", 0)) for result in values)
    resolved = sum(int(result.payload.get("resolved_attacks", 0)) for result in values)
    unresolved = sum(int(result.payload.get("unresolved_attacks", 0)) for result in values)
    unknown = sum(int(result.payload.get("unknown_prize_attacks", 0)) for result in values)
    non_prize = sum(int(result.payload.get("non_prize_attacks", 0)) for result in values)
    powerful = {
        key: sum(
            int(result.payload.get("powerful_hand", {}).get(key, 0))
            for result in values
        )
        for key in ("resolved_attacks", "unknown_prize_attacks", "non_prize_attacks")
    }
    error_games = sum(
        int(result.payload.get("error_games", result.status == "error"))
        for result in values
    )
    unfinished_games = sum(
        int(result.payload.get("unfinished_games", result.status == "unfinished"))
        for result in values
    )
    return {
        "games": len(values),
        "error_games": error_games,
        "unfinished_games": unfinished_games,
        "lifecycle_status_counts": {
            "finished": len(values) - error_games - unfinished_games,
            "unfinished": unfinished_games,
            "error": error_games,
        },
        "attack_submissions": submissions,
        "resolved_attacks": resolved,
        "unresolved_attacks": unresolved,
        "unknown_prize_attacks": unknown,
        "non_prize_attacks": {
            "numerator": non_prize,
            "denominator": resolved - unknown,
            "rate": non_prize / (resolved - unknown) if resolved - unknown else None,
        },
        "powerful_hand": powerful,
        "attacks": [
            attack
            for result in values
            for attack in result.payload.get("attacks", [])
            if isinstance(attack, dict)
        ],
    }


def _by_opponent(results: Iterable[GameMetric]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[GameMetric]] = defaultdict(list)
    for result in results:
        diagnostic = result.diagnostics[0] if result.diagnostics else {}
        opponent = str(diagnostic.get("opponent", "unknown")) if isinstance(diagnostic, dict) else "unknown"
        grouped[opponent].append(result)
    return {opponent: _aggregate_payload(values) for opponent, values in grouped.items()}
