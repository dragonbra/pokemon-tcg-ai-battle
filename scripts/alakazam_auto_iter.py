#!/usr/bin/env python3
"""Analyze Alakazam evaluator traces and compare AutoIter candidates.

This module deliberately does not choose simulator actions. It turns the
adjacent evaluator's JSON output into rule-aware metrics and compact cases so
strategy changes can be reviewed one hypothesis at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable


ALAKAZAM = 743
KADABRA = 742
ABRA = 741
DUDUNSPARCE = 66
POWERFUL_HAND = 1072
PSYCHIC_ENERGY = 5
ATTACK_LINE = {ABRA, KADABRA, ALAKAZAM}
READY_ATTACKERS = {KADABRA, ALAKAZAM}

# Keep this in sync with ptcg-agent-kaggle/eval/matchup_test.py. Unknown
# opponents intentionally use the evaluator's 0.05 fallback weight.
META_WEIGHTS = {
    "romanrozen_v9": 0.10,
    "pilkwang_v2": 0.08,
    "kokinn_search": 0.06,
    "penguin_915": 0.06,
    "crustle_wall": 0.08,
    "crustle_v1": 0.05,
    "kiyotah_lucario": 0.08,
    "kiyotah_dragapult": 0.06,
    "kiyotah_iono": 0.04,
    "kiyotah_abomasnow": 0.04,
    "kacchan_anti_wall": 0.06,
    "nursrijan_lucario": 0.05,
    "yakitori_raging_bolt": 0.03,
    "zoli_dragapult": 0.04,
    "sue_alakazam": 0.05,
    "maktha_1084": 0.08,
}


@dataclass(frozen=True)
class EvaluationMetrics:
    games: int
    wins: int
    losses: int
    draws: int
    errors: int
    win_rate: float
    meta_weighted_win_rate: float
    second_turn_powerful_hand_games: int
    second_turn_powerful_hand_rate: float
    post_ko_count: int
    post_ko_zero_ready_count: int
    post_ko_zero_ready_event_rate: float
    games_with_post_ko_break: int
    games_with_post_ko_break_rate: float
    empty_bench_run_away_draw_count: int
    first_alakazam_turns: tuple[int, ...]


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    source: dict[str, Any]
    failure_class: str
    state_summary: dict[str, Any]
    legal_options: list[dict[str, Any]]
    actual_action: list[int]
    expected_action: list[int]
    expected_reason: str
    case_status: str


@dataclass(frozen=True)
class AnalysisResult:
    metrics: EvaluationMetrics
    cases: tuple[CaseRecord, ...]
    source_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromotionDecision:
    status: str
    reasons: tuple[str, ...]
    regressions: tuple[str, ...]


def _as_int(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _trace_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    trace = record.get("trace")
    return trace if isinstance(trace, list) else []


def _infer_agent_label(records: Iterable[dict[str, Any]], explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for record in records:
        label = record.get("label")
        if isinstance(label, str) and label:
            return label
        for step in _trace_for_record(record):
            role = step.get("role")
            if isinstance(role, str) and role not in {"opponent", "finished"}:
                return role
    return None


def _observation(step: dict[str, Any]) -> dict[str, Any]:
    observation = step.get("observation")
    return observation if isinstance(observation, dict) else {}


def _current(step: dict[str, Any]) -> dict[str, Any]:
    current = _observation(step).get("current")
    if isinstance(current, dict):
        return current
    return {
        "turn": step.get("turn", 0),
        "yourIndex": step.get("yourIndex", 0),
        "players": step.get("players") or [],
    }


def _players(step: dict[str, Any]) -> list[dict[str, Any]]:
    players = _current(step).get("players")
    return players if isinstance(players, list) else []


def _player_at(step: dict[str, Any], index: int) -> dict[str, Any]:
    players = _players(step)
    if 0 <= index < len(players) and isinstance(players[index], dict):
        return players[index]
    return {}


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    cards = player.get(area) or []
    return [card for card in cards if isinstance(card, dict)]


def _active(player: dict[str, Any]) -> dict[str, Any] | None:
    cards = _cards(player, "active")
    return cards[0] if cards else None


def _bench(player: dict[str, Any]) -> list[dict[str, Any]]:
    return _cards(player, "bench")


def _option_list(step: dict[str, Any]) -> list[dict[str, Any]]:
    select = step.get("select")
    if not isinstance(select, dict):
        select = _observation(step).get("select")
    if not isinstance(select, dict):
        return []
    options = select.get("options")
    if options is None:
        options = select.get("option")
    return [option for option in options or [] if isinstance(option, dict)]


def _action_list(step: dict[str, Any]) -> list[int]:
    action = step.get("action")
    if not isinstance(action, list):
        return []
    return [value for value in action if isinstance(value, int)]


def _selected_options(step: dict[str, Any]) -> list[dict[str, Any]]:
    options = _option_list(step)
    return [
        options[index]
        for index in _action_list(step)
        if 0 <= index < len(options)
    ]


def _option_type(option: dict[str, Any]) -> int | str | None:
    value = option.get("type")
    if isinstance(value, (int, str)):
        return value
    return None


def _option_card_id(option: dict[str, Any]) -> int | None:
    for key in ("cardId", "card_id", "id"):
        value = _as_int(option.get(key))
        if value is not None:
            return value
    return None


def _option_card_id_in_step(
    option: dict[str, Any], step: dict[str, Any], player_index: int
) -> int | None:
    """Resolve field-target options whose card id is implicit in the area."""
    card_id = _option_card_id(option)
    if card_id is not None:
        return card_id
    area = _as_int(option.get("area", option.get("inPlayArea")))
    if area not in {4, 5}:
        return None
    owner = _as_int(option.get("playerIndex"), player_index)
    index = next(
        (
            _as_int(option.get(key))
            for key in ("indexInArea", "inPlayIndex")
            if _as_int(option.get(key)) is not None
        ),
        None,
    )
    if index is None:
        return None
    player = _player_at(step, owner if owner is not None else player_index)
    cards = _cards(player, "active" if area == 4 else "bench")
    return _as_int(cards[index].get("id")) if 0 <= index < len(cards) else None


def _option_attack_id(option: dict[str, Any]) -> int | None:
    return _as_int(option.get("attackId", option.get("attack_id")))


def _selected_attack_id(step: dict[str, Any]) -> int | None:
    for option in _selected_options(step):
        attack_id = _option_attack_id(option)
        if attack_id is not None:
            return attack_id
    return None


def _logs(step: dict[str, Any]) -> list[dict[str, Any]]:
    logs = _observation(step).get("logs")
    return logs if isinstance(logs, list) else []


def _field_state(step: dict[str, Any], player_index: int) -> dict[int, dict[str, Any]]:
    """Return the last visible field snapshot keyed by Pokémon serial."""
    player = _player_at(step, player_index)
    state: dict[int, dict[str, Any]] = {}
    for pokemon in [_active(player), *_bench(player)]:
        if not pokemon:
            continue
        serial = _as_int(pokemon.get("serial"))
        if serial is not None:
            state[serial] = pokemon
    return state


def _damage_counter_serials(step: dict[str, Any], player_index: int) -> set[int]:
    """Return Pokémon that received a real damage counter in this observation."""
    return {
        serial
        for log in _logs(step)
        if _as_int(log.get("type")) == 16
        and _as_int(log.get("playerIndex")) == player_index
        and log.get("putDamageCounter") is True
        for serial in [_as_int(log.get("serial"))]
        if serial is not None
    }


def _knockout_is_confirmed(
    step: dict[str, Any],
    log: dict[str, Any],
    player_index: int,
    previous_field: dict[int, dict[str, Any]],
) -> bool:
    """Require damage or an observed zero-HP state before calling a move KO."""
    serial = _as_int(log.get("serial"))
    if serial is None:
        return False
    if serial in _damage_counter_serials(step, player_index):
        return True
    previous = previous_field.get(serial)
    previous_hp = _as_int(previous.get("hp")) if previous is not None else None
    if previous_hp is not None and previous_hp <= 0:
        return True
    current = _field_state(step, player_index).get(serial)
    current_hp = _as_int(current.get("hp")) if current is not None else None
    return current_hp is not None and current_hp <= 0


def _agent_index(record: dict[str, Any]) -> int:
    value = _as_int(record.get("alakazamPhysicalIndex"), 0)
    return value if value in (0, 1) else 0


def _agent_turn_target(record: dict[str, Any]) -> int | None:
    agent_index = _agent_index(record)
    for step in _trace_for_record(record):
        first_player = _as_int(_current(step).get("firstPlayer"))
        if first_player in (0, 1):
            return 3 if first_player == agent_index else 4
    return None


def _is_agent_step(step: dict[str, Any], agent_label: str | None) -> bool:
    if agent_label is None:
        return step.get("role") not in {"opponent", "finished"}
    return step.get("role") == agent_label


def _pokemon_summary(card: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(card, dict):
        return None
    return {
        "id": card.get("id"),
        "serial": card.get("serial"),
        "hp": card.get("hp"),
        "maxHp": card.get("maxHp"),
        "energies": list(card.get("energies") or []),
        "energy_count": len(card.get("energies") or card.get("energyCards") or []),
    }


def _state_summary(player: dict[str, Any]) -> dict[str, Any]:
    return {
        "active": _pokemon_summary(_active(player)),
        "bench": [_pokemon_summary(card) for card in _bench(player)],
        "hand_count": _as_int(player.get("handCount"), len(player.get("hand") or [])) or 0,
        "deck_count": _as_int(player.get("deckCount"), 0) or 0,
        "prize_count": len(player.get("prize") or []),
    }


def _ready_attacker_count(player: dict[str, Any]) -> int:
    count = 0
    for pokemon in [_active(player), *_bench(player)]:
        if pokemon and _as_int(pokemon.get("id")) in READY_ATTACKERS:
            energies = pokemon.get("energies") or []
            if PSYCHIC_ENERGY in energies:
                count += 1
    return count


def _attack_line_state(player: dict[str, Any]) -> dict[str, int]:
    field = [_active(player), *_bench(player)]
    return {
        "abra_count": sum(_as_int(card.get("id")) == ABRA for card in field if card),
        "kadabra_count": sum(_as_int(card.get("id")) == KADABRA for card in field if card),
        "alakazam_count": sum(_as_int(card.get("id")) == ALAKAZAM for card in field if card),
        "ready_attacker_count": _ready_attacker_count(player),
    }


def _case_source(record: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    return {
        "opponent": record.get("opponent", "unknown"),
        "game": record.get("game"),
        "turn": _as_int(step.get("turn"), _as_int(_current(step).get("turn"), 0)) or 0,
    }


def _case(
    record: dict[str, Any],
    step: dict[str, Any],
    failure_class: str,
    reason: str,
    status: str = "diagnostic",
) -> CaseRecord:
    source = _case_source(record, step)
    case_id = f"{source['opponent']}-game-{source['game']}-turn-{source['turn']}-{failure_class}"
    agent_player = _player_at(step, _agent_index(record))
    return CaseRecord(
        case_id=case_id,
        source=source,
        failure_class=failure_class,
        state_summary={
            **_state_summary(agent_player),
            "attack_line": _attack_line_state(agent_player),
        },
        legal_options=_option_list(step),
        actual_action=_action_list(step),
        expected_action=[],
        expected_reason=reason,
        case_status=status,
    )


def _record_win(record: dict[str, Any]) -> bool:
    return _as_int(record.get("winner"), -1) == 0


def _weighted_win_rate(records: list[dict[str, Any]]) -> float:
    wins_by_opponent: Counter[str] = Counter()
    games_by_opponent: Counter[str] = Counter()
    for record in records:
        opponent = str(record.get("opponent", "unknown"))
        games_by_opponent[opponent] += 1
        if _record_win(record):
            wins_by_opponent[opponent] += 1
    weighted_sum = 0.0
    total_weight = 0.0
    for opponent, games in games_by_opponent.items():
        weight = META_WEIGHTS.get(opponent, 0.05)
        weighted_sum += (wins_by_opponent[opponent] / games) * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight else 0.0


def analyze_records(
    records: list[dict[str, Any]], agent_label: str | None = None
) -> AnalysisResult:
    """Analyze already-loaded evaluator records without touching the engine."""
    label = _infer_agent_label(records, agent_label)
    wins = sum(_record_win(record) for record in records)
    losses = sum(_as_int(record.get("winner"), -1) == 1 for record in records)
    draws = len(records) - wins - losses
    errors = sum(bool(record.get("error")) for record in records)
    powerful_games = 0
    post_ko_count = 0
    post_ko_zero_ready_count = 0
    games_with_post_ko_break = 0
    empty_bench_draws = 0
    first_alakazam_turns: list[int] = []
    cases: list[CaseRecord] = []

    for record in records:
        trace = _trace_for_record(record)
        target_turn = _agent_turn_target(record)
        second_turn_seen = False
        second_turn_step: dict[str, Any] | None = None
        seen_knockouts: set[tuple[Any, Any]] = set()
        game_has_break = False
        first_alakazam: int | None = None
        agent_index = _agent_index(record)
        previous_field: dict[int, dict[str, Any]] = {}

        for step in trace:
            player = _player_at(step, agent_index)
            if first_alakazam is None and any(
                _as_int(card.get("id")) == ALAKAZAM
                for card in [_active(player), *_bench(player)]
                if card
            ):
                first_alakazam = _as_int(step.get("turn"), 0) or 0

            if not _is_agent_step(step, label):
                # KO logs are attached to the first observation after the
                # opponent action, even if the next selection belongs to us.
                pass

            if _is_agent_step(step, label):
                selected = _selected_options(step)
                attack_id = _selected_attack_id(step)
                is_second_turn = (
                    target_turn in (3, 4) and _as_int(step.get("turn")) == target_turn
                ) or (
                    target_turn is None and _as_int(step.get("turn")) in (3, 4)
                )
                if is_second_turn:
                    second_turn_step = step
                    if attack_id == POWERFUL_HAND and not second_turn_seen:
                        second_turn_seen = True
                        powerful_games += 1

                active = _active(player)
                if (
                    active
                    and _as_int(active.get("id")) == DUDUNSPARCE
                    and not _bench(player)
                    and any(
                        _option_type(option) == 10
                        and _option_card_id_in_step(option, step, agent_index)
                        == DUDUNSPARCE
                        for option in selected
                    )
                ):
                    empty_bench_draws += 1
                    cases.append(
                        _case(
                            record,
                            step,
                            "empty_bench_run_away_draw",
                            "Active Dudunsparce 在空 Bench 时不应选择 Run Away Draw",
                            status="fail",
                        )
                    )

            for log in _logs(step):
                if not isinstance(log, dict):
                    continue
                if (
                    _as_int(log.get("type")) != 6
                    or _as_int(log.get("playerIndex")) != agent_index
                    or _as_int(log.get("fromArea")) not in {4, 5}
                    or _as_int(log.get("toArea")) != 3
                    or _as_int(log.get("cardId")) not in ATTACK_LINE
                    or not _knockout_is_confirmed(step, log, agent_index, previous_field)
                ):
                    continue
                key = (log.get("serial"), log.get("toArea"))
                if key in seen_knockouts:
                    continue
                seen_knockouts.add(key)
                post_ko_count += 1
                ready_count = _ready_attacker_count(player)
                if ready_count == 0:
                    post_ko_zero_ready_count += 1
                    game_has_break = True
                    cases.append(
                        _case(
                            record,
                            step,
                            "post_ko_no_ready_attacker",
                            "我方 Pokémon 被击倒后没有可立即接班的 Kadabra/Alakazam",
                        )
                    )

            previous_field = _field_state(step, agent_index)

        if first_alakazam is not None:
            first_alakazam_turns.append(first_alakazam)
        if game_has_break:
            games_with_post_ko_break += 1
        if not second_turn_seen:
            cases.append(
                _case(
                    record,
                    second_turn_step or (trace[-1] if trace else {}),
                    "second_turn_powerful_hand_missing",
                    "本局在目标第二回合没有实际选择 Alakazam 的 Powerful Hand；需要结合 trace 判断是资源不可得还是策略顺序错误",
                )
            )

    games = len(records)
    return AnalysisResult(
        metrics=EvaluationMetrics(
            games=games,
            wins=wins,
            losses=losses,
            draws=draws,
            errors=errors,
            win_rate=wins / games if games else 0.0,
            meta_weighted_win_rate=_weighted_win_rate(records),
            second_turn_powerful_hand_games=powerful_games,
            second_turn_powerful_hand_rate=powerful_games / games if games else 0.0,
            post_ko_count=post_ko_count,
            post_ko_zero_ready_count=post_ko_zero_ready_count,
            post_ko_zero_ready_event_rate=(
                post_ko_zero_ready_count / post_ko_count if post_ko_count else 0.0
            ),
            games_with_post_ko_break=games_with_post_ko_break,
            games_with_post_ko_break_rate=(
                games_with_post_ko_break / games if games else 0.0
            ),
            empty_bench_run_away_draw_count=empty_bench_draws,
            first_alakazam_turns=tuple(first_alakazam_turns),
        ),
        cases=tuple(cases),
    )


def _iter_game_files(report_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in report_dir.rglob("game_*.json")
        if path.is_file() and path.name != "summary.json"
    )


def analyze_report(report_dir: Path, agent_label: str | None = None) -> AnalysisResult:
    """Load summary/game JSON files recursively and analyze their full traces."""
    report_dir = report_dir.resolve()
    files = _iter_game_files(report_dir)
    records: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            records.append(payload)

    if not records:
        summary_path = report_dir / "summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            records = payload.get("results") or []
            if isinstance(payload.get("label"), str) and agent_label is None:
                agent_label = payload["label"]

    result = analyze_records(records, agent_label=agent_label)
    return replace(
        result,
        source_files=tuple(str(path.relative_to(report_dir)) for path in files),
    )


def _percentage(value: float) -> str:
    return f"{value:.1%}"


def _metrics_payload(result: AnalysisResult) -> dict[str, Any]:
    return {
        "metrics": asdict(result.metrics),
        "source_files": list(result.source_files),
        "case_count": len(result.cases),
    }


def _render_analysis(result: AnalysisResult, manifest: dict[str, Any]) -> str:
    metrics = result.metrics
    by_failure = Counter(case.failure_class for case in result.cases)
    lines = [
        "# Alakazam AutoIter 分析",
        "",
        "## 样本与结果",
        "",
        f"- 实际读取 trace 文件：{len(result.source_files)}",
        f"- 对局：{metrics.games}；胜 / 负 / 平：{metrics.wins} / {metrics.losses} / {metrics.draws}",
        f"- 我方错误：{metrics.errors}",
        f"- 胜率：{_percentage(metrics.win_rate)}",
        f"- Meta 加权胜率：{_percentage(metrics.meta_weighted_win_rate)}",
        f"- 第二回合 Powerful Hand：{metrics.second_turn_powerful_hand_games}/{metrics.games} ({_percentage(metrics.second_turn_powerful_hand_rate)})",
        f"- 我方被击倒事件：{metrics.post_ko_count}",
        f"- 击倒后无 ready attacker：{metrics.post_ko_zero_ready_count}/{metrics.post_ko_count} ({_percentage(metrics.post_ko_zero_ready_event_rate)})",
        f"- 出现过打手断档的对局：{metrics.games_with_post_ko_break}/{metrics.games} ({_percentage(metrics.games_with_post_ko_break_rate)})",
        f"- 空 Bench Run Away Draw：{metrics.empty_bench_run_away_draw_count}",
        "",
        "## Case 摘要",
        "",
        "| failure_class | 数量 |",
        "|---|---:|",
    ]
    for failure_class, count in sorted(by_failure.items()):
        lines.append(f"| {failure_class} | {count} |")
    if not by_failure:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            "## 评测配置",
            "",
            "```json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。",
            "",
        ]
    )
    return "\n".join(lines)


def write_analysis(
    result: AnalysisResult, output_dir: Path, manifest: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(_metrics_payload(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in result.cases:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    (output_dir / "analysis.md").write_text(
        _render_analysis(result, manifest), encoding="utf-8"
    )


def _load_metrics(path: Path) -> EvaluationMetrics:
    payload = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    values = payload.get("metrics", payload)
    values = dict(values)
    values["first_alakazam_turns"] = tuple(values.get("first_alakazam_turns") or [])
    fields = set(EvaluationMetrics.__dataclass_fields__)
    return EvaluationMetrics(**{key: value for key, value in values.items() if key in fields})


def _load_cases(path: Path) -> list[dict[str, Any]]:
    case_path = path / "cases.jsonl"
    if not case_path.exists():
        return []
    cases: list[dict[str, Any]] = []
    for line in case_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def decide_promotion(
    control: EvaluationMetrics,
    candidate: EvaluationMetrics,
    target_case_improved: bool,
    independent_pairs: list[tuple[float, float]] | None = None,
) -> PromotionDecision:
    """Apply correctness, case, signal and soft outcome gates."""
    reasons: list[str] = []
    regressions: list[str] = []
    if candidate.errors > 0:
        reasons.append("correctness")
    if candidate.empty_bench_run_away_draw_count > 0:
        reasons.append("empty_bench_run_away_draw")
    if not target_case_improved:
        reasons.append("target_case")

    if candidate.second_turn_powerful_hand_rate < control.second_turn_powerful_hand_rate:
        regressions.append("second_turn_powerful_hand")
    if candidate.post_ko_zero_ready_event_rate > control.post_ko_zero_ready_event_rate:
        regressions.append("post_ko_zero_ready")
    if candidate.meta_weighted_win_rate < control.meta_weighted_win_rate:
        regressions.append("meta_weighted_win_rate")

    repeated_win_decline = bool(independent_pairs) and all(
        candidate_rate < control_rate
        for control_rate, candidate_rate in independent_pairs or []
    )
    if repeated_win_decline:
        reasons.append("win_rate")

    if reasons and any(reason in {"correctness", "empty_bench_run_away_draw", "win_rate"} for reason in reasons):
        status = "reject"
    elif not target_case_improved or regressions:
        status = "observe"
    else:
        status = "accept"
    return PromotionDecision(status, tuple(reasons), tuple(regressions))


def compare_reports(control_dir: Path, candidate_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Write comparison.json and decision.md for one candidate/control pair."""
    control_metrics = _load_metrics(control_dir)
    candidate_metrics = _load_metrics(candidate_dir)
    control_cases = _load_cases(control_dir)
    candidate_cases = _load_cases(candidate_dir)
    control_failures = Counter(
        case.get("failure_class")
        for case in control_cases
        if case.get("case_status") == "fail"
    )
    candidate_failures = Counter(
        case.get("failure_class")
        for case in candidate_cases
        if case.get("case_status") == "fail"
    )
    target_case_improved = sum(candidate_failures.values()) < sum(control_failures.values())
    decision = decide_promotion(
        control_metrics,
        candidate_metrics,
        target_case_improved=target_case_improved,
    )
    payload = {
        "status": decision.status,
        "reasons": list(decision.reasons),
        "regressions": list(decision.regressions),
        "control": asdict(control_metrics),
        "candidate": asdict(candidate_metrics),
        "control_failure_counts": dict(control_failures),
        "candidate_failure_counts": dict(candidate_failures),
        "target_case_improved": target_case_improved,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    decision_lines = [
        "# AutoIter Candidate/Control 决策",
        "",
        f"- 状态：**{decision.status}**",
        f"- 目标 case 改善：`{target_case_improved}`",
        f"- 原因：{', '.join(decision.reasons) or 'none'}",
        f"- 回退信号：{', '.join(decision.regressions) or 'none'}",
        "",
        "本文件只代表当前两份报告的比较；完整晋级仍需要独立随机批次复核。",
        "",
    ]
    (output_dir / "decision.md").write_text("\n".join(decision_lines), encoding="utf-8")
    return payload


def build_replay_command(
    evaluator_root: Path,
    agent: Path,
    label: str,
    opponents: list[str],
    games: int,
    output: Path,
    cg_path: Path,
    save_traces: bool = True,
) -> list[str]:
    """Build, but do not execute, the external alakazam_replay.py command."""
    command = [
        sys.executable,
        str((evaluator_root / "eval" / "alakazam_replay.py").resolve()),
        "--agent",
        str(agent.resolve()),
        "--label",
        label,
        "--opponents",
        ",".join(opponents),
        "--games",
        str(games),
        "--output",
        str(output.resolve()),
        "--cg-path",
        str(cg_path.resolve()),
    ]
    if save_traces:
        command.insert(command.index("--output"), "--save-traces")
    return command


def _command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="分析已有 evaluator report")
    analyze.add_argument("--report-dir", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--agent-label", default=None)

    compare = subparsers.add_parser("compare", help="比较 control 与 candidate")
    compare.add_argument("--control", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)

    run = subparsers.add_parser("run", help="运行隔壁 evaluator 并分析结果")
    run.add_argument("--evaluator-root", type=Path, required=True)
    run.add_argument("--agent", type=Path, required=True)
    run.add_argument("--cg-path", type=Path, required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--opponents", required=True)
    run.add_argument("--games", type=int, default=10)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--no-save-traces",
        dest="save_traces",
        action="store_false",
        help="只保存 evaluator summary，避免完整 trace 占用大量空间",
    )
    run.set_defaults(save_traces=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _command_parser().parse_args(argv)
    if args.command == "analyze":
        result = analyze_report(args.report_dir, agent_label=args.agent_label)
        manifest = {
            "command": "analyze",
            "report_dir": str(args.report_dir.resolve()),
            "agent_label": args.agent_label,
        }
        write_analysis(result, args.output_dir, manifest)
        return 0
    if args.command == "compare":
        compare_reports(args.control, args.candidate, args.output_dir)
        return 0

    opponents = [item.strip() for item in args.opponents.split(",") if item.strip()]
    command = build_replay_command(
        args.evaluator_root,
        args.agent,
        args.label,
        opponents,
        args.games,
        args.output_dir,
        args.cg_path,
        save_traces=args.save_traces,
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(command, cwd=args.evaluator_root, env=env, check=True)
    result = analyze_report(args.output_dir, agent_label=args.label)
    write_analysis(
        result,
        args.output_dir,
        {
            "command": command,
            "evaluator_root": str(args.evaluator_root.resolve()),
            "agent": str(args.agent.resolve()),
            "label": args.label,
            "opponents": opponents,
            "games": args.games,
            "trace_mode": "full" if args.save_traces else "summary",
            "swap": True,
            "seed_policy": "evaluator_default_independent_randomness",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
