from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluation.cards import card_image_url, load_card_catalog


REPORT_DATA_PATTERN = re.compile(
    r'<script id="report-data" type="application/json">(.*?)</script>',
    re.DOTALL,
)
ARCHETYPE_SUFFIX = re.compile(r"_\d{2}(?:_v\d{8})?(?:_(?:bc|rl|sota))?$")
DISPLAY_SUFFIX = re.compile(r" \d{2}(?: · \d{4}-\d{2}-\d{2})?$")


def build_combat_matrix_data(catalog_path: Path, reports_root: Path) -> dict[str, object]:
    entries = _catalog_entries(catalog_path)
    names = [entry["name"] for entry in entries]
    reports = {name: _load_latest_report(reports_root / name) for name in names}
    matrix: dict[str, dict[str, dict[str, object]]] = {}
    candidate_runs = []
    total_games = 0
    total_wall_time = 0.0
    total_turns = 0
    total_turn_samples = 0
    total_errors = 0
    total_unfinished = 0

    for candidate in entries:
        name = candidate["name"]
        report_path, report = reports[name]
        manifest = _mapping(report.get("manifest"))
        summary = _mapping(report.get("summary"))
        metrics = _mapping(report.get("metrics"))
        games = _mapping_sequence(report.get("games"))
        _validate_report_identity(name, names, manifest, summary)
        by_opponent = _mapping(summary.get("by_opponent"))
        turns_by_opponent = _turns_by_opponent(metrics)
        turn_order_by_opponent = _turn_order_by_opponent(games, names)
        row: dict[str, dict[str, object]] = {}
        for opponent in names:
            outcome = _mapping(by_opponent.get(opponent))
            turns = _mapping(turns_by_opponent.get(opponent))
            turn_order = _mapping(turn_order_by_opponent.get(opponent))
            cell = {
                "games": _integer(outcome.get("games")),
                "wins": _integer(outcome.get("wins")),
                "losses": _integer(outcome.get("losses")),
                "draws": _integer(outcome.get("draws")),
                "errors": _integer(outcome.get("errors")),
                "unfinished": _integer(outcome.get("unfinished")),
                "win_rate": _number(outcome.get("win_rate")),
                "turn_numerator": _integer(turns.get("numerator")),
                "turn_denominator": _integer(turns.get("denominator")),
                "average_turns": _number(turns.get("value")),
                "first_games": _integer(turn_order.get("first_games")),
                "first_wins": _integer(turn_order.get("first_wins")),
                "first_win_rate": _number(turn_order.get("first_win_rate")),
                "second_games": _integer(turn_order.get("second_games")),
                "second_wins": _integer(turn_order.get("second_wins")),
                "second_win_rate": _number(turn_order.get("second_win_rate")),
            }
            row[opponent] = cell
            total_games += int(cell["games"])
            total_turns += int(cell["turn_numerator"])
            total_turn_samples += int(cell["turn_denominator"])
            total_errors += int(cell["errors"])
            total_unfinished += int(cell["unfinished"])
        matrix[name] = row
        wall_time = _number(manifest.get("wall_time_seconds")) or 0.0
        total_wall_time += wall_time
        first_games = sum(_integer(cell.get("first_games")) for cell in row.values())
        first_wins = sum(_integer(cell.get("first_wins")) for cell in row.values())
        second_games = sum(_integer(cell.get("second_games")) for cell in row.values())
        second_wins = sum(_integer(cell.get("second_wins")) for cell in row.values())
        candidate_runs.append(
            {
                "name": name,
                "display_name": candidate["display_name"],
                "games": _integer(summary.get("total_games")),
                "wins": _integer(summary.get("wins")),
                "losses": _integer(summary.get("losses")),
                "draws": _integer(summary.get("draws")),
                "errors": _integer(summary.get("errors")),
                "unfinished": _integer(summary.get("unfinished")),
                "win_rate": _number(summary.get("win_rate")),
                "first_games": first_games,
                "first_wins": first_wins,
                "first_win_rate": first_wins / first_games if first_games else None,
                "second_games": second_games,
                "second_wins": second_wins,
                "second_win_rate": second_wins / second_games if second_games else None,
                "wall_time_seconds": wall_time,
                "run_id": str(manifest.get("run_id", "")),
                "report_path": report_path.relative_to(reports_root.parent).as_posix(),
            }
        )

    archetypes = _archetypes(entries)
    archetype_matrix = _aggregate_archetypes(matrix, archetypes)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "packages": len(entries),
            "archetypes": len(archetypes),
            "games_per_cell": 10,
            "matrix_shape": f"{len(entries)}×{len(entries)}",
            "archetype_matrix_shape": f"{len(archetypes)}×{len(archetypes)}",
            "workers": 8,
            "worker_cpu_threads": 1,
        },
        "summary": {
            "total_games": total_games,
            "total_wall_time_seconds": total_wall_time,
            "average_turns": (
                total_turns / total_turn_samples if total_turn_samples else None
            ),
            "turn_samples": total_turn_samples,
            "errors": total_errors,
            "unfinished": total_unfinished,
        },
        "packages": entries,
        "archetypes": archetypes,
        "candidate_runs": candidate_runs,
        "package_matrix": matrix,
        "archetype_matrix": archetype_matrix,
    }


def build_frozen_combat_matrix_data(
    catalog_path: Path, reports_root: Path, *, pool_id: str
) -> dict[str, object]:
    """Build a full two-perspective matrix from one sampled unordered matchup."""
    entries = _catalog_entries(catalog_path)
    names = [entry["name"] for entry in entries]
    matrix: dict[str, dict[str, dict[str, object]]] = {name: {} for name in names}
    report_paths: dict[str, Path] = {}
    wall_times: dict[str, float] = {}
    physical_games = physical_turns = physical_turn_samples = 0
    physical_errors = physical_unfinished = 0

    for index, candidate in enumerate(entries):
        name = str(candidate["name"])
        report_path, report = _load_latest_report(reports_root / name)
        report_paths[name] = report_path
        manifest = _mapping(report.get("manifest"))
        summary = _mapping(report.get("summary"))
        metrics = _mapping(report.get("metrics"))
        games = _mapping_sequence(report.get("games"))
        expected_opponents = names[index:]
        _validate_frozen_unordered_report(
            name, expected_opponents, pool_id, manifest, summary
        )
        wall_times[name] = _number(
            _mapping(summary.get("performance")).get("wall_time_seconds")
        ) or 0.0
        by_opponent = _mapping(summary.get("by_opponent"))
        turns_by_opponent = _turns_by_opponent(metrics)
        turn_order = _turn_order_by_opponent(games, expected_opponents)
        for opponent in expected_opponents:
            cell = _matrix_cell(
                _mapping(by_opponent.get(opponent)),
                _mapping(turns_by_opponent.get(opponent)),
                _mapping(turn_order.get(opponent)),
            )
            matrix[name][opponent] = cell
            if opponent != name:
                matrix[opponent][name] = _mirror_cell(cell)
            physical_games += int(cell["games"])
            physical_turns += int(cell["turn_numerator"])
            physical_turn_samples += int(cell["turn_denominator"])
            physical_errors += int(cell["errors"])
            physical_unfinished += int(cell["unfinished"])

    candidate_runs = []
    for candidate in entries:
        name = str(candidate["name"])
        row = matrix[name]
        games = sum(_integer(cell.get("games")) for cell in row.values())
        wins = sum(_integer(cell.get("wins")) for cell in row.values())
        losses = sum(_integer(cell.get("losses")) for cell in row.values())
        draws = sum(_integer(cell.get("draws")) for cell in row.values())
        first_games = sum(_integer(cell.get("first_games")) for cell in row.values())
        first_wins = sum(_integer(cell.get("first_wins")) for cell in row.values())
        second_games = sum(_integer(cell.get("second_games")) for cell in row.values())
        second_wins = sum(_integer(cell.get("second_wins")) for cell in row.values())
        candidate_runs.append({
            "name": name,
            "display_name": candidate["display_name"],
            "games": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "errors": sum(_integer(cell.get("errors")) for cell in row.values()),
            "unfinished": sum(_integer(cell.get("unfinished")) for cell in row.values()),
            "win_rate": wins / games if games else None,
            "first_games": first_games,
            "first_wins": first_wins,
            "first_win_rate": first_wins / first_games if first_games else None,
            "second_games": second_games,
            "second_wins": second_wins,
            "second_win_rate": second_wins / second_games if second_games else None,
            "wall_time_seconds": wall_times[name],
            "run_id": "unordered-frozen",
            "report_path": report_paths[name].relative_to(reports_root).as_posix(),
        })

    archetypes = _archetypes(entries)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "pool_kind": "frozen",
            "pool_id": pool_id,
            "packages": len(entries),
            "archetypes": len(archetypes),
            "games_per_cell": 10,
            "matrix_shape": f"{len(entries)}×{len(entries)}",
            "sampling": "one unordered matchup projected to both player perspectives",
            "physical_games": physical_games,
            "report_href_prefix": f"../../reports_frozen/{pool_id}",
        },
        "summary": {
            "total_games": physical_games,
            "projected_directional_games": len(entries) * len(entries) * 10,
            "total_wall_time_seconds": sum(wall_times.values()),
            "average_turns": physical_turns / physical_turn_samples if physical_turn_samples else None,
            "turn_samples": physical_turn_samples,
            "errors": physical_errors,
            "unfinished": physical_unfinished,
        },
        "packages": entries,
        "archetypes": archetypes,
        "candidate_runs": candidate_runs,
        "package_matrix": matrix,
        "archetype_matrix": _aggregate_archetypes(matrix, archetypes),
    }


def _matrix_cell(outcome: Mapping[str, object], turns: Mapping[str, object], turn_order: Mapping[str, object]) -> dict[str, object]:
    return {
        "games": _integer(outcome.get("games")), "wins": _integer(outcome.get("wins")),
        "losses": _integer(outcome.get("losses")), "draws": _integer(outcome.get("draws")),
        "errors": _integer(outcome.get("errors")), "unfinished": _integer(outcome.get("unfinished")),
        "win_rate": _number(outcome.get("win_rate")),
        "turn_numerator": _integer(turns.get("numerator")), "turn_denominator": _integer(turns.get("denominator")),
        "average_turns": _number(turns.get("value")),
        "first_games": _integer(turn_order.get("first_games")), "first_wins": _integer(turn_order.get("first_wins")),
        "first_win_rate": _number(turn_order.get("first_win_rate")),
        "second_games": _integer(turn_order.get("second_games")), "second_wins": _integer(turn_order.get("second_wins")),
        "second_win_rate": _number(turn_order.get("second_win_rate")),
    }


def _mirror_cell(cell: Mapping[str, object]) -> dict[str, object]:
    first_games = _integer(cell.get("second_games"))
    second_games = _integer(cell.get("first_games"))
    first_wins = first_games - _integer(cell.get("second_wins"))
    second_wins = second_games - _integer(cell.get("first_wins"))
    games = _integer(cell.get("games"))
    wins = _integer(cell.get("losses"))
    return {
        **cell, "wins": wins, "losses": _integer(cell.get("wins")),
        "win_rate": wins / games if games else None,
        "first_games": first_games, "first_wins": first_wins,
        "first_win_rate": first_wins / first_games if first_games else None,
        "second_games": second_games, "second_wins": second_wins,
        "second_win_rate": second_wins / second_games if second_games else None,
    }


def _validate_frozen_unordered_report(candidate_name: str, opponent_names: list[str], pool_id: str, manifest: Mapping[str, object], summary: Mapping[str, object]) -> None:
    candidate = _mapping(manifest.get("candidate"))
    actual_pool = _mapping(manifest.get("opponent_pool")).get("pool_id")
    actual_opponents = [str(item.get("name")) for item in _mapping_sequence(manifest.get("opponents"))]
    if candidate.get("name") != candidate_name or actual_pool != pool_id:
        raise ValueError(f"Frozen identity mismatch for {candidate_name}")
    if actual_opponents != opponent_names:
        raise ValueError(f"Frozen opponent suffix mismatch for {candidate_name}")
    if _integer(summary.get("total_games")) != len(opponent_names) * 10:
        raise ValueError(f"incomplete Frozen game count for {candidate_name}")
    if _integer(summary.get("errors")) or _integer(summary.get("unfinished")):
        raise ValueError(f"invalid Frozen games for {candidate_name}")


def write_combat_matrix(
    data: Mapping[str, object], output: Path, source_data_path: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    source_data_path.parent.mkdir(parents=True, exist_ok=True)
    if _mapping(data.get("protocol")).get("pool_kind") == "frozen":
        data = json.loads(json.dumps(data, ensure_ascii=False))
        _write_frozen_deck_reports(data, output.parent)
    source_data_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output.write_text(render_combat_matrix(data, output.parent), encoding="utf-8")


def _write_frozen_deck_reports(data: dict[str, object], output_root: Path) -> None:
    packages = _mapping_sequence(data.get("packages"))
    names = [str(package["name"]) for package in packages]
    package_by_name = {str(package["name"]): package for package in packages}
    matrix = _mapping(data.get("package_matrix"))
    protocol = _mapping(data.get("protocol"))
    prefix = str(protocol.get("report_href_prefix", ""))
    runs = _mapping_sequence(data.get("candidate_runs"))
    raw_reports = {
        str(run["name"]): (Path(prefix) / str(run["report_path"])).as_posix()
        for run in runs
    }
    details_root = output_root / "decks"
    details_root.mkdir(parents=True, exist_ok=True)
    for run in runs:
        name = str(run["name"])
        detail_path = details_root / f"{name}.html"
        detail_path.write_text(
            _render_frozen_deck_report(
                package_by_name[name], names, package_by_name,
                _mapping(matrix.get(name)), raw_reports,
            ),
            encoding="utf-8",
        )
        run["report_path"] = f"decks/{name}.html"
    protocol["report_href_prefix"] = ""


def _render_frozen_deck_report(
    package: Mapping[str, object], names: list[str],
    package_by_name: Mapping[str, Mapping[str, object]],
    row: Mapping[str, object], raw_reports: Mapping[str, str],
) -> str:
    name = str(package["name"])
    name_index = names.index(name)
    comparative = [_mapping(row.get(opponent)) for opponent in names if opponent != name]
    games = sum(_integer(cell.get("games")) for cell in comparative)
    wins = sum(_integer(cell.get("wins")) for cell in comparative)
    losses = sum(_integer(cell.get("losses")) for cell in comparative)
    first_games = sum(_integer(cell.get("first_games")) for cell in comparative)
    first_wins = sum(_integer(cell.get("first_wins")) for cell in comparative)
    second_games = sum(_integer(cell.get("second_games")) for cell in comparative)
    second_wins = sum(_integer(cell.get("second_wins")) for cell in comparative)
    table_rows = []
    for opponent_index, opponent in enumerate(names):
        opponent_package = package_by_name[opponent]
        cell = _mapping(row.get(opponent))
        self_play = opponent == name
        source = name if name_index <= opponent_index else opponent
        evidence = "../" + raw_reports[source]
        rate = _number(cell.get("win_rate"))
        rate_text = "-" if self_play else _percentage(rate)
        record = "-" if self_play else f"{cell.get('wins', 0)}–{cell.get('losses', 0)}"
        bar = 0 if self_play or rate is None else max(0.0, min(100.0, rate * 100))
        table_rows.append(
            f'<tr><td>{_identity(opponent_package)}</td><td>{record}</td>'
            f'<td><div class="match-rate"><span style="width:{bar:.1f}%"></span>'
            f'<strong>{rate_text}</strong></div></td>'
            f'<td>{"-" if self_play else _percentage(cell.get("first_win_rate"))}</td>'
            f'<td>{"-" if self_play else _percentage(cell.get("second_win_rate"))}</td>'
            f'<td>{"-" if self_play else _decimal(cell.get("average_turns"))}</td>'
            f'<td><a href="{_escape(evidence)}">Raw evidence</a></td></tr>'
        )
    title = str(package.get("display_name", name))
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape(title)} · Frozen Matchups</title><style>{_styles()}
.match-rate{{position:relative;width:150px;height:26px;overflow:hidden;border-radius:5px;background:#edf2ef}}
.match-rate span{{position:absolute;inset:0 auto 0 0;background:#66b58d}}
.match-rate strong{{position:relative;z-index:1;display:block;padding:3px 8px;text-align:right}}
.deck-nav{{margin-bottom:14px}}.deck-nav a{{font-weight:700}}.runs td{{vertical-align:middle}}
</style></head><body><main><div class="deck-nav"><a href="../index.html">← Frozen Combat Matrix</a></div>
<header class="hero"><div><p class="eyebrow">FROZEN ARENA · 50-DECK FULL ROW</p>
<h1>{_escape(title)}</h1><p>完整 50 卡组 matchup 视图；self-play 对角线显示为 -</p></div></header>
<section><h2>对阵总览</h2><div class="summary-grid">
{_summary_card('比较对局', games)}{_summary_card('胜 / 负', f'{wins} / {losses}')}
{_summary_card('比较胜率', _percentage(wins / games if games else None))}
{_summary_card('先攻胜率', _percentage(first_wins / first_games if first_games else None))}
{_summary_card('后攻胜率', _percentage(second_wins / second_games if second_games else None))}
{_summary_card('Opponent 数', 50)}</div></section>
<section><h2>对 50 套 Frozen 卡组的胜率</h2><p class="note">每个非对角 matchup 固定 10 局；Raw evidence 指向实际承载该组对局的不可变采样报告。</p>
<div class="table-scroll"><table class="runs"><thead><tr><th>Opponent</th><th>W–L</th><th>胜率</th><th>先攻</th><th>后攻</th><th>平均回合</th><th>证据</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></div></section></main></body></html>"""


def render_combat_matrix(data: Mapping[str, object], output_root: Path) -> str:
    summary = _mapping(data.get("summary"))
    protocol = _mapping(data.get("protocol"))
    packages = _mapping_sequence(data.get("packages"))
    archetypes = _mapping_sequence(data.get("archetypes"))
    package_matrix = _mapping(data.get("package_matrix"))
    archetype_matrix = _mapping(data.get("archetype_matrix"))
    embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    frozen = protocol.get("pool_kind") == "frozen"
    page_title = "Frozen Arena Combat Matrix" if frozen else "Arena Combat Matrix"
    page_note = (
        "Frozen 无序 matchup 共用一组 10 局，并镜像生成双方视角"
        if frozen else "正式 opponents 全量循环评测"
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{page_title}</title><style>{_styles()}</style></head>
<body><main>
<header class="hero"><div><p class="eyebrow">POKÉMON TCG · EVALUATION ARENA</p>
<h1>{page_title}</h1><p>{page_note}</p></div>
<div class="run-id">{_escape(data.get('generated_at', ''))}</div></header>
<section><h2>评测总览</h2><div class="summary-grid">
{_summary_card('正式卡组', protocol.get('packages'))}
{_summary_card('卡组类别', protocol.get('archetypes'))}
{_summary_card('总对局', summary.get('total_games'))}
{_summary_card('累计评测耗时', _duration(summary.get('total_wall_time_seconds')))}
{_summary_card('平均完整回合数', _decimal(summary.get('average_turns')))}
{_summary_card('错误 / 未完成', f"{summary.get('errors', 0)} / {summary.get('unfinished', 0)}")}
</div><p class="note">每个有向单元格固定 10 局，包含每个 package 对自己的 10 局；
胜率以行卡组为 candidate、列卡组为 opponent。累计耗时为各完整评测 wall time 之和。<br>
完整回合定义：先攻玩家阶段与紧随其后的后攻玩家阶段共同组成 1 个完整回合；结束回合数按
<code>ceil(engine_turn / 2)</code> 计算，不使用 action selection 次数。</p></section>
{_run_table(data, output_root)}
{_matrix_section('Package 对局胜率', 'N×N · 行 candidate 对列 opponent', packages, packages, package_matrix, 'win_rate')}
{_matrix_section('Package 先攻胜率', 'N×N · 每格 5 局，candidate 被分配为先攻', packages, packages, package_matrix, 'first_win_rate')}
{_matrix_section('Package 后攻胜率', 'N×N · 每格 5 局，candidate 被分配为后攻', packages, packages, package_matrix, 'second_win_rate')}
{_matrix_section('Archetype 对局胜率', 'M×M · 按关键宝可梦类别加权聚合', archetypes, archetypes, archetype_matrix, 'win_rate')}
{_matrix_section('Package 平均完整回合数', 'N×N · 先后手 phase 合并后的结束回合均值', packages, packages, package_matrix, 'average_turns')}
{_matrix_section('Archetype 平均完整回合数', 'M×M · 所有底层对局按样本数加权', archetypes, archetypes, archetype_matrix, 'average_turns')}
<script>{_sort_script()}</script>
<script id="combat-mat-data" type="application/json">{embedded}</script>
</main></body></html>"""


def _catalog_entries(catalog_path: Path) -> list[dict[str, object]]:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    raw_entries = payload.get("opponents")
    if not isinstance(raw_entries, list):
        raise ValueError("catalog must contain opponents")
    card_catalog = load_card_catalog(
        catalog_path.resolve().parents[2] / "data" / "official" / "EN_Card_Data.csv"
    )
    entries = []
    for raw in raw_entries:
        if not isinstance(raw, dict) or not raw.get("enabled"):
            continue
        name = str(raw["name"])
        cards = []
        for card_id in raw["representative_card_ids"]:
            metadata = card_catalog[int(card_id)]
            cards.append(
                {
                    "card_id": int(card_id),
                    "name": metadata["name"],
                    "image_url": card_image_url(
                        metadata["expansion"], metadata["collection_number"]
                    ),
                }
            )
        entries.append(
            {
                "name": name,
                "display_name": str(raw["display_name"]),
                "archetype_id": ARCHETYPE_SUFFIX.sub("", name),
                "archetype_name": DISPLAY_SUFFIX.sub("", str(raw["display_name"])),
                "representative_cards": cards,
            }
        )
    return entries


def _load_latest_report(candidate_root: Path) -> tuple[Path, dict[str, object]]:
    reports = sorted(
        candidate_root.glob("run-*/report.html"), key=lambda path: path.stat().st_mtime
    )
    if not reports:
        raise ValueError(f"missing report for {candidate_root.name}")
    path = reports[-1]
    match = REPORT_DATA_PATTERN.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"report-data is missing: {path}")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError(f"report-data must be an object: {path}")
    return path, payload


def _validate_report_identity(
    candidate_name: str,
    opponent_names: list[str],
    manifest: Mapping[str, object],
    summary: Mapping[str, object],
) -> None:
    candidate = _mapping(manifest.get("candidate"))
    if candidate.get("name") != candidate_name:
        raise ValueError(f"candidate mismatch for {candidate_name}")
    opponents = [
        str(item.get("name"))
        for item in _mapping_sequence(manifest.get("opponents"))
    ]
    if opponents != opponent_names:
        raise ValueError(f"opponent catalog mismatch for {candidate_name}")
    if _integer(summary.get("total_games")) != len(opponent_names) * 10:
        raise ValueError(f"incomplete game count for {candidate_name}")


def _turns_by_opponent(metrics: Mapping[str, object]) -> Mapping[str, object]:
    length = _mapping(metrics.get("length"))
    payload = _mapping(length.get("payload"))
    turns = _mapping(payload.get("rounds", payload.get("turns")))
    by_opponent = turns.get("by_opponent")
    if not isinstance(by_opponent, Mapping):
        raise ValueError("length metric is missing turn aggregation")
    return by_opponent


def _turn_order_by_opponent(
    games: list[Mapping[str, Any]], opponent_names: list[str]
) -> dict[str, dict[str, object]]:
    counts = {
        name: {"first_games": 0, "first_wins": 0, "second_games": 0, "second_wins": 0}
        for name in opponent_names
    }
    for game in games:
        opponent = str(game.get("opponent", ""))
        if opponent not in counts or not isinstance(game.get("candidate_first"), bool):
            continue
        prefix = "first" if game["candidate_first"] else "second"
        counts[opponent][f"{prefix}_games"] += 1
        metric_refs = _mapping(game.get("metric_refs"))
        outcome = _mapping(metric_refs.get("outcome"))
        if outcome.get("value") == "win":
            counts[opponent][f"{prefix}_wins"] += 1
    result = {}
    for opponent, values in counts.items():
        first_games = values["first_games"]
        second_games = values["second_games"]
        result[opponent] = {
            **values,
            "first_win_rate": values["first_wins"] / first_games if first_games else None,
            "second_win_rate": (
                values["second_wins"] / second_games if second_games else None
            ),
        }
    return result


def _archetypes(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for entry in entries:
        grouped.setdefault(str(entry["archetype_id"]), []).append(entry)
    return [
        {
            "name": archetype_id,
            "display_name": values[0]["archetype_name"],
            "representative_cards": values[0]["representative_cards"],
            "members": [value["name"] for value in values],
        }
        for archetype_id, values in grouped.items()
    ]


def _aggregate_archetypes(
    package_matrix: Mapping[str, object], archetypes: list[dict[str, object]]
) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for candidate_group in archetypes:
        row = {}
        for opponent_group in archetypes:
            cells = []
            for candidate in candidate_group["members"]:
                package_row = _mapping(package_matrix.get(str(candidate)))
                for opponent in opponent_group["members"]:
                    cells.append(_mapping(package_row.get(str(opponent))))
            games = sum(_integer(cell.get("games")) for cell in cells)
            wins = sum(_integer(cell.get("wins")) for cell in cells)
            turns = sum(_integer(cell.get("turn_numerator")) for cell in cells)
            turn_samples = sum(_integer(cell.get("turn_denominator")) for cell in cells)
            first_games = sum(_integer(cell.get("first_games")) for cell in cells)
            first_wins = sum(_integer(cell.get("first_wins")) for cell in cells)
            second_games = sum(_integer(cell.get("second_games")) for cell in cells)
            second_wins = sum(_integer(cell.get("second_wins")) for cell in cells)
            row[str(opponent_group["name"])] = {
                "games": games,
                "wins": wins,
                "losses": sum(_integer(cell.get("losses")) for cell in cells),
                "draws": sum(_integer(cell.get("draws")) for cell in cells),
                "errors": sum(_integer(cell.get("errors")) for cell in cells),
                "unfinished": sum(_integer(cell.get("unfinished")) for cell in cells),
                "win_rate": wins / games if games else None,
                "turn_numerator": turns,
                "turn_denominator": turn_samples,
                "average_turns": turns / turn_samples if turn_samples else None,
                "first_games": first_games,
                "first_wins": first_wins,
                "first_win_rate": first_wins / first_games if first_games else None,
                "second_games": second_games,
                "second_wins": second_wins,
                "second_win_rate": (
                    second_wins / second_games if second_games else None
                ),
            }
        result[str(candidate_group["name"])] = row
    return result


def _run_table(data: Mapping[str, object], output_root: Path) -> str:
    rows = []
    report_prefix = str(_mapping(data.get("protocol")).get("report_href_prefix", ""))
    for order, run in enumerate(_mapping_sequence(data.get("candidate_runs"))):
        report_path = Path(str(run["report_path"]))
        href = (
            os.path.relpath(report_path, output_root)
            if report_path.is_absolute()
            else (Path(report_prefix) / report_path).as_posix()
        )
        win_rate = _number(run.get("win_rate"))
        rows.append(
            f'<tr data-default-order="{order}" data-win-rate="{win_rate or 0}" '
            f'data-wall-time="{_number(run.get("wall_time_seconds")) or 0}">'
            f"<td>{_escape(run.get('display_name'))}</td>"
            f"<td>{run.get('wins', 0)}–{run.get('losses', 0)}–{run.get('draws', 0)}</td>"
            f'<td><span class="win-rate-pill" style="{_win_heat(win_rate)}">'
            f"{_percentage(win_rate)}</span></td>"
            f"<td>{_percentage(run.get('first_win_rate'))} "
            f"<span class=\"sample\">({run.get('first_wins', 0)}/{run.get('first_games', 0)})</span></td>"
            f"<td>{_percentage(run.get('second_win_rate'))} "
            f"<span class=\"sample\">({run.get('second_wins', 0)}/{run.get('second_games', 0)})</span></td>"
            f"<td>{_duration(run.get('wall_time_seconds'))}</td>"
            f'<td><a href="{_escape(href)}">Report</a></td></tr>'
        )
    return (
        '<section><div class="run-table-heading"><div><h2>每个卡组的完整评测</h2>'
        '<p class="section-note">每行对应一次面对全 catalog 的完整 run；点击按钮切换排序。</p>'
        '</div><div class="sort-controls"><span>排序</span>'
        '<button type="button" data-sort="default">默认</button>'
        '<button type="button" data-sort="win_rate">胜率</button>'
        '<button type="button" data-sort="wall_time">耗时</button></div></div>'
        '<div class="table-scroll"><table class="runs" id="candidate-runs">'
        '<thead><tr><th>卡组</th><th>W–L–D</th><th>胜率</th>'
        "<th>先攻胜率</th><th>后攻胜率</th><th>耗时</th><th>源报告</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


def _matrix_section(
    title: str,
    subtitle: str,
    rows: list[Mapping[str, object]],
    columns: list[Mapping[str, object]],
    matrix: Mapping[str, object],
    metric: str,
) -> str:
    values = [
        _number(_mapping(_mapping(matrix.get(str(row["name"]))).get(str(column["name"]))).get(metric))
        for row in rows
        for column in columns
    ]
    numeric = [value for value in values if value is not None]
    low = min(numeric) if numeric else 0.0
    high = max(numeric) if numeric else 1.0
    headers = "".join(f"<th>{_identity(column, compact=True)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        matrix_row = _mapping(matrix.get(str(row["name"])))
        for column in columns:
            cell = _mapping(matrix_row.get(str(column["name"])))
            value = _number(cell.get(metric))
            is_self_play = row["name"] == column["name"]
            if is_self_play:
                display = "-"
                style = "background:#f2f5f3;color:#7b8983"
            elif metric.endswith("win_rate"):
                display = _percentage(value)
                style = _win_heat(value)
            else:
                display = _decimal(value)
                style = _turn_heat(value, low, high)
            details = (
                f"{row['display_name']} → {column['display_name']} · "
                f"{cell.get('games', 0)} games"
            )
            cells.append(
                f'<td style="{style}" title="{_escape(details)}">{display}</td>'
            )
        body.append(f"<tr><th>{_identity(row)}</th>{''.join(cells)}</tr>")
    legend = (
        '<div class="legend"><span>低</span><i class="gradient win"></i><span>高</span></div>'
        if metric.endswith("win_rate")
        else '<div class="legend"><span>短</span><i class="gradient turns"></i><span>长</span></div>'
    )
    return (
        f"<section><div class=\"section-heading\"><div><h2>{_escape(title)}</h2>"
        f'<p class="section-note">{_escape(subtitle)}</p></div>{legend}</div>'
        '<div class="matrix-scroll"><table class="matrix"><thead><tr><th class="corner">'
        f"Candidate ↓ / Opponent →</th>{headers}</tr></thead><tbody>{''.join(body)}"
        "</tbody></table></div></section>"
    )


def _sort_script() -> str:
    return """
(() => {
  const table = document.getElementById('candidate-runs');
  if (!table) return;
  const body = table.tBodies[0];
  const buttons = document.querySelectorAll('.sort-controls button[data-sort]');
  let active = 'default';
  let direction = 1;
  const defaults = {default: 1, win_rate: -1, wall_time: 1};
  for (const button of buttons) {
    button.addEventListener('click', () => {
      const key = button.dataset.sort;
      direction = active === key ? -direction : defaults[key];
      active = key;
      const attribute = key === 'default' ? 'defaultOrder' : key === 'win_rate' ? 'winRate' : 'wallTime';
      const rows = Array.from(body.rows);
      rows.sort((left, right) => direction * (Number(left.dataset[attribute]) - Number(right.dataset[attribute])));
      for (const row of rows) body.appendChild(row);
      for (const item of buttons) item.classList.toggle('active', item === button);
    });
  }
  buttons[0]?.classList.add('active');
})();
""".strip()


def _identity(item: Mapping[str, object], compact: bool = False) -> str:
    images = "".join(
        f'<img src="{_escape(card.get("image_url"))}" alt="{_escape(card.get("name"))}" '
        'loading="lazy" onerror="this.hidden=true">'
        for card in _mapping_sequence(item.get("representative_cards"))
        if card.get("image_url")
    )
    mode = " compact" if compact else ""
    return (
        f'<div class="identity{mode}" title="{_escape(item.get("display_name"))}">'
        f'<span class="card-stack">{images}</span>'
        f'<span>{_escape(item.get("display_name"))}</span></div>'
    )


def _summary_card(label: str, value: object) -> str:
    return (
        '<div class="summary-card">'
        f'<span class="label">{_escape(label)}</span><strong>{_escape(value)}</strong></div>'
    )


def _styles() -> str:
    return """
:root{--bg:#f3f7f5;--surface:#fff;--soft:#f7faf8;--ink:#172b25;--muted:#60736c;
--line:#dce7e2;--brand:#217a58;--brand-dark:#14563d;--shadow:0 12px 32px rgba(26,71,55,.08)}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 12% 0%,rgba(58,155,112,.12),transparent 32rem),linear-gradient(180deg,#f8fbf9 0,var(--bg) 24rem);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI","PingFang SC",sans-serif}main{max-width:1680px;margin:auto;padding:36px 28px 64px}.hero{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:22px;padding:30px 32px;border-radius:18px;color:#fff;background:linear-gradient(135deg,var(--brand-dark),#23835d 68%,#3b9c71);box-shadow:0 18px 44px rgba(20,86,61,.2)}.hero h1{margin:0;font-size:34px}.hero p{margin:4px 0 0;color:#d9f0e6}.eyebrow{font-size:12px!important;font-weight:700;letter-spacing:.12em}.run-id{padding:8px 12px;border:1px solid rgba(255,255,255,.22);border-radius:999px;background:rgba(255,255,255,.1);font:12px ui-monospace,monospace}section{margin:18px 0;padding:22px;border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.97);box-shadow:var(--shadow)}h2{margin:0 0 6px;font-size:20px}.note,.section-note{margin:4px 0;color:var(--muted)}.summary-grid{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:12px;margin-top:14px}.summary-card{padding:15px 16px;border:1px solid var(--line);border-radius:11px;background:linear-gradient(180deg,#fff,var(--soft))}.summary-card .label{display:block;color:var(--muted);font-size:12px}.summary-card strong{display:block;margin-top:5px;font-size:22px}.table-scroll,.matrix-scroll{overflow:auto;margin-top:14px;border:1px solid var(--line);border-radius:10px}table{border-collapse:separate;border-spacing:0;background:#fff}.runs{width:100%}.runs th,.runs td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left}.runs th{background:var(--soft);color:#496159}.runs tr:last-child td{border-bottom:0}.sample{color:var(--muted);font-size:11px}.win-rate-pill{display:inline-block;min-width:66px;padding:4px 8px;border-radius:999px;text-align:center}.run-table-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:20px}.sort-controls{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:12px}.sort-controls button{padding:6px 10px;border:1px solid var(--line);border-radius:999px;color:var(--brand-dark);background:#fff;cursor:pointer}.sort-controls button:hover,.sort-controls button.active{border-color:#70ad91;background:#e8f5ee}a{color:var(--brand);font-weight:700;text-decoration:none}.section-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:20px}.matrix{min-width:max-content;font-size:11px}.matrix th,.matrix td{width:62px;min-width:62px;height:48px;padding:4px;border-right:1px solid rgba(220,231,226,.8);border-bottom:1px solid rgba(220,231,226,.8);text-align:center;font-variant-numeric:tabular-nums}.matrix thead th{position:sticky;top:0;z-index:3;height:128px;background:var(--soft);vertical-align:bottom}.matrix tbody th{position:sticky;left:0;z-index:2;width:210px;min-width:210px;background:var(--soft);text-align:left}.matrix .corner{left:0;z-index:4;width:210px;min-width:210px}.identity{display:flex;align-items:center;gap:7px;min-width:0}.identity.compact{width:54px;flex-direction:column;gap:3px}.identity.compact>span:last-child{width:104px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;transform:rotate(-55deg);transform-origin:50% 50%;font-size:9px}.card-stack{display:flex;flex:none;padding-left:3px}.card-stack img{width:25px;height:34px;margin-left:-3px;object-fit:cover;border:1px solid rgba(23,43,37,.2);border-radius:3px;background:#e6eee9;box-shadow:0 2px 4px rgba(23,43,37,.12)}.identity.compact .card-stack img{width:22px;height:30px}.legend{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:11px}.gradient{display:block;width:120px;height:8px;border-radius:999px}.gradient.win{background:linear-gradient(90deg,hsl(8 62% 58%),hsl(74 56% 76%),hsl(140 55% 48%))}.gradient.turns{background:linear-gradient(90deg,hsl(165 45% 94%),hsl(222 55% 45%))}@media(max-width:900px){main{padding:18px 12px 40px}.hero{align-items:flex-start;flex-direction:column;padding:24px 20px}.summary-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.section-heading,.run-table-heading{align-items:flex-start;flex-direction:column}}
"""


def _win_heat(value: float | None) -> str:
    if value is None:
        return "background:#eef2f0;color:#718078"
    hue = 8 + 132 * max(0.0, min(1.0, value))
    lightness = 91 - abs(value - 0.5) * 62
    color = "#fff" if lightness < 67 else "#173129"
    return f"background:hsl({hue:.0f} 58% {lightness:.0f}%);color:{color};font-weight:700"


def _turn_heat(value: float | None, low: float, high: float) -> str:
    if value is None:
        return "background:#eef2f0;color:#718078"
    ratio = (value - low) / (high - low) if high > low else 0.5
    lightness = 95 - max(0.0, min(1.0, ratio)) * 50
    color = "#fff" if lightness < 64 else "#173129"
    return f"background:hsl(210 48% {lightness:.0f}%);color:{color};font-weight:700"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _integer(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _number(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _escape(value: object) -> str:
    return html.escape(str(value if value is not None else "-"), quote=True)


def _percentage(value: object) -> str:
    number = _number(value)
    return f"{number * 100:.2f}%" if number is not None else "-"


def _decimal(value: object) -> str:
    number = _number(value)
    return f"{number:.1f}" if number is not None else "-"


def _duration(value: object) -> str:
    seconds = _number(value)
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    minutes, remaining = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} 分 {remaining:.1f} 秒"
    hours, remaining_minutes = divmod(int(minutes), 60)
    return f"{hours} 小时 {remaining_minutes} 分"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the formal arena combat matrix")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--frozen-pool-id")
    args = parser.parse_args(argv)
    data = (
        build_frozen_combat_matrix_data(
            args.catalog, args.reports_root, pool_id=args.frozen_pool_id
        )
        if args.frozen_pool_id
        else build_combat_matrix_data(args.catalog, args.reports_root)
    )
    write_combat_matrix(data, args.output, args.source_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
