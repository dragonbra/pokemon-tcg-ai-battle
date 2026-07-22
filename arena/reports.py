from __future__ import annotations

import html
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from .storage import ArenaStore


def build_report_snapshot(
    store: ArenaStore,
    run_id: str,
    *,
    include_demoted: bool = True,
) -> dict[str, object]:
    decks = {str(item["deck_id"]): item for item in store.load_decks()}
    events = store.load_rating_events(run_id)
    games = store.load_games(run_id)
    gaussian: dict[str, dict[str, object]] = {}
    elo: dict[str, float] = {}
    for event in events:
        after = event.get("after", {})
        if not isinstance(after, Mapping):
            continue
        engine = str(event.get("engine", ""))
        for side, player_key in (("a", "player_a"), ("b", "player_b")):
            player = str(event.get(player_key, ""))
            state = after.get(side, {})
            if not isinstance(state, Mapping):
                continue
            if engine == "official_gaussian_approx":
                gaussian[player] = {
                    "mu": float(state.get("mu", 600)),
                    "sigma": float(state.get("sigma", 200)),
                    "games": int(state.get("games", 0)),
                    "status": str(state.get("status", "active")),
                }
            elif engine == "elo_compat":
                elo[player] = float(state.get("elo", 600))
    checkpoint = store.load_latest_checkpoint(run_id)
    if checkpoint is not None and isinstance(checkpoint.get("ratings"), Mapping):
        for player, raw in checkpoint["ratings"].items():
            if not isinstance(raw, Mapping):
                continue
            player = str(player)
            gaussian[player] = {
                "mu": float(raw.get("mu", gaussian.get(player, {}).get("mu", 600))),
                "sigma": float(raw.get("sigma", gaussian.get(player, {}).get("sigma", 200))),
                "games": int(raw.get("games", gaussian.get(player, {}).get("games", 0))),
                "status": str(raw.get("status", gaussian.get(player, {}).get("status", "active"))),
            }
            elo[player] = float(raw.get("elo", elo.get(player, 600)))
    arena_decks = {
        deck_id
        for deck_id, deck in decks.items()
        if str(deck.get("status")) == "valid" or str(deck.get("role")) == "internal_reference"
    }
    all_players = arena_decks | set(gaussian) | set(elo)
    player_stats = {
        player: {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "errors": 0,
            "first_games": 0,
            "first_wins": 0,
            "second_games": 0,
            "second_wins": 0,
        }
        for player in all_players
    }
    for game in games:
        a, b = str(game.get("player_a", "")), str(game.get("player_b", ""))
        for player in (a, b):
            player_stats.setdefault(player, {"wins": 0, "losses": 0, "draws": 0, "errors": 0, "first_games": 0, "first_wins": 0, "second_games": 0, "second_wins": 0})
        if game.get("status") != "finished":
            player_stats[a]["errors"] += 1
            player_stats[b]["errors"] += 1
            continue
        winner = game.get("winner")
        if winner is None:
            player_stats[a]["draws"] += 1
            player_stats[b]["draws"] += 1
        else:
            winner = str(winner)
            loser = b if winner == a else a
            player_stats[winner]["wins"] += 1
            player_stats[loser]["losses"] += 1
        first, second = (a, b) if bool(game.get("player_a_first")) else (b, a)
        player_stats[first]["first_games"] += 1
        player_stats[second]["second_games"] += 1
        if winner == first:
            player_stats[first]["first_wins"] += 1
        elif winner == second:
            player_stats[second]["second_wins"] += 1
    ratings: list[dict[str, object]] = []
    for player in sorted(all_players):
        deck = decks.get(player, {})
        state = gaussian.get(player, {})
        status = str(state.get("status", deck.get("status", "active")))
        pokemon = deck.get("metadata", {}).get("primary_pokemon", []) if isinstance(deck.get("metadata"), Mapping) else []
        stats = player_stats.get(player, {})
        valid_games = int(stats.get("wins", 0)) + int(stats.get("losses", 0)) + int(stats.get("draws", 0))
        ratings.append(
            {
                "deck_id": player,
                "display_name": deck.get("display_name", player),
                "archetype": deck.get("archetype", "Unknown Archetype"),
                "mu": state.get("mu", 600.0),
                "sigma": state.get("sigma", 200.0),
                "elo": elo.get(player, 600.0),
                "games": state.get("games", 0),
                "status": status,
                "role": deck.get("role", "public"),
                "primary_pokemon": pokemon,
                **stats,
                "win_rate": (int(stats.get("wins", 0)) + 0.5 * int(stats.get("draws", 0))) / valid_games if valid_games else 0.0,
            }
        )
    if not include_demoted:
        quality_filter = "eligible"
    else:
        quality_filter = "all"
    visible_ids = {
        str(item["deck_id"])
        for item in ratings
        if include_demoted or str(item.get("status")) != "demoted"
    }
    matrix: dict[tuple[str, str], dict[str, object]] = {}
    wins = losses = draws = 0
    for game in games:
        if game.get("status") != "finished":
            continue
        a, b = str(game["player_a"]), str(game["player_b"])
        if a not in visible_ids or b not in visible_ids:
            continue
        key = tuple(sorted((a, b)))
        cell = matrix.setdefault(key, {"a": key[0], "b": key[1], "wins_a": 0, "wins_b": 0, "draws": 0, "games": 0})
        cell["games"] = int(cell["games"]) + 1
        winner = game.get("winner")
        if winner is None:
            cell["draws"] = int(cell["draws"]) + 1
            draws += 1
        elif str(winner) == a:
            if a == key[0]:
                cell["wins_a"] = int(cell["wins_a"]) + 1
            else:
                cell["wins_b"] = int(cell["wins_b"]) + 1
            wins += 1
        else:
            if a == key[0]:
                cell["wins_b"] = int(cell["wins_b"]) + 1
            else:
                cell["wins_a"] = int(cell["wins_a"]) + 1
            losses += 1
    total_games = wins + losses + draws
    sources = []
    for deck_id, deck in decks.items():
        metadata = deck.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        sources.append(
            {
                "deck_id": deck_id,
                "title": metadata.get("source_title", deck_id),
                "url": metadata.get("source_url", "#"),
                "author": metadata.get("author", ""),
                "votes": metadata.get("votes", ""),
                "public_score": metadata.get("public_score", ""),
                "status": deck.get("status", ""),
            }
        )
    return {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "quality_filter": quality_filter,
        "ratings": ratings,
        "pair_matrix": tuple(matrix.values()),
        "sources": sources,
        "summary": {
            "total_games": total_games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / total_games if total_games else 0.0,
        },
        "demoted": [
            {"deck_id": item["deck_id"], "display_name": item["display_name"], "mu": item["mu"]}
            for item in ratings
            if str(item.get("status")) == "demoted"
        ],
        "config": {"rating_method": "official_gaussian_approx"},
    }


def render_html(snapshot: Mapping[str, object]) -> str:
    quality_filter = str(snapshot.get("quality_filter", "all"))
    ratings = _visible_ratings(snapshot, quality_filter)
    embedded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str)
    embedded = embedded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="zh-CN"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_escape(snapshot.get('run_id', 'Arena'))} - Arena 报告</title>",
            _STYLE,
            "</head><body><main>",
            f"<h1>Kaggle 公共卡组 Arena</h1><p class=\"muted\">运行：{_escape(snapshot.get('run_id', '-'))}；视图：{_escape(quality_filter)}；更新时间：{_escape(snapshot.get('generated_at', '-'))}</p>",
            _summary_html(snapshot),
            _rating_html(ratings, snapshot),
            _matrix_html(snapshot, quality_filter),
            _source_html(snapshot),
            _demoted_html(snapshot),
            f'<script id="arena-data" type="application/json">{embedded}</script>',
            "</main></body></html>",
        )
    )


def render_markdown(snapshot: Mapping[str, object]) -> str:
    quality_filter = str(snapshot.get("quality_filter", "all"))
    ratings = _visible_ratings(snapshot, quality_filter)
    lines = [
        f"# Kaggle 公共卡组 Arena - {snapshot.get('run_id', '-')}",
        "",
        f"- 视图：{quality_filter}",
        f"- 更新时间：{snapshot.get('generated_at', '-')}",
        f"- 默认 Rating：{snapshot.get('config', {}).get('rating_method', 'official_gaussian_approx')}",
        "- 兼容 Rating：`elo_compat`",
        "",
        "## 排行榜",
        "",
        "| 排名 | 卡组 | Archetype | Gaussian mu | sigma | Elo | 对局 | 胜率 | 状态 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, rating in enumerate(ratings, start=1):
        lines.append(
            f"| {rank} | {rating.get('display_name', rating.get('deck_id', '-'))} | "
            f"{rating.get('archetype', '-')} | {rating.get('mu', '-')} | {rating.get('sigma', '-')} | "
            f"{rating.get('elo', '-')} | {rating.get('games', 0)} | {float(rating.get('win_rate', 0)):.1%} | {rating.get('status', '-')} |"
        )
    lines.extend(("", "## 卡组克制关系", ""))
    for cell in snapshot.get("pair_matrix", ()):
        lines.append(
            f"- `{cell.get('a', '-')}` vs `{cell.get('b', '-')}`："
            f"{cell.get('wins_a', 0)}-{cell.get('wins_b', 0)}-{cell.get('draws', 0)}"
            f"（{cell.get('games', 0)} 局）"
        )
    lines.extend(("", "## 来源", "", "| 卡组 | 标题 | 作者 | Votes | Public score | 状态 |", "|---|---|---|---:|---:|---|"))
    for source in snapshot.get("sources", ()):
        lines.append(
            f"| {source.get('deck_id', '-')} | [{source.get('title', '-')}]({source.get('url', '#')}) | "
            f"{source.get('author', '-')} | {source.get('votes', '-')} | {source.get('public_score', '-')} | {source.get('status', '-')} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(snapshot: Mapping[str, object], destination: Path, *, latest: bool = True) -> tuple[Path, Path]:
    run_id = str(snapshot.get("run_id", "run-unknown"))
    run_destination = destination / run_id
    run_destination.mkdir(parents=True, exist_ok=True)
    html_path = run_destination / "report.html"
    markdown_path = run_destination / "report.md"
    html_path.write_text(render_html(snapshot), encoding="utf-8")
    markdown_path.write_text(render_markdown(snapshot), encoding="utf-8")
    if latest:
        _atomic_replace(destination / "latest.html", html_path.read_text(encoding="utf-8"))
        _atomic_replace(destination / "latest.md", markdown_path.read_text(encoding="utf-8"))
    return html_path, markdown_path


def write_report_pair(snapshot: Mapping[str, object], destination: Path) -> dict[str, Path]:
    run_id = str(snapshot.get("run_id", "run-unknown"))
    run_destination = destination / run_id
    run_destination.mkdir(parents=True, exist_ok=True)
    all_snapshot = dict(snapshot) | {"quality_filter": "all"}
    eligible_snapshot = dict(snapshot) | {"quality_filter": "eligible"}
    paths = {
        "all_html": run_destination / "report-all.html",
        "all_markdown": run_destination / "report-all.md",
        "eligible_html": run_destination / "report-eligible.html",
        "eligible_markdown": run_destination / "report-eligible.md",
    }
    paths["all_html"].write_text(render_html(all_snapshot), encoding="utf-8")
    paths["all_markdown"].write_text(render_markdown(all_snapshot), encoding="utf-8")
    paths["eligible_html"].write_text(render_html(eligible_snapshot), encoding="utf-8")
    paths["eligible_markdown"].write_text(render_markdown(eligible_snapshot), encoding="utf-8")
    _atomic_replace(destination / "latest-all.html", paths["all_html"].read_text(encoding="utf-8"))
    _atomic_replace(destination / "latest-all.md", paths["all_markdown"].read_text(encoding="utf-8"))
    _atomic_replace(destination / "latest-eligible.html", paths["eligible_html"].read_text(encoding="utf-8"))
    _atomic_replace(destination / "latest-eligible.md", paths["eligible_markdown"].read_text(encoding="utf-8"))
    _atomic_replace(destination / "latest.html", paths["all_html"].read_text(encoding="utf-8"))
    _atomic_replace(destination / "latest.md", paths["all_markdown"].read_text(encoding="utf-8"))
    return paths


def _visible_ratings(snapshot: Mapping[str, object], quality_filter: str) -> list[Mapping[str, object]]:
    ratings = [item for item in snapshot.get("ratings", ()) if isinstance(item, Mapping)]
    if quality_filter != "eligible":
        return sorted(ratings, key=lambda item: float(item.get("mu", 0)), reverse=True)
    return sorted(
        (item for item in ratings if str(item.get("status", "active")) != "demoted"),
        key=lambda item: float(item.get("mu", 0)),
        reverse=True,
    )


def _summary_html(snapshot: Mapping[str, object]) -> str:
    summary = snapshot.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    cards = (
        ("总对局", summary.get("total_games", 0)),
        ("胜", summary.get("wins", 0)),
        ("负", summary.get("losses", 0)),
        ("平", summary.get("draws", 0)),
        ("胜率", f"{float(summary.get('win_rate', 0)) * 100:.2f}%"),
    )
    return '<section><h2>总体结果</h2><div class="summary">' + "".join(
        f'<div class="card"><div class="label">{_escape(label)}</div><div class="value">{_escape(value)}</div></div>'
        for label, value in cards
    ) + "</div></section>"


def _rating_html(ratings: list[Mapping[str, object]], snapshot: Mapping[str, object]) -> str:
    rows: list[str] = []
    for rank, rating in enumerate(ratings, start=1):
        deck_id = str(rating.get("deck_id", ""))
        images = []
        for pokemon in rating.get("primary_pokemon", ()):
            if not isinstance(pokemon, Mapping):
                continue
            image_url = pokemon.get("image_url")
            if image_url:
                images.append(
                    f'<img class="pokemon" loading="lazy" src="{_escape(image_url)}" alt="{_escape(pokemon.get("name", "Pokémon"))}">'
                )
        rows.append(
            f'<tr data-deck-id="{_escape(deck_id)}"><td>{rank}</td>'
            f'<td><strong>{_escape(rating.get("display_name", deck_id))}</strong><div>{"".join(images)}</div></td>'
            f'<td>{_escape(rating.get("archetype", "-"))}</td><td>{_escape(rating.get("mu", "-"))}</td>'
            f'<td>{_escape(rating.get("sigma", "-"))}</td><td>{_escape(rating.get("elo", "-"))}</td>'
            f'<td>{_escape(rating.get("games", 0))}</td><td>{float(rating.get("win_rate", 0)):.1%}</td>'
            f'<td>{_escape(rating.get("status", "-"))}</td></tr>'
        )
    method = snapshot.get("config", {}).get("rating_method", "official_gaussian_approx")
    return (
        f"<section><h2>实时排行榜</h2><p class=\"muted\">默认：{_escape(method)}；兼容结果：elo_compat。"
        "官方完整 sigma 更新公式未公开，本页展示可重放的语义近似。</p>"
        '<table><thead><tr><th>排名</th><th>卡组</th><th>Archetype</th><th>Gaussian μ</th><th>σ</th><th>Elo</th><th>对局</th><th>胜率</th><th>状态</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def _matrix_html(snapshot: Mapping[str, object], quality_filter: str) -> str:
    allowed = {str(item.get("deck_id")) for item in _visible_ratings(snapshot, quality_filter)}
    rows = []
    for cell in snapshot.get("pair_matrix", ()):
        if not isinstance(cell, Mapping):
            continue
        if quality_filter == "eligible" and (str(cell.get("a")) not in allowed or str(cell.get("b")) not in allowed):
            continue
        games = int(cell.get("games", 0) or 0)
        effective = (int(cell.get("wins_a", 0)) + 0.5 * int(cell.get("draws", 0))) / games if games else 0.5
        shade = int(235 - 120 * effective)
        rows.append(
            f"<tr><td>{_escape(cell.get('a', '-'))}</td><td>{_escape(cell.get('b', '-'))}</td>"
            f"<td>{_escape(cell.get('wins_a', 0))}-{_escape(cell.get('wins_b', 0))}-{_escape(cell.get('draws', 0))}</td>"
            f"<td style=\"background:rgb({shade},245,238)\">{_escape(games)}（A 得分率 {effective:.1%}）</td></tr>"
        )
    return '<section><h2>克制关系热力图</h2><table><thead><tr><th>卡组 A</th><th>卡组 B</th><th>W-L-D</th><th>样本</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></section>"


def _source_html(snapshot: Mapping[str, object]) -> str:
    rows = []
    for source in snapshot.get("sources", ()):
        if not isinstance(source, Mapping):
            continue
        rows.append(
            f"<tr><td>{_escape(source.get('deck_id', '-'))}</td><td><a href=\"{_escape(source.get('url', '#'))}\">{_escape(source.get('title', '-'))}</a></td>"
            f"<td>{_escape(source.get('author', '-'))}</td><td>{_escape(source.get('votes', '-'))}</td><td>{_escape(source.get('public_score', '-'))}</td><td>{_escape(source.get('status', '-'))}</td></tr>"
        )
    return '<section><h2>Kaggle 来源和元数据</h2><table><thead><tr><th>卡组</th><th>标题</th><th>作者</th><th>Votes</th><th>Public score</th><th>状态</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></section>"


def _demoted_html(snapshot: Mapping[str, object]) -> str:
    rows = "".join(
        f"<li data-demoted-id=\"{_escape(item.get('deck_id', '-'))}\">{_escape(item.get('display_name', item.get('deck_id', '-')))}：mu={_escape(item.get('mu', '-'))}</li>"
        for item in snapshot.get("demoted", ())
        if isinstance(item, Mapping)
    )
    return f"<section><h2>低质量附录</h2><ul>{rows or '<li>暂无稳定低于 300 分的卡组。</li>'}</ul></section>"


def _atomic_replace(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


_STYLE = """<style>
body{margin:0;background:#f5f7f7;color:#172321;font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px}h1,h2{margin:0 0 12px}section{margin:24px 0}
table{width:100%;border-collapse:collapse;background:#fff}th,td{padding:8px 10px;border:1px solid #d7dfdc;text-align:left;vertical-align:top}th{background:#e9f0ed}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}.card{background:#fff;border:1px solid #d7dfdc;padding:12px;border-radius:6px}.label,.muted{color:#52615d;font-size:12px}.value{font-size:21px;font-weight:700}.pokemon{height:78px;margin:4px;border-radius:5px}
</style>"""
