"""Build a timestamp-bound Kaggle Top 100 environment snapshot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "pokemon-tcg-ai-battle"
DRAGAPULT_EX = 121
MARNIES_GRIMMSNARL_EX = 648

REPORT_DATE = "2026-07-27"
REPORT_ID = "0727"
UI_BASELINE = "2026-07-26.html"
CARD_POOL_BASELINE = "2026-07-25.html"

CARD_IMAGE_SET_BY_EXPANSION = {
    "BLK": "zsv10pt5", "DRI": "sv10", "JTG": "sv9", "MEG": "me1",
    "PAL": "sv2", "PFL": "me2", "PRE": "sv8pt5", "SCR": "sv7",
    "SFA": "sv6pt5", "SSP": "sv8", "SVE": "sve", "SVI": "sv1",
    "TEF": "sv5", "TWM": "sv6", "WHT": "rsv10pt5",
}
SCRYDEX_SET_BY_EXPANSION = {"ASC": "me2pt5", "POR": "me3"}


def _card_catalog() -> dict[int, dict[str, str]]:
    path = Path(__file__).parents[2] / "official" / "EN_Card_Data.csv"
    catalog: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            catalog.setdefault(int(row["Card ID"]), row)
    return catalog


def _rate_call(fn):
    while True:
        try:
            return fn()
        except Exception as exc:  # Kaggle currently rate-limits team-submission lookups.
            if "429" not in str(exc):
                raise
            time.sleep(65)


def _result(episodes, submission_id: int) -> tuple[int, int, int, int]:
    from archive.train_legacy.kaggle_bc_top20.training.download_top100_exact_replays import (
        _model_value,
        episode_player_index,
    )

    wins = losses = draws = 0
    for episode in episodes:
        agent = (_model_value(episode, "agents", default=[]) or [])[episode_player_index(episode, submission_id)]
        reward = _model_value(agent, "reward")
        if reward is None:
            continue
        if float(reward) > 0:
            wins += 1
        elif float(reward) < 0:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws, wins + losses + draws


def _archetype(deck: list[int], catalog: dict[int, dict[str, str]]) -> str:
    """Name a deck from its primary evolution line plus its strongest distinct partner."""
    counts: dict[int, int] = {card_id: deck.count(card_id) for card_id in set(deck)}
    pokemon = []
    for card_id, count in counts.items():
        card = catalog.get(card_id, {})
        stage = card.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
        if "Pokémon" not in stage:
            continue
        level = 3 if "Stage 2" in stage else 2 if "Stage 1" in stage else 1
        if "ex" in card.get("Card Name", "") or "Mega" in card.get("Card Name", ""):
            level += 1
        pokemon.append((level, count, card_id, card))
    pokemon.sort(key=lambda item: (-item[0], -item[1], item[3].get("Card Name", "")))
    if not pokemon:
        return "Unknown Pokémon"
    primary = pokemon[0]
    ancestors = {primary[3].get("Card Name", "")}
    previous = primary[3].get("Previous stage", "")
    while previous and previous != "n/a":
        ancestors.add(previous)
        previous = next(
            (row.get("Previous stage", "") for row in catalog.values()
             if row.get("Card Name") == previous),
            "",
        )
    partner = next((item for item in pokemon[1:] if item[3].get("Card Name") not in ancestors), None)
    if partner is None:
        return primary[3]["Card Name"]
    return f"{primary[3]['Card Name']} / {partner[3]['Card Name']}"


def _card_group(row: dict[str, str]) -> str:
    kind = row.get("Stage (Pokémon)/Type (Energy and Trainer)", "").strip()
    category = row.get("Category", "").strip()
    if kind.endswith("Energy"):
        return "Energy"
    if kind.endswith("Pokémon") and category not in {"Fossil", "Technical Machine"}:
        return "Pokémon"
    return "Trainer"


def _image_url(row: dict[str, str] | None, *, hires: bool = False) -> str:
    if not row:
        return ""
    number = row["Collection No."]
    scrydex_set = SCRYDEX_SET_BY_EXPANSION.get(row["Expansion"])
    if scrydex_set:
        return f"https://images.scrydex.com/pokemon/{scrydex_set}-{number}/{'large' if hires else 'small'}"
    set_id = CARD_IMAGE_SET_BY_EXPANSION.get(row["Expansion"])
    if not set_id:
        return ""
    return f"https://images.pokemontcg.io/{set_id}/{number}{'_hires' if hires else ''}.png"


def _card_thumb(card_id: int, catalog: dict[int, dict[str, str]], *, compact: bool) -> str:
    row = catalog.get(card_id)
    name = row["Card Name"] if row else f"Card ID {card_id}"
    source = _image_url(row)
    preview = _image_url(row, hires=True)
    cls = "card-thumb compact" if compact else "card-thumb"
    image = (
        f'<img src="{html.escape(source)}" alt="{html.escape(name)}" loading="lazy" '
        'decoding="async" referrerpolicy="no-referrer" '
        'onerror="this.hidden=true;this.parentElement.classList.add(\'thumb-failed\')">'
        if source else ""
    )
    return (
        f'<span class="{cls}" tabindex="0" data-card-name="{html.escape(name)}" '
        f'data-card-preview-url="{html.escape(preview)}" title="{html.escape(name)} · Card ID {card_id}">'
        f'{image}<span class="thumb-fallback">ID {card_id}</span></span>'
    )


def _deck_hash(deck: list[int]) -> str:
    encoded = ",".join(str(card_id) for card_id in sorted(deck)).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _baseline_css(report: Path) -> str:
    baseline = report.parent / UI_BASELINE
    match = re.search(r"<style>(.*?)</style>", baseline.read_text(encoding="utf-8"), re.S)
    if not match:
        raise ValueError(f"missing style block in UI baseline: {baseline}")
    return match.group(1)


def _previous_players(report: Path) -> dict[str, dict[str, str | int | float]]:
    text = (report.parent / UI_BASELINE).read_text(encoding="utf-8")
    result: dict[str, dict[str, str | int | float]] = {}
    for row in re.findall(r'<tr data-index-row.*?</tr>', text, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if len(cells) < 10:
            continue
        plain = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", cell)).strip() for cell in cells]
        rank = re.search(r"#(\d+)", plain[0])
        archetype = re.search(r'data-archetype="([^"]*)"', row)
        deck_hash = re.search(r"([0-9a-f]{10,12})", plain[-1])
        if rank:
            result[plain[1].split("Team ")[0].strip()] = {
                "rank": int(rank.group(1)),
                "archetype": html.unescape(archetype.group(1)) if archetype else "",
                "deck_hash": deck_hash.group(1) if deck_hash else "",
            }
    return result


def _render_report(
    state: dict[str, object],
    players: list[dict[str, object]],
    catalog: dict[int, dict[str, str]],
    report: Path,
) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    css = _baseline_css(report) + """
    .snapshot-note{padding:16px 18px;border:1px solid #e5bb76;border-radius:14px;background:#fff8e9;color:#5e4218}
    .archetype-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.archetype-card{padding:15px;border:1px solid var(--line);border-radius:14px;background:#fff}.archetype-card h3{margin:0 0 8px}.bar-track{height:9px;border-radius:99px;background:#e8eee9;overflow:hidden}.bar-fill{height:100%;background:linear-gradient(90deg,#1e7c60,#8bbf9a)}
    .pool-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(132px,1fr));gap:13px}.pool-card{min-width:0;padding:10px;border:1px solid var(--line);border-radius:12px;background:#fff}.pool-card .card-art{position:relative;aspect-ratio:2.5/3.5;overflow:hidden;border-radius:8px;background:#edf2ef}.pool-card .card-thumb{width:100%!important;height:100%!important;min-width:0!important;max-width:none!important;min-height:0!important;max-height:none!important}.pool-card .card-thumb img{width:100%!important;height:100%!important;object-fit:contain!important}.pool-card b,.pool-card small{display:block}.pool-card b{margin-top:7px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.pool-card small{color:var(--muted)}.pool-rank{position:absolute;z-index:4;top:7px;left:7px;padding:3px 7px;border-radius:999px;color:#fff;background:#173e31e8;font-weight:900}.pool-count{position:absolute;z-index:4;top:7px;right:7px;padding:3px 7px;border-radius:999px;color:#fff;background:#9b6218e8;font-weight:900}
    .deck-group{margin:18px 0 28px}.deck-group h4{padding-bottom:7px;border-bottom:2px solid #a8cdbd}.card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:12px}.card-tile{min-width:0}.card-tile .card-art{position:relative;aspect-ratio:2.5/3.5;overflow:hidden;border-radius:8px;background:#edf2ef}.card-tile .card-thumb{width:100%!important;height:100%!important;min-width:0!important;max-width:none!important;min-height:0!important;max-height:none!important}.card-tile .card-thumb img{width:100%!important;height:100%!important;object-fit:contain!important}.card-tile b,.card-tile small{display:block}.card-tile b{margin-top:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.card-tile small{color:var(--muted)}.deck-count{position:absolute;z-index:4;top:7px;right:7px;padding:3px 7px;border-radius:999px;color:#fff;background:#173e31e8;font-weight:900}
    .archetype-profile{margin:10px 0;border:1px solid var(--line);border-radius:14px}.archetype-profile summary{display:flex;justify-content:space-between;gap:12px;padding:15px;cursor:pointer;font-weight:800}.archetype-profile>div{padding:0 15px 15px}.tag-list{display:flex;flex-wrap:wrap;gap:7px}.tag{padding:5px 8px;border-radius:999px;background:#edf6f1;color:#0d6349;font-size:12px}.unavailable-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.unavailable-grid article{padding:16px;border:1px dashed #c9d2cd;border-radius:14px;background:#f7f8f7}.comparison-table small,.summary-table small{display:block;color:var(--muted)}
    @media(max-width:900px){.archetype-grid,.unavailable-grid{grid-template-columns:1fr 1fr}}@media(max-width:680px){.archetype-grid,.unavailable-grid{grid-template-columns:1fr}}
    """
    css = "\n".join(line.rstrip() for line in css.splitlines())
    for player in players:
        player["deck_hash"] = _deck_hash(player["deck"])
        player["card_counts"] = Counter(int(card_id) for card_id in player["deck"])

    distribution = Counter(str(player["archetype"]) for player in players)
    presence: Counter[int] = Counter()
    archetype_presence: dict[str, Counter[int]] = defaultdict(Counter)
    for player in players:
        unique_ids = set(player["deck"])
        presence.update(unique_ids)
        archetype_presence[str(player["archetype"])].update(unique_ids)

    ordered_archetypes = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
    previous = _previous_players(report)
    current_names = {str(player["team_name"]) for player in players}
    joined = [player for player in players if player["team_name"] in previous]
    entered = [player for player in players if player["team_name"] not in previous]
    exited = sorted(set(previous) - current_names)
    changed = [
        player for player in joined
        if previous[str(player["team_name"])]["deck_hash"] != player["deck_hash"][:10]
    ]

    total_games = sum(int(player["valid_games"]) for player in players)
    total_wins = sum(int(player["wins"]) for player in players)
    total_losses = sum(int(player["losses"]) for player in players)
    total_draws = sum(int(player["draws"]) for player in players)
    overall = (total_wins + 0.5 * total_draws) / total_games if total_games else None

    def badge(player: dict[str, object]) -> str:
        return f'<span class="archetype-badge">{html.escape(str(player["archetype"]))}</span>'

    def player_row(player: dict[str, object], attribute: str) -> str:
        search = f'{player["rank"]} {player["team_name"]} {player["team_id"]} {player["archetype"]}'.lower()
        return (
            f'<tr {attribute} data-archetype="{html.escape(str(player["archetype"]))}" '
            f'data-search="{html.escape(search)}"><td><b>#{player["rank"]}</b></td>'
            f'<td><a href="#player-{int(player["rank"]):03d}">{html.escape(str(player["team_name"]))}</a>'
            f'<small>Team {player["team_id"]} · submission {player["submission_id"]}</small></td>'
            f'<td>{badge(player)}</td><td>{float(player["score"]):.1f}</td>'
            f'<td>{int(player["valid_games"]):,}</td><td>{player["wins"]}-{player["losses"]}-{player["draws"]}</td>'
            f'<td><b>{_pct(player["win_rate"])}</b></td><td><code>{player["deck_hash"]}</code>'
            f'<small>Episode {player["episode_id"]}</small></td></tr>'
        )

    archetype_cards = []
    max_count = max(distribution.values(), default=1)
    for name, count in ordered_archetypes:
        ids = archetype_presence[name].most_common(5)
        tags = "".join(
            f'<span class="tag">{html.escape(catalog[card_id]["Card Name"])} · {n}/{count}</span>'
            for card_id, n in ids if card_id in catalog
        )
        archetype_cards.append(
            f'<article class="archetype-card"><h3>{html.escape(name)} <small>{count} 人</small></h3>'
            f'<p>{_pct(count / 100)} · 代表 replay exact deck</p>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{100 * count / max_count:.1f}%"></div></div>'
            f'<div class="tag-list">{tags}</div></article>'
        )

    comparison_rows = []
    for player in joined:
        old = previous[str(player["team_name"])]
        is_changed = old["deck_hash"] != player["deck_hash"][:10]
        comparison_rows.append(
            f'<tr class="{"changed-deck" if is_changed else "same-deck"}"><td><b>{html.escape(str(player["team_name"]))}</b></td>'
            f'<td>#{old["rank"]} → <b>#{player["rank"]}</b></td>'
            f'<td>{html.escape(str(old["archetype"]))} → <b>{html.escape(str(player["archetype"]))}</b></td>'
            f'<td><span class="status {"provisional" if is_changed else "strict"}">{"卡组变化" if is_changed else "卡组不变"}</span>'
            f'<small>{old["deck_hash"] or "—"} → {player["deck_hash"][:10]}</small></td></tr>'
        )

    pool_cards = []
    for rank, (card_id, count) in enumerate(presence.most_common(), 1):
        row = catalog.get(card_id, {})
        name = row.get("Card Name", f"Card ID {card_id}")
        pool_cards.append(
            f'<article class="pool-card" data-pool-card><div class="card-art">'
            f'{_card_thumb(card_id, catalog, compact=False)}<span class="pool-rank">#{rank}</span>'
            f'<span class="pool-count">{count}/100</span></div><b title="{html.escape(name)}">{html.escape(name)}</b>'
            f'<small>Card ID {card_id} · {_card_group(row)} · {_pct(count / 100)}</small></article>'
        )

    archetype_profiles = []
    for name, count in ordered_archetypes:
        typical = archetype_presence[name].most_common(16)
        tags = "".join(
            f'<span class="tag">{html.escape(catalog.get(card_id, {}).get("Card Name", str(card_id)))} · {n}/{count}</span>'
            for card_id, n in typical
        )
        archetype_profiles.append(
            f'<details class="archetype-profile"><summary><span>{html.escape(name)}</span><span>{count} 套 · 展开构筑卡池</span></summary>'
            f'<div class="tag-list">{tags}</div></details>'
        )

    detail_blocks = []
    for player in players:
        groups: dict[str, list[str]] = {"Pokémon": [], "Trainer": [], "Energy": []}
        totals: Counter[str] = Counter()
        for card_id, count in sorted(player["card_counts"].items(), key=lambda item: catalog.get(item[0], {}).get("Card Name", "")):
            row = catalog.get(card_id, {})
            group = _card_group(row)
            totals[group] += count
            name = row.get("Card Name", f"Card ID {card_id}")
            groups[group].append(
                f'<article class="card-tile"><div class="card-art">{_card_thumb(card_id, catalog, compact=False)}'
                f'<span class="deck-count">×{count}</span></div><b title="{html.escape(name)}">{html.escape(name)}</b>'
                f'<small>Card ID {card_id} · {count} 张</small></article>'
            )
        deck_html = "".join(
            f'<section class="deck-group"><h4>{group} <small>{totals[group]} 张 · {len(groups[group])} 种</small></h4>'
            f'<div class="card-grid">{"".join(groups[group])}</div></section>'
            for group in ("Pokémon", "Trainer", "Energy")
        )
        search = f'{player["rank"]} {player["team_name"]} {player["team_id"]} {player["archetype"]}'.lower()
        detail_blocks.append(
            f'<details class="person-card" id="player-{int(player["rank"]):03d}" data-player-detail '
            f'data-archetype="{html.escape(str(player["archetype"]))}" data-search="{html.escape(search)}">'
            f'<summary><span><b>#{player["rank"]} {html.escape(str(player["team_name"]))}</b> · {badge(player)}</span>'
            f'<span>{player["wins"]}-{player["losses"]}-{player["draws"]} · {_pct(player["win_rate"])}</span></summary>'
            f'<div class="detail-body"><p><b>submission {player["submission_id"]}</b> · Episode {player["episode_id"]} · '
            f'deck hash <code>{player["deck_hash"]}</code></p>{deck_html}</div></details>'
        )

    report_html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Top 100 今日环境快照 · {REPORT_ID} · {REPORT_DATE}</title><style>{css}</style></head>
<body id="top"><main class="page">
<nav class="section-nav" aria-label="{REPORT_ID} 完整分析导航"><a href="#new-summary">新版总览</a><a href="#personal-winrates">个人胜率</a><a href="#construction-distribution">构筑分布</a><a href="#snapshot-comparison">跨日变化</a><a href="#matchup-boundary">Match-up</a><a href="#card-pool">构筑卡池</a><a href="#archetype-builds">牌型卡池</a><a href="#rank-index">静态排名</a><a href="#player-details">逐人展开</a></nav>

<section class="panel" id="new-summary"><div class="heading"><div><p class="eyebrow">{REPORT_DATE} SNAPSHOT · {REPORT_ID}</p><h2>新版环境分析总览</h2></div><p>整体组件与 UI 继承 0726；构筑卡池专项恢复 0725 的卡图网格与覆盖率表达。</p></div>
<div class="metrics new-metrics"><div class="metric"><b>100</b><span>最终 submissions</span></div><div class="metric"><b>{total_games:,}</b><span>公开 Meta 玩家视角</span></div><div class="metric"><b>{len(presence)}</b><span>Card Pool 并集</span></div><div class="metric"><b>{len(distribution)}</b><span>实际牌型</span></div><div class="metric"><b>100</b><span>exact decks 已审计</span></div><div class="metric"><b>100</b><span>代表 replays</span></div></div>
<div class="evidence-grid"><article><b>榜单层</b><p>冻结官方 Top100、score 与 submissionDate。</p></article><article><b>Meta 层</b><p>只计 exact submission 的 PUBLIC + COMPLETED Episode Meta。</p></article><article><b>Replay 层</b><p>逐 submission 最新合格 replay 的 exact 60-card deck。</p></article></div>
<h3>读数摘要</h3><div class="finding-grid"><article><b>榜首与分差</b><p>{html.escape(str(players[0]["team_name"]))} · {float(players[0]["score"]):.1f}；第 100 名 {html.escape(str(players[-1]["team_name"]))} · {float(players[-1]["score"]):.1f}。</p></article><article><b>公开 Meta</b><p>{total_wins:,}-{total_losses:,}-{total_draws:,}，玩家视角胜率 {_pct(overall)}（n={total_games:,}）。</p></article><article><b>环境集中度</b><p>{html.escape(ordered_archetypes[0][0])} {ordered_archetypes[0][1]} 人；前两类合计 {sum(count for _, count in ordered_archetypes[:2])}/100。</p></article><article><b>卡池审计</b><p>100 份代表 deck 均为 60 张，共覆盖 {len(presence)} 个 Card ID。</p></article></div>
<div class="snapshot-note"><b>证据护栏：</b>单 submission 的 Episode Meta 最多暴露约 1,000 局；本次冻结文件没有持久化逐局对手字段，因此不伪造 0727 matchup、先后攻或墙钟矩阵。</div></section>

<section class="panel" id="personal-winrates"><div class="heading"><div><p class="eyebrow">PLAYER CONSTRUCTION UI · FINAL SUBMISSION META</p><h2>个人构筑与公开 Meta 胜率</h2></div><p>筛选会同步作用于本表和逐人 exact deck。</p></div><div class="toolbar"><input id="search" type="search" placeholder="搜索选手、Team ID、牌型或 rank…"><select id="archetype-filter"><option value="">全部牌型</option>{''.join(f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name, _ in ordered_archetypes)}</select><span id="visible-count" class="count-visible">显示 100</span></div><div class="table-scroll"><table class="wide-table summary-table"><thead><tr><th>Rank</th><th>选手</th><th>牌型</th><th>榜分</th><th>Meta 场次</th><th>W-L-D</th><th>胜率</th><th>deck evidence</th></tr></thead><tbody>{''.join(player_row(player, 'data-player-row') for player in players)}</tbody></table></div></section>

<section class="panel" id="construction-distribution"><div class="heading"><div><p class="eyebrow">CONSTRUCTION DISTRIBUTION</p><h2>构筑使用分布</h2></div><p>牌型只由 0727 最终 submission 的代表 replay exact 60-card deck 分类。</p></div><div class="archetype-grid">{''.join(archetype_cards)}</div></section>

<section class="panel" id="snapshot-comparison"><div class="heading"><div><p class="eyebrow">SNAPSHOT DELTA</p><h2>2026-07-26 → 2026-07-27 同榜选手与卡组变化</h2></div><p>以 0726 已发布报告的 rank、牌型和 deck hash 前缀为基线。</p></div><div class="metrics comparison-metrics"><div class="metric"><b>{len(joined)}</b><span>两次同时上榜</span></div><div class="metric"><b>{len(changed)}</b><span>deck hash 变化</span></div><div class="metric"><b>{len(entered)}</b><span>新进榜</span></div><div class="metric"><b>{len(exited)}</b><span>退出榜</span></div></div><p class="muted"><b>新进榜：</b>{html.escape('、'.join(str(player['team_name']) for player in entered) or '—')}<br><b>退出榜：</b>{html.escape('、'.join(exited) or '—')}</p><label><input id="changed-only" type="checkbox"> 只看卡组变化</label><div class="table-scroll"><table class="comparison-table"><thead><tr><th>选手</th><th>Rank</th><th>牌型</th><th>deck hash</th></tr></thead><tbody>{''.join(comparison_rows)}</tbody></table></div></section>

<section class="panel" id="matchup-boundary"><div class="heading"><div><p class="eyebrow">MATCH-UP / PACING / TURN ORDER</p><h2>对局矩阵与节奏证据边界</h2></div><p>保留 0726 的分析入口，但只展示 0727 快照实际持久化的证据。</p></div><div class="unavailable-grid"><article><b>牌型 Match-up 矩阵</b><p>本次快照只保留逐 submission W-L-D 聚合，没有逐局 opponent submission，无法诚实重建。</p></article><article><b>Top 20 直接 Match-up</b><p>同上；n=0 不等于 0% 胜率，因此不生成伪矩阵。</p></article><article><b>先后攻与墙钟</b><p>代表 replay 可证明 deck，但当前 snapshot 未保留完整 firstPlayer/turn/endTime 审计字段。</p></article></div></section>

<section class="panel" id="card-pool"><div class="heading"><div><p class="eyebrow">100 AUDITED EXACT DECKS · 0725 CARD-POOL UI</p><h2>Top 100 全局构筑卡池</h2></div><p>恢复 0725 的卡图式 Card Pool：按“包含该卡的代表 deck 数”排序，并显示 Card ID、类别和覆盖率。</p></div><div class="pool-grid">{''.join(pool_cards)}</div></section>

<section class="panel" id="archetype-builds"><div class="heading"><div><p class="eyebrow">ARCHETYPE BUILD AUDIT</p><h2>各牌型典型构筑卡池</h2></div><p>每张卡的 n 表示该牌型中有多少套代表 deck 包含它；不是平均投入张数或因果强度。</p></div>{''.join(archetype_profiles)}</section>

<section class="panel" id="rank-index"><div class="heading"><div><p class="eyebrow">STATIC RANK INDEX</p><h2>Top 100 静态排名索引</h2></div><p>固定为 {html.escape(str(state["captured_at_utc"]))} 的榜单，不随后续 leaderboard 变化重排。</p></div><div class="table-scroll"><table class="summary-table"><thead><tr><th>Rank</th><th>选手</th><th>牌型</th><th>榜分</th><th>Meta 场次</th><th>W-L-D</th><th>胜率</th><th>deck evidence</th></tr></thead><tbody>{''.join(player_row(player, 'data-index-row') for player in players)}</tbody></table></div></section>

<section class="panel" id="player-details"><div class="heading"><div><p class="eyebrow">PLAYER DETAIL / EXACT DECK</p><h2>逐人 exact 60-card deck 展开</h2></div><p>点击展开；卡图、分组、张数、Card ID 与大图预览沿用 0726。</p></div><div class="toolbar"><button id="open-visible" type="button">展开当前筛选</button><button id="close-all" type="button">全部收起</button></div><div class="person-grid">{''.join(detail_blocks)}</div></section>

<section class="panel provenance" id="boundaries"><div class="heading"><div><p class="eyebrow">BOUNDARIES</p><h2>数据边界与复现</h2></div></div><ul><li>报告 ID：<code>{REPORT_ID}</code>；leaderboard 冻结：<code>{html.escape(str(state["captured_at_utc"]))}</code>。</li><li>100 行均绑定最终 submission；只统计 PUBLIC + COMPLETED 且 submission ID 精确匹配的公开 Episode Meta。</li><li>每个 deck 来自该 submission 最新合格代表 replay 的自身 player index，且恰为 60 个 Card ID。</li><li>UI 基准：<a href="{UI_BASELINE}">0726</a>；构筑卡池专项基准：<a href="{CARD_POOL_BASELINE}">0725</a>；归档入口：<a href="../index.html">环境日报索引</a>。</li><li>本地冻结事实源：<code>.tmp/environment_daily_0727/snapshot.json</code>；生成入口：<code>data/processed/environment_daily/generate_live_snapshot.py</code>。</li></ul></section>
</main><script>
const search=document.getElementById('search'), archetype=document.getElementById('archetype-filter'), visible=document.getElementById('visible-count');
function apply(){{const q=search.value.trim().toLowerCase();let n=0;document.querySelectorAll('[data-player-row],[data-player-detail]').forEach(row=>{{const ok=(!q||row.dataset.search.includes(q))&&(!archetype.value||row.dataset.archetype===archetype.value);row.hidden=!ok;if(ok&&row.matches('[data-player-row]'))n++;}});visible.textContent=`显示 ${{n}}`;}}search.addEventListener('input',apply);archetype.addEventListener('change',apply);apply();
document.getElementById('changed-only').addEventListener('change',event=>document.querySelectorAll('.comparison-table tbody tr').forEach(row=>row.hidden=event.target.checked&&!row.classList.contains('changed-deck')));
document.getElementById('open-visible').addEventListener('click',()=>document.querySelectorAll('[data-player-detail]:not([hidden])').forEach(row=>row.open=true));document.getElementById('close-all').addEventListener('click',()=>document.querySelectorAll('[data-player-detail]').forEach(row=>row.open=false));
const preview=document.createElement('div');preview.className='card-preview';preview.innerHTML='<img alt="卡牌大图预览" width="234" height="328"><b></b>';document.body.appendChild(preview);const previewImg=preview.querySelector('img'),previewName=preview.querySelector('b');let active=null;function position(e){{const w=preview.offsetWidth||250,h=preview.offsetHeight||370,m=14;let l=e.clientX+18,t=e.clientY+18;if(l+w+m>innerWidth)l=e.clientX-w-18;if(t+h+m>innerHeight)t=innerHeight-h-m;preview.style.left=`${{Math.max(m,l)}}px`;preview.style.top=`${{Math.max(m,t)}}px`;}}function show(thumb,e){{const img=thumb.querySelector('img');if(!img||img.hidden)return;active=thumb;previewImg.src=thumb.dataset.cardPreviewUrl||img.src;previewName.textContent=thumb.dataset.cardName||img.alt;preview.classList.add('visible');position(e);}}function hide(){{active=null;preview.classList.remove('visible');}}document.addEventListener('pointerover',e=>{{const t=e.target.closest&&e.target.closest('.card-thumb');if(t&&t!==active)show(t,e);}});document.addEventListener('pointermove',e=>{{if(active)position(e);}});document.addEventListener('pointerout',e=>{{if(active&&!e.relatedTarget?.closest?.('.card-thumb'))hide();}});
</script></body></html>'''
    report.write_text(report_html, encoding="utf-8")


def collect(work: Path, report: Path) -> None:
    from archive.train_legacy.kaggle_bc_top20.training.download_top100_exact_replays import (
        _decks_from_replay,
        _eligible_episodes,
        episode_player_index,
        select_leaderboard_submission,
    )

    work.mkdir(parents=True, exist_ok=True)
    replay_root = work / "representative_replays"
    replay_root.mkdir(exist_ok=True)
    state_path = work / "snapshot.json"
    api = KaggleApi()
    api.authenticate()
    catalog = _card_catalog()
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        leaderboard = api.competition_leaderboard_view(COMPETITION, page_size=100)[:100]
        state = {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": [
                {"rank": rank, "team_id": row.team_id, "team_name": row.team_name,
                 "score": row.score, "submission_date": row.submission_date.isoformat()}
                for rank, row in enumerate(leaderboard, 1)
            ],
            "players": {},
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    for row in state["rows"]:
        key = str(row["rank"])
        if key in state["players"]:
            continue
        submissions = _rate_call(lambda: api.competition_team_submissions(row["team_id"]))
        class Model:
            pass
        model = Model()
        model.team_id = row["team_id"]
        model.team_name = row["team_name"]
        model.score = row["score"]
        model.submission_date = datetime.fromisoformat(row["submission_date"])
        selected, binding = select_leaderboard_submission(model, list(submissions))
        submission_id = int(selected.id)
        episodes = _rate_call(lambda: _eligible_episodes(api, submission_id, 1000))
        wins, losses, draws, known = _result(episodes, submission_id)
        deck: list[int] = []
        episode_id = None
        if episodes:
            episode_id = int(episodes[0].id)
            replay = replay_root / f"episode-{episode_id}.json"
            if not replay.exists():
                _rate_call(lambda: api.competition_episode_replay(episode_id, path=str(replay_root), quiet=True))
                (replay_root / f"episode-{episode_id}-replay.json").replace(replay)
            deck = _decks_from_replay(json.loads(replay.read_text(encoding="utf-8")))[
                episode_player_index(episodes[0], submission_id)
            ]
        state["players"][key] = {
            **row, "submission_id": submission_id, "binding": binding, "episode_id": episode_id,
            "valid_games": len(episodes), "wins": wins, "losses": losses, "draws": draws,
            "win_rate": (wins + 0.5 * draws) / known if known else None,
            "archetype": _archetype(deck, catalog), "deck": deck,
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        time.sleep(2.5)
    players = [state["players"][str(i)] for i in range(1, 101)]
    for player in players:
        player["archetype"] = _archetype(player["deck"], catalog)
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    _render_report(state, players, catalog, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        snapshot = json.loads((args.work / "snapshot.json").read_text(encoding="utf-8"))
        snapshot_players = [snapshot["players"][str(rank)] for rank in range(1, 101)]
        _render_report(snapshot, snapshot_players, _card_catalog(), args.report)
    else:
        collect(args.work, args.report)
