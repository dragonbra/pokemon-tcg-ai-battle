"""Render the frozen Top-100 ladder research and BC candidate report."""

from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from .render_top20_evaluation import card_image_url, load_card_catalog


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAMPAIGN = REPO_ROOT / "rl_runs/dataset/top100_research_20260723/campaign.json"
DEFAULT_CARD_DATA = REPO_ROOT / "data/official/EN_Card_Data.csv"
DEFAULT_OUTPUT = REPO_ROOT / "docs/reports/rl/top100_bc_candidate_research-20260723.html"
DEFAULT_INDEX = REPO_ROOT / "docs/reports/rl/top100_static_rank_index-20260723.csv"
STRICT_EPISODE_MINIMUM = 500
CANDIDATE_EPISODE_FLOOR = 200
TARGET_ROSTER_SIZE = 20
MANDATORY_TEAM = "LumenLiquidity"
WIN_QUALITY_WEIGHT = 0.65
SAMPLE_VOLUME_WEIGHT = 0.35

ARCHETYPE_ORDER = (
    "Alakazam / Dudunsparce",
    "Marnie's Grimmsnarl ex / Froslass",
    "Team Rocket's Mewtwo ex / Spidops",
    "Cynthia's Garchomp ex / Roserade",
    "Mega Kangaskhan ex / Crustle",
    "Dragapult ex / Dusknoir",
    "Dragapult ex / Dudunsparce",
    "Dragapult ex",
    "Mega Lopunny ex / Mega Froslass ex",
    "Archaludon ex / Cinderace",
    "Mega Lucario ex / Solrock",
    "N's Zoroark ex",
    "Festival Lead / Dipplin",
    "Mega Starmie ex / Dusknoir",
    "Mega Gardevoir ex / toolbox",
    "Hydrapple ex / Ogerpon",
    "Brambleghast / Comfey",
)

CORE_POKEMON_SLUG = {
    "Alakazam / Dudunsparce": "alakazam",
    "Marnie's Grimmsnarl ex / Froslass": "marnies_grimmsnarl_ex",
    "Team Rocket's Mewtwo ex / Spidops": "team_rockets_mewtwo_ex",
    "Cynthia's Garchomp ex / Roserade": "cynthias_garchomp_ex",
    "Mega Kangaskhan ex / Crustle": "mega_kangaskhan_ex",
    "Dragapult ex / Dusknoir": "dragapult_ex_dusknoir",
    "Dragapult ex / Dudunsparce": "dragapult_ex_dudunsparce",
    "Dragapult ex": "dragapult_ex",
    "Mega Lopunny ex / Mega Froslass ex": "mega_lopunny_ex",
    "Archaludon ex / Cinderace": "archaludon_ex",
    "Mega Lucario ex / Solrock": "mega_lucario_ex",
    "N's Zoroark ex": "ns_zoroark_ex",
    "Festival Lead / Dipplin": "festival_lead_dipplin",
    "Mega Starmie ex / Dusknoir": "mega_starmie_ex",
    "Mega Gardevoir ex / toolbox": "mega_gardevoir_ex",
    "Hydrapple ex / Ogerpon": "hydrapple_ex",
    "Brambleghast / Comfey": "brambleghast",
}


def classify_archetype(deck_profile: dict[str, Any]) -> str:
    names = {str(row["name"]) for row in deck_profile.get("pokemon", [])}
    if "Marnie's Grimmsnarl ex" in names:
        return "Marnie's Grimmsnarl ex / Froslass"
    if "Alakazam" in names:
        return "Alakazam / Dudunsparce"
    if "Dragapult ex" in names:
        if "Dusknoir" in names:
            return "Dragapult ex / Dusknoir"
        if "Dudunsparce" in names:
            return "Dragapult ex / Dudunsparce"
        return "Dragapult ex"
    if "Cynthia's Garchomp ex" in names:
        return "Cynthia's Garchomp ex / Roserade"
    if "Team Rocket's Mewtwo ex" in names:
        return "Team Rocket's Mewtwo ex / Spidops"
    if "Mega Kangaskhan ex" in names:
        return "Mega Kangaskhan ex / Crustle"
    if "Mega Gardevoir ex" in names:
        return "Mega Gardevoir ex / toolbox"
    if "N’s Zoroark ex" in names or "N's Zoroark ex" in names:
        return "N's Zoroark ex"
    if "Mega Lopunny ex" in names:
        return "Mega Lopunny ex / Mega Froslass ex"
    if "Archaludon ex" in names:
        return "Archaludon ex / Cinderace"
    if "Mega Lucario ex" in names:
        return "Mega Lucario ex / Solrock"
    if "Mega Starmie ex" in names:
        return "Mega Starmie ex / Dusknoir"
    if "Hydrapple ex" in names:
        return "Hydrapple ex / Ogerpon"
    if {"Dipplin", "Thwackey"}.issubset(names):
        return "Festival Lead / Dipplin"
    if "Brambleghast" in names:
        return "Brambleghast / Comfey"
    raise ValueError(f"unclassified Pokémon profile: {sorted(names)}")


def package_name(player: dict[str, Any], archetype: str) -> str:
    return f"rank_{int(player['rank']):03d}_{CORE_POKEMON_SLUG[archetype]}_bc"


def wilson_lcb95(episode_window: dict[str, Any]) -> float:
    wins = int(episode_window["wins"])
    losses = int(episode_window["losses"])
    draws = int(episode_window["draws"])
    sample_count = wins + losses + draws
    if not sample_count:
        return 0.0
    probability = (wins + 0.5 * draws) / sample_count
    z_score = 1.959963984540054
    z_squared = z_score * z_score
    center = probability + z_squared / (2 * sample_count)
    margin = z_score * math.sqrt(
        probability * (1 - probability) / sample_count
        + z_squared / (4 * sample_count * sample_count)
    )
    return (center - margin) / (1 + z_squared / sample_count)


def sample_sufficiency(episode_window: dict[str, Any]) -> float:
    episodes = int(episode_window["episodes"])
    return min(episodes, STRICT_EPISODE_MINIMUM) / STRICT_EPISODE_MINIMUM


def selection_score(player: dict[str, Any]) -> float:
    episode_window = player["episode_window"]
    return (
        WIN_QUALITY_WEIGHT * wilson_lcb95(episode_window)
        + SAMPLE_VOLUME_WEIGHT * sample_sufficiency(episode_window)
    )


def select_rosters(
    players: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strict = sorted(
        (
            player
            for player in players
            if int(player["episode_window"]["episodes"]) >= STRICT_EPISODE_MINIMUM
        ),
        key=lambda player: (-float(player["episode_window"]["win_rate"]), player["rank"]),
    )
    mandatory = next(
        player for player in players if str(player["team_name"]) == MANDATORY_TEAM
    )
    ordinary_candidates = sorted(
        (
            player
            for player in players
            if int(player["episode_window"]["episodes"])
            >= CANDIDATE_EPISODE_FLOOR
            and player is not mandatory
        ),
        key=lambda player: (-selection_score(player), player["rank"]),
    )
    selected = [mandatory]
    selected.extend(ordinary_candidates[: TARGET_ROSTER_SIZE - 1])
    return strict, selected


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _percent(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def _card_group(metadata: dict[str, str]) -> str:
    if metadata["hp"] not in {"", "n/a"}:
        return "Pokémon"
    if "Energy" in metadata["stage_or_type"]:
        return "Energy"
    return "Trainer"


def _render_card(card: dict[str, Any], catalog: dict[int, dict[str, str]]) -> str:
    card_id = int(card["card_id"])
    metadata = catalog[card_id]
    name = metadata["name"]
    expansion = metadata["expansion"]
    number = metadata["collection_number"]
    image_url = card_image_url(expansion, number)
    if image_url is None:
        raise ValueError(f"no image mapping for Card ID {card_id}: {expansion} {number}")
    return f"""
      <article class="card-tile">
        <div class="card-art">
          <div class="fallback"><b>ID {card_id}</b><span>{_escape(name)}</span></div>
          <img src="{_escape(image_url)}" alt="{_escape(name)}" loading="lazy"
            decoding="async" referrerpolicy="no-referrer"
            onerror="this.hidden=true;this.parentElement.classList.add('failed')">
          <span class="count">×{int(card['count'])}</span>
        </div>
        <b class="card-name" title="{_escape(name)}">{_escape(name)}</b>
        <small>Card ID {card_id} · {_escape(expansion)} {_escape(number)}</small>
      </article>"""


def _render_deck(player: dict[str, Any], catalog: dict[int, dict[str, str]]) -> str:
    groups: dict[str, list[dict[str, Any]]] = {
        "Pokémon": [],
        "Trainer": [],
        "Energy": [],
    }
    for card in player["deck_profile"]["cards"]:
        metadata = catalog[int(card["card_id"])]
        groups[_card_group(metadata)].append(card)
    sections = []
    for name in ("Pokémon", "Trainer", "Energy"):
        cards = groups[name]
        total = sum(int(card["count"]) for card in cards)
        tiles = "".join(_render_card(card, catalog) for card in cards)
        sections.append(
            f"""
            <section class="deck-group">
              <h4>{name} <small>{total} 张 · {len(cards)} 种</small></h4>
              <div class="card-grid">{tiles}</div>
            </section>"""
        )
    return "".join(sections)


def aggregate_matchups(
    player: dict[str, Any], archetype_by_hash: dict[str, str]
) -> dict[str, collections.Counter[str]]:
    result = {archetype: collections.Counter() for archetype in ARCHETYPE_ORDER}
    for row in player["known_top_deck_matchups"]["by_opponent_deck"]:
        deck_hash = str(row["opponent_deck_sha256"])
        archetype = archetype_by_hash[deck_hash]
        for field in ("episodes", "wins", "losses", "draws", "unknown"):
            result[archetype][field] += int(row.get(field, 0))
    return result


def _render_matchups(matchups: dict[str, collections.Counter[str]]) -> str:
    populated = [
        (archetype, row)
        for archetype, row in matchups.items()
        if int(row["episodes"]) > 0
    ]
    if not populated:
        return '<p class="empty">没有可由 Top 100 submission 映射出的对手牌型样本。</p>'
    rows = []
    for archetype, row in populated:
        decided = row["wins"] + row["losses"] + row["draws"]
        rate = row["wins"] / decided if decided else None
        rows.append(
            f"<tr><td>{_escape(archetype)}</td>"
            f"<td>{row['wins']}-{row['losses']}-{row['draws']}</td>"
            f"<td>{row['episodes']}</td><td><b>{_percent(rate)}</b></td></tr>"
        )
    return f"""
      <div class="table-scroll"><table class="compact">
        <thead><tr><th>已识别对手牌型</th><th>W-L-D</th><th>样本</th><th>胜率</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table></div>"""


def _selection_status(
    player: dict[str, Any], strict_ids: set[int], shadow_ids: set[int]
) -> tuple[str, str]:
    submission_id = int(player["submission_id"])
    if str(player["team_name"]) == MANDATORY_TEAM:
        return "mandatory", "Lumen 强制入选"
    if submission_id in strict_ids:
        return "strict", "严格合格"
    if submission_id in shadow_ids:
        return "provisional", "200+ 综合入选"
    return "insufficient", "样本不足"


def _render_selection_rows(
    players: Iterable[dict[str, Any]],
    archetype_by_source: dict[str, str],
    *,
    show_order: bool,
) -> str:
    rows = []
    for order, player in enumerate(players, 1):
        episode = player["episode_window"]
        archetype = archetype_by_source[str(player["source_id"])]
        prefix = f"<td>{order}</td>" if show_order else ""
        lower_bound = wilson_lcb95(episode)
        sufficiency = sample_sufficiency(episode)
        score = selection_score(player)
        rows.append(
            f"""
            <tr>{prefix}<td>#{int(player['rank'])}</td>
              <td><a href="#player-{int(player['rank']):03d}">
                {_escape(player['team_name'])}</a></td>
              <td>{_escape(archetype)}</td><td>{int(episode['episodes']):,}</td>
              <td>{episode['wins']}-{episode['losses']}-{episode['draws']}</td>
              <td><b>{_percent(episode['win_rate'], 2)}</b></td>
              <td>{_percent(lower_bound, 2)}</td>
              <td>{_percent(sufficiency, 0)}</td><td><b>{score:.4f}</b></td>
              <td><code>{_escape(package_name(player, archetype))}</code></td></tr>"""
        )
    return "".join(rows)


def _render_index_rows(
    players: list[dict[str, Any]],
    archetype_by_source: dict[str, str],
    strict_ids: set[int],
    shadow_ids: set[int],
) -> str:
    rows = []
    for player in players:
        episode = player["episode_window"]
        archetype = archetype_by_source[str(player["source_id"])]
        status, label = _selection_status(player, strict_ids, shadow_ids)
        search = f"{player['team_name']} {player['source_id']} {archetype}".lower()
        rows.append(
            f"""
            <tr data-index-row data-status="{status}" data-archetype="{_escape(archetype)}"
              data-search="{_escape(search)}">
              <td><b>#{int(player['rank'])}</b></td>
              <td><a href="#player-{int(player['rank']):03d}">
                {_escape(player['team_name'])}</a></td>
              <td>{_escape(archetype)}</td><td>{_escape(player['score'])}</td>
              <td>{int(episode['episodes']):,}</td>
              <td>{episode['wins']}-{episode['losses']}-{episode['draws']}</td>
              <td><b>{_percent(episode['win_rate'], 2)}</b></td>
              <td>{_percent(wilson_lcb95(episode), 2)}</td>
              <td><b>{selection_score(player):.4f}</b></td>
              <td><span class="status {status}">{label}</span></td>
              <td><code>{_escape(player['deck_profile']['deck_sha256'][:10])}</code></td>
            </tr>"""
        )
    return "".join(rows)


def _render_player_details(
    players: list[dict[str, Any]],
    archetype_by_source: dict[str, str],
    matchups_by_source: dict[str, dict[str, collections.Counter[str]]],
    strict_ids: set[int],
    shadow_ids: set[int],
    catalog: dict[int, dict[str, str]],
) -> str:
    sections = []
    for player in players:
        rank = int(player["rank"])
        source_id = str(player["source_id"])
        episode = player["episode_window"]
        known = player["known_top_deck_matchups"]
        archetype = archetype_by_source[source_id]
        status, status_label = _selection_status(player, strict_ids, shadow_ids)
        search = f"{player['team_name']} {source_id} {archetype}".lower()
        deck = _render_deck(player, catalog)
        matchups = _render_matchups(matchups_by_source[source_id])
        capped = (
            '<span class="status capped">1,000 局截断</span>'
            if episode["window_censored"]
            else ""
        )
        sections.append(
            f"""
            <details class="player-detail" id="player-{rank:03d}" data-player-detail
              data-status="{status}" data-archetype="{_escape(archetype)}"
              data-search="{_escape(search)}">
              <summary>
                <span class="rank">#{rank}</span>
                <span class="summary-name"><b>{_escape(player['team_name'])}</b>
                  <small>{_escape(archetype)}</small></span>
                <span class="summary-stat"><b>{int(episode['episodes']):,}</b>
                  <small>场</small></span>
                <span class="summary-stat"><b>{_percent(episode['win_rate'])}</b>
                  <small>胜率</small></span>
                <span class="status {status}">{status_label}</span>{capped}
              </summary>
              <div class="detail-body">
                <div class="stat-grid">
                  <div><span>静态 Rank</span><b>#{rank}</b></div>
                  <div><span>W-L-D</span>
                    <b>{episode['wins']}-{episode['losses']}-{episode['draws']}</b></div>
                  <div><span>已识别 matchup</span>
                    <b>{known['known_episode_count']} / {episode['episodes']}</b>
                    <small>{_percent(known['coverage'], 2)}</small></div>
                  <div><span>榜单分</span><b>{_escape(player['score'])}</b></div>
                  <div><span>胜率 LCB95</span><b>{_percent(wilson_lcb95(episode), 2)}</b></div>
                  <div><span>选择分</span><b>{selection_score(player):.4f}</b></div>
                  <div><span>Submission ID</span><b>{int(player['submission_id'])}</b></div>
                  <div><span>训练包名</span>
                    <code>{_escape(package_name(player, archetype))}</code></div>
                </div>
                <div class="subsection"><h3>已识别 matchup</h3>{matchups}</div>
                <div class="subsection">
                  <h3>Exact 60-card deck</h3>
                  <p class="muted">{player['deck_profile']['unique_card_count']} 种卡；数量徽标合计 60。
                    Deck hash <code>{_escape(player['deck_profile']['deck_sha256'])}</code></p>
                  {deck}
                </div>
              </div>
            </details>"""
        )
    return "".join(sections)


def _render_archetype_cards(
    players: list[dict[str, Any]], archetype_by_source: dict[str, str]
) -> str:
    counts = collections.Counter(archetype_by_source.values())
    hashes: dict[str, set[str]] = collections.defaultdict(set)
    for player in players:
        archetype = archetype_by_source[str(player["source_id"])]
        hashes[archetype].add(str(player["deck_profile"]["deck_sha256"]))
    return "".join(
        f"""
        <div class="archetype-card" style="--h:{index * 31 % 360}">
          <b>{_escape(archetype)}</b>
          <span>{counts[archetype]} 位 · {len(hashes[archetype])} 个 exact deck</span>
        </div>"""
        for index, archetype in enumerate(ARCHETYPE_ORDER)
        if counts[archetype]
    )


def _render_selected_matchup_matrix(
    players: list[dict[str, Any]],
    matchups_by_source: dict[str, dict[str, collections.Counter[str]]],
) -> str:
    rows = []
    for order, player in enumerate(players, 1):
        source_matchups = matchups_by_source[str(player["source_id"])]
        cells = []
        for archetype in ARCHETYPE_ORDER:
            matchup = source_matchups[archetype]
            games = int(matchup["episodes"])
            decided = matchup["wins"] + matchup["losses"] + matchup["draws"]
            if games and decided:
                rate = matchup["wins"] / decided
                record = f"{matchup['wins']}-{matchup['losses']}-{matchup['draws']}"
                cells.append(
                    f'<td title="{record} / {games} 场"><b>{_percent(rate)}</b>'
                    f"<small>n={games}</small></td>"
                )
            else:
                cells.append('<td class="no-sample">—<small>n=0</small></td>')
        rows.append(
            f"""
            <tr><th><a href="#player-{int(player['rank']):03d}">
              {order}. #{int(player['rank'])} {_escape(player['team_name'])}</a></th>
              {''.join(cells)}</tr>"""
        )
    return "".join(rows)


def _validate_campaign(campaign: dict[str, Any]) -> None:
    players = campaign["players"]
    if len(players) != 100:
        raise ValueError(f"expected 100 players, found {len(players)}")
    if [int(player["rank"]) for player in players] != list(range(1, 101)):
        raise ValueError("static ranks are not contiguous 1..100")
    for player in players:
        if player["deck_profile"] is None or len(player["deck"]) != 60:
            raise ValueError(f"{player['source_id']} has no audited 60-card deck")
        if sum(int(row["count"]) for row in player["deck_profile"]["cards"]) != 60:
            raise ValueError(f"{player['source_id']} deck profile does not sum to 60")


def render_report(campaign: dict[str, Any], catalog: dict[int, dict[str, str]]) -> str:
    _validate_campaign(campaign)
    players = list(campaign["players"])
    strict, shadow = select_rosters(players)
    strict_ids = {int(player["submission_id"]) for player in strict}
    shadow_ids = {int(player["submission_id"]) for player in shadow}
    archetype_by_source = {
        str(player["source_id"]): classify_archetype(player["deck_profile"])
        for player in players
    }
    archetype_by_hash: dict[str, str] = {}
    for player in players:
        deck_hash = str(player["deck_profile"]["deck_sha256"])
        archetype = archetype_by_source[str(player["source_id"])]
        previous = archetype_by_hash.setdefault(deck_hash, archetype)
        if previous != archetype:
            raise ValueError(f"deck hash {deck_hash} maps to two archetypes")
    matchups_by_source = {
        str(player["source_id"]): aggregate_matchups(player, archetype_by_hash)
        for player in players
    }
    total_games = sum(int(player["episode_window"]["episodes"]) for player in players)
    total_wins = sum(int(player["episode_window"]["wins"]) for player in players)
    total_losses = sum(int(player["episode_window"]["losses"]) for player in players)
    total_draws = sum(int(player["episode_window"]["draws"]) for player in players)
    known_games = sum(
        int(player["known_top_deck_matchups"]["known_episode_count"])
        for player in players
    )
    known_from_rows = sum(
        int(row["episodes"])
        for source_rows in matchups_by_source.values()
        for row in source_rows.values()
    )
    if known_games != known_from_rows:
        raise ValueError(f"matchup total mismatch: {known_games} != {known_from_rows}")
    captured = datetime.fromisoformat(str(campaign["captured_at"]))
    captured_shanghai = captured.astimezone(ZoneInfo("Asia/Shanghai"))
    captured_text = captured_shanghai.strftime("%Y-%m-%d %H:%M:%S %Z")
    archetype_cards = _render_archetype_cards(players, archetype_by_source)
    strict_rows = _render_selection_rows(strict, archetype_by_source, show_order=True)
    shadow_rows = _render_selection_rows(shadow, archetype_by_source, show_order=True)
    index_rows = _render_index_rows(
        players, archetype_by_source, strict_ids, shadow_ids
    )
    details = _render_player_details(
        players,
        archetype_by_source,
        matchups_by_source,
        strict_ids,
        shadow_ids,
        catalog,
    )
    selected_matrix_rows = _render_selected_matchup_matrix(
        shadow, matchups_by_source
    )
    matrix_headers = "".join(
        f"<th>{_escape(archetype)}</th>" for archetype in ARCHETYPE_ORDER
    )
    archetype_options = "".join(
        f'<option value="{_escape(archetype)}">{_escape(archetype)}</option>'
        for archetype in ARCHETYPE_ORDER
    )
    coverage = known_games / total_games
    weighted_win_rate = total_wins / (total_wins + total_losses + total_draws)
    strict_500_count = len(strict)
    shadow_shortfall = sum(
        int(player["episode_window"]["episodes"]) < STRICT_EPISODE_MINIMUM
        for player in shadow
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kaggle Top 100 BC 候选调研 · 2026-07-23</title>
  <style>
    :root {{
      --bg:#f3f4f0; --paper:#fff; --ink:#18231f; --muted:#68746e; --line:#dce3df;
      --green:#16785d; --green-soft:#dff2e9; --amber:#9b6218; --amber-soft:#fff0d1;
      --red:#a4473e; --red-soft:#fae5e2; --blue:#315f9b; --blue-soft:#e3edfa;
      --shadow:0 15px 40px rgba(27,49,40,.08);
    }}
    * {{ box-sizing:border-box; }} html {{ scroll-behavior:smooth; }}
    body {{ margin:0; color:var(--ink); background:var(--bg); line-height:1.55;
      font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
    a {{ color:var(--green); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.88em; }}
    .page {{ width:min(1580px,calc(100% - 28px)); margin:16px auto 80px; }}
    .hero {{ padding:clamp(28px,5vw,74px); border-radius:28px; color:#effcf6;
      background:radial-gradient(circle at 87% 12%,#50a884 0,transparent 27%),
      linear-gradient(135deg,#102b24,#175e49); box-shadow:var(--shadow); }}
    .eyebrow {{ margin:0; color:var(--green); font-size:12px; font-weight:850;
      letter-spacing:.14em; text-transform:uppercase; }} .hero .eyebrow {{ color:#a8e9cd; }}
    h1 {{ margin:8px 0 15px; font-size:clamp(36px,5.5vw,76px); line-height:1.02; }}
    .lead {{ max-width:950px; color:#d4eade; font-size:clamp(16px,2vw,21px); }}
    .metrics {{ display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin-top:30px; }}
    .metric {{ padding:14px; border:1px solid #ffffff2b; border-radius:15px;
      background:#ffffff12; }} .metric b {{ display:block; font-size:clamp(22px,2.5vw,33px); }}
    .metric span {{ color:#c5e4d6; font-size:12px; }}
    .alert {{ margin:18px 0; padding:18px 20px; border:1px solid #e5bb76;
      border-radius:16px; background:#fff8e9; color:#5e4218; }}
    .alert strong {{ color:#7b4608; }} .panel {{ margin:18px 0; padding:clamp(18px,3vw,34px);
      border:1px solid var(--line); border-radius:23px; background:var(--paper);
      box-shadow:var(--shadow); }}
    .heading {{ display:flex; justify-content:space-between; align-items:end; gap:24px;
      margin-bottom:18px; }} .heading h2 {{ margin:4px 0 0; font-size:clamp(25px,3vw,38px); }}
    .heading > p {{ max-width:650px; margin:0; color:var(--muted); text-align:right; }}
    .toolbar {{ position:sticky; z-index:20; top:8px; display:flex; flex-wrap:wrap; gap:9px;
      margin:18px 0; padding:11px; border:1px solid var(--line); border-radius:16px;
      background:#fffffff0; box-shadow:var(--shadow); backdrop-filter:blur(12px); }}
    .toolbar input,.toolbar select,.toolbar button {{ min-height:41px; padding:8px 11px;
      border:1px solid var(--line); border-radius:9px; background:#fff; color:var(--ink);
      font:inherit; }} .toolbar input {{ flex:1 1 280px; }} .toolbar button {{ cursor:pointer; }}
    .toolbar .count-visible {{ align-self:center; margin-left:auto; color:var(--muted); }}
    .table-scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:13px; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; }}
    th,td {{ padding:10px 11px; border-bottom:1px solid var(--line); text-align:left;
      vertical-align:middle; }} thead th {{ color:#435149; background:#f3f7f4; font-size:12px; }}
    tbody tr:hover {{ background:#f8fbf9; }} tbody tr:last-child td {{ border-bottom:0; }}
    .wide-table {{ min-width:1220px; }} .compact {{ min-width:660px; }}
    .matrix {{ min-width:2800px; font-size:12px; }}
    .matrix th:first-child {{ position:sticky; left:0; z-index:2; min-width:240px;
      background:#f3f7f4; }} .matrix tbody th:first-child {{ background:#fff; }}
    .matrix td {{ min-width:115px; text-align:center; }}
    .matrix td small {{ display:block; color:var(--muted); }}
    .matrix .no-sample {{ color:#a4ada8; }}
    .status {{ display:inline-block; padding:4px 8px; border-radius:999px; font-size:11px;
      font-weight:800; white-space:nowrap; }}
    .status.strict {{ color:#0d6349; background:var(--green-soft); }}
    .status.mandatory {{ color:#245087; background:var(--blue-soft); }}
    .status.provisional,.status.capped {{ color:#80500f; background:var(--amber-soft); }}
    .status.insufficient {{ color:#8e4038; background:var(--red-soft); }}
    .archetypes {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }}
    .archetype-card {{ padding:13px; border:1px solid var(--line); border-left:5px solid
      hsl(var(--h) 50% 48%); border-radius:11px; }}
    .archetype-card b,.archetype-card span {{ display:block; }}
    .archetype-card span {{ margin-top:3px; color:var(--muted); font-size:12px; }}
    .player-detail {{ margin:11px 0; scroll-margin-top:90px; border:1px solid var(--line);
      border-radius:17px; background:#fff; box-shadow:0 7px 24px rgba(27,49,40,.05); }}
    .player-detail[hidden],tr[hidden] {{ display:none; }}
    .player-detail summary {{ display:grid;
      grid-template-columns:auto minmax(200px,1fr) auto auto auto auto;
      gap:14px; align-items:center; padding:15px 18px; cursor:pointer; list-style:none; }}
    .player-detail summary::-webkit-details-marker {{ display:none; }}
    .player-detail[open] summary {{ border-bottom:1px solid var(--line);
      background:linear-gradient(110deg,#f7fbf8,#edf6f1); }}
    .rank {{ width:50px; height:50px; display:grid; place-items:center; border-radius:14px;
      color:#fff; background:var(--green); font-size:17px; font-weight:900; }}
    .summary-name b,.summary-name small,.summary-stat b,.summary-stat small {{ display:block; }}
    .summary-name b {{ font-size:17px; }}
    .summary-name small,.summary-stat small {{ color:var(--muted); }}
    .summary-stat {{ min-width:75px; text-align:right; }}
    .detail-body {{ padding:clamp(18px,3vw,32px); }} .stat-grid {{ display:grid;
      grid-template-columns:repeat(4,1fr); gap:9px; }} .stat-grid > div {{ min-width:0;
      padding:13px; border:1px solid var(--line); border-radius:11px; background:#fafcfb; }}
    .stat-grid span,.stat-grid small {{ display:block; color:var(--muted); font-size:11px; }}
    .stat-grid b {{ display:block; overflow:hidden; font-size:17px; text-overflow:ellipsis; }}
    .subsection {{ margin-top:26px; }} .subsection h3 {{ margin:0 0 10px; }}
    .muted,.empty {{ color:var(--muted); }} .deck-group {{ margin:20px 0 30px; }}
    .deck-group h4 {{ margin:0 0 11px; padding-bottom:7px; border-bottom:2px solid #a8cdbd;
      font-size:19px; }} .deck-group h4 small {{ color:var(--muted); font-weight:500; }}
    .card-grid {{ display:grid;
      grid-template-columns:repeat(auto-fill,minmax(128px,1fr)); gap:13px; }}
    .card-tile {{ min-width:0; }} .card-art {{ position:relative; aspect-ratio:2.5/3.5;
      overflow:hidden; border-radius:8px; background:linear-gradient(145deg,#e4ebe7,#cbd6d0);
      box-shadow:0 7px 17px rgba(25,43,35,.14); }} .card-art img {{ position:absolute;
      z-index:1; inset:0; width:100%; height:100%; object-fit:contain; background:#edf2ef; }}
    .fallback {{ position:absolute; inset:0; display:grid; align-content:center; gap:8px;
      padding:12px; color:#405149; text-align:center; }} .failed {{ outline:2px dashed #929e98;
      outline-offset:-5px; }} .count {{ position:absolute; z-index:3; top:6px; right:6px;
      min-width:34px; padding:3px 7px; border-radius:999px; color:#fff; background:#122b23e8;
      font-size:13px; font-weight:900; text-align:center; }} .card-name {{ display:block;
      margin-top:7px; overflow:hidden; font-size:12px; white-space:nowrap;
      text-overflow:ellipsis; }}
    .card-tile > small {{ display:block; color:var(--muted); font-size:10px; }}
    .provenance {{ color:var(--muted); font-size:13px; }} .provenance li {{ margin:6px 0; }}
    @media (max-width:1100px) {{ .metrics {{ grid-template-columns:repeat(3,1fr); }}
      .archetypes {{ grid-template-columns:repeat(2,1fr); }}
      .stat-grid {{ grid-template-columns:repeat(3,1fr); }}
    }}
    @media (max-width:700px) {{ .page {{ width:calc(100% - 16px); }} .hero {{ border-radius:20px; }}
      .metrics {{ grid-template-columns:repeat(2,1fr); }} .heading {{ align-items:start;
      flex-direction:column; gap:7px; }} .heading > p {{ text-align:left; }}
      .archetypes,.stat-grid {{ grid-template-columns:1fr; }} .player-detail summary {{
      grid-template-columns:auto 1fr auto; }} .summary-stat {{ display:none; }}
      .card-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
  </style>
</head>
<body id="top"><main class="page">
  <section class="hero">
    <p class="eyebrow">FROZEN TOP 100 · {_escape(captured_text)}</p>
    <h1>Top 100 BC 候选<br>卡组与样本审计</h1>
    <p class="lead">静态 Rank、当前 leaderboard submission 的公开 Episode 窗口、exact 60-card
      deck、关键 Pokémon 牌型和局部 matchup。用于选择独立单专家 BC policy，不跨 team 或
      submission 混合动作标签。</p>
    <div class="metrics">
      <div class="metric"><b>100</b><span>静态榜单选手</span></div>
      <div class="metric"><b>{total_games:,}</b><span>player-episode 记录</span></div>
      <div class="metric"><b>{len(archetype_by_hash)}</b><span>exact deck hashes</span></div>
      <div class="metric"><b>{len(set(archetype_by_source.values()))}</b>
        <span>关键 Pokémon 牌型</span></div>
      <div class="metric"><b>{strict_500_count}</b><span>严格满足 500 局</span></div>
      <div class="metric"><b>{_percent(coverage,2)}</b><span>matchup 已识别覆盖</span></div>
    </div>
  </section>

  <aside class="alert"><strong>门槛冲突：</strong>本次快照只有 <b>{strict_500_count}</b> 位当前
    submission 达到 500 局；加上强制入选但只有
    {next(p for p in players if p['team_name'] == MANDATORY_TEAM)['episode_window']['episodes']} 局的
    LumenLiquidity，严格可启动队列为 6 人，无法组成 20 人。页面另列一个
    <b>20 人推荐排程</b>使用新授权的放宽规则：Lumen 固定第一优先，其余选手至少 200 局，按
    <code>0.65 × Wilson LCB95 + 0.35 × min(场次,500)/500</code> 排序。这样同时奖励保守胜率
    与标签量，并让 500 局以上的样本奖励封顶；其中 {shadow_shortfall} 人仍低于原始 500 局目标，
    训练后必须接受更严格的离线复现审计。</aside>

  <nav class="toolbar">
    <input id="search" type="search" placeholder="搜索选手、source ID 或牌型…">
    <select id="status-filter"><option value="">全部状态</option>
      <option value="strict">严格合格</option><option value="mandatory">Lumen 强制入选</option>
      <option value="provisional">200+ 综合入选</option><option value="insufficient">样本不足</option>
    </select>
    <select id="archetype-filter"><option value="">全部牌型</option>{archetype_options}</select>
    <button id="open-visible" type="button">展开可见卡组</button>
    <button id="close-all" type="button">全部收起</button>
    <span class="count-visible"><b id="visible-count">100</b> / 100 位</span>
  </nav>

  <section class="panel">
    <div class="heading"><div><p class="eyebrow">STRICT QUEUE</p><h2>严格合格：≥500 局，按胜率排序</h2></div>
      <p>这是唯一满足用户原始样本门槛的训练队列；Lumen 作为单独例外，不混入此排序。</p></div>
    <div class="table-scroll"><table class="wide-table"><thead><tr><th>严格序</th><th>静态 Rank</th>
      <th>选手</th><th>牌型</th><th>场次</th><th>W-L-D</th><th>胜率</th>
      <th>LCB95</th><th>样本充足度</th><th>选择分</th><th>归档名</th></tr></thead>
      <tbody>{strict_rows}</tbody></table></div>
  </section>

  <section class="panel">
    <div class="heading"><div><p class="eyebrow">RECOMMENDED TRAINING ROSTER</p>
      <h2>20 人综合排程</h2></div>
      <p>Lumen 固定第一；其余使用 200 局硬底线和双维度选择分。选择分仅用于数据源排程，
      不等同于模型训练后的强度。</p></div>
    <div class="table-scroll"><table class="wide-table"><thead><tr><th>排程序</th><th>静态 Rank</th>
      <th>选手</th><th>牌型</th><th>场次</th><th>W-L-D</th><th>胜率</th>
      <th>LCB95</th><th>样本充足度</th><th>选择分</th><th>归档名</th></tr></thead>
      <tbody>{shadow_rows}</tbody></table></div>
  </section>

  <section class="panel">
    <div class="heading"><div><p class="eyebrow">TRAINING QUALITY CONTRACT</p>
      <h2>复现合格线与救援预算</h2></div>
      <p>先完成 20 个共享配置 V1，再把剩余时间用于失败来源；不会让前几副卡组的反复调参
      挤掉“至少完成 20 套”的目标。</p></div>
    <div class="archetypes">
      <div class="archetype-card" style="--h:155"><b>≥75%</b>
        <span>best validation exact action</span></div>
      <div class="archetype-card" style="--h:205"><b>≥65%</b>
        <span>validation multi-action exact</span></div>
      <div class="archetype-card" style="--h:265"><b>≥98%</b>
        <span>selection-count accuracy</span></div>
      <div class="archetype-card" style="--h:10"><b>100%</b>
        <span>validation/test legal action</span></div>
      <div class="archetype-card" style="--h:42"><b>V1 + 最多 2 次</b>
        <span>预设相邻学习率救援</span></div>
      <div class="archetype-card" style="--h:82"><b>禁止反馈</b>
        <span>test/evaluation 不用于选择学习率</span></div>
    </div>
  </section>

  <section class="panel"><div class="heading"><div><p class="eyebrow">ARCHETYPE MAP</p>
    <h2>17 类关键 Pokémon 牌型</h2></div><p>100 位选手对应 44 个 exact deck hash；相同构筑的不同
    submission 仍视为不同 BC policy source。</p></div>
    <div class="archetypes">{archetype_cards}</div></section>

  <section class="panel"><div class="heading"><div><p class="eyebrow">SELECTED MATCHUP MATRIX</p>
    <h2>推荐 20 人的局部 matchup</h2></div><p>单元格为胜率和已识别样本数；悬停查看 W-L-D。
    只覆盖 Top 100 已知 submission 对手，当前总覆盖 {_percent(coverage,2)}，不能当作完整
    meta matchup。</p></div><div class="table-scroll"><table class="matrix">
      <thead><tr><th>训练排程 / 静态 Rank / 选手</th>{matrix_headers}</tr></thead>
      <tbody>{selected_matrix_rows}</tbody></table></div></section>

  <section class="panel"><div class="heading"><div><p class="eyebrow">STATIC RANK INDEX</p>
    <h2>Top 100 静态排名索引</h2></div><p>固定为 {_escape(captured_text)} 的 Rank，不随之后榜单变化重排。
    加权窗口战绩 {total_wins}-{total_losses}-{total_draws}，胜率 {_percent(weighted_win_rate,2)}。</p></div>
    <div class="table-scroll"><table class="wide-table"><thead><tr><th>Rank</th>
      <th>选手</th><th>牌型</th>
      <th>榜单分</th><th>场次</th><th>W-L-D</th><th>胜率</th><th>LCB95</th>
      <th>选择分</th><th>状态</th><th>Deck hash</th></tr></thead>
      <tbody>{index_rows}</tbody></table></div></section>

  <section class="panel"><div class="heading"><div><p class="eyebrow">100 EXACT DECKS</p>
    <h2>逐人卡组图与 matchup</h2></div><p>点击展开；所有卡图均由官方 Card ID 对应的 Expansion +
    Collection No. 生成远程图片 URL，失败时回退为文字卡片。</p></div>{details}</section>

  <section class="panel provenance"><div class="heading"><div><p class="eyebrow">BOUNDARIES</p>
    <h2>数据边界与复现</h2></div></div><ul>
      <li>冻结时间：<code>{_escape(campaign['captured_at'])}</code> / {_escape(captured_text)}；
        排名只代表该时点。</li>
      <li>100 人共 {total_games:,} 条当前 submission 的公开已完成 Episode metadata；Yushin Ito
        与 mitomeat823 达到 API 最近 1,000 局上限，实际历史可能更多。</li>
      <li>Matchup 只在对手 submission 也属于本次 Top 100 且已有审计 deck 时识别：
        {known_games:,}/{total_games:,} = {_percent(coverage,2)}。未识别对局不猜测牌型。</li>
      <li>本轮保存 96 个唯一代表 replay；共享对局只保存一次，但双方 policy source 保持独立。</li>
      <li>完整原始 campaign：
        <a href="../../../rl_runs/dataset/top100_research_20260723/campaign.json">
        campaign.json</a>；静态索引：<a href="top100_static_rank_index-20260723.csv">CSV</a>。</li>
    </ul></section>
  </main>
  <script>
    (() => {{
      const search=document.getElementById('search');
      const status=document.getElementById('status-filter');
      const archetype=document.getElementById('archetype-filter');
      const visibleCount=document.getElementById('visible-count');
      function apply() {{
        const query=search.value.trim().toLowerCase(); let visible=0;
        document.querySelectorAll('[data-index-row],[data-player-detail]').forEach(item=>{{
          const show=(!query||item.dataset.search.includes(query))&&
            (!status.value||item.dataset.status===status.value)&&
            (!archetype.value||item.dataset.archetype===archetype.value);
          item.hidden=!show;
          if(show&&item.matches('[data-player-detail]')) visible+=1;
        }});
        visibleCount.textContent=visible;
      }}
      search.addEventListener('input',apply); status.addEventListener('change',apply);
      archetype.addEventListener('change',apply);
      document.getElementById('open-visible').addEventListener('click',()=>
        document.querySelectorAll('[data-player-detail]:not([hidden])').forEach(x=>x.open=true));
      document.getElementById('close-all').addEventListener('click',()=>
        document.querySelectorAll('[data-player-detail]').forEach(x=>x.open=false));
    }})();
  </script>
</body></html>
"""


def write_static_index(
    path: Path,
    campaign: dict[str, Any],
    archetype_by_source: dict[str, str],
    strict: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> None:
    strict_order = {
        int(player["submission_id"]): index for index, player in enumerate(strict, 1)
    }
    selected_order = {
        int(player["submission_id"]): index for index, player in enumerate(selected, 1)
    }
    fields = (
        "captured_at",
        "static_rank",
        "team_name",
        "team_id",
        "submission_id",
        "leaderboard_score",
        "episodes",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "win_rate_wilson_lcb95",
        "sample_sufficiency_to_500",
        "selection_score_65_lcb_35_volume",
        "window_censored",
        "archetype",
        "deck_sha256",
        "strict_500_eligible",
        "strict_win_rate_order",
        "mandatory_lumen",
        "meets_200_episode_floor",
        "selected_roster_order",
        "package_name",
        "sample_episode_id",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for player in campaign["players"]:
            submission_id = int(player["submission_id"])
            episode = player["episode_window"]
            archetype = archetype_by_source[str(player["source_id"])]
            writer.writerow(
                {
                    "captured_at": campaign["captured_at"],
                    "static_rank": player["rank"],
                    "team_name": player["team_name"],
                    "team_id": player["team_id"],
                    "submission_id": submission_id,
                    "leaderboard_score": player["score"],
                    "episodes": episode["episodes"],
                    "wins": episode["wins"],
                    "losses": episode["losses"],
                    "draws": episode["draws"],
                    "win_rate": episode["win_rate"],
                    "win_rate_wilson_lcb95": wilson_lcb95(episode),
                    "sample_sufficiency_to_500": sample_sufficiency(episode),
                    "selection_score_65_lcb_35_volume": selection_score(player),
                    "window_censored": episode["window_censored"],
                    "archetype": archetype,
                    "deck_sha256": player["deck_profile"]["deck_sha256"],
                    "strict_500_eligible": submission_id in strict_order,
                    "strict_win_rate_order": strict_order.get(submission_id, ""),
                    "mandatory_lumen": player["team_name"] == MANDATORY_TEAM,
                    "meets_200_episode_floor": int(episode["episodes"])
                    >= CANDIDATE_EPISODE_FLOOR,
                    "selected_roster_order": selected_order.get(submission_id, ""),
                    "package_name": package_name(player, archetype),
                    "sample_episode_id": player["sample_episode_id"],
                }
            )


def build_report(
    campaign_path: Path,
    card_data_path: Path,
    output_path: Path,
    index_path: Path,
) -> None:
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    catalog = load_card_catalog(card_data_path)
    markup = render_report(campaign, catalog)
    players = list(campaign["players"])
    strict, selected = select_rosters(players)
    archetype_by_source = {
        str(player["source_id"]): classify_archetype(player["deck_profile"])
        for player in players
    }
    write_static_index(index_path, campaign, archetype_by_source, strict, selected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markup, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--card-data", type=Path, default=DEFAULT_CARD_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index-output", type=Path, default=DEFAULT_INDEX)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_report(args.campaign, args.card_data, args.output, args.index_output)
    print(json.dumps({"html": str(args.output), "index": str(args.index_output)}))


if __name__ == "__main__":
    main()
