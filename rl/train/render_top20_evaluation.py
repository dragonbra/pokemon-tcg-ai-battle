"""Render the frozen Top-20 ladder deck and metadata report as one HTML page."""

from __future__ import annotations

import argparse
import collections
import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = REPO_ROOT / "rl/artifact/dataset/top20_arena_20260723/campaign.json"
DEFAULT_CARD_DATA = REPO_ROOT / "data/official/EN_Card_Data.csv"
DEFAULT_OUTPUT = REPO_ROOT / "docs/reports/rl/top20_evaluation-20260723.html"

# PokemonTCG.io uses API set ids rather than the abbreviations in the official
# competition card table. The two newest sets use the image URLs published by
# the same PokemonTCG data catalog on Scrydex.
SET_IMAGE_IDS = {
    "ASC": "me2pt5",
    "BLK": "zsv10pt5",
    "DRI": "sv10",
    "JTG": "sv9",
    "MEG": "me1",
    "PAL": "sv2",
    "PFL": "me2",
    "POR": "me3",
    "PRE": "sv8pt5",
    "SCR": "sv7",
    "SFA": "sv6pt5",
    "SSP": "sv8",
    "SVE": "sve",
    "SVI": "sv1",
    "SVP": "svp",
    "TEF": "sv5",
    "TWM": "sv6",
    "WHT": "rsv10pt5",
}
SCRYDEX_SET_IDS = {"me2pt5", "me3"}

ARCHETYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Dragapult ex", ("Dragapult ex",)),
    ("Cynthia's Garchomp ex", ("Cynthia's Garchomp ex",)),
    ("Alakazam", ("Alakazam",)),
    ("Marnie's Grimmsnarl ex", ("Marnie's Grimmsnarl ex",)),
    (
        "Mega Lopunny ex / Mega Froslass ex",
        ("Mega Lopunny ex", "Mega Froslass ex"),
    ),
    (
        "Team Rocket's Mewtwo ex / Spidops",
        ("Team Rocket's Mewtwo ex", "Team Rocket's Spidops"),
    ),
    ("Archaludon ex / Cinderace", ("Archaludon ex", "Cinderace")),
    ("Mega Kangaskhan ex / Crustle", ("Mega Kangaskhan ex", "Crustle")),
)
ARCHETYPE_ORDER = tuple(label for label, _ in ARCHETYPE_RULES)
ARCHETYPE_KEYS = {label: keys for label, keys in ARCHETYPE_RULES}


def load_card_catalog(path: Path) -> dict[int, dict[str, str]]:
    """Load one metadata row per numeric competition Card ID."""
    result: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                card_id = int(row.get("Card ID", ""))
            except (TypeError, ValueError):
                continue
            result.setdefault(
                card_id,
                {
                    "name": str(row.get("Card Name", "")),
                    "expansion": str(row.get("Expansion", "")),
                    "collection_number": str(row.get("Collection No.", "")),
                    "stage_or_type": str(
                        row.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
                    ),
                    "hp": str(row.get("HP", "")),
                },
            )
    return result


def card_image_url(expansion: str, collection_number: str) -> str | None:
    """Build a card image URL from the official expansion and collection number."""
    set_id = SET_IMAGE_IDS.get(expansion)
    number = collection_number.strip()
    if not set_id or not number:
        return None
    if set_id in SCRYDEX_SET_IDS:
        return f"https://images.scrydex.com/pokemon/{set_id}-{number}/small"
    return f"https://images.pokemontcg.io/{set_id}/{number}.png"


def classify_archetype(deck_profile: dict[str, Any]) -> str:
    """Classify a deck using auditable key-Pokémon rules."""
    pokemon_names = {str(row["name"]) for row in deck_profile.get("pokemon", [])}
    matches = [
        label
        for label, required_names in ARCHETYPE_RULES
        if all(name in pokemon_names for name in required_names)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one archetype for {sorted(pokemon_names)}, found {matches}"
        )
    return matches[0]


def aggregate_matchups(
    player: dict[str, Any], archetype_by_hash: dict[str, str]
) -> dict[str, dict[str, int | float | None]]:
    """Aggregate audited exact-deck matchup rows into key-Pokémon archetypes."""
    counters: dict[str, collections.Counter[str]] = {
        label: collections.Counter() for label in ARCHETYPE_ORDER
    }
    rows = player["known_top_deck_matchups"]["by_opponent_deck"]
    for row in rows:
        deck_hash = str(row["opponent_deck_sha256"])
        if deck_hash not in archetype_by_hash:
            raise ValueError(f"no archetype for opponent deck {deck_hash}")
        archetype = archetype_by_hash[deck_hash]
        for key in ("episodes", "wins", "losses", "draws", "unknown"):
            counters[archetype][key] += int(row.get(key, 0))

    result: dict[str, dict[str, int | float | None]] = {}
    for archetype in ARCHETYPE_ORDER:
        counter = counters[archetype]
        decided = counter["wins"] + counter["losses"] + counter["draws"]
        result[archetype] = {
            "episodes": counter["episodes"],
            "wins": counter["wins"],
            "losses": counter["losses"],
            "draws": counter["draws"],
            "unknown": counter["unknown"],
            "win_rate": counter["wins"] / decided if decided else None,
        }
    return result


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _percent(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def _card_group(metadata: dict[str, str]) -> str:
    stage = metadata["stage_or_type"]
    hp = metadata["hp"]
    if hp not in {"", "n/a"}:
        return "Pokémon"
    if "Energy" in stage:
        return "Energy"
    return "Trainer"


def _render_card(
    card: dict[str, Any], catalog: dict[int, dict[str, str]]
) -> str:
    card_id = int(card["card_id"])
    metadata = catalog.get(
        card_id,
        {
            "name": str(card.get("name", f"Unknown card {card_id}")),
            "expansion": "",
            "collection_number": "",
            "stage_or_type": str(card.get("stage_or_type", "unknown")),
            "hp": "",
        },
    )
    name = metadata["name"] or str(card.get("name", f"Unknown card {card_id}"))
    expansion = metadata["expansion"]
    number = metadata["collection_number"]
    image_url = card_image_url(expansion, number)
    image = ""
    if image_url:
        image = (
            f'<img src="{_escape(image_url)}" alt="{_escape(name)}" loading="lazy" '
            'decoding="async" referrerpolicy="no-referrer" '
            'onerror="this.hidden=true;this.parentElement.classList.add(\'image-failed\')">'
        )
    code = " · ".join(part for part in (expansion, number) if part)
    return f"""
      <article class="card-tile">
        <div class="card-art">
          <div class="card-fallback"><b>ID {card_id}</b><span>{_escape(name)}</span></div>
          {image}
          <span class="count-badge">×{int(card['count'])}</span>
        </div>
        <div class="card-name">{_escape(name)}</div>
        <div class="card-code">Card ID {card_id}{' · ' + _escape(code) if code else ''}</div>
      </article>"""


def _render_deck(
    player: dict[str, Any], catalog: dict[int, dict[str, str]]
) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {
        "Pokémon": [],
        "Trainer": [],
        "Energy": [],
    }
    for card in player["deck_profile"]["cards"]:
        metadata = catalog.get(int(card["card_id"]))
        if metadata is None:
            raise ValueError(f"Card ID {card['card_id']} is absent from the card catalog")
        grouped[_card_group(metadata)].append(card)

    sections = []
    for group_name in ("Pokémon", "Trainer", "Energy"):
        cards = grouped[group_name]
        card_total = sum(int(card["count"]) for card in cards)
        rendered_cards = "".join(_render_card(card, catalog) for card in cards)
        sections.append(
            f"""
            <section class="deck-group deck-group-{group_name.lower()}">
              <h4>{group_name} <span>{card_total} 张 · {len(cards)} 种</span></h4>
              <div class="card-grid">{rendered_cards}</div>
            </section>"""
        )
    return "".join(sections)


def _render_matchup_rows(matchups: dict[str, dict[str, int | float | None]]) -> str:
    rows = []
    for index, archetype in enumerate(ARCHETYPE_ORDER):
        row = matchups[archetype]
        games = int(row["episodes"] or 0)
        rate = row["win_rate"]
        width = 0.0 if rate is None else float(rate) * 100
        record = f"{row['wins']}-{row['losses']}-{row['draws']}"
        sample_note = "无已识别样本" if not games else f"{games} 场"
        rows.append(
            f"""
            <tr>
              <td><span class="archetype-dot color-{index}"></span>{_escape(archetype)}</td>
              <td class="record-cell">{record}</td>
              <td>{sample_note}</td>
              <td class="rate-cell">
                <div class="rate-line"><b>{_percent(rate)}</b>
                  <span class="rate-track"><i style="width:{width:.1f}%"></i></span>
                </div>
              </td>
            </tr>"""
        )
    return "".join(rows)


def _render_player(
    player: dict[str, Any],
    archetype: str,
    matchups: dict[str, dict[str, int | float | None]],
    catalog: dict[int, dict[str, str]],
) -> str:
    rank = int(player["rank"])
    episode = player["episode_window"]
    known = player["known_top_deck_matchups"]
    deck_profile = player["deck_profile"]
    known_count = int(known["known_episode_count"])
    games = int(episode["episodes"])
    key_names = " + ".join(ARCHETYPE_KEYS[archetype])
    censored = (
        '<span class="pill warning">API 上限截断</span>'
        if episode.get("window_censored")
        else ""
    )
    deck_html = _render_deck(player, catalog)
    matchup_rows = _render_matchup_rows(matchups)
    search = " ".join(
        (str(player["team_name"]), str(player["source_id"]), archetype)
    ).lower()
    return f"""
    <article class="player-detail" id="player-{rank:02d}" data-player-detail
      data-rank="{rank}" data-games="{games}" data-win-rate="{episode['win_rate']}"
      data-archetype="{_escape(archetype)}" data-search="{_escape(search)}">
      <header class="player-header">
        <div class="rank-medal">#{rank}</div>
        <div class="player-title">
          <p class="eyebrow">{_escape(player['source_id'])}</p>
          <h2>{_escape(player['team_name'])}</h2>
          <div class="pills">
            <span class="pill">{_escape(archetype)}</span>
            <span class="pill subtle">关键卡：{_escape(key_names)}</span>
            {censored}
          </div>
        </div>
        <div class="player-identifiers">
          <span>榜单分 <b>{_escape(player['score'])}</b></span>
          <span>Submission <b>{int(player['submission_id'])}</b></span>
          <span>Deck hash <code>{_escape(deck_profile['deck_sha256'][:12])}</code></span>
        </div>
      </header>

      <div class="player-stat-grid">
        <div class="player-stat"><span>公开对局窗口</span><b>{games:,}</b><small>场</small></div>
        <div class="player-stat accent"><span>总胜率</span>
          <b>{_percent(episode['win_rate'])}</b><small>W / 已完成对局</small></div>
        <div class="player-stat"><span>W-L-D</span>
          <b>{episode['wins']}-{episode['losses']}-{episode['draws']}</b>
          <small>{episode['unknown']} unknown</small></div>
        <div class="player-stat"><span>已识别 matchup</span>
          <b>{known_count} / {games}</b><small>覆盖率 {_percent(known['coverage'], 2)}</small></div>
      </div>

      <section class="matchup-panel">
        <div class="section-heading">
          <div><p class="eyebrow">MATCHUP WINDOW</p><h3>对主流牌型的已识别战绩</h3></div>
          <p>只统计对手 submission 能映射到本次 Top 20 审计卡组的对局。</p>
        </div>
        <div class="table-scroll">
          <table class="matchup-table">
            <thead><tr><th>对手牌型</th><th>W-L-D</th><th>样本</th><th>胜率</th></tr></thead>
            <tbody>{matchup_rows}</tbody>
          </table>
        </div>
      </section>

      <section class="deck-panel">
        <div class="section-heading">
          <div><p class="eyebrow">EXACT 60-CARD DECK</p><h3>完整卡组图</h3></div>
          <p>{deck_profile['card_count']} 张 · {deck_profile['unique_card_count']} 种卡；
            数量徽标合计为 60。</p>
        </div>
        {deck_html}
      </section>
      <div class="back-link"><a href="#top">↑ 返回顶部</a></div>
    </article>"""


def _render_overview_rows(
    players: Iterable[dict[str, Any]], archetype_by_source: dict[str, str]
) -> str:
    rows = []
    for player in players:
        rank = int(player["rank"])
        episode = player["episode_window"]
        known = player["known_top_deck_matchups"]
        archetype = archetype_by_source[str(player["source_id"])]
        search = f"{player['team_name']} {player['source_id']} {archetype}".lower()
        rows.append(
            f"""
            <tr data-player-row data-rank="{rank}" data-archetype="{_escape(archetype)}"
              data-search="{_escape(search)}">
              <td><b>#{rank}</b></td>
              <td><a href="#player-{rank:02d}">{_escape(player['team_name'])}</a></td>
              <td>{_escape(archetype)}</td>
              <td>{int(episode['episodes']):,}</td>
              <td>{episode['wins']}-{episode['losses']}-{episode['draws']}</td>
              <td><b>{_percent(episode['win_rate'])}</b></td>
              <td>{int(known['known_episode_count'])} / {int(episode['episodes'])}
                <small>({_percent(known['coverage'], 2)})</small></td>
            </tr>"""
        )
    return "".join(rows)


def _render_matrix_rows(
    players: Iterable[dict[str, Any]],
    archetype_by_source: dict[str, str],
    matchups_by_source: dict[str, dict[str, dict[str, int | float | None]]],
) -> str:
    rows = []
    for player in players:
        source_id = str(player["source_id"])
        rank = int(player["rank"])
        archetype = archetype_by_source[source_id]
        search = f"{player['team_name']} {source_id} {archetype}".lower()
        cells = []
        for opponent_archetype in ARCHETYPE_ORDER:
            matchup = matchups_by_source[source_id][opponent_archetype]
            games = int(matchup["episodes"] or 0)
            if games:
                title = (
                    f"{matchup['wins']}-{matchup['losses']}-{matchup['draws']} / {games} 场"
                )
                cells.append(
                    f'<td title="{_escape(title)}"><b>{_percent(matchup["win_rate"])}</b>'
                    f"<small>n={games}</small></td>"
                )
            else:
                cells.append('<td class="no-sample">—<small>n=0</small></td>')
        rows.append(
            f"""
            <tr data-player-row data-rank="{rank}" data-archetype="{_escape(archetype)}"
              data-search="{_escape(search)}">
              <th><a href="#player-{rank:02d}">#{rank} {_escape(player['team_name'])}</a></th>
              {''.join(cells)}
            </tr>"""
        )
    return "".join(rows)


def _render_archetype_chips(
    players: list[dict[str, Any]], archetype_by_source: dict[str, str]
) -> str:
    player_counts = collections.Counter(archetype_by_source.values())
    hashes: dict[str, set[str]] = collections.defaultdict(set)
    for player in players:
        archetype = archetype_by_source[str(player["source_id"])]
        hashes[archetype].add(str(player["deck_profile"]["deck_sha256"]))
    return "".join(
        f"""
        <div class="archetype-chip color-border-{index}">
          <b>{_escape(archetype)}</b>
          <span>{player_counts[archetype]} 位选手 · {len(hashes[archetype])} 个 exact deck</span>
        </div>"""
        for index, archetype in enumerate(ARCHETYPE_ORDER)
    )


def _validate_campaign(campaign: dict[str, Any]) -> None:
    players = campaign.get("players", [])
    if len(players) != int(campaign.get("leaderboard_size", len(players))):
        raise ValueError("campaign player count does not match leaderboard_size")
    ranks = [int(player["rank"]) for player in players]
    if ranks != list(range(1, len(players) + 1)):
        raise ValueError(f"campaign ranks are not contiguous: {ranks}")
    for player in players:
        deck = [int(card_id) for card_id in player["deck"]]
        profile = player["deck_profile"]
        profile_count = sum(int(card["count"]) for card in profile["cards"])
        if len(deck) != 60 or int(profile["card_count"]) != 60 or profile_count != 60:
            raise ValueError(f"{player['source_id']} does not contain one exact 60-card deck")


def render_report(campaign: dict[str, Any], catalog: dict[int, dict[str, str]]) -> str:
    """Return the complete self-contained report markup (card images remain remote)."""
    _validate_campaign(campaign)
    players = list(campaign["players"])
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
            raise ValueError(f"deck {deck_hash} maps to both {previous} and {archetype}")

    matchups_by_source = {
        str(player["source_id"]): aggregate_matchups(player, archetype_by_hash)
        for player in players
    }
    total_games = sum(int(player["episode_window"]["episodes"]) for player in players)
    total_wins = sum(int(player["episode_window"]["wins"]) for player in players)
    known_games = sum(
        int(player["known_top_deck_matchups"]["known_episode_count"])
        for player in players
    )
    matchup_games = sum(
        int(row["episodes"] or 0)
        for source_matchups in matchups_by_source.values()
        for row in source_matchups.values()
    )
    if matchup_games != known_games:
        raise ValueError(
            f"aggregated matchup rows contain {matchup_games} games, expected {known_games}"
        )
    exact_decks = len(archetype_by_hash)
    matchup_coverage = known_games / total_games if total_games else 0.0
    weighted_win_rate = total_wins / total_games if total_games else None
    overview_rows = _render_overview_rows(players, archetype_by_source)
    matrix_rows = _render_matrix_rows(players, archetype_by_source, matchups_by_source)
    archetype_chips = _render_archetype_chips(players, archetype_by_source)
    player_sections = "".join(
        _render_player(
            player,
            archetype_by_source[str(player["source_id"])],
            matchups_by_source[str(player["source_id"])],
            catalog,
        )
        for player in players
    )
    archetype_options = "".join(
        f'<option value="{_escape(label)}">{_escape(label)}</option>'
        for label in ARCHETYPE_ORDER
    )
    matrix_headers = "".join(
        f"<th>{_escape(label)}</th>" for label in ARCHETYPE_ORDER
    )
    captured_at = str(campaign.get("captured_at", "unknown"))

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kaggle Top 20 卡组与公开战绩审计 · 2026-07-23</title>
  <style>
    :root {{
      --bg: #f3f5f2; --paper: #fff; --ink: #17221e; --muted: #64726b;
      --line: #dce3df; --accent: #16785d; --accent-soft: #dff2e9;
      --warning: #9a5a13; --shadow: 0 14px 40px rgba(31, 55, 45, .08);
      --c0: #8357c5; --c1: #d6712c; --c2: #2d7ebd; --c3: #3d9364;
      --c4: #b54b72; --c5: #5c68bd; --c6: #bb8b22; --c7: #54736b;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0; background: var(--bg); color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em; }}
    .page {{ width: min(1540px, calc(100% - 32px)); margin: 0 auto 80px; }}
    .hero {{
      margin: 18px auto 0; padding: clamp(28px, 5vw, 72px); border-radius: 28px;
      color: #f6fff9; overflow: hidden; position: relative;
      background: radial-gradient(circle at 92% 10%, #5eb58e 0, transparent 28%),
        linear-gradient(135deg, #102b24, #174f40 62%, #13644d);
      box-shadow: var(--shadow);
    }}
    .hero:after {{
      content: "20"; position: absolute; right: 4%; bottom: -38%; font-weight: 900;
      font-size: clamp(170px, 27vw, 430px); color: rgba(255,255,255,.055); line-height: 1;
    }}
    .hero-content {{ position: relative; z-index: 1; max-width: 1000px; }}
    .hero .eyebrow {{ color: #a7e6c9; }}
    .hero h1 {{ margin: 8px 0 14px; font-size: clamp(34px, 5vw, 70px); line-height: 1.04; }}
    .hero .lead {{ max-width: 850px; color: #d7eee4; font-size: clamp(16px, 2vw, 21px); }}
    .eyebrow {{
      margin: 0; color: var(--accent); font-size: 12px; font-weight: 800;
      letter-spacing: .14em; text-transform: uppercase;
    }}
    .hero-metrics {{
      display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-top: 34px;
    }}
    .hero-metric {{
      padding: 16px; border: 1px solid rgba(255,255,255,.17); border-radius: 16px;
      background: rgba(255,255,255,.08); backdrop-filter: blur(8px);
    }}
    .hero-metric b {{ display: block; font-size: clamp(22px, 3vw, 34px); line-height: 1.1; }}
    .hero-metric span {{ color: #c9e7da; font-size: 12px; }}
    .method-note {{
      margin: 20px 0; padding: 18px 20px; display: grid; grid-template-columns: auto 1fr;
      gap: 14px; align-items: start; border: 1px solid #eccf9e; border-radius: 16px;
      background: #fff8e9; color: #604318;
    }}
    .method-note strong {{ white-space: nowrap; }}
    .toolbar {{
      position: sticky; z-index: 20; top: 10px; display: flex; flex-wrap: wrap; gap: 10px;
      margin: 20px 0; padding: 12px; background: rgba(255,255,255,.92);
      border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }}
    .toolbar input, .toolbar select {{
      min-height: 42px; padding: 8px 12px; border: 1px solid var(--line);
      border-radius: 10px; background: #fff; color: var(--ink); font: inherit;
    }}
    .toolbar input {{ flex: 1 1 260px; }}
    .toolbar .visible-count {{ align-self: center; margin-left: auto; color: var(--muted); }}
    .panel, .player-detail {{
      margin: 20px 0; padding: clamp(18px, 3vw, 34px); background: var(--paper);
      border: 1px solid var(--line); border-radius: 24px; box-shadow: var(--shadow);
    }}
    .section-heading {{
      display: flex; justify-content: space-between; align-items: end; gap: 24px;
      margin-bottom: 18px;
    }}
    .section-heading h2, .section-heading h3 {{ margin: 4px 0 0; line-height: 1.2; }}
    .section-heading h2 {{ font-size: clamp(25px, 3vw, 38px); }}
    .section-heading h3 {{ font-size: clamp(21px, 2.4vw, 30px); }}
    .section-heading > p {{ max-width: 610px; margin: 0; color: var(--muted); text-align: right; }}
    .archetype-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
    .archetype-chip {{
      padding: 14px; border: 1px solid var(--line); border-left-width: 5px;
      border-radius: 12px;
    }}
    .archetype-chip b, .archetype-chip span {{ display: block; }}
    .archetype-chip span {{ margin-top: 3px; color: var(--muted); font-size: 13px; }}
    .color-border-0 {{ border-left-color: var(--c0); }}
    .color-border-1 {{ border-left-color: var(--c1); }}
    .color-border-2 {{ border-left-color: var(--c2); }}
    .color-border-3 {{ border-left-color: var(--c3); }}
    .color-border-4 {{ border-left-color: var(--c4); }}
    .color-border-5 {{ border-left-color: var(--c5); }}
    .color-border-6 {{ border-left-color: var(--c6); }}
    .color-border-7 {{ border-left-color: var(--c7); }}
    .table-scroll {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 14px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; }}
    th, td {{
      padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: left;
      vertical-align: middle;
    }}
    thead th {{
      position: sticky; top: 0; z-index: 1; color: #435149; background: #f4f7f5;
      font-size: 12px;
    }}
    tbody tr:last-child td, tbody tr:last-child th {{ border-bottom: 0; }}
    tbody tr:hover {{ background: #f8fbf9; }}
    td small {{ display: block; color: var(--muted); }}
    .matrix-table {{ min-width: 1380px; font-size: 13px; }}
    .matrix-table th:first-child {{
      position: sticky; left: 0; z-index: 2; min-width: 220px; background: #f4f7f5;
    }}
    .matrix-table tbody th:first-child {{ background: #fff; }}
    .matrix-table td {{ min-width: 128px; text-align: center; }}
    .matrix-table td small {{ font-size: 11px; }}
    .matrix-table .no-sample {{ color: #a3aca7; }}
    .player-detail {{ scroll-margin-top: 92px; padding: 0; overflow: hidden; }}
    .player-detail[hidden], tr[hidden] {{ display: none; }}
    .player-header {{
      display: grid; grid-template-columns: auto 1fr auto; gap: 20px; align-items: center;
      padding: clamp(20px, 3vw, 34px); background: linear-gradient(115deg, #f8fbf9, #eef7f2);
      border-bottom: 1px solid var(--line);
    }}
    .rank-medal {{
      width: 68px; height: 68px; display: grid; place-items: center; border-radius: 20px;
      color: #fff; background: var(--accent); font-size: 24px; font-weight: 900;
      box-shadow: 0 10px 22px rgba(22,120,93,.22);
    }}
    .player-title h2 {{ margin: 3px 0 8px; font-size: clamp(25px, 3vw, 38px); line-height: 1.1; }}
    .pills {{ display: flex; flex-wrap: wrap; gap: 7px; }}
    .pill {{
      padding: 5px 9px; border-radius: 999px; color: #0d5f48;
      background: var(--accent-soft); font-size: 12px; font-weight: 700;
    }}
    .pill.subtle {{ color: #59655f; background: #e8ece9; }}
    .pill.warning {{ color: #84500e; background: #f8dfb7; }}
    .player-identifiers {{
      display: grid; gap: 4px; color: var(--muted); font-size: 13px; text-align: right;
    }}
    .player-stat-grid {{
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
      padding: clamp(20px, 3vw, 34px); padding-bottom: 0;
    }}
    .player-stat {{
      padding: 16px; border: 1px solid var(--line); border-radius: 14px;
      background: #fbfcfb;
    }}
    .player-stat span, .player-stat small {{ display: block; color: var(--muted); }}
    .player-stat b {{
      display: inline-block; margin: 3px 5px 2px 0;
      font-size: clamp(21px, 2.5vw, 31px);
    }}
    .player-stat.accent {{ border-color: #abd7c7; background: var(--accent-soft); }}
    .matchup-panel, .deck-panel {{ padding: clamp(20px, 3vw, 34px); padding-bottom: 0; }}
    .matchup-table {{ min-width: 740px; }}
    .matchup-table td:first-child {{ width: 34%; font-weight: 650; }}
    .archetype-dot {{
      display: inline-block; width: 9px; height: 9px; margin-right: 8px;
      border-radius: 50%;
    }}
    .color-0 {{ background: var(--c0); }} .color-1 {{ background: var(--c1); }}
    .color-2 {{ background: var(--c2); }} .color-3 {{ background: var(--c3); }}
    .color-4 {{ background: var(--c4); }} .color-5 {{ background: var(--c5); }}
    .color-6 {{ background: var(--c6); }} .color-7 {{ background: var(--c7); }}
    .record-cell {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .rate-line {{ display: flex; align-items: center; gap: 12px; min-width: 180px; }}
    .rate-line b {{ width: 56px; }}
    .rate-track {{
      flex: 1; height: 8px; overflow: hidden; border-radius: 999px; background: #e5ebe7;
    }}
    .rate-track i {{
      display: block; height: 100%; border-radius: inherit; background: var(--accent);
    }}
    .deck-group {{ margin: 20px 0 34px; }}
    .deck-group h4 {{
      margin: 0 0 12px; padding-bottom: 8px; border-bottom: 2px solid var(--line);
      font-size: 20px;
    }}
    .deck-group h4 span {{ color: var(--muted); font-size: 13px; font-weight: 500; }}
    .deck-group-pokémon h4 {{ border-color: #7bc1a3; }}
    .deck-group-trainer h4 {{ border-color: #e6b66e; }}
    .deck-group-energy h4 {{ border-color: #8d96d8; }}
    .card-grid {{
      display: grid; grid-template-columns: repeat(auto-fill, minmax(134px, 1fr));
      gap: 14px;
    }}
    .card-tile {{ min-width: 0; }}
    .card-art {{
      position: relative; aspect-ratio: 2.5 / 3.5; overflow: hidden; border-radius: 8px;
      background: linear-gradient(145deg, #e4ebe7, #cbd6d0);
      box-shadow: 0 8px 18px rgba(25,43,35,.14);
    }}
    .card-art img {{
      position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain;
      z-index: 1; background: #edf2ef;
    }}
    .card-fallback {{
      position: absolute; inset: 0; display: grid; align-content: center; gap: 10px;
      padding: 14px; color: #405149; text-align: center;
    }}
    .card-fallback b {{ font-size: 20px; }}
    .card-fallback span {{ font-size: 12px; }}
    .image-failed {{ outline: 2px dashed #9eaaa4; outline-offset: -5px; }}
    .count-badge {{
      position: absolute; z-index: 3; top: 7px; right: 7px; min-width: 36px;
      padding: 4px 7px; border-radius: 999px; color: #fff; background: #122b23e8;
      box-shadow: 0 3px 8px rgba(0,0,0,.25); font-size: 14px; font-weight: 900; text-align: center;
    }}
    .card-name {{
      margin-top: 8px; overflow: hidden; font-size: 13px; font-weight: 750;
      white-space: nowrap; text-overflow: ellipsis;
    }}
    .card-code {{ color: var(--muted); font-size: 11px; }}
    .back-link {{ padding: 0 34px 26px; text-align: right; }}
    .provenance {{ color: var(--muted); font-size: 13px; }}
    .provenance li {{ margin: 7px 0; }}
    .no-results {{ display: none; padding: 30px; text-align: center; color: var(--muted); }}
    .no-results.visible {{ display: block; }}
    @media (max-width: 1050px) {{
      .hero-metrics {{ grid-template-columns: repeat(3, 1fr); }}
      .archetype-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .player-header {{ grid-template-columns: auto 1fr; }}
      .player-identifiers {{ grid-column: 2; text-align: left; }}
      .player-stat-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    @media (max-width: 650px) {{
      .page {{ width: min(100% - 18px, 1540px); }}
      .hero {{ border-radius: 20px; }}
      .hero-metrics {{ grid-template-columns: repeat(2, 1fr); }}
      .method-note {{ grid-template-columns: 1fr; }}
      .section-heading {{ align-items: start; flex-direction: column; gap: 8px; }}
      .section-heading > p {{ text-align: left; }}
      .archetype-grid, .player-stat-grid {{ grid-template-columns: 1fr; }}
      .player-header {{ grid-template-columns: 1fr; }}
      .player-identifiers {{ grid-column: 1; }}
      .rank-medal {{ width: 54px; height: 54px; border-radius: 15px; }}
      .card-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 11px; }}
    }}
    @media print {{
      body {{ background: #fff; }}
      .page {{ width: 100%; }}
      .toolbar, .back-link {{ display: none; }}
      .hero, .panel, .player-detail {{ box-shadow: none; break-inside: avoid; }}
      .card-art img {{ print-color-adjust: exact; }}
    }}
  </style>
</head>
<body id="top">
  <main class="page">
    <section class="hero">
      <div class="hero-content">
        <p class="eyebrow">KAGGLE LADDER SNAPSHOT · 2026-07-23</p>
        <h1>Top 20 卡组图<br>与公开战绩审计</h1>
        <p class="lead">每位选手一副从官方 replay 还原的 exact 60-card deck，配合该榜单
          submission 的公开 Episode metadata 窗口战绩，以及按关键 Pokémon 分类的 matchup。</p>
        <div class="hero-metrics">
          <div class="hero-metric"><b>{len(players)}</b><span>排行榜选手</span></div>
          <div class="hero-metric"><b>{total_games:,}</b><span>player-episode 记录</span></div>
          <div class="hero-metric"><b>{exact_decks}</b><span>exact deck hashes</span></div>
          <div class="hero-metric"><b>{len(ARCHETYPE_ORDER)}</b><span>关键 Pokémon 牌型</span></div>
          <div class="hero-metric"><b>{_percent(matchup_coverage, 2)}</b>
            <span>matchup 已识别覆盖</span></div>
        </div>
      </div>
    </section>

    <aside class="method-note">
      <strong>统计边界</strong>
      <div>总胜率来自冻结 submission 的公开已完成 Episode metadata；不是选手生涯战绩。
        Kaggle API 每个 submission 最多返回最近 1,000 局，因此达到 1,000 局的窗口已截断。
        对手 metadata 不含卡组，只有对手 submission 同样属于本次 Top 20 且已审计卡组时才归入牌型；
        当前为 <b>{known_games} / {total_games:,}（{_percent(matchup_coverage, 2)}）</b>。
        未识别的 {total_games - known_games:,} 条记录没有被猜测分配。</div>
    </aside>

    <nav class="toolbar" aria-label="报告筛选">
      <input id="search" type="search" placeholder="搜索选手、source ID 或牌型…">
      <select id="archetype-filter">
        <option value="">全部牌型</option>{archetype_options}
      </select>
      <select id="sort-order">
        <option value="rank">按排名</option>
        <option value="win-rate">按总胜率</option>
        <option value="games">按公开场次</option>
      </select>
      <span class="visible-count"><b id="visible-count">{len(players)}</b> / {len(players)} 位</span>
    </nav>

    <section class="panel">
      <div class="section-heading">
        <div><p class="eyebrow">ARCHETYPE MAP</p><h2>8 种主流牌型</h2></div>
        <p>按关键 Pokémon 确定性分类；“exact deck”仍按完整 60 张规范化 hash 区分。</p>
      </div>
      <div class="archetype-grid">{archetype_chips}</div>
    </section>

    <section class="panel">
      <div class="section-heading">
        <div><p class="eyebrow">LADDER OVERVIEW</p><h2>排行榜与总窗口战绩</h2></div>
        <p>加权窗口胜率 {_percent(weighted_win_rate)}；不同选手公开窗口长度不同，不能把场次简单理解为同长度测试。</p>
      </div>
      <div class="table-scroll">
        <table>
          <thead><tr><th>Rank</th><th>选手</th><th>卡组</th><th>场次</th>
            <th>W-L-D</th><th>胜率</th><th>已识别 matchup</th></tr></thead>
          <tbody>{overview_rows}</tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <div class="section-heading">
        <div><p class="eyebrow">MATCHUP MATRIX</p><h2>主流牌型胜率矩阵</h2></div>
        <p>单元格显示胜率与样本数；悬停查看 W-L-D。小样本只作审计线索，不代表稳定 matchup 结论。</p>
      </div>
      <div class="table-scroll">
        <table class="matrix-table">
          <thead><tr><th>选手</th>{matrix_headers}</tr></thead>
          <tbody>{matrix_rows}</tbody>
        </table>
      </div>
    </section>

    <div id="player-sections">{player_sections}</div>
    <div id="no-results" class="no-results panel">没有符合当前筛选条件的选手。</div>

    <section class="panel provenance">
      <div class="section-heading">
        <div><p class="eyebrow">PROVENANCE & LIMITS</p><h2>数据来源与复现口径</h2></div>
      </div>
      <ul>
        <li>冻结时间：<code>{_escape(captured_at)}</code>（即 Asia/Shanghai 2026-07-23 04:24 左右）。</li>
        <li>卡组：20 位选手各一场官方 replay；共 19 个唯一 replay、14 个 exact deck hash，
          每副都校验为 60 张。共享 replay 的双方各自还原自己的卡组。</li>
        <li>战绩：共 {total_games:,} 条 player-episode metadata；W-L-D 由双方 reward 比较得出。
          其中达到 API 1,000 局上限的选手会显示“API 上限截断”。</li>
        <li>Matchup：共 {known_games} 条 player-episode 能通过对手 submission ID 映射到已审计卡组；
          再按关键 Pokémon 聚合为 8 类。它不是完整 meta matchup 数据。</li>
        <li>卡图：以 <code>EN_Card_Data.csv</code> 的 Expansion + Collection No. 拼接
          PokemonTCG 数据目录所用图片 URL；ASC/POR 使用该目录发布的 Scrydex URL。
          图片保持远程引用，加载失败时页面回退显示卡名与 Card ID。</li>
        <li>源数据：
          <a href="../../../rl/artifact/dataset/top20_arena_20260723/campaign.json">
            campaign.json</a> 与
          <a href="../../../data/replays/kaggle_top20_bc_20260723/manifest.json">
            raw manifest</a>。</li>
      </ul>
    </section>
  </main>
  <script>
    (() => {{
      const search = document.getElementById('search');
      const filter = document.getElementById('archetype-filter');
      const sort = document.getElementById('sort-order');
      const container = document.getElementById('player-sections');
      const count = document.getElementById('visible-count');
      const empty = document.getElementById('no-results');

      function apply() {{
        const query = search.value.trim().toLowerCase();
        const archetype = filter.value;
        const details = Array.from(container.querySelectorAll('[data-player-detail]'));
        let visible = 0;
        for (const item of document.querySelectorAll('[data-player-detail], [data-player-row]')) {{
          const matchesQuery = !query || item.dataset.search.includes(query);
          const matchesArchetype = !archetype || item.dataset.archetype === archetype;
          item.hidden = !(matchesQuery && matchesArchetype);
          if (item.matches('[data-player-detail]') && !item.hidden) visible += 1;
        }}
        const direction = sort.value === 'rank' ? 1 : -1;
        const key = sort.value === 'win-rate' ? 'winRate' : sort.value;
        details.sort((a, b) => direction * (Number(a.dataset[key]) - Number(b.dataset[key])));
        details.forEach(item => container.appendChild(item));
        count.textContent = visible;
        empty.classList.toggle('visible', visible === 0);
      }}
      search.addEventListener('input', apply);
      filter.addEventListener('change', apply);
      sort.addEventListener('change', apply);
    }})();
  </script>
</body>
</html>
"""


def build_report(campaign_path: Path, card_data_path: Path, output_path: Path) -> None:
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    catalog = load_card_catalog(card_data_path)
    markup = render_report(campaign, catalog)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markup, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--card-data", type=Path, default=DEFAULT_CARD_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_report(args.campaign, args.card_data, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
