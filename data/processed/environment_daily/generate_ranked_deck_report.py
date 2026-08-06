"""Render a construction-first report from a frozen ranked exact-deck snapshot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import statistics
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT = (
    REPOSITORY_ROOT
    / "docs/environment-daily_kaggle_top100/ranked/data/2026-08-06-top500-exact.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "docs/environment-daily_kaggle_top100/ranked/2026-08-06-top500.html"
)
CARD_DATA = REPOSITORY_ROOT / "data/official/EN_Card_Data.csv"
EXACT_BINDINGS = {"leaderboard_score", "leaderboard_score_submission_date"}
CARD_IMAGE_SET_BY_EXPANSION = {
    "BLK": "zsv10pt5",
    "DRI": "sv10",
    "JTG": "sv9",
    "MEG": "me1",
    "PAL": "sv2",
    "PFL": "me2",
    "PRE": "sv8pt5",
    "SCR": "sv7",
    "SFA": "sv6pt5",
    "SSP": "sv8",
    "SVE": "sve",
    "SVI": "sv1",
    "TEF": "sv5",
    "TWM": "sv6",
    "WHT": "rsv10pt5",
}
SCRYDEX_SET_BY_EXPANSION = {"ASC": "me2pt5", "POR": "me3"}


def _catalog() -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    with CARD_DATA.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(int(row["Card ID"]), row)
    return result


def _deck_hash(deck: Iterable[int]) -> str:
    encoded = ",".join(str(card_id) for card_id in sorted(deck)).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _load_snapshot(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "pokemon_tcg_top500_exact_v1":
        raise ValueError(f"unsupported snapshot schema: {payload.get('schema')!r}")
    players = list(payload.get("players") or [])
    if len(players) != 500:
        raise ValueError(f"expected 500 players, got {len(players)}")
    ranks = sorted(int(player["rank"]) for player in players)
    if ranks != list(range(1, 501)):
        raise ValueError("snapshot must contain each rank from 1 through 500 exactly once")
    for player in players:
        deck = [int(card_id) for card_id in player.get("deck") or []]
        if len(deck) != 60:
            raise ValueError(f"rank {player['rank']} deck has {len(deck)} cards, expected 60")
        player["deck"] = sorted(deck)
        player["deck_hash"] = _deck_hash(deck)
        player["card_counts"] = Counter(deck)
    players.sort(key=lambda player: int(player["rank"]))
    return payload, players


def _card_group(row: dict[str, str] | None) -> str:
    if not row:
        return "Unknown"
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
        size = "large" if hires else "small"
        return f"https://images.scrydex.com/pokemon/{scrydex_set}-{number}/{size}"
    set_id = CARD_IMAGE_SET_BY_EXPANSION.get(row["Expansion"])
    if not set_id:
        return ""
    suffix = "_hires" if hires else ""
    return f"https://images.pokemontcg.io/{set_id}/{number}{suffix}.png"


def _card_thumb(card_id: int, catalog: dict[int, dict[str, str]], size: str = "small") -> str:
    row = catalog.get(card_id)
    name = row["Card Name"] if row else f"Card ID {card_id}"
    source = _image_url(row)
    preview = _image_url(row, hires=True)
    dimensions = {"tiny": (24, 34), "small": (34, 48), "large": (112, 157)}
    width, height = dimensions[size]
    image = ""
    if source:
        image = (
            f'<img src="{html.escape(source)}" alt="{html.escape(name)}" loading="lazy" '
            f'width="{width}" height="{height}" decoding="async" referrerpolicy="no-referrer" '
            'onerror="this.hidden=true;this.parentElement.classList.add(\'thumb-failed\')">'
        )
    return (
        f'<span class="card-thumb {size}" tabindex="0" '
        f'data-card-name="{html.escape(name)}" data-card-preview-url="{html.escape(preview)}" '
        f'title="{html.escape(name)} · Card ID {card_id}">{image}'
        f'<span class="thumb-fallback">ID {card_id}</span></span>'
    )


def _representative_cards(
    archetype: str,
    players: list[dict[str, object]],
    catalog: dict[int, dict[str, str]],
) -> list[int]:
    presence: Counter[int] = Counter()
    for player in players:
        presence.update(set(player["deck"]))
    requested = [part.strip() for part in archetype.split("/")]
    aliases = {
        "Festival Lead": "Dipplin",
        "Froslass": "Froslass",
        "Roserade": "Cynthia's Roserade",
        "Solrock": "Solrock",
        "Spidops": "Team Rocket's Spidops",
        "Hero’s Cape": "Teal Mask Ogerpon ex",
    }
    selected: list[int] = []
    for name in requested:
        name = aliases.get(name, name)
        match = next(
            (
                card_id
                for card_id in presence
                if catalog.get(card_id, {}).get("Card Name") == name
            ),
            None,
        )
        if match is not None and match not in selected:
            selected.append(match)
    candidates: list[tuple[int, int, int]] = []
    for card_id, count in presence.items():
        row = catalog.get(card_id)
        if _card_group(row) != "Pokémon":
            continue
        stage = (row or {}).get("Stage (Pokémon)/Type (Energy and Trainer)", "")
        priority = 3 if "Stage 2" in stage else 2 if "Stage 1" in stage else 1
        name = (row or {}).get("Card Name", "")
        priority += 2 if " ex" in name or name.startswith("Mega ") else 0
        candidates.append((priority, count, card_id))
    candidates.sort(reverse=True)
    selected.extend(card_id for _, _, card_id in candidates if card_id not in selected)
    return selected[:2]


def _archetype_visual(
    archetype: str,
    representatives: dict[str, list[int]],
    catalog: dict[int, dict[str, str]],
) -> str:
    thumbs = "".join(
        _card_thumb(card_id, catalog, "tiny") for card_id in representatives[archetype]
    )
    return (
        f'<span class="archetype-visual"><span class="archetype-thumbs">{thumbs}</span>'
        f'<span>{html.escape(archetype)}</span></span>'
    )


def _variant_groups(players: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for player in players:
        grouped[str(player["archetype"])][str(player["deck_hash"])].append(player)
    result: dict[str, list[dict[str, object]]] = {}
    for archetype, variants in grouped.items():
        rows = []
        for deck_hash, users in variants.items():
            users.sort(key=lambda player: int(player["rank"]))
            rows.append(
                {
                    "hash": deck_hash,
                    "players": users,
                    "deck": users[0]["deck"],
                    "count": len(users),
                    "top100": sum(int(player["rank"]) <= 100 for player in users),
                    "best_rank": int(users[0]["rank"]),
                    "median_rank": statistics.median(int(player["rank"]) for player in users),
                }
            )
        rows.sort(key=lambda row: (-int(row["count"]), int(row["best_rank"]), str(row["hash"])))
        result[archetype] = rows
    return result


def _card_pool(
    players: list[dict[str, object]], catalog: dict[int, dict[str, str]]
) -> list[dict[str, object]]:
    investments: dict[int, list[int]] = defaultdict(list)
    for player in players:
        for card_id, copies in player["card_counts"].items():
            investments[int(card_id)].append(int(copies))
    rows = []
    for card_id, copies in investments.items():
        row = catalog.get(card_id)
        distribution = Counter(copies)
        rows.append(
            {
                "card_id": card_id,
                "name": row["Card Name"] if row else f"Card ID {card_id}",
                "group": _card_group(row),
                "decks": len(copies),
                "coverage": len(copies) / len(players),
                "total": sum(copies),
                "mean": statistics.mean(copies),
                "median": statistics.median(copies),
                "minimum": min(copies),
                "maximum": max(copies),
                "distribution": distribution,
            }
        )
    rows.sort(key=lambda item: (-int(item["decks"]), -int(item["total"]), str(item["name"])))
    return rows


def _deck_grid(deck: list[int], catalog: dict[int, dict[str, str]]) -> str:
    counts = Counter(deck)
    groups = {"Pokémon": [], "Trainer": [], "Energy": [], "Unknown": []}
    for card_id, count in counts.items():
        groups[_card_group(catalog.get(card_id))].append((card_id, count))
    blocks = []
    for group in ("Pokémon", "Trainer", "Energy", "Unknown"):
        cards = groups[group]
        if not cards:
            continue
        cards.sort(key=lambda item: catalog.get(item[0], {}).get("Card Name", str(item[0])))
        tiles = "".join(
            f'<article class="card-tile"><div class="card-art">{_card_thumb(card_id, catalog, "large")}'
            f'<b class="count">×{count}</b></div><b>{html.escape(catalog.get(card_id, {}).get("Card Name", f"Card ID {card_id}"))}</b>'
            f'<small>ID {card_id}</small></article>'
            for card_id, count in cards
        )
        blocks.append(f'<div class="deck-group"><h4>{group} · {sum(n for _, n in cards)} 张</h4><div class="card-grid">{tiles}</div></div>')
    return "".join(blocks)


def _pool_table(
    scope: str,
    players: list[dict[str, object]],
    catalog: dict[int, dict[str, str]],
) -> str:
    rows = []
    for item in _card_pool(players, catalog):
        distribution = " · ".join(
            f"{copies}×:{decks}套" for copies, decks in sorted(item["distribution"].items())
        )
        rows.append(
            f'<tr data-pool-card data-search="{html.escape(str(item["name"]).lower())}"><td><span class="pool-card">'
            f'{_card_thumb(int(item["card_id"]), catalog)}<span><b>{html.escape(str(item["name"]))}</b>'
            f'<small>ID {item["card_id"]}</small></span></span></td><td>{item["group"]}</td>'
            f'<td><b>{item["decks"]}</b></td><td>{100 * float(item["coverage"]):.1f}%</td>'
            f'<td>{item["total"]}</td><td>{float(item["mean"]):.2f}</td>'
            f'<td>{float(item["median"]):g}</td><td>{item["minimum"]}–{item["maximum"]}</td>'
            f'<td>{distribution}</td></tr>'
        )
    return (
        f'<details class="pool-scope" {"open" if scope == "Top 100" else ""}><summary><b>{scope} 全局构筑卡池</b>'
        f'<span>{len(players)} 套 exact deck · {len(rows)} 种卡</span></summary><div class="pool-tools">'
        f'<label>筛选 {scope} 卡牌<input type="search" data-pool-search placeholder="卡名"></label></div>'
        f'<div class="table-wrap pool-scroll"><table class="pool-table"><thead><tr><th>卡牌资料</th><th>类别</th>'
        '<th>使用构筑数</th><th>覆盖率</th><th>合计投入</th><th>使用时均值</th><th>中位数</th><th>范围</th><th>投入分布</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div></details>'
    )


def _render_html(
    payload: dict[str, object], players: list[dict[str, object]], catalog: dict[int, dict[str, str]]
) -> str:
    top100 = players[:100]
    variants = _variant_groups(players)
    by_archetype: dict[str, list[dict[str, object]]] = defaultdict(list)
    for player in players:
        by_archetype[str(player["archetype"])].append(player)
    ordered = sorted(by_archetype, key=lambda name: (-len(by_archetype[name]), name))
    representatives = {
        name: _representative_cards(name, by_archetype[name], catalog) for name in ordered
    }
    top100_counts = Counter(str(player["archetype"]) for player in top100)
    top500_counts = Counter(str(player["archetype"]) for player in players)
    exact_bindings = sum(str(player.get("binding")) in EXACT_BINDINGS for player in players)
    unique_decks = len({str(player["deck_hash"]) for player in players})

    distribution_rows = []
    max_count = max(top500_counts.values())
    for name in ordered:
        t100 = top100_counts[name]
        t500 = top500_counts[name]
        top100_share = t100 / 100
        top500_share = t500 / 500
        index = top100_share / top500_share if top500_share else 0.0
        ranks = [int(player["rank"]) for player in by_archetype[name]]
        distribution_rows.append(
            f'<tr data-archetype-row><td>{_archetype_visual(name, representatives, catalog)}</td>'
            f'<td><b>{t100}</b><small>{100 * top100_share:.1f}%</small></td>'
            f'<td><b>{t500}</b><small>{100 * top500_share:.1f}%</small></td>'
            f'<td>{index:.2f}×</td><td>{len(variants[name])}</td><td>#{min(ranks)}</td>'
            f'<td>#{statistics.median(ranks):g}</td><td><span class="share-track"><i style="width:{100*t500/max_count:.1f}%"></i></span></td></tr>'
        )

    variant_sections = []
    variant_number = 0
    for name in ordered:
        rows = []
        for variant in variants[name]:
            variant_number += 1
            users = variant["players"]
            user_links = "、".join(
                f'<a href="#player-{int(player["rank"]):03d}">#{player["rank"]} {html.escape(str(player["team_name"]))}</a>'
                for player in users
            )
            rows.append(
                f'<details class="variant" data-variant-row data-search="{html.escape((name + " " + str(variant["hash"]) + " " + " ".join(str(p["team_name"]) for p in users)).lower())}">'
                f'<summary><span><b>V{variant_number:03d}</b><code>{variant["hash"]}</code></span>'
                f'<span><b>{variant["count"]} 人完全相同</b> · T100 {variant["top100"]} · 最佳 #{variant["best_rank"]} · 中位 #{float(variant["median_rank"]):g}</span></summary>'
                f'<div class="variant-body"><p>{user_links}</p>{_deck_grid(variant["deck"], catalog)}</div></details>'
            )
        leading = variants[name][0]
        canonical_share = int(leading["count"]) / len(by_archetype[name])
        open_attribute = (
            "open"
            if name in {"Marnie's Grimmsnarl ex / Froslass", "Alakazam / Dudunsparce"}
            else ""
        )
        variant_sections.append(
            f'<details class="archetype-variants" {open_attribute}>'
            f'<summary>{_archetype_visual(name, representatives, catalog)}<span>{len(by_archetype[name])} 人 · {len(rows)} 种 exact 变种 · 主流同构率 {100*canonical_share:.1f}%</span></summary>'
            f'<div class="variant-list">{"".join(rows)}</div></details>'
        )

    over_indexed = sorted(
        (
            (name, top100_counts[name] / 100 / (top500_counts[name] / 500))
            for name in ordered
            if top500_counts[name] >= 5
        ),
        key=lambda item: -item[1],
    )[:5]
    rare_top = [
        name for name in ordered if top100_counts[name] and top500_counts[name] <= 2
    ]
    clones = sorted(
        (variant for rows in variants.values() for variant in rows),
        key=lambda row: (-int(row["count"]), int(row["best_rank"])),
    )[:8]
    over_cards = "".join(
        f'<li>{_archetype_visual(name, representatives, catalog)}<b>{ratio:.2f}×</b><small>T100 {top100_counts[name]} / T500 {top500_counts[name]}</small></li>'
        for name, ratio in over_indexed
    )
    rare_cards = "".join(
        f'<li>{_archetype_visual(name, representatives, catalog)}<b>#{min(int(p["rank"]) for p in by_archetype[name])}</b>'
        f'<small>T500 仅 {top500_counts[name]} 人</small></li>' for name in rare_top
    ) or '<li><span>没有满足“Top 100 出现且 T500 ≤2 人”的孤例</span></li>'
    clone_rows = "".join(
        f'<tr><td><code>{variant["hash"]}</code></td><td>{html.escape(str(variant["players"][0]["archetype"]))}</td>'
        f'<td><b>{variant["count"]}</b></td><td>{variant["top100"]}</td><td>#{variant["best_rank"]}</td></tr>'
        for variant in clones
    )

    rank_rows = []
    player_details = []
    for player in players:
        name = str(player["archetype"])
        binding_ok = str(player.get("binding")) in EXACT_BINDINGS
        search = f'{player["rank"]} {player["team_name"]} {name} {player["deck_hash"]}'.lower()
        rank_rows.append(
            f'<tr data-player-row data-scope="{100 if int(player["rank"]) <= 100 else 500}" data-search="{html.escape(search)}">'
            f'<td><b>#{player["rank"]}</b></td><td><a href="#player-{int(player["rank"]):03d}">{html.escape(str(player["team_name"]))}</a></td>'
            f'<td>{_archetype_visual(name, representatives, catalog)}</td><td>{float(player["score"]):.1f}</td>'
            f'<td><code>{player["deck_hash"]}</code></td><td><span class="status {"strict" if binding_ok else "limited"}">'
            f'{"榜分绑定" if binding_ok else "日期回退"}</span></td></tr>'
        )
        player_details.append(
            f'<details class="player-detail" id="player-{int(player["rank"]):03d}" data-player-detail data-scope="{100 if int(player["rank"]) <= 100 else 500}" data-search="{html.escape(search)}">'
            f'<summary><span><b>#{player["rank"]} {html.escape(str(player["team_name"]))}</b>{_archetype_visual(name, representatives, catalog)}</span>'
            f'<span>{float(player["score"]):.1f} 分 · <code>{player["deck_hash"]}</code> · {"榜分绑定" if binding_ok else "日期回退"}</span></summary>'
            f'<div class="player-body"><p>submission {player["submission_id"]} · 60/60 cards · 数据绑定：{html.escape(str(player.get("binding")))}</p>'
            f'{_deck_grid(player["deck"], catalog)}</div></details>'
        )

    css = """
:root{color-scheme:light;--bg:#eef1ed;--paper:#fff;--ink:#17221e;--muted:#64716b;--line:#d6ded9;--green:#12664d;--green2:#228566;--amber:#b87811;--soft:#e9f4ee;--blue:#315f89;--danger:#9b4b32}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}a{color:var(--green);text-decoration:none}a:hover{text-decoration:underline}nav{position:sticky;top:0;z-index:20;display:flex;gap:4px;overflow:auto;padding:9px max(16px,calc((100% - 1440px)/2));border-bottom:1px solid #2f5547;background:#153d31;color:#fff}nav a{flex:0 0 auto;padding:7px 10px;border-radius:6px;color:#eaf7f1}nav a:hover{background:#ffffff18;text-decoration:none}.hero{padding:38px max(20px,calc((100% - 1400px)/2));background:#174c3d;color:#fff;border-bottom:6px solid #d29b36}.hero .eyebrow{color:#b9ddcf;font-size:12px;font-weight:800;text-transform:uppercase}.hero h1{max-width:980px;margin:6px 0 10px;font-size:42px;line-height:1.08}.hero p{max-width:950px;color:#d7e9e2;font-size:16px}.metrics{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:8px;max-width:1160px;margin-top:22px}.metric{min-height:82px;padding:12px;border:1px solid #ffffff30;border-radius:8px;background:#ffffff10}.metric b,.metric span{display:block}.metric b{font-size:25px}.metric span{color:#cce1d9}.band{padding:34px max(18px,calc((100% - 1440px)/2))}.band:nth-of-type(even){background:#f8faf8}.band-head{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:18px}.band-head h2{margin:0;font-size:27px}.band-head p{max-width:820px;margin:0;color:var(--muted)}.notice{padding:14px 16px;border-left:4px solid var(--amber);border-radius:4px;background:#fff5df}.scope-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.scope{min-height:150px;padding:16px;border:1px solid var(--line);border-radius:8px;background:#fff}.scope.available{border-top:4px solid var(--green)}.scope.unavailable{border-style:dashed;border-top:4px solid #9da9a3}.scope b,.scope span{display:block}.scope b{font-size:25px}.scope p{color:var(--muted)}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px;background:#fff}table{width:100%;border-collapse:collapse}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}th{position:sticky;top:0;z-index:2;background:#e8eee9;white-space:nowrap}td small{display:block;color:var(--muted)}.distribution-table{min-width:1050px}.share-track{display:block;width:110px;height:8px;border-radius:99px;background:#e3e9e5;overflow:hidden}.share-track i{display:block;height:100%;background:var(--green2)}.archetype-visual{display:flex;align-items:center;gap:7px;min-width:0}.archetype-thumbs{display:flex;flex:0 0 auto}.card-thumb{position:relative;display:inline-block;overflow:hidden;border:1px solid #c7d1cc;border-radius:4px;background:#edf1ef;vertical-align:middle}.card-thumb.tiny{width:24px;height:34px}.card-thumb.small{width:34px;height:48px}.card-thumb.large{width:112px;height:157px}.card-thumb img{position:absolute;z-index:1;inset:0;width:100%;height:100%;object-fit:contain}.thumb-fallback{position:absolute;inset:0;display:grid;place-items:center;color:#5a6761;font-size:7px}.thumb-failed{outline:1px dashed var(--danger)}.archetype-thumbs .card-thumb+.card-thumb{margin-left:-4px}.archetype-variants,.pool-scope,.player-detail{margin:10px 0;border:1px solid var(--line);border-radius:8px;background:#fff}.archetype-variants>summary,.pool-scope>summary,.player-detail>summary{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:14px;cursor:pointer}.archetype-variants>summary>span:last-child,.pool-scope>summary>span:last-child,.player-detail>summary>span:last-child{color:var(--muted)}.variant-list{padding:0 12px 12px}.variant{border-top:1px solid var(--line)}.variant>summary{display:flex;justify-content:space-between;gap:16px;padding:11px 6px;cursor:pointer}.variant>summary span{display:flex;align-items:center;gap:9px}.variant-body,.player-body{padding:0 14px 18px}.variant-body>p{max-width:1100px}.deck-group{margin:18px 0}.deck-group h4{margin:0 0 8px;color:var(--green)}.card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(106px,1fr));gap:10px}.card-tile{min-width:0}.card-art{position:relative;aspect-ratio:2.5/3.5;border-radius:6px;background:#edf1ef}.card-art .card-thumb{width:100%;height:100%}.card-art .count{position:absolute;z-index:3;right:5px;top:5px;padding:2px 6px;border-radius:99px;background:#143e32;color:#fff}.card-tile>b{display:block;margin-top:5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px}.card-tile>small{color:var(--muted)}code{padding:2px 5px;border-radius:4px;background:#edf1ef}.insight-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.insight{padding:18px;border:1px solid var(--line);border-radius:8px;background:#fff}.insight h3{margin-top:0}.insight ul{margin:0;padding:0;list-style:none}.insight li{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:2px 12px;padding:9px 0;border-top:1px solid var(--line)}.insight li:first-child{border-top:0}.insight li small{grid-column:1/-1;color:var(--muted)}.clone-table{min-width:700px}.pool-scope>summary b,.pool-scope>summary span{display:block}.pool-tools{padding:0 14px 12px}.pool-tools label{display:flex;align-items:center;gap:10px}.pool-tools input,.filters input,.filters select{min-height:38px;padding:7px 10px;border:1px solid #b9c6c0;border-radius:6px;background:#fff}.pool-scroll{max-height:720px;margin:0 12px 12px}.pool-table{min-width:1120px}.pool-card{display:flex;align-items:center;gap:8px;min-width:210px}.pool-card span:last-child{min-width:0}.pool-card b,.pool-card small{display:block}.filters{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px}.filters input{flex:1 1 280px}.rank-table{min-width:900px}.status{display:inline-block;padding:3px 7px;border-radius:99px;font-size:11px;font-weight:800}.status.strict{background:#daf1e5;color:#0e6348}.status.limited{background:#fff0d4;color:#80520a}.player-detail>summary>span:first-child{display:flex;align-items:center;gap:12px}.evidence{display:grid;grid-template-columns:1fr 1fr;gap:12px}.evidence article{padding:16px;border-left:4px solid var(--blue);background:#fff}.evidence article.warning{border-color:var(--amber)}.evidence h3{margin-top:0}.card-preview{position:fixed;z-index:100;display:none;width:250px;padding:7px;border:1px solid #afbeb7;border-radius:8px;background:#fff;box-shadow:0 16px 45px #10231c44;pointer-events:none}.card-preview.visible{display:block}.card-preview img{display:block;width:234px;height:328px;object-fit:contain}.card-preview b{display:block;margin-top:5px;text-align:center}.empty{display:none;padding:18px;color:var(--muted)}footer{padding:24px max(18px,calc((100% - 1440px)/2));background:#163d32;color:#d8e9e2}.nowrap{white-space:nowrap}@media(max-width:980px){.metrics{grid-template-columns:repeat(3,1fr)}.scope-grid{grid-template-columns:1fr 1fr}.insight-grid,.evidence{grid-template-columns:1fr}}@media(max-width:620px){.hero h1{font-size:32px}.metrics{grid-template-columns:1fr 1fr}.scope-grid{grid-template-columns:1fr}.band-head,.archetype-variants>summary,.variant>summary,.player-detail>summary{align-items:flex-start;flex-direction:column}.card-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.card-thumb.large{width:100%;height:100%}.card-preview{width:190px}.card-preview img{width:174px;height:244px}}
"""
    script = """
const q=(s,r=document)=>r.querySelector(s),qa=(s,r=document)=>[...r.querySelectorAll(s)];
const preview=q('#card-preview'),previewImg=q('img',preview),previewName=q('b',preview);
function movePreview(e){const pad=14,w=preview.offsetWidth||250,h=preview.offsetHeight||360;preview.style.left=Math.min(innerWidth-w-pad,e.clientX+16)+'px';preview.style.top=Math.min(innerHeight-h-pad,e.clientY+16)+'px'}
document.addEventListener('pointerover',e=>{const card=e.target.closest('.card-thumb');if(!card||!card.dataset.cardPreviewUrl)return;previewImg.src=card.dataset.cardPreviewUrl;previewName.textContent=card.dataset.cardName;preview.classList.add('visible')});
document.addEventListener('pointermove',e=>{if(preview.classList.contains('visible'))movePreview(e)});document.addEventListener('pointerout',e=>{if(e.target.closest('.card-thumb'))preview.classList.remove('visible')});
qa('[data-pool-search]').forEach(input=>input.addEventListener('input',()=>{const root=input.closest('.pool-scope'),term=input.value.trim().toLowerCase();qa('[data-pool-card]',root).forEach(row=>row.hidden=!row.dataset.search.includes(term))}));
const rankSearch=q('#rank-search'),rankScope=q('#rank-scope');function filterRank(){const term=rankSearch.value.trim().toLowerCase(),limit=Number(rankScope.value);qa('[data-player-row]').forEach(row=>row.hidden=Number(row.dataset.scope)>limit||!row.dataset.search.includes(term));qa('[data-player-detail]').forEach(row=>row.hidden=Number(row.dataset.scope)>limit||!row.dataset.search.includes(term))}rankSearch.addEventListener('input',filterRank);rankScope.addEventListener('change',filterRank);
"""
    captured = html.escape(str(payload["captured_at_utc"]))
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026-08-06 Kaggle Top 500 exact deck 环境分布</title><style>{css}</style></head><body>
<nav><a href="#summary">摘要</a><a href="#rank-bands">排名分层</a><a href="#archetype-distribution">牌型分布</a><a href="#exact-variants">exact 变种</a><a href="#notable-decks">关注构筑</a><a href="#card-pool">卡池</a><a href="#rank-index">排名</a><a href="#evidence-boundary">证据</a></nav>
<header class="hero" id="summary"><div class="eyebrow">Pokémon TCG AI Battle · Ranked exact-deck snapshot</div><h1>2026-08-06 Top 500 构筑环境分布</h1><p>以排名和 exact 60-card deck 为主轴，分析高手实际采用的牌型、完全相同构筑与卡牌投入。近期 15 场 Episode reward 不进入强度判断，也不参与任何排序。</p><div class="metrics"><div class="metric"><b>500</b><span>完整 exact deck</span></div><div class="metric"><b>{len(ordered)}</b><span>牌型分类</span></div><div class="metric"><b>{unique_decks}</b><span>全局 exact 变种</span></div><div class="metric"><b>{exact_bindings}</b><span>榜分严格绑定</span></div><div class="metric"><b>{500-exact_bindings}</b><span>日期回退样本</span></div><div class="metric"><b>{captured[:10]}</b><span>冻结日期 UTC</span></div></div></header>
<main>
<section class="band" id="rank-bands"><div class="band-head"><div><h2>排名分层</h2><p>本次数据只覆盖前 500。Top 100 与 Top 500 可以精确重算；更深排名没有 exact deck，保留入口但不外推。</p></div></div><div class="scope-grid"><article class="scope available"><span>可计算</span><b>Top 100</b><p>前 100 名 exact deck；适合观察高排名集中度与高位过度代表。</p></article><article class="scope available"><span>可计算</span><b>Top 500</b><p>500 名完整构筑；适合估计 Kaggle ladder 的主流厚度和长尾。</p></article><article class="scope unavailable"><span>当前快照不可重建</span><b>Top 1000</b><p>缺少第 501–1000 名 exact deck，禁止用 Top 500 比例补齐。</p></article><article class="scope unavailable"><span>当前快照不可重建</span><b>Top 5000</b><p>缺少第 501–5000 名 exact deck，禁止把 leaderboard 人数当构筑证据。</p></article></div></section>
<section class="band" id="archetype-distribution"><div class="band-head"><div><h2>Top 100 vs Top 500 牌型分布</h2><p>“T100 指数”=Top 100 份额 ÷ Top 500 份额。大于 1 表示该牌型在高排名段更集中，不等同于因果强度。</p></div></div><div class="table-wrap"><table class="distribution-table"><thead><tr><th>牌型</th><th>Top 100</th><th>Top 500</th><th>T100 指数</th><th>exact 变种</th><th>最佳排名</th><th>中位排名</th><th>规模</th></tr></thead><tbody>{''.join(distribution_rows)}</tbody></table></div></section>
<section class="band" id="exact-variants"><div class="band-head"><div><h2>牌型内部 exact 变种</h2><p>同一变种要求全部 60 个 card ID 和投入数量完全相同。每行列出完全相同的使用人数、Top 100 人数、最佳/中位排名与所有使用者。</p></div></div><div class="notice"><b>如何读：</b>“变种多”不一定代表策略更丰富，也可能是训练者自行微调；“主流同构率高”则说明大量 Kaggle 选手直接复用了同一张 exact list。</div>{''.join(variant_sections)}</section>
<section class="band" id="notable-decks"><div class="band-head"><div><h2>特别值得关注的构筑</h2><p>只根据排名位置、稀有度和 exact-list 复用关系筛选，不使用 15 局胜率。</p></div></div><div class="insight-grid"><article class="insight"><h3>Top 100 过度代表</h3><ul>{over_cards}</ul></article><article class="insight"><h3>高位稀有构筑</h3><ul>{rare_cards}</ul></article></div><h3>复用人数最多的完全相同构筑</h3><div class="table-wrap"><table class="clone-table"><thead><tr><th>deck hash</th><th>牌型</th><th>完全相同人数</th><th>其中 Top 100</th><th>最佳排名</th></tr></thead><tbody>{clone_rows}</tbody></table></div><div class="notice"><b>Frozen opponent pool 启示：</b>环境权重不应简单按“牌型数量”均匀分配。更合理的第一版是同时保留牌型份额和 exact-list 复用率：主流同构 list 作为高权重锚点，牌型内的真实变种作为低权重多样性，Top 100 过度代表牌型再作为 curriculum 强化层。具体接模型与 frozen pool 更新应在下一阶段单独设计和验证。</div></section>
<section class="band" id="card-pool"><div class="band-head"><div><h2>全局构筑卡池</h2><p>沿用日报的 9 列审计表，分别重算 Top 100 和 Top 500。覆盖率按“包含该卡的 exact deck 数 / 分层总 deck 数”。</p></div></div>{_pool_table("Top 100", top100, catalog)}{_pool_table("Top 500", players, catalog)}</section>
<section class="band" id="rank-index"><div class="band-head"><div><h2>按排名查看 exact deck</h2><p>默认展示全部 500 名；可以切换 Top 100 并按选手、牌型或 hash 筛选。分数只用于复原排名，不作为本文的构筑强度指标。</p></div></div><div class="filters"><select id="rank-scope" aria-label="排名范围"><option value="500">Top 500</option><option value="100">Top 100</option></select><input id="rank-search" type="search" placeholder="筛选选手、牌型或 deck hash" aria-label="筛选排名"></div><div class="table-wrap"><table class="rank-table"><thead><tr><th>排名</th><th>选手</th><th>牌型</th><th>榜分</th><th>deck hash</th><th>身份链</th></tr></thead><tbody>{''.join(rank_rows)}</tbody></table></div></section>
<section class="band" id="player-details"><div class="band-head"><div><h2>500 位选手 exact 60-card deck</h2><p>逐人展开查看带卡面的完整构筑；同一个 hash 即完全相同构筑。</p></div></div>{''.join(player_details)}</section>
<section class="band" id="evidence-boundary"><div class="band-head"><div><h2>证据边界</h2><p>这份报告是构筑环境快照，不是 matchup、策略质量或模型表现评测。</p></div></div><div class="evidence"><article><h3>可以确认</h3><p>500 个排名槽位齐全；每套牌均为 60 张；Top 100 / Top 500 的牌型、exact 变种与卡池统计均直接由本次 JSON 重算。</p></article><article class="warning"><h3>不能确认</h3><p>仅 {exact_bindings}/500 行以 leaderboard score 严格绑定 submission；其余 {500-exact_bindings} 行为 submissionDate 回退且存在 score mismatch。因此全体分布应称为“这批排名选手的已观测 exact deck”，不能声称 500/500 都是产生当前榜分的那次提交。</p></article><article><h3>明确排除</h3><p>不使用 15 局胜率、W-L-D、结构性克制猜测或单局 replay 结果来判断强弱。排名本身也受提交次数、时间和 ladder 过程影响。</p></article><article><h3>来源</h3><p>冻结时间：{captured}。<a href="data/2026-08-06-top500-exact.json">exact snapshot JSON</a>；<a href="source/2026-08-06-top500-rough.html">原始粗报告</a>。卡名来自仓库官方 `EN_Card_Data.csv`；卡图只用于识别。</p></article></div></section>
</main><footer>2026-08-06 Top 500 ranked construction report · 不使用 15 局胜率作为强度证据</footer><div class="card-preview" id="card-preview"><img alt="卡牌大图预览"><b></b></div><script>{script}</script></body></html>'''


def render_report(snapshot_path: Path, output_path: Path) -> dict[str, object]:
    payload, players = _load_snapshot(snapshot_path)
    catalog = _catalog()
    missing = sorted({card_id for player in players for card_id in player["deck"] if card_id not in catalog})
    if missing:
        raise ValueError(f"snapshot contains card IDs absent from official catalog: {missing}")
    source = _render_html(payload, players, catalog)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")
    return {
        "players": len(players),
        "top100": sum(int(player["rank"]) <= 100 for player in players),
        "top500": len(players),
        "archetypes": len({str(player["archetype"]) for player in players}),
        "exact_variants": len({str(player["deck_hash"]) for player in players}),
        "exact_score_bindings": sum(
            str(player.get("binding")) in EXACT_BINDINGS for player in players
        ),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(render_report(args.snapshot, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
