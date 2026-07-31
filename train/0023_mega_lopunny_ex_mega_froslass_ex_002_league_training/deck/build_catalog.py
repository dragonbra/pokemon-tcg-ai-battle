"""Build a readable exact-deck catalog from a frozen daily report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


EXPANSION_IMAGE_SET = {
    "BLK": "zsv10pt5", "DRI": "sv10", "JTG": "sv9", "MEG": "me1",
    "PAL": "sv2", "PFL": "me2", "PRE": "sv8pt5", "SCR": "sv7",
    "SFA": "sv6pt5", "SSP": "sv8", "SVE": "sve", "SVI": "sv1",
    "TEF": "sv5", "TWM": "sv6", "WHT": "rsv10pt5",
}
SCRYDEX_IMAGE_SET = {"ASC": "me2pt5", "POR": "me3"}
def _slug(value: str) -> str:
    value = html.unescape(value).replace("&", " and ")
    value = value.replace("'", "").replace("’", "")
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value or "unknown"


def _deck_hash(deck: list[int]) -> str:
    payload = ",".join(str(card_id) for card_id in sorted(deck)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _read_catalog(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {int(row["Card ID"]): row for row in csv.DictReader(handle)}


def _parse_report(report: Path) -> tuple[list[dict], dict[str, dict]]:
    text = report.read_text(encoding="utf-8")
    audit_match = re.search(
        r'<script type="application/json" id="snapshot-audit">(.*?)</script>',
        text,
        re.DOTALL,
    )
    if not audit_match:
        raise ValueError("daily report has no snapshot-audit payload")
    audit = json.loads(html.unescape(audit_match.group(1)))
    selected = {int(row["rank"]): row for row in audit["selected"]}

    detail_chunks = re.findall(
        r'<details[^>]*data-player-detail[^>]*>.*?</details>',
        text,
        re.DOTALL,
    )
    if len(detail_chunks) != int(audit["audited_players"]):
        raise ValueError(
            f"expected {audit['audited_players']} details, got {len(detail_chunks)}"
        )

    rows: list[dict] = []
    for chunk in detail_chunks:
        rank_match = re.search(r'<summary>.*?#(\d+)\s+(.*?)</b>', chunk, re.DOTALL)
        archetype_match = re.search(r'data-archetype="([^"]+)"', chunk)
        record_match = re.search(
            r'<summary>.*?<span>(\d+)-(\d+)-(\d+) · ([0-9.]+)%</span>',
            chunk,
            re.DOTALL,
        )
        if not rank_match or not archetype_match or not record_match:
            raise ValueError("could not parse a player detail header")
        rank = int(rank_match.group(1))
        selected_row = selected.get(rank)
        if selected_row is None:
            raise ValueError(f"rank {rank} is missing from snapshot audit")
        card_matches = re.findall(
            r'<article class="card-tile">.*?'
            r'data-card-name="([^"]+)".*?Card ID (\d+)".*?'
            r'class="deck-count">×(\d+)</span>.*?'
            r'<small>Card ID \d+ · \d+ 张</small></article>',
            chunk,
            re.DOTALL,
        )
        cards = []
        for name, card_id, count in card_matches:
            cards.append({"name": html.unescape(name), "card_id": int(card_id), "count": int(count)})
        if sum(card["count"] for card in cards) != 60:
            raise ValueError(f"rank {rank} deck is not exact 60 cards")
        rows.append(
            {
                "rank": rank,
                "team_name": html.unescape(selected_row["team_name"]),
                "submission_id": int(selected_row["submission_id"]),
                "episode_id": int(selected_row["episode_id"]),
                "episode_create_time": selected_row["episode_create_time"],
                "player_index": int(selected_row["episode_player_index"]),
                "leaderboard_score": float(selected_row["leaderboard_score"]),
                "deck_sha256": selected_row["deck_sha256"],
                "archetype": html.unescape(archetype_match.group(1)),
                "wins": int(record_match.group(1)),
                "losses": int(record_match.group(2)),
                "draws": int(record_match.group(3)),
                "reported_win_rate": float(record_match.group(4)) / 100,
                "cards": cards,
            }
        )
    return rows, audit


def _image_url(row: dict[str, str], hires: bool = False) -> str:
    expansion = row.get("Expansion", "")
    number = row.get("Collection No.", "")
    suffix = "_hires.png" if hires else ".png"
    if expansion in SCRYDEX_IMAGE_SET:
        size = "large" if hires else "small"
        return f"https://images.scrydex.com/pokemon/{SCRYDEX_IMAGE_SET[expansion]}-{number}/{size}"
    image_set = EXPANSION_IMAGE_SET.get(expansion, expansion.lower())
    return f"https://images.pokemontcg.io/{image_set}/{number}{suffix}"


def _card_group(row: dict[str, str]) -> str:
    card_type = row.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
    if "Pokémon" in card_type:
        return "Pokemon"
    if "Energy" in card_type:
        return "Energy"
    return "Trainer"


def _card_html(card: dict, catalog: dict[int, dict[str, str]]) -> str:
    row = catalog[card["card_id"]]
    name = html.escape(row["Card Name"])
    thumb = _image_url(row)
    large = _image_url(row, hires=True)
    cid = card["card_id"]
    return (
        '<article class="card-tile">'
        f'<button class="card-thumb" type="button" data-preview="{html.escape(large)}" '
        f'title="{name} · Card ID {cid}"><img src="{html.escape(thumb)}" alt="{name}" '
        'loading="lazy"><span class="thumb-fallback">'
        f'ID {cid}<small>{name}</small></span></button>'
        f'<b>{name}</b><small>Card ID {cid} · x{card["count"]}</small></article>'
    )


def _catalog_identity(best: dict, cards: list[dict]) -> tuple[str, str]:
    names = {card["name"] for card in cards}
    if best["deck_sha256"] == "f50fa3a23cdf21be7cf7d3f558b8ff0b82e8d4e7ba8f61b7b4cacc1a0080c16a":
        return (
            "Raging Bolt ex / Mega Kangaskhan ex",
            "raging_bolt_ex_james_cox_henry_chao",
        )
    if best["archetype"] == "Dragapult ex" and names & {"Dunsparce", "Dudunsparce", "Dudunsparce ex"}:
        return "Dragapult ex / Dunsparce", "dragapult_ex_dunsparce"
    return best["archetype"], _slug(best["archetype"].replace(" / ", "_"))


def _aggregate(rows: list[dict], catalog: dict[int, dict[str, str]]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["deck_sha256"]].append(row)
    result = []
    for full_hash, members in grouped.items():
        members.sort(key=lambda item: item["rank"])
        cards = members[0]["cards"]
        ids = [card["card_id"] for card in cards for _ in range(card["count"])]
        if _deck_hash(ids) != full_hash:
            raise ValueError(f"full hash mismatch for rank {members[0]['rank']}")
        if any(
            Counter((card["card_id"], card["count"]) for card in member["cards"])
            != Counter((card["card_id"], card["count"]) for card in cards)
            for member in members[1:]
        ):
            raise ValueError(f"inconsistent card contents for hash {full_hash}")
        wins = sum(member["wins"] for member in members)
        losses = sum(member["losses"] for member in members)
        draws = sum(member["draws"] for member in members)
        n = wins + losses + draws
        best = members[0]
        display_archetype, key_slug = _catalog_identity(best, cards)
        result.append(
            {
                "deck_sha256": full_hash,
                "short_hash": full_hash[:12],
                "archetype": display_archetype,
                "source_archetype": best["archetype"],
                "cards": cards,
                "members": members,
                "count": len(members),
                "best_rank": best["rank"],
                "best_score": best["leaderboard_score"],
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "games": n,
                "effective_win_rate": (wins + 0.5 * draws) / n if n else None,
                "key_slug": key_slug,
            }
        )
    result.sort(key=lambda item: (item["best_rank"], item["short_hash"]))
    counters: Counter[str] = Counter()
    for item in result:
        counters[item["key_slug"]] += 1
        item["directory"] = f"{item['key_slug']}_{counters[item['key_slug']]:03d}"
        for card in item["cards"]:
            if card["card_id"] not in catalog:
                raise ValueError(f"unknown official Card ID {card['card_id']}")
    return result


def _write_deck(
    item: dict,
    output: Path,
    report: Path,
    catalog: dict[int, dict[str, str]],
    refresh: bool = False,
) -> None:
    directory = output / item["directory"]
    directory.mkdir(parents=True, exist_ok=refresh)
    deck_ids = [card["card_id"] for card in item["cards"] for _ in range(card["count"])]
    (directory / "deck.csv").write_text("\n".join(map(str, sorted(deck_ids))) + "\n", encoding="ascii")
    manifest = {
        "schema_version": "daily_deck_catalog_v1",
        "directory": item["directory"],
        "deck_sha256": item["deck_sha256"],
        "short_hash": item["short_hash"],
        "archetype": item["archetype"],
        "source_archetype": item["source_archetype"],
        "source_report": str(report),
        "source_date": "2026-07-30",
        "member_count": item["count"],
        "best_rank": item["best_rank"],
        "best_score": item["best_score"],
        "aggregate_record": {"wins": item["wins"], "losses": item["losses"], "draws": item["draws"]},
        "aggregate_effective_win_rate": item["effective_win_rate"],
        "members": [
            {key: member[key] for key in (
                "rank", "team_name", "submission_id", "episode_id", "episode_create_time",
                "player_index", "leaderboard_score", "archetype", "wins", "losses", "draws",
            )}
            for member in item["members"]
        ],
        "cards": [
            {"card_id": card["card_id"], "name": catalog[card["card_id"]]["Card Name"], "count": card["count"]}
            for card in item["cards"]
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_external_references(output: Path, catalog: dict[int, dict[str, str]]) -> list[dict]:
    references = []
    if not output.is_dir():
        return references
    for manifest_path in sorted(output.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "external_deck_reference_v1":
            continue
        directory = manifest_path.parent
        if manifest.get("directory") != directory.name:
            raise ValueError(f"external manifest directory mismatch: {manifest_path}")
        deck = [int(line) for line in (directory / "deck.csv").read_text(encoding="ascii").splitlines()]
        if len(deck) != 60:
            raise ValueError(f"external deck is not exact 60 cards: {directory}")
        if _deck_hash(deck) != manifest.get("deck_sha256"):
            raise ValueError(f"external deck hash mismatch: {directory}")
        card_counts = Counter(deck)
        cards = []
        for card_id, count in sorted(card_counts.items()):
            if card_id not in catalog:
                raise ValueError(f"unknown official Card ID {card_id} in {directory}")
            cards.append({"card_id": card_id, "name": catalog[card_id]["Card Name"], "count": count})
        recorded_cards = {
            int(card["card_id"]): (card["name"], int(card["count"]))
            for card in manifest.get("cards", [])
        }
        expected_cards = {
            card["card_id"]: (card["name"], card["count"])
            for card in cards
        }
        if recorded_cards != expected_cards:
            raise ValueError(f"external manifest card list mismatch: {directory}")
        references.append({**manifest, "cards": cards})
    return references


def _external_reference_html(item: dict, catalog: dict[int, dict[str, str]]) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    for card in item["cards"]:
        groups[_card_group(catalog[card["card_id"]])].append(card)
    group_html = ""
    for group in ("Pokemon", "Trainer", "Energy"):
        cards = groups.get(group, [])
        if cards:
            tiles = "".join(_card_html(card, catalog) for card in cards)
            group_html += (
                f'<section class="deck-group"><h3>{group} '
                f'<small>{sum(card["count"] for card in cards)} cards</small></h3>'
                f'<div class="card-grid">{tiles}</div></section>'
            )
    source_bits = [item.get("source_label") or item["source_kind"]]
    if item.get("source_url"):
        source_bits.append(f'<a href="{html.escape(item["source_url"])}">source record</a>')
    if item.get("source_path"):
        source_bits.append(f'<code>{html.escape(item["source_path"])}</code>')
    facts = []
    if item.get("player_name"):
        facts.append(f'<span><b>{html.escape(item["player_name"])}</b><small>player</small></span>')
    if item.get("event"):
        facts.append(f'<span><b>{html.escape(item["event"])}</b><small>event</small></span>')
    if item.get("placement") is not None:
        facts.append(f'<span><b>#{int(item["placement"])}</b><small>placement</small></span>')
    if item.get("record"):
        record = item["record"]
        facts.append(
            f'<span><b>{record["wins"]}-{record["losses"]}-{record["draws"]}</b>'
            '<small>record</small></span>'
        )
    facts_html = f'<div class="record-grid">{"".join(facts)}</div>' if facts else ""
    notes = "".join(
        f'<p class="note"><b>{label}:</b> {html.escape(item[key])}</p>'
        for key, label in (("classification_note", "Classification"), ("usage_note", "Use"))
        if item.get(key)
    )
    label = item.get("catalog_label", item["archetype"])
    search = f'{label} {item["directory"]} {item["short_hash"]}'.lower()
    return (
        f'<details class="deck-card external" data-archetype="{html.escape(item["archetype"])}" '
        f'data-search="{html.escape(search)}"><summary><span class="deck-title">'
        f'<b>{html.escape(label)}</b><small>{item["directory"]} · {item["short_hash"]}</small></span>'
        '<span class="deck-stats"><b>External reference</b> · no Kaggle snapshot metrics</span></summary>'
        f'<div class="detail-body"><p class="provenance">Full hash <code>{item["deck_sha256"]}</code> · '
        f'{" · ".join(source_bits)}</p>{facts_html}{notes}{group_html}'
        f'<p><a href="{item["directory"]}/deck.csv">Open deck.csv</a> · '
        f'<a href="{item["directory"]}/manifest.json">Open manifest</a></p></div></details>'
    )


def _render_index(
    items: list[dict],
    external_references: list[dict],
    row_count: int,
    output: Path,
    report: Path,
    catalog: dict[int, dict[str, str]],
) -> None:
    cards_by_deck = []
    for item in items:
        groups: dict[str, list[dict]] = defaultdict(list)
        for card in item["cards"]:
            groups[_card_group(catalog[card["card_id"]])].append(card)
        group_html = ""
        for group in ("Pokemon", "Trainer", "Energy"):
            cards = groups.get(group, [])
            if not cards:
                continue
            label = {"Pokemon": "Pokemon", "Trainer": "Trainer", "Energy": "Energy"}[group]
            tiles = "".join(_card_html(card, catalog) for card in cards)
            group_html += f'<section class="deck-group"><h3>{label} <small>{sum(c["count"] for c in cards)} cards</small></h3><div class="card-grid">{tiles}</div></section>'
        rate = "-" if item["effective_win_rate"] is None else f'{item["effective_win_rate"] * 100:.1f}%'
        members = "".join(
            f'<tr><td>#{member["rank"]}</td><td>{html.escape(member["team_name"])}</td>'
            f'<td>{member["leaderboard_score"]:.1f}</td><td>{member["wins"]}-{member["losses"]}-{member["draws"]}</td></tr>'
            for member in item["members"]
        )
        cards_by_deck.append(
            f'<details class="deck-card" data-archetype="{html.escape(item["archetype"])}" data-search="{html.escape((item["archetype"] + " " + item["short_hash"]).lower())}">'
            f'<summary><span class="deck-title"><b>{html.escape(item["archetype"])}</b><small>{item["directory"]} · {item["short_hash"]}</small></span>'
            f'<span class="deck-stats"><b>{item["count"]} deck(s)</b> · best #{item["best_rank"]} · {item["best_score"]:.1f} · {rate}</span></summary>'
            f'<div class="detail-body"><p class="provenance">Full hash <code>{item["deck_sha256"]}</code> · {html.escape(str(report))}</p>'
            f'<div class="record-grid"><span><b>{item["count"]}</b><small>Top 100 entries</small></span><span><b>#{item["best_rank"]}</b><small>best rank</small></span><span><b>{item["best_score"]:.1f}</b><small>best score</small></span><span><b>{rate}</b><small>aggregate W-L-D {item["wins"]}-{item["losses"]}-{item["draws"]}</small></span></div>'
            f'<h3>Users and leaderboard evidence</h3><div class="table-scroll"><table><thead><tr><th>Rank</th><th>Team</th><th>Score</th><th>W-L-D</th></tr></thead><tbody>{members}</tbody></table></div>{group_html}'
            f'<p><a href="{item["directory"]}/deck.csv">Open deck.csv</a> · <a href="{item["directory"]}/manifest.json">Open manifest</a></p></div></details>'
        )
    external_cards = [_external_reference_html(item, catalog) for item in external_references]
    archetypes = sorted({item["archetype"] for item in items + external_references})
    options = ''.join(f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name in archetypes)
    html_text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Daily Deck Catalog</title>
<style>
:root{{--bg:#f4f1ea;--ink:#1d2b2b;--muted:#66716e;--panel:#fffdf8;--line:#d9d4c9;--accent:#0d7b75;--accent2:#d66a3d}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}main{{max-width:1420px;margin:auto;padding:28px 22px 80px}}.eyebrow{{color:var(--accent);font-size:12px;font-weight:800;letter-spacing:.08em}}h1{{font-size:clamp(28px,4vw,48px);margin:5px 0}}h2{{margin:0 0 8px}}h3{{margin:20px 0 8px}}p{{margin:8px 0}}.section-heading{{margin:30px 0 10px}}.hero,.panel,.deck-card{{background:var(--panel);border:1px solid var(--line);box-shadow:0 8px 22px #1d2b2b0b}}.hero,.panel{{padding:22px;margin:16px 0}}.summary-grid,.record-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}.metric,.record-grid span{{border:1px solid var(--line);padding:12px;background:#faf7f0}}.metric b,.record-grid b{{display:block;font-size:25px;color:var(--accent)}}small,.muted{{color:var(--muted);display:block}}.toolbar{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}input,select{{padding:10px;border:1px solid var(--line);background:white;min-width:230px;font:inherit}}.deck-list{{display:grid;gap:12px}}.deck-card{{overflow:hidden}}.deck-card.external{{border-left:4px solid var(--accent2)}}.deck-card summary{{cursor:pointer;list-style:none;display:flex;align-items:center;justify-content:space-between;gap:16px;padding:17px 20px}}.deck-card summary::-webkit-details-marker{{display:none}}.deck-title small{{font-family:ui-monospace,monospace}}.deck-stats{{color:var(--muted);white-space:nowrap}}.external .deck-stats b{{color:var(--accent2)}}.detail-body{{border-top:1px solid var(--line);padding:4px 20px 22px}}.note{{padding:10px 12px;border-left:3px solid var(--accent2);background:#fff6ef}}.provenance code,code{{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}}.table-scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left}}th{{background:#eee9dd}}.card-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:9px}}.card-tile{{border:1px solid var(--line);padding:9px;background:#fff;min-width:0}}.card-art{{display:flex;align-items:center;gap:8px;min-height:94px}}.card-thumb{{position:relative;border:0;background:none;padding:0;width:62px;height:87px;cursor:zoom-in;flex:0 0 auto}}.card-thumb img{{width:62px;height:87px;object-fit:cover;border-radius:4px;display:block}}.thumb-fallback{{display:none;position:absolute;inset:0;background:#e9e3d8;padding:5px;font-size:11px}}.thumb-fallback small{{font-size:9px}}.card-tile b{{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.card-tile small{{font-size:12px}}.modal{{display:none;position:fixed;inset:0;background:#101c1ccc;align-items:center;justify-content:center;padding:20px;z-index:5}}.modal.open{{display:flex}}.modal img{{max-width:min(92vw,520px);max-height:88vh;border-radius:8px}}.modal button{{position:absolute;top:18px;right:22px;font-size:28px;color:white;background:none;border:0;cursor:pointer}}a{{color:var(--accent)}}@media(max-width:700px){{main{{padding:18px 12px 50px}}.deck-card summary{{display:block}}.deck-stats{{margin-top:6px;white-space:normal}}}}
</style></head><body><main><section class="hero"><div class="eyebrow">KAGGLE DAILY SNAPSHOT · EXACT DECK CATALOG</div><h1>Unique Top 100 Decks</h1><p>Frozen from <code>{html.escape(str(report))}</code>. One directory per full <code>deck_sha256</code>; Card IDs remain machine-readable in <code>deck.csv</code>, while this page uses card names and images.</p><div class="summary-grid"><div class="metric"><b>{row_count}</b><span>audited leaderboard rows</span></div><div class="metric"><b>{len(items)}</b><span>unique Kaggle full-hash decks</span></div><div class="metric"><b>{len(external_references)}</b><span>external reference decks</span></div><div class="metric"><b>60</b><span>cards per deck</span></div><div class="metric"><b>2026-07-30</b><span>report date</span></div></div></section><section class="panel"><h2>Filter catalog</h2><div class="toolbar"><input id="search" placeholder="Search archetype or hash"><select id="archetype"><option value="">All archetypes</option>{options}</select><button id="expand">Expand visible</button><button id="collapse">Collapse all</button></div><span id="count" class="muted"></span></section><h2 class="section-heading">Kaggle Top 100 exact decks</h2><p class="muted">These 32 unique hashes account for all 100 audited leaderboard entries. Rank, score, usage, and W-L-D come only from the frozen daily snapshot.</p><section class="deck-list" id="kaggle-decks">{"".join(cards_by_deck)}</section><h2 class="section-heading">External reference decks</h2><p class="muted">These exact lists are useful training or benchmark references. They are not members of the frozen Kaggle Top 100 and therefore have no Kaggle snapshot rank, score, usage count, or win rate.</p><section class="deck-list" id="external-decks">{"".join(external_cards)}</section></main><div class="modal" id="modal"><button id="close">×</button><img id="large" alt="Card preview"></div><script>const rows=[...document.querySelectorAll('.deck-card')];const search=document.querySelector('#search');const archetype=document.querySelector('#archetype');const count=document.querySelector('#count');function apply(){{const q=search.value.toLowerCase().trim();let n=0;for(const row of rows){{const ok=(!q||row.dataset.search.includes(q))&&(!archetype.value||row.dataset.archetype===archetype.value);row.hidden=!ok;if(ok)n++}}count.textContent=n+' exact deck(s) visible';}}search.addEventListener('input',apply);archetype.addEventListener('change',apply);document.querySelector('#expand').onclick=()=>rows.filter(r=>!r.hidden).forEach(r=>r.open=true);document.querySelector('#collapse').onclick=()=>rows.forEach(r=>r.open=false);document.querySelectorAll('[data-preview]').forEach(b=>b.onclick=()=>{{document.querySelector('#large').src=b.dataset.preview;document.querySelector('#modal').classList.add('open')}});document.querySelector('#close').onclick=()=>document.querySelector('#modal').classList.remove('open');document.querySelector('#modal').onclick=e=>{{if(e.target.id==='modal')e.currentTarget.classList.remove('open')}};apply();</script></body></html>'''
    (output / "index.html").write_text(html_text, encoding="utf-8")


def build(report: Path, output: Path, check: bool = False, refresh: bool = False) -> list[dict]:
    repository_root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "data/official/EN_Card_Data.csv").is_file()
    )
    catalog_path = repository_root / "data/official/EN_Card_Data.csv"
    catalog = _read_catalog(catalog_path)
    rows, audit = _parse_report(report)
    items = _aggregate(rows, catalog)
    if output.exists() and not check and not refresh:
        existing = [
            path for path in output.iterdir()
            if path.name not in {"build_catalog.py", "__pycache__"}
        ]
        if existing:
            raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    external_references = _load_external_references(output, catalog)
    if not check:
        output.mkdir(parents=True, exist_ok=True)
        for item in items:
            _write_deck(item, output, report, catalog, refresh=refresh)
        _render_index(items, external_references, len(rows), output, report, catalog)
        manifest = {
            "schema_version": "daily_deck_catalog_v1",
            "source_report": str(report),
            "source_date": "2026-07-30",
            "audited_rows": len(rows),
            "unique_full_hashes": len(items),
            "audit_contract": audit["contract"],
            "decks": [
                {key: item[key] for key in ("directory", "deck_sha256", "short_hash", "archetype", "source_archetype", "count", "best_rank", "best_score", "effective_win_rate")}
                for item in items
            ],
            "external_references": [
                {key: item[key] for key in (
                    "directory", "deck_sha256", "short_hash", "archetype",
                    "catalog_label", "source_kind",
                )}
                for item in external_references
            ],
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not refresh:
            (output / "README.md").write_text(
            "# Daily Deck Catalog\n\n"
            "This catalog is generated from the frozen 2026-07-30 daily report. "
            "Each directory is one unique full `deck_sha256`; `deck.csv` contains exactly 60 engine Card IDs. "
            "Use `index.html` for readable card names, images, leaderboard evidence, and aggregate records.\n\n"
            "Regenerate with `python3 deck/build_catalog.py --report docs/environment-daily_kaggle_top100/daily/2026-07-30.html --output deck`.\n",
                encoding="utf-8",
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.check and args.refresh:
        parser.error("--check and --refresh are mutually exclusive")
    items = build(args.report, args.output, args.check, args.refresh)
    print(json.dumps({"audited_rows": 100, "unique_full_hashes": len(items), "directories": [item["directory"] for item in items]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
