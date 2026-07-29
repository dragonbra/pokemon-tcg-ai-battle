from __future__ import annotations

import html
import json
from collections import defaultdict
from typing import Any


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def _card_image(card: dict[str, Any], size: str = "SM") -> str:
    card_set = card["set"].upper()
    number = str(card["number"]).zfill(3)
    return (
        "https://limitlesstcg.nyc3.cdn.digitaloceanspaces.com/tpci/"
        f"{card_set}/{card_set}_{number}_R_EN_{size}.png"
    )


def _heat_style(rate: float, evidence: str) -> str:
    alpha = 0.14 if evidence == "insufficient" else 0.24 if evidence == "directional" else 0.38
    if rate >= 0.56:
        color = f"24, 133, 120, {alpha}"
    elif rate <= 0.44:
        color = f"190, 70, 61, {alpha}"
    else:
        color = f"207, 151, 39, {alpha * 0.75}"
    return f"background:rgba({color})"


def _matrix_table(
    rows: list[str],
    columns: list[str],
    names: dict[str, str],
    matrix_rows: list[dict[str, Any]],
    table_id: str,
) -> str:
    cells = {(row["deck_id"], row["opponent_id"]): row for row in matrix_rows}
    header = "".join(f"<th><span>{_e(names.get(key, key))}</span></th>" for key in columns)
    body: list[str] = []
    labels = {
        "strong_advantage": "可信优势",
        "strong_disadvantage": "可信劣势",
        "directional": "方向信号",
        "insufficient": "样本不足",
        "self_matchup": "同型内战",
    }
    for row_key in rows:
        values = [f"<th>{_e(names.get(row_key, row_key))}</th>"]
        for column_key in columns:
            cell = cells.get((row_key, column_key))
            if cell is None:
                values.append('<td class="heat missing"><span>—</span><small>无公开样本</small></td>')
                continue
            if cell.get("self_matchup"):
                title = (
                    f"{names.get(row_key, row_key)} 同型内战 | "
                    f"分出胜负 {cell['decisive']} / 平局 {cell['ties']} | n={cell['n']} | "
                    "无方向统计，不计算优势等级"
                )
                values.append(
                    f'<td class="heat self-matchup" data-n="{cell["n"]}" title="{_e(title)}" '
                    f'tabindex="0" aria-label="{_e(title)}"><b data-rate>内战</b>'
                    f'<small>决胜 {cell["decisive"]} · 平局 {cell["ties"]}</small>'
                    f'<i>n={cell["n"]} · 无方向</i></td>'
                )
                continue
            raw_rate = cell["wins"] / cell["n"] if cell["n"] else 0.0
            title = (
                f"{names.get(row_key, row_key)} vs {names.get(column_key, column_key)} | "
                f"{cell['wins']}-{cell['losses']}-{cell['ties']} | n={cell['n']} | "
                f"95% Wilson {_pct(cell['wilson_low'])}–{_pct(cell['wilson_high'])} | "
                f"{labels[cell['evidence']]}"
            )
            values.append(
                f'<td class="heat {cell["evidence"]}" style="{_heat_style(cell["effective_win_rate"], cell["evidence"])}" '
                f'data-effective="{cell["effective_win_rate"]:.6f}" data-raw="{raw_rate:.6f}" '
                f'data-n="{cell["n"]}" title="{_e(title)}" tabindex="0" aria-label="{_e(title)}">'
                f'<b data-rate>{_pct(cell["effective_win_rate"])}</b>'
                f'<small>{cell["wins"]}-{cell["losses"]}-{cell["ties"]} · n={cell["n"]}</small>'
                f'<i>{_pct(cell["wilson_low"])}–{_pct(cell["wilson_high"])}</i></td>'
            )
        body.append("<tr>" + "".join(values) + "</tr>")
    return (
        f'<div class="matrix-wrap"><table class="matrix" id="{table_id}">'
        f'<thead><tr><th>我方 ↓ / 对手 →</th>{header}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def _record(row: dict[str, Any]) -> str:
    return f"{row['wins']}-{row['losses']}-{row['ties']}"


def render_report(snapshot: dict[str, Any]) -> str:
    primary_points = snapshot["primary_points_share"]
    variant_points = snapshot["variant_points_share"]
    primary_meta = snapshot["labs_primary_meta"]
    variant_meta = snapshot["labs_variant_meta"]
    kaggle = snapshot["kaggle"]
    comparison = snapshot["comparison"]
    pairing = snapshot["pairing_summary"]
    event_dates = [row["date"] for row in snapshot["events"]]

    primary_by_name = {row["name"]: row for row in primary_points}
    labs_primary_by_id = {row["deck_id"]: row for row in primary_meta}
    dragapult = labs_primary_by_id["dragapult-ex"]
    grimmsnarl = labs_primary_by_id["marnie-grimmsnarl-ex"]
    drag_points = next(row for row in primary_points if row["name"].startswith("Dragapult"))
    grim_points = next(row for row in primary_points if row["name"].startswith("Marnie's Grimmsnarl"))
    kaggle_counts = {row["name"]: row["count"] for row in kaggle["archetypes"]}
    kaggle_dragapult = sum(
        count for name, count in kaggle_counts.items() if name.startswith("Dragapult")
    )
    kaggle_grimmsnarl = sum(
        count for name, count in kaggle_counts.items() if name.startswith("Marnie's Grimmsnarl")
    )
    variant_matchups = {
        (row["deck_id"], row["opponent_id"]): row
        for row in snapshot["matchups_variant"]
    }

    def matchup(deck: str, opponent: str) -> dict[str, Any]:
        return variant_matchups[(deck, opponent)]

    base_vs_dusk = matchup("dragapult-ex", "dragapult-dusknoir")
    base_vs_blaziken = matchup("dragapult-ex", "dragapult-blaziken")
    base_vs_dudun = matchup("dragapult-ex", "dragapult-dudunsparce")
    base_vs_bolt = matchup("dragapult-ex", "raging-bolt-ogerpon")
    dudun_vs_bolt = matchup("dragapult-dudunsparce", "raging-bolt-ogerpon")
    dusk_vs_alakazam = matchup("dragapult-dusknoir", "alakazam-dudunsparce")
    dusk_vs_bolt = matchup("dragapult-dusknoir", "raging-bolt-ogerpon")
    dusk_vs_lucario = matchup("dragapult-dusknoir", "mega-lucario-ex")
    base_vs_alakazam = matchup("dragapult-ex", "alakazam-dudunsparce")
    blaziken_vs_alakazam = matchup("dragapult-blaziken", "alakazam-dudunsparce")
    base_vs_grim = matchup("dragapult-ex", "grimmsnarl-froslass")
    dusk_vs_grim = matchup("dragapult-dusknoir", "grimmsnarl-froslass")

    event_rows = "".join(
        "<tr data-event-row>"
        f'<td><a href="https://limitlesstcg.com/tournaments/{row["tournament_id"]}" target="_blank">{_e(row["name"])}</a></td>'
        f'<td>{row["players"]:,}</td><td>{_e(row.get("lab_event_id") or "—")}</td>'
        f'<td>{row.get("rounds", "—")}</td>'
        f'<td>{format(row["accepted_matches"], ",") if row["pairings_available"] else "—"}</td>'
        f'<td>{format(row["unresolved"], ",") if row["pairings_available"] else "—"}</td>'
        f'<td>{format(row["no_result"], ",") if row["pairings_available"] else "—"}</td>'
        f'<td><span class="status {"ok" if row["pairings_available"] else "warn"}">'
        f'{"完整逐轮" if row["pairings_available"] else "不可重建"}</span></td>'
        "</tr>"
        for row in snapshot["coverage"]
    )

    point_rows = "".join(
        "<tr>"
        f'<td>{row["rank"]}</td><td><a href="{_e(row["url"])}" target="_blank">{_e(row["name"])}</a></td>'
        f'<td>{row["points"]:,}</td><td><b>{_pct(row["share"], 2)}</b></td>'
        "</tr>"
        for row in primary_points
    )
    meta_rows = "".join(
        "<tr>"
        f'<td>{index}</td><td>{_e(row["name"])}</td><td>{row["players"]:,}</td>'
        f'<td>{_pct(row["players"] / pairing["players_in_pairing_events"], 2)}</td>'
        f'<td>{row["day2s"]:,}</td><td>{_pct(row["day2_rate"])}</td>'
        f'<td>{_record(row)}</td><td><b>{_pct(row["effective_win_rate"])}</b></td>'
        "</tr>"
        for index, row in enumerate(primary_meta, start=1)
    )
    comparison_rows = "".join(
        "<tr>"
        f'<td>{_e(row["name"])}</td><td>{_pct(row["limitless_share"], 2)}</td>'
        f'<td>{_pct(row["kaggle_share"], 2)}</td>'
        f'<td><b class="delta {"up" if row["delta"] > 0 else "down"}">{row["delta"] * 100:+.2f} pp</b></td>'
        "</tr>"
        for row in comparison["rows"]
    )
    mapping_rows = "".join(
        "<tr>"
        f'<td>{_e(row["source"])}</td><td>{_e(row["original"])}</td>'
        f'<td>{_e(row["canonical"])}</td><td><code>{_e(row["rule"])}</code></td>'
        f'<td>{_pct(row["mass"], 2)}</td></tr>'
        for row in comparison["mapping_audit"]
    )

    drag_variants = [row for row in variant_meta if row["primary_id"] == "dragapult-ex"]
    drag_variant_by_id = {row["deck_id"]: row for row in drag_variants}
    variant_point_by_name = {row["name"]: row for row in variant_points}
    drag_variant_rows = "".join(
        "<tr>"
        f'<td>{_e(row["name"])}</td><td>{row["players"]:,}</td>'
        f'<td>{_pct(row["players"] / pairing["players_in_pairing_events"], 2)}</td>'
        f'<td>{_pct(variant_point_by_name.get(row["name"], {}).get("share"))}</td>'
        f'<td>{row["day2s"]:,}</td><td>{_pct(row["day2_rate"])}</td>'
        f'<td>{_record(row)}</td><td><b>{_pct(row["effective_win_rate"])}</b></td>'
        "</tr>"
        for row in drag_variants
    )
    all_variant_rows = "".join(
        f'<tr data-variant-row data-search="{_e((row["name"] + " " + row["primary_name"]).lower())}">'
        f'<td>{index}</td><td>{_e(row["name"])}</td><td>{_e(row["primary_name"])}</td>'
        f'<td>{row["players"]:,}</td><td>{row["events"]}</td><td>{row["day2s"]:,}</td>'
        f'<td>{_pct(row["day2_rate"])}</td><td>{_record(row)}</td>'
        f'<td><b>{_pct(row["effective_win_rate"])}</b></td></tr>'
        for index, row in enumerate(variant_meta, start=1)
    )

    player_rows = "".join(
        "<tr>"
        f'<td>{index}</td><td><b>{_e(row["name"])}</b><small>{_e(row["country"] or "—")}</small></td>'
        f'<td>{row["events"]}</td><td>{row["points"]}</td><td>{row["wins"]}-{row["losses"]}-{row["ties"]}</td>'
        f'<td>{row["best_placement"] or "—"}</td><td>{_e(" / ".join(row["decks"]))}</td></tr>'
        for index, row in enumerate(snapshot["players"][:40], start=1)
    )

    deck_blocks: list[str] = []
    ordered_decks = sorted(
        snapshot["representative_decklists"],
        key=lambda row: (
            not row["deck_id"].startswith("dragapult"),
            row["placement"] or 10**9,
            row["deck_name"],
        ),
    )
    for deck in ordered_decks:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for card in deck["cards"]:
            groups[card["group"]].append(card)
        pokemon = "".join(
            f'<button class="pokemon-card" data-preview="{_e(_card_image(card, "MD"))}" '
            f'data-card-name="{_e(card["name"])}"><img src="{_e(_card_image(card))}" '
            f'alt="{_e(card["name"])}" loading="lazy" onerror="this.hidden=true">'
            f'<span>{card["count"]}× {_e(card["name"])}<small>{_e(card["set"])} {card["number"]}</small></span></button>'
            for card in groups["pokemon"]
        )
        text_groups = "".join(
            f'<div><b>{"Trainer" if group == "trainer" else "Energy"}</b><p>'
            + " · ".join(
                f'{card["count"]}× {_e(card["name"])} <small>{_e(card["set"])} {card["number"]}</small>'
                for card in groups[group]
            )
            + "</p></div>"
            for group in ("trainer", "energy")
        )
        history = deck["history"]
        classification = deck["classification_audit"]
        key_cards = "、".join(
            f'{row["name"]}×{row["count"]}' for row in classification["key_cards"]
        ) or "无单列关键副轴"
        open_attribute = " open" if deck["deck_id"].startswith("dragapult") else ""
        deck_blocks.append(
            f'<details class="deck-detail"{open_attribute}>'
            f'<summary><span><b>{_e(deck["deck_name"])}</b><small>{_e(deck["player_name"])} · '
            f'{_e(deck["event_name"])} #{deck["placement"]}</small></span>'
            f'<span class="deck-record">{deck["record"][0]}-{deck["record"][1]}-{deck["record"][2]} · 60 cards</span></summary>'
            '<div class="deck-audit">'
            f'<p><b>样本内履历：</b>{history["events"]} 场赛事，{history["wins"]}-{history["losses"]}-{history["ties"]}，'
            f'最佳 #{history["best_placement"]}。<a href="{_e(deck["source_url"])}" target="_blank">核对原始卡表</a></p>'
            f'<p><b>分类交叉审计：</b>{_e(classification["note"])} '
            f'<small>关键副轴：{_e(key_cards)}；范围：{_e(classification["scope"])}。</small></p>'
            f'<div class="pokemon-strip">{pokemon}</div><div class="list-lines">{text_groups}</div>'
            "</div></details>"
        )

    primary_ids = [row["deck_id"] for row in primary_meta[:16]]
    primary_names = {row["deck_id"]: row["name"] for row in primary_meta}
    primary_matrix = _matrix_table(
        primary_ids, primary_ids, primary_names, snapshot["matchups_primary"], "primary-heatmap"
    )
    variant_names = {row["deck_id"]: row["name"] for row in variant_meta}
    drag_ids = [row["deck_id"] for row in drag_variants]
    opponent_ids = [row["deck_id"] for row in variant_meta if row["deck_id"] not in drag_ids][:16]
    drag_matrix = _matrix_table(
        drag_ids, opponent_ids, variant_names, snapshot["matchups_variant"], "dragapult-heatmap"
    )

    strong_rows = [
        row
        for row in snapshot["matchups_variant"]
        if row["deck_id"] != row["opponent_id"]
        and row["evidence"] in {"strong_advantage", "strong_disadvantage"}
    ]
    strong_advantages = sorted(
        (row for row in strong_rows if row["evidence"] == "strong_advantage"),
        key=lambda row: (-row["n"], -row["effective_win_rate"]),
    )[:35]
    counter_rows = "".join(
        "<tr>"
        f'<td>{_e(row["deck_name"])}</td><td>{_e(row["opponent_name"])}</td>'
        f'<td>{row["wins"]}-{row["losses"]}-{row["ties"]}</td><td>{row["n"]}</td>'
        f'<td><b>{_pct(row["effective_win_rate"])}</b></td>'
        f'<td>{_pct(row["wilson_low"])}–{_pct(row["wilson_high"])}</td>'
        f'<td>{row["events"]} 场赛事 / {row["players"]} 名我方选手</td></tr>'
        for row in strong_advantages
    )

    css = """
:root{--ink:#17202a;--muted:#65707c;--line:#d8dee4;--paper:#f7f8fa;--panel:#fff;--red:#aa3935;--teal:#126f67;--amber:#9a6815;--blue:#255d8b}*{box-sizing:border-box}html{scroll-behavior:smooth;max-width:100%;overflow-x:clip}body{margin:0;max-width:100%;overflow-x:clip;background:var(--paper);color:var(--ink);font-family:Inter,"PingFang SC","Microsoft YaHei",system-ui,sans-serif;font-size:14px;line-height:1.55;letter-spacing:0}.page{max-width:1540px;margin:auto;padding:0 22px 80px}.topbar{position:sticky;top:0;z-index:20;max-width:100%;overflow:hidden;background:rgba(247,248,250,.96);border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}.nav{width:100%;max-width:1540px;margin:auto;padding:10px 22px;display:flex;gap:6px;overflow:auto}.nav a{white-space:nowrap;color:var(--ink);text-decoration:none;padding:6px 9px;border-radius:5px}.nav a:hover{background:#e8edf1}.hero{padding:42px 0 26px;border-bottom:3px solid var(--ink)}.eyebrow{margin:0 0 8px;color:var(--red);font-weight:800;text-transform:uppercase}.hero h1{font-size:clamp(30px,4vw,58px);line-height:1.02;margin:0;max-width:1000px}.hero .lede{font-size:18px;max-width:1040px;color:#3f4b55}.stamp{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px}.panel{min-width:0;padding:34px 0;border-bottom:1px solid var(--line);scroll-margin-top:58px}.heading{display:flex;align-items:end;justify-content:space-between;gap:24px;margin-bottom:18px}.heading>*,.split>*{min-width:0}.heading h2{font-size:26px;margin:0}.heading p{max-width:760px;color:var(--muted);margin:0}.metrics{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));border:1px solid var(--line);background:var(--panel)}.metric{min-width:0;padding:16px;border-right:1px solid var(--line);min-height:108px}.metric:last-child{border-right:0}.metric span,.metric small{display:block;color:var(--muted)}.metric b{display:block;font-size:25px;margin:7px 0}.note{padding:14px 16px;border-left:4px solid var(--amber);background:#fff7e6;margin:18px 0}.finding-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line)}.finding{background:var(--panel);padding:18px}.finding h3{font-size:16px;margin:0 0 8px}.finding p{margin:0;color:#3f4b55}.table-wrap,.matrix-wrap{max-width:100%;overflow:auto;border:1px solid var(--line);background:var(--panel)}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}thead th{position:sticky;top:0;background:#eef1f4;z-index:2;font-weight:800}tbody tr:hover{background:#f3f6f8}td small{display:block;color:var(--muted)}a{color:var(--blue)}.status{font-weight:800}.status.ok{color:var(--teal)}.status.warn,.delta.down{color:var(--red)}.delta.up{color:var(--teal)}.split{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}.bar-list{display:grid;gap:9px}.bar-row{display:grid;grid-template-columns:180px 1fr 58px;gap:10px;align-items:center}.bar{height:11px;background:#e4e8ec}.bar i{display:block;height:100%;background:var(--teal)}.bar.kaggle i{background:var(--red)}.controls{display:flex;gap:8px;align-items:center;margin:12px 0;flex-wrap:wrap}.controls input{padding:8px 10px;border:1px solid var(--line);min-width:280px}.segmented{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}.segmented button{border:0;border-right:1px solid var(--line);padding:7px 10px;background:#fff}.segmented button:last-child{border-right:0}.segmented button.active{background:var(--ink);color:#fff}.matrix{width:max-content;min-width:100%}.matrix th:first-child{position:sticky;left:0;z-index:4;min-width:150px}.matrix thead th:first-child{z-index:6}.matrix thead th{height:150px;min-width:98px;max-width:98px;vertical-align:bottom}.matrix thead th span{display:block;writing-mode:vertical-rl;transform:rotate(180deg);max-height:130px}.matrix td{min-width:98px;text-align:center;padding:8px 5px}.heat b,.heat small,.heat i{display:block}.heat b{font-size:15px}.heat small{font-size:10px}.heat i{font-size:10px;color:#4c5964;font-style:normal}.heat.missing{color:#89939c;background:#fafafa}.heat:focus{outline:3px solid var(--blue);outline-offset:-3px}.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:10px 0}.legend span:before{content:"";display:inline-block;width:10px;height:10px;margin-right:5px}.legend .good:before{background:rgba(24,133,120,.38)}.legend .mid:before{background:rgba(207,151,39,.18)}.legend .bad:before{background:rgba(190,70,61,.38)}[data-sortable] thead th{cursor:pointer}[data-sortable] thead th:after{content:" ↕";color:var(--muted);font-size:10px}[data-sortable] thead th[aria-sort="ascending"]:after{content:" ↑"}[data-sortable] thead th[aria-sort="descending"]:after{content:" ↓"}.deck-detail{background:#fff;border:1px solid var(--line);margin-bottom:8px}.deck-detail summary{cursor:pointer;padding:12px 14px;display:flex;justify-content:space-between;gap:12px;align-items:center}.deck-detail summary span,.deck-detail summary small{display:block}.deck-detail summary small{color:var(--muted)}.deck-record{font-weight:800;color:var(--teal)}.deck-audit{padding:0 14px 16px;border-top:1px solid var(--line)}.pokemon-strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(145px,1fr));gap:6px}.pokemon-card{display:flex;gap:8px;align-items:center;text-align:left;border:1px solid var(--line);background:#fff;padding:6px;min-height:72px}.pokemon-card img{width:38px;height:53px;object-fit:contain}.pokemon-card span,.pokemon-card small{display:block}.pokemon-card small{color:var(--muted)}.list-lines{display:grid;grid-template-columns:2fr 1fr;gap:16px;margin-top:12px}.list-lines p{margin:5px 0}.reason-flow{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;border:1px solid var(--line)}.reason{padding:16px;background:#fff;border-right:1px solid var(--line)}.reason:last-child{border:0}.reason b{display:block;font-size:22px;color:var(--red)}.training-table td:first-child{font-weight:800}.source-list{columns:2}.source-list li{break-inside:avoid;margin-bottom:7px}.modal{border:0;border-radius:6px;padding:12px;box-shadow:0 18px 60px rgba(0,0,0,.3)}.modal::backdrop{background:rgba(0,0,0,.72)}.modal img{max-height:75vh;max-width:80vw;display:block}.modal button{width:100%;margin-top:8px;padding:8px}@media(max-width:1000px){.metrics{grid-template-columns:repeat(3,minmax(0,1fr))}.metric:nth-child(3){border-right:0}.finding-grid{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.reason-flow{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.reason:nth-child(2){border-right:0}.split{grid-template-columns:minmax(0,1fr)}}@media(max-width:650px){.page{padding:0 12px 60px}.nav{padding:8px 12px}.hero{padding-top:26px}.metrics{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.metric:nth-child(3){border-right:1px solid var(--line)}.metric:nth-child(even){border-right:0}.finding-grid,.reason-flow{grid-template-columns:minmax(0,1fr)}.reason{border-right:0;border-bottom:1px solid var(--line)}.heading{display:block}.heading p{margin-top:8px}.list-lines{grid-template-columns:minmax(0,1fr)}.source-list{columns:1}.bar-row{grid-template-columns:120px minmax(0,1fr) 48px}.controls input{width:100%;min-width:0}}@media print{.topbar,.controls,.pokemon-card img{display:none}.page{max-width:none}.panel{break-inside:avoid}.matrix-wrap{overflow:visible}.matrix{transform:scale(.72);transform-origin:top left}}
"""
    js = """
const search=document.querySelector('#variant-search');
search?.addEventListener('input',()=>{const q=search.value.trim().toLowerCase();document.querySelectorAll('[data-variant-row]').forEach(row=>row.hidden=!row.dataset.search.includes(q));});
document.querySelectorAll('[data-heat-mode]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-heat-mode]').forEach(b=>b.classList.toggle('active',b===button));const mode=button.dataset.heatMode;document.querySelectorAll('.heat[data-n]:not(.self-matchup)').forEach(cell=>{const value=Number(cell.dataset[mode]);cell.querySelector('[data-rate]').textContent=(value*100).toFixed(1)+'%';});}));
const modal=document.querySelector('#card-modal'),modalImg=modal?.querySelector('img'),modalTitle=modal?.querySelector('b');
document.querySelectorAll('[data-preview]').forEach(button=>button.addEventListener('click',()=>{modalImg.src=button.dataset.preview;modalImg.alt=button.dataset.cardName;modalTitle.textContent=button.dataset.cardName;modal.showModal();}));
modal?.querySelector('button')?.addEventListener('click',()=>modal.close());
const sortValue=value=>{const compact=value.trim().replaceAll(',','').replaceAll('%','').replace(/^#/, '');const number=Number(compact);return Number.isFinite(number)&&compact!==''?number:value.trim().toLowerCase();};
document.querySelectorAll('table[data-sortable]').forEach(table=>{const headers=[...table.querySelectorAll('thead th')];headers.forEach((header,index)=>{header.tabIndex=0;header.title='点击排序';const sort=()=>{const direction=header.getAttribute('aria-sort')==='ascending'?'descending':'ascending';headers.forEach(item=>item.setAttribute('aria-sort','none'));header.setAttribute('aria-sort',direction);const rows=[...table.tBodies[0].rows];rows.sort((left,right)=>{const a=sortValue(left.cells[index]?.textContent||''),b=sortValue(right.cells[index]?.textContent||'');const result=typeof a==='number'&&typeof b==='number'?a-b:String(a).localeCompare(String(b),'zh-CN');return direction==='ascending'?result:-result;});rows.forEach(row=>table.tBodies[0].append(row));};header.addEventListener('click',sort);header.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();sort();}});});});
"""

    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Limitless TEF-POR × Kaggle 0730 环境深度审计</title><style>{css}.table-wrap table{{min-width:720px}}.heat.self-matchup{{background:#e8edf1;color:#394653}}</style></head><body>
<div class="topbar"><nav class="nav" aria-label="报告导航"><a href="#summary">结论</a><a href="#filter-audit">Filter</a><a href="#metagame">环境</a><a href="#kaggle-gap">差距</a><a href="#dragapult-core">多龙</a><a href="#variants">变种</a><a href="#players">选手与卡表</a><a href="#matchup-matrix">热力图</a><a href="#counter-evidence">克制</a><a href="#causes">成因</a><a href="#training">训练</a><a href="#methodology">方法</a></nav></div>
<main class="page"><header class="hero"><p class="eyebrow">LIMITLESS TEF–POR · {snapshot['filter_contract']['events']} EVENTS · KAGGLE TOP {kaggle['players']}</p><h1>Perfect Order 线下赛环境，与 Kaggle 0730 到底差多远？</h1><p class="lede">本报告冻结 Limitless 的 Temporal Forces–Perfect Order filter，以 {snapshot['filter_contract']['events']} 场、{snapshot['filter_contract']['players']:,} 名参赛者的环境边界为总体；其中 {pairing['events_with_pairings']} 场 Labs 数据可重建 {pairing['accepted_matches']:,} 场公开 BO3 match。多龙按 Limitless Labs 的五个站点标签分别统计，再与 {kaggle['date']} Kaggle Top {kaggle['players']} exact-deck 快照并列。</p><div class="stamp"><span>生成时间 {_e(snapshot['generated_at'])}</span><span>Schema v{snapshot['schema_version']}</span><span>只读公开数据</span></div></header>

<section class="panel" id="summary"><div class="heading"><div><p class="eyebrow">EXECUTIVE VERDICT</p><h2>先给结论：这不是同一个竞争生态</h2></div><p>卡池相近，不代表策略分布可直接迁移。这里同时量化“选择了什么”“打得怎样”和“榜单留下什么”。</p></div>
<div class="metrics"><div class="metric"><span>Limitless 多龙积分份额</span><b>{_pct(drag_points['share'],2)}</b><small>{drag_points['points']:,} / {snapshot['filter_contract']['points']:,} Points</small></div><div class="metric"><span>{pairing['events_with_pairings']} 场多龙实际参赛</span><b>{dragapult['players']:,}</b><small>{_pct(dragapult['players']/pairing['players_in_pairing_events'],2)} · {len(drag_variants)} 个站点标签</small></div><div class="metric"><span>Kaggle 多龙席位</span><b>{kaggle_dragapult}/{kaggle['players']}</b><small>相对积分份额 {(kaggle_dragapult/kaggle['players']-drag_points['share'])*100:+.2f} pp</small></div><div class="metric"><span>Kaggle Grimmsnarl</span><b>{kaggle_grimmsnarl}/{kaggle['players']}</b><small>Limitless Points {_pct(grim_points['share'],2)}</small></div><div class="metric"><span>分布距离</span><b>{comparison['jensen_shannon_divergence']:.3f}</b><small>Jensen–Shannon · 0 相同 / 1 完全分离</small></div><div class="metric"><span>公开 matchup</span><b>{pairing['accepted_matches']:,}</b><small>{pairing['events_with_pairings']}/{pairing['events_total']} 赛事 · {pairing['unresolved_rows'] + pairing['no_result_rows']} 条赛果排除</small></div></div>
<div class="note"><b>最重要的口径：</b>{_pct(drag_points['share'],2)} 是 Limitless 主站的 <b>Points Share</b>，不是参赛套数。{pairing['events_with_pairings']} 场 Labs 中多龙实际为 {dragapult['players']:,}/{pairing['players_in_pairing_events']:,}（{_pct(dragapult['players']/pairing['players_in_pairing_events'],2)}），有效胜率 {_pct(dragapult['effective_win_rate'])}。Kaggle 的 {_pct(kaggle_dragapult/kaggle['players'])} 则是 Top {kaggle['players']} 席位，不是所有参赛 submission 的选择率。</div>
<div class="finding-grid"><article class="finding"><h3>差距是结构性的</h3><p>Kaggle HHI {comparison['kaggle_hhi']:.3f}，Limitless 为 {comparison['limitless_hhi']:.3f}；Top 2 由 {_pct(comparison['limitless_top2'])} 升至 {_pct(comparison['kaggle_top2'])}。自动对战榜单明显更集中。</p></article><article class="finding"><h3>多龙不是一个 matchup</h3><p>基础/未单列多龙对黑夜魔灵多龙 {_pct(base_vs_dusk['effective_win_rate'])}（{_record(base_vs_dusk)}，n={base_vs_dusk['n']}），反向则为 {_pct(1-base_vs_dusk['effective_win_rate'])}。子型合并会直接掩盖方向。</p></article><article class="finding"><h3>Grimmsnarl 的 Kaggle 统治不能外推</h3><p>线下 {pairing['events_with_pairings']} 场仅 {grimmsnarl['players']} 人，有效胜率 {_pct(grimmsnarl['effective_win_rate'])}、Day 2 {_pct(grimmsnarl['day2_rate'])}；{kaggle_grimmsnarl}/{kaggle['players']} 更像 agent 实现和榜分选择压力的产物，而不是线下公开数据证明的通用最强牌。</p></article></div></section>

<section class="panel" id="filter-audit"><div class="heading"><div><p class="eyebrow">FILTER AUDIT</p><h2>筛选是否真的生效？</h2></div><p>刷新时同时验证五个 URL 参数、活动 format、赛事详情页卡池链接，并从详情页发现 Labs ID。下面每一行都可回溯到独立响应哈希。</p></div><div class="metrics"><div class="metric"><span>活动格式</span><b>TEF–POR</b><small>format: tef-por</small></div><div class="metric"><span>赛事</span><b>{snapshot['filter_contract']['events']}</b><small>{min(event_dates)} → {max(event_dates)}</small></div><div class="metric"><span>参赛者边界</span><b>{snapshot['filter_contract']['players']:,}</b><small>主站 tournament filter</small></div><div class="metric"><span>可重建赛事</span><b>{pairing['events_with_pairings']}/{pairing['events_total']}</b><small>{pairing['players_in_pairing_events']:,} 名参赛者</small></div><div class="metric"><span>Points 守恒</span><b>{snapshot['filter_contract']['points']:,}</b><small>一级 = variant 拆分</small></div><div class="metric"><span>赛果排除</span><b>{pairing['unresolved_rows'] + pairing['no_result_rows']}</b><small>{pairing['unresolved_rows']} 牌型缺失 / {pairing['no_result_rows']} no-result</small></div></div><div class="table-wrap"><table><thead><tr><th>赛事</th><th>参赛者</th><th>Labs ID</th><th>轮次</th><th>纳入 match</th><th>牌型缺失</th><th>No-result</th><th>覆盖</th></tr></thead><tbody>{event_rows}</tbody></table></div><div class="note"><b>赛果边界：</b>另有 {pairing['byes']} 个 bye；`winner=-1` 的 {pairing['no_result_rows']} 条双方有牌型记录均核验为双方各记一负，不按平局计入。<br><b>韩国联赛边界：</b>{next(row['players'] for row in snapshot['coverage'] if not row['pairings_available']):,} 名参赛者进入总体与主站 Points Share；其赛事详情页已核验 TEF–POR，但没有 Labs standings/pairings 链接，因此不进入逐对局热力图。它不是 0 场，而是“当前公开数据不可重建”。</div></section>

<section class="panel" id="metagame"><div class="heading"><div><p class="eyebrow">THREE LENSES</p><h2>环境分布：积分、参赛与赛果要分开看</h2></div><p>左表是 8 场主站 Points Share；右表是 7 场 Labs 的实际参赛套数、Day 2 与玩家视角 W-L-D。</p></div><div class="split"><div><h3>8 场 Points Share</h3><div class="table-wrap"><table><thead><tr><th>#</th><th>牌型</th><th>Points</th><th>Share</th></tr></thead><tbody>{point_rows}</tbody></table></div></div><div><h3>7 场实际参赛与表现</h3><div class="table-wrap"><table><thead><tr><th>#</th><th>一级牌型</th><th>人数</th><th>占比</th><th>D2</th><th>D2率</th><th>W-L-D</th><th>有效胜率</th></tr></thead><tbody>{meta_rows}</tbody></table></div></div></div></section>

<section class="panel" id="kaggle-gap"><div class="heading"><div><p class="eyebrow">DISTRIBUTION SHIFT</p><h2>与 Kaggle 0730 的差距</h2></div><p>版本化包含规则按主要攻击轴归并复合名称；正值表示 Kaggle 过度代表，负值表示线下过度代表。</p></div><div class="metrics"><div class="metric"><span>Limitless HHI</span><b>{comparison['limitless_hhi']:.3f}</b><small>Points Share</small></div><div class="metric"><span>Kaggle HHI</span><b>{comparison['kaggle_hhi']:.3f}</b><small>Top {kaggle['players']} 席位</small></div><div class="metric"><span>Limitless Top 5</span><b>{_pct(comparison['limitless_top5'])}</b></div><div class="metric"><span>Kaggle Top 5</span><b>{_pct(comparison['kaggle_top5'])}</b></div><div class="metric"><span>多龙差</span><b>{(kaggle_dragapult/kaggle['players']-drag_points['share'])*100:+.2f} pp</b></div><div class="metric"><span>Grimmsnarl 差</span><b>{(kaggle_grimmsnarl/kaggle['players']-grim_points['share'])*100:+.2f} pp</b></div></div><div class="table-wrap"><table><thead><tr><th>映射牌型</th><th>Limitless Points</th><th>Kaggle 席位</th><th>差值</th></tr></thead><tbody>{comparison_rows}</tbody></table></div><details id="mapping-audit" class="deck-detail"><summary><span><b>展开逐标签映射审计</b><small>{len(comparison['mapping_audit'])} 个原始标签；规则按优先级匹配，未命中者原样保留</small></span></summary><div class="deck-audit"><div class="table-wrap"><table data-sortable><thead><tr><th aria-sort="none">来源</th><th aria-sort="none">原标签</th><th aria-sort="none">归并标签</th><th aria-sort="none">规则</th><th aria-sort="none">质量</th></tr></thead><tbody>{mapping_rows}</tbody></table></div></div></details></section>

<section class="panel" id="dragapult-core"><div class="heading"><div><p class="eyebrow">THE FORMAT CENTER</p><h2>多龙：{_pct(drag_points['share'],2)} 积分份额背后的 {len(drag_variants)} 个站点标签</h2></div><p>这些是 Limitless Labs 站点标签，不是逐份卡表重分类。基础/未单列、黑夜魔灵、土龙节节、火焰鸡和雪妖女在速度、资源循环、补刀方式及弱点上不同，不能合成一个训练 opponent。</p></div><div class="note"><b>分类证据边界：</b>统计按 Labs 标签直接聚合；下方 {len(snapshot['representative_decklists'])} 份代表表只交叉核对每型最佳名次卡表。特别是“Dragapult”应读作“基础/未单列”，其代表表也含 1-1 土龙 tech，不能解释成所有 {drag_variant_by_id['dragapult-ex']['players']:,} 份都没有副轴。</div><div class="table-wrap"><table><thead><tr><th>细分子型</th><th>{pairing['events_with_pairings']} 场人数</th><th>参赛占比</th><th>Points Share</th><th>Day 2</th><th>D2率</th><th>W-L-D</th><th>有效胜率</th></tr></thead><tbody>{drag_variant_rows}</tbody></table></div><div class="finding-grid"><article class="finding"><h3>基础/未单列型内战表现突出</h3><p>{drag_variant_by_id['dragapult-ex']['players']:,} 人，是最大站点标签；对黑夜魔灵型 {_pct(base_vs_dusk['effective_win_rate'])}、对火焰鸡型 {_pct(base_vs_blaziken['effective_win_rate'])}、对土龙型 {_pct(base_vs_dudun['effective_win_rate'])}。</p></article><article class="finding"><h3>土龙型改变部分外部 matchup</h3><p>土龙多龙对 Raging Bolt/Ogerpon 为 {_pct(dudun_vs_bolt['effective_win_rate'])}（{_record(dudun_vs_bolt)}，n={dudun_vs_bolt['n']}），而基础/未单列型只有 {_pct(base_vs_bolt['effective_win_rate'])}（{_record(base_vs_bolt)}，n={base_vs_bolt['n']}）。“多龙怕/不怕 Raging Bolt”取决于子型。</p></article><article class="finding"><h3>黑夜魔灵不是普遍升级</h3><p>它对 Alakazam/Dudunsparce 为 {_pct(dusk_vs_alakazam['effective_win_rate'])}，但对基础/未单列多龙 {_pct(1-base_vs_dusk['effective_win_rate'])}、Raging Bolt/Ogerpon {_pct(dusk_vs_bolt['effective_win_rate'])}，对 Mega Lucario {_pct(dusk_vs_lucario['effective_win_rate'])}（n={dusk_vs_lucario['n']}）。加入爆伤线会改变资源和稳定性，不是无条件增强。</p></article></div></section>

<section class="panel" id="variants"><div class="heading"><div><p class="eyebrow">VARIANT CATALOG</p><h2>全部公开细分牌型</h2></div><p>默认按 {pairing['events_with_pairings']} 场实际参赛人数排序；低样本仍保留，但不进入克制结论。点击表头可重新排序。</p></div><div class="controls"><input id="variant-search" type="search" placeholder="筛选牌型或一级 archetype…"><span>{len(variant_meta)} 个细分类</span></div><div class="table-wrap"><table data-sortable><thead><tr>{''.join(f'<th aria-sort="none">{label}</th>' for label in ('#','细分牌型','一级牌型','人数','赛事','D2','D2率','W-L-D','有效胜率'))}</tr></thead><tbody>{all_variant_rows}</tbody></table></div></section>

<section class="panel" id="players"><div class="heading"><div><p class="eyebrow">PLAYERS & EXACT 60</p><h2>代表选手、样本内实力与真实构筑</h2></div><p>选手表默认按 {pairing['events_with_pairings']} 场累计 Points/胜场排序；卡表按各细分牌型的最佳公开名次选择。这里的“历史”严格指 TEF-POR 可审计样本。</p></div><div class="table-wrap"><table data-sortable><thead><tr>{''.join(f'<th aria-sort="none">{label}</th>' for label in ('#','选手','赛事','Points','W-L-D','最佳','使用牌型'))}</tr></thead><tbody>{player_rows}</tbody></table></div><h3>{len(snapshot['representative_decklists'])} 份代表 exact 60-card deck</h3>{''.join(deck_blocks)}</section>

<section class="panel" id="matchup-matrix"><div class="heading"><div><p class="eyebrow">PRIMARY MATCHUP MATRIX</p><h2>一级牌型对战热力图</h2></div><p>每格为我方视角，平局计半胜；对角线仅表示内战样本。悬停查看 95% Wilson、赛事数和选手数。</p></div><div class="controls"><div class="segmented"><button class="active" data-heat-mode="effective">有效胜率</button><button data-heat-mode="raw">纯胜率</button></div></div><div class="legend"><span class="good">优势</span><span class="mid">接近五五开/方向信号</span><span class="bad">劣势</span><span>粗色仅在证据更强时加深</span></div>{primary_matrix}</section>

<section class="panel" id="dragapult-matchups"><div class="heading"><div><p class="eyebrow">DRAGAPULT DEEP DIVE</p><h2>多龙五个站点标签 × 主流细分对手</h2></div><p>这是本页最重要的矩阵。相同“Dragapult”主标签在不同副轴下会出现方向相反的 matchup。</p></div>{drag_matrix}<div class="note"><b>读法示例：</b>基础/未单列多龙对 Raging Bolt/Ogerpon 为 {_pct(base_vs_bolt['effective_win_rate'])}，土龙多龙则为 {_pct(dudun_vs_bolt['effective_win_rate'])}；三型对 Alakazam/Dudunsparce 分别为 {_pct(base_vs_alakazam['effective_win_rate'])}、{_pct(dusk_vs_alakazam['effective_win_rate'])}、{_pct(blaziken_vs_alakazam['effective_win_rate'])}。训练时至少需要把四个高样本站点标签拆成独立 opponent，再用实际卡表审计确认 candidate 边界。</div></section>

<section class="panel" id="counter-evidence"><div class="heading"><div><p class="eyebrow">COUNTER EVIDENCE</p><h2>哪些优势足以称为“较可信信号”</h2></div><p>只列 n≥30 且 95% Wilson 完全高于 50% 的有向细分 matchup；仍然是观察性赛果，不消除选手水平和赛事阶段混杂。</p></div><div class="table-wrap"><table><thead><tr><th>我方</th><th>对手</th><th>W-L-D</th><th>n</th><th>有效胜率</th><th>95% Wilson</th><th>覆盖</th></tr></thead><tbody>{counter_rows}</tbody></table></div></section>

<section class="panel" id="causes"><div class="heading"><div><p class="eyebrow">WHY THIS SHAPE</p><h2>环境为什么呈现这种状态？</h2></div></div><div class="reason-flow"><article class="reason"><b>01</b><h3>卡组强度与工具箱</h3><p>多龙拥有分散伤害、进化抽牌和多种副轴，线下玩家可以随预期环境选择不同版本；大样本下这种可调节性提高了积分产出。</p></article><article class="reason"><b>02</b><h3>BO3 与平局压力</h3><p>公开记录含大量 tie。线下瑞士轮不仅奖励赢局，也惩罚慢速和复杂决策的时间成本；本页使用有效胜率保留平局，而 Kaggle 自动 BO1 不存在同样的时间管理。</p></article><article class="reason"><b>03</b><h3>Agent 实现成本</h3><p>多龙的伤害指示物分配、进化线、副轴选择与跨回合规划扩大 action/credit assignment 难度。Grimmsnarl/Froslass 的计划更集中，更容易被规则策略或单专家 BC 稳定复制。</p></article><article class="reason"><b>04</b><h3>榜单反馈回路</h3><p>Kaggle 高分构筑会被反复复现，形成 {_pct(kaggle_grimmsnarl/kaggle['players'])} Grimmsnarl 与 {_pct(comparison['kaggle_top2'])} Top-2 集中；线下 {snapshot['filter_contract']['players']:,} 人跨地区分散选择，Top-2 Points 为 {_pct(comparison['limitless_top2'])}。这是选择机制差异，不是单一克制链能解释。</p></article></div><div class="note"><b>不能下的结论：</b>不能因为 Kaggle Grimmsnarl 占 {kaggle_grimmsnarl} 席就说它在线下天然克制多龙。公开线下基础/未单列多龙对 Grimmsnarl/Froslass 为 {_pct(base_vs_grim['effective_win_rate'])}（{_record(base_vs_grim)}，n={base_vs_grim['n']}）；黑夜魔灵多龙为 {_pct(dusk_vs_grim['effective_win_rate'])}（{_record(dusk_vs_grim)}，n={dusk_vs_grim['n']}）。子型与实现质量都比粗标签重要。</div></section>

<section class="panel" id="training"><div class="heading"><div><p class="eyebrow">BC + RL IMPLICATIONS</p><h2>对训练路线的具体调整</h2></div><p>建议遵守 deck-specific policy、candidate package 和固定 opponent catalog 边界，不把不同专家动作标签无条件混训。</p></div><div class="table-wrap"><table class="training-table"><thead><tr><th>优先级</th><th>工作包</th><th>证据</th><th>实施边界</th></tr></thead><tbody><tr><td>P0</td><td>建立基础/未单列、黑夜魔灵、土龙、火焰鸡四套隔离 BC policy</td><td>四个高样本标签分别有 {drag_variant_by_id['dragapult-ex']['players']:,} / {drag_variant_by_id['dragapult-dusknoir']['players']:,} / {drag_variant_by_id['dragapult-dudunsparce']['players']:,} / {drag_variant_by_id['dragapult-blaziken']['players']:,} 名选手，且多个 matchup 方向相反</td><td>先按标签收集，再用 exact deck 审计各自 expert、dataset manifest、version、candidate package；不合并冲突标签</td></tr><tr><td>P0</td><td>将四型作为独立 arena curriculum 候选</td><td>多龙一级 {dragapult['players']:,} 人、{_pct(drag_points['share'],2)} Points，覆盖线下核心环境</td><td>先生成候选 package，经官方 engine 真实评测和用户确认后再准入</td></tr><tr><td>P1</td><td>补充 Raging Bolt、N's Zoroark、Crustle、Alakazam 对手</td><td>它们分别揭示多龙子型的速度、Prize trade、墙与资源循环弱点</td><td>固定 pool snapshot 与采样权重，RL run 中显式记录</td></tr><tr><td>P1</td><td>为多龙加入跨回合伤害分配与副轴 conditioning</td><td>仅对 Raging Bolt/Ogerpon，土龙与基础/未单列标签就相差 {abs(dudun_vs_bolt['effective_win_rate']-base_vs_bolt['effective_win_rate'])*100:.1f} pp</td><td>若共享模型，必须显式 deck/source conditioning 和分来源评测</td></tr><tr><td>P2</td><td>保留 Kaggle Grimmsnarl 高频 curriculum，但降低其代表“真实环境”的权重</td><td>Kaggle {_pct(kaggle_grimmsnarl/kaggle['players'])}，线下实际 {_pct(grimmsnarl['players']/pairing['players_in_pairing_events'])}、有效胜率 {_pct(grimmsnarl['effective_win_rate'])}</td><td>把“榜单适应性”与“广谱牌型强度”设为两个实验变量</td></tr><tr><td>暂缓</td><td>雪妖女多龙专门训练</td><td>仅 {drag_variant_by_id['dragapult-froslass']['players']} 名选手，公开表现与 matchup 样本不足</td><td>保留 catalog 观察，不据此重排训练预算</td></tr></tbody></table></div></section>

<section class="panel" id="methodology"><div class="heading"><div><p class="eyebrow">METHOD & LIMITS</p><h2>统计方法、来源与不可比较部分</h2></div></div><div class="split"><div><h3>计算合同</h3><ul><li>有效胜率 = (W + 0.5×D) / n；纯胜率 = W / n。</li><li>每格显示 95% Wilson 区间；n≥30 且区间排除 50% 才标为可信优势/劣势。</li><li>15≤n&lt;30 仅为方向信号；n&lt;15 不推断。bye、未完成、牌型缺失和 `winner=-1` no-result 均排除。</li><li>每场 BO3 match 是一个样本，不把一场 match 伪装成三局 game；同型内战只计一次，仅展示决胜/平局数量，不赋予任意 player1 牌型胜率。</li><li>Jensen–Shannon 在版本化名称映射后的并集类别计算，单位为 bit；0 相同，1 完全分离。</li></ul></div><div><h3>残余局限</h3><ul><li>韩国联赛没有 Labs pairings，进入总体但不进入矩阵。</li><li>{pairing['accepted_matches']:,} 场仍有选手能力、地区、轮次与 drop 选择偏差。</li><li>主站 Points Share、Labs 参赛套数和 Kaggle Top {kaggle['players']} 席位是三种不同分母。</li><li>细分统计遵循 Labs 站点标签；仅代表卡表经过关键副轴交叉审计，不能外推到该型每位参赛者。</li><li>卡牌机理解释来自构筑结构与公开赛果；严格因果需要控制选手与随机性的实验。</li></ul></div></div><h3>关键来源</h3><ul class="source-list"><li><a href="{_e(primary_points[0]['url'].split('/decks/')[0] + '/decks?' + snapshot['filter_contract']['query'])}" target="_blank">Limitless TEF-POR Metagame Filter</a></li><li><a href="https://limitlesstcg.com/tournaments?{_e(snapshot['filter_contract']['query'])}" target="_blank">Limitless TEF-POR Tournament Filter</a></li><li><a href="https://labs.limitlesstcg.com/0068/standings" target="_blank">Limitless Labs Indianapolis 示例</a></li><li><a href="../environment-daily_kaggle_top100/daily/2026-07-30.html">Kaggle 2026-07-30 Top {kaggle['players']} 日报</a></li><li>快照中保留 {len(snapshot['sources'])} 条源 URL、抓取时间与 SHA-256（含 8 个赛事详情页）。</li><li>聚合事实：<code>data/processed/environment_limitless/snapshot.json</code></li></ul></section>
</main><dialog class="modal" id="card-modal"><b></b><img alt=""><button type="button">关闭</button></dialog><script>{js}</script></body></html>"""
    return report
