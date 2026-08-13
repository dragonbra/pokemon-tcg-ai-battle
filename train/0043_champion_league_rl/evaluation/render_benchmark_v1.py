"""Render image-rich Benchmark V1 reports and their top-level index."""

from __future__ import annotations

from collections import Counter
import html
import json
import os
from pathlib import Path
from typing import Any

from .render_g2_candidate_gate import (
    ROOT, _deck_representative_art, _meta_taxonomy, _pct, _representative_cards,
)
from .run_benchmark_v1 import REPORT_ROOT, validate_report


OUTPUT_ROOT = ROOT / "docs/evaluation/combat_mat/benchmark_v1"
def _aggregate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(entries)
    wins = sum(row["outcome"] == 1 for row in entries)
    losses = sum(row["outcome"] == -1 for row in entries)
    draws = games - wins - losses
    return {
        "games": games, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / games if games else None,
    }


def aggregate(report: dict[str, Any]) -> dict[str, Any]:
    validate_report(report)
    taxonomy = _meta_taxonomy()
    art = _deck_representative_art()
    by_deck = []
    for deck_id in (f"{value:03d}" for value in range(1, 68)):
        entries = [row for row in report["entries"] if row["opponent_id"] == deck_id]
        by_deck.append({
            "deck_id": deck_id, "representative_cards": art[deck_id],
            **_aggregate(entries),
        })
    by_meta = []
    for row in taxonomy["classes"]:
        class_id = int(row["archetype_id"])
        deck_ids = list(map(str, row["deck_ids"]))
        entries = [
            item for item in report["entries"]
            if item["opponent_meta_archetype_id"] == class_id
        ]
        by_meta.append({
            "archetype_id": class_id, "name": row["name"],
            "display_name": row["display_name"], "definition": row["definition"],
            "parent_archetype": row.get("parent_archetype"), "deck_ids": deck_ids,
            "representative_decks": [
                {"deck_id": deck_id, "representative_cards": art[deck_id]}
                for deck_id in deck_ids
            ],
            **_aggregate(entries),
        })
    return {
        "schema_version": "0043_benchmark_v1_published_report_v1",
        "status": "PASS", "report": report,
        "focal_representative_cards": _representative_cards(report["focal_deck_cards"]),
        "by_meta_archetype": by_meta, "by_opponent_deck": by_deck,
        "meta_taxonomy": taxonomy,
    }


def _art(cards: list[dict[str, Any]], css: str = "art") -> str:
    return f'<span class="{css}">' + "".join(
        f'<img src="{html.escape(str(card["image_url"]), quote=True)}" '
        f'alt="{html.escape(str(card["name"]), quote=True)}" loading="lazy" '
        f'title="{html.escape(str(card["name"]), quote=True)}">'
        for card in cards
    ) + "</span>"


def _result(row: dict[str, Any]) -> str:
    if not row["games"]:
        return '<span class="no-sample">n=0 · 无已注册卡组/样本</span>'
    return (
        f'<b>{_pct(row["win_rate"])}</b><small>'
        f'{row["wins"]}-{row["losses"]}-{row["draws"]} · n={row["games"]}</small>'
    )


def render(payload: dict[str, Any]) -> str:
    report = payload["report"]
    summary = report["summary"]
    meta_rows = "".join(
        f'<tr><td><span class="meta-id">{row["archetype_id"]:02d}</span> '
        f'<b>{html.escape(row["display_name"])}</b><small>{html.escape(row["definition"])}</small></td>'
        f'<td><div class="deck-chips">' + "".join(
            f'<span class="chip">{_art(deck["representative_cards"], "thumbs")}<b>{deck["deck_id"]}</b></span>'
            for deck in row["representative_decks"]
        ) + f'</div><small>{", ".join(row["deck_ids"]) or "无注册 deck"}</small></td>'
        f'<td>{_result(row)}</td></tr>'
        for row in payload["by_meta_archetype"]
    )
    deck_rows = "".join(
        f'<tr><td><span class="deck-id">{row["deck_id"]}</span></td>'
        f'<td>{_art(row["representative_cards"])}<span>{row["deck_id"]}</span></td>'
        f'<td>{_result(row)}</td></tr>'
        for row in payload["by_opponent_deck"]
    )
    audit = "".join(
        f'<tr><th>{html.escape(label)}</th><td><code>{html.escape(str(value))}</code></td></tr>'
        for label, value in (
            ("Benchmark contract", report["schedule"]["contract_id"]),
            ("Schedule SHA-256", report["schedule"]["schedule_sha256"]),
            ("Focal policy", report["focal_policy_id"]),
            ("Focal effective", report["focal_policy_identity_audit"].get(
                "effective_candidate_sha256",
                report["focal_policy_identity_audit"].get("effective_policy_sha256"),
            )),
            ("Focal deck-bound deployment effective", report["focal_deployment_effective_sha256"]),
            ("Focal exact deck", report["focal_exact_deck_sha256"]),
            ("Opponent policy", report["schedule"]["opponent_policy_id"]),
            ("Opponent effective", report["schedule"]["opponent_effective_policy_sha256"]),
            ("Master seed", report["schedule"]["master_seed"]),
            ("CUDA Engine 2.0", report["cuda_engine_identity_audit"]["engine_source_sha256"]),
        )
    )
    embedded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    style = """
:root{--bg:#f2f6f4;--paper:#fff;--ink:#172a24;--muted:#65766f;--line:#d8e4de;--brand:#186647;--dark:#153e30}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,"PingFang SC",sans-serif}main{max-width:1380px;margin:auto;padding:28px}.hero{display:flex;justify-content:space-between;align-items:end;gap:24px;padding:30px;border-radius:12px;background:var(--dark);color:#fff}.hero h1{margin:3px 0;font-size:32px}.hero p{margin:0;color:#cde2d9}.hero .art img{width:82px;height:114px}.summary{display:grid;grid-template-columns:repeat(4,1fr);margin:18px 0;border:1px solid var(--line);background:#fff}.metric{padding:17px;border-right:1px solid var(--line)}.metric:last-child{border:0}.metric b{display:block;font-size:24px}.metric small,small{display:block;color:var(--muted)}section{margin-top:18px;padding:20px;border:1px solid var(--line);background:#fff;overflow:auto}h2{margin:0 0 8px}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}th{background:#edf3f0;color:#50645c}.art,.thumbs{display:inline-flex;vertical-align:middle;width:60px}.art img{width:36px;height:50px}.thumbs{width:33px}.thumbs img{width:24px;height:34px;margin-right:-12px}.art img,.thumbs img{object-fit:cover;border:1px solid #c8d5cf;border-radius:3px;margin-right:-8px;background:#e5ece8}.deck-chips{display:flex;flex-wrap:wrap;gap:6px}.chip{display:inline-flex;align-items:center;gap:5px;padding:3px 7px 3px 3px;border:1px solid var(--line);border-radius:6px;background:#f7faf8}.meta-id,.deck-id{display:inline-block;padding:2px 6px;border-radius:4px;background:#e2f0e9;color:var(--brand);font-weight:850}.no-sample{color:var(--muted);font-style:italic}code{overflow-wrap:anywhere}@media(max-width:800px){.summary{grid-template-columns:1fr 1fr}.hero>.art{display:none}}
"""
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Benchmark V1 · {report['focal_deck_id']}</title><style>{style}</style></head><body><main>
<div class="hero"><div><p>0043 · BENCHMARK V1 · CHAMPION-G2 MIRROR LEAGUE</p><h1>{report['focal_deck_id']} · {html.escape(report['focal_deck_display_name'])}</h1><p>Focal: Champion-G2 U407 · Opponent: Meta-balanced (deck_i, Champion-G2) · CUDA-2048</p></div>{_art(payload['focal_representative_cards'])}</div>
<div class="summary"><div class="metric"><small>Overall win rate</small><b>{_pct(summary['win_rate'])}</b><span>{summary['wins']}-{summary['losses']}-{summary['draws']} · n=2048</span></div><div class="metric"><small>Actual first</small><b>{_pct(summary['focal_first_win_rate'])}</b><span>n={summary['focal_first_games']}</span></div><div class="metric"><small>Actual second</small><b>{_pct(summary['focal_second_win_rate'])}</b><span>n={summary['focal_second_games']}</span></div><div class="metric"><small>Throughput</small><b>{summary['games_per_second']:.2f}</b><span>games/s · {summary['elapsed_seconds']:.1f}s</span></div></div>
<section><h2>合同边界</h2><p>共 2,048 局 official CUDA Engine 2.0 greedy 对战。先在 28 个非空 Meta Archetype 间分配 73/74 局，再在类内 exact deck 间均衡 seeded random。Opponent policy 始终是完整 immutable Champion-G2；class 14 Other 没有注册 deck，因此显示 n=0 而不是 0% 胜率。</p></section>
<section><h2>Against Meta Archetype</h2><table><thead><tr><th>Meta Archetype</th><th>成员 deck / 卡图</th><th>Focal 胜率与局数</th></tr></thead><tbody>{meta_rows}</tbody></table></section>
<section><h2>By opponent exact deck</h2><table><thead><tr><th>Deck</th><th>代表卡图</th><th>Focal 胜率与局数</th></tr></thead><tbody>{deck_rows}</tbody></table></section>
<section><h2>Deployment / schedule identity</h2><table>{audit}</table></section>
<script id="report-data" type="application/json">{embedded}</script></main></body></html>'''


def publish(deck_id: str, *, report_root: Path = REPORT_ROOT) -> Path:
    report_path = report_root / deck_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    payload = aggregate(report)
    slug = f"champion_g2_deck_{deck_id}"
    output_root = OUTPUT_ROOT / slug
    output_root.mkdir(parents=True, exist_ok=True)
    for name, content in (
        ("index.html", render(payload)),
        ("manifest.json", json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"),
    ):
        target = output_root / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, target)
    publish_index()
    return output_root / "index.html"


def publish_index() -> Path:
    rows = []
    manifests = sorted(OUTPUT_ROOT.glob("champion_g2_deck_*/manifest.json"))
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        report = payload["report"]
        deck_id = report["focal_deck_id"]
        rows.append(
            f'<tr><td>{deck_id}</td><td>{_art(payload["focal_representative_cards"])}</td>'
            f'<td><a href="champion_g2_deck_{deck_id}/index.html">Champion-G2 · {html.escape(report["focal_deck_display_name"])}</a></td>'
            f'<td>{_pct(report["summary"]["win_rate"])}</td><td>{report["summary"]["games"]}</td></tr>'
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0043 Benchmark V1</title><style>body{{font:15px/1.6 system-ui;max-width:1100px;margin:auto;padding:36px;background:#f3f7f5;color:#172a24}}section{{padding:24px;background:#fff;border:1px solid #d8e4de}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #d8e4de;text-align:left}}a{{color:#186647;font-weight:700}}.art{{display:inline-flex;width:60px}}.art img{{width:36px;height:50px;object-fit:cover;border-radius:3px;margin-right:-8px}}</style></head><body><h1>0043 · Benchmark V1</h1><p>Meta-balanced 001–067 · fixed (deck_i, Champion-G2) opponents · CUDA-2048 per focal identity.</p><section><table><thead><tr><th>Deck</th><th>卡图</th><th>Focal identity</th><th>Win rate</th><th>Games</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section></body></html>'''
    target = OUTPUT_ROOT / "index.html"
    temporary = target.with_suffix(".html.tmp")
    temporary.write_text(page, encoding="utf-8")
    os.replace(temporary, target)
    return target


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-id")
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    print(publish(args.deck_id, report_root=args.report_root) if args.deck_id else publish_index())
