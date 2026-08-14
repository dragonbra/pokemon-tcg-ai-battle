"""Render the progressive Frozen-0045-Init cross-deck Benchmark V2 page."""

from __future__ import annotations

from collections import defaultdict
import argparse
from datetime import UTC, datetime
import html
import json
import os
from pathlib import Path
import time
from typing import Any

from .render_g2_candidate_gate import (
    _deck_representative_art,
    _meta_taxonomy,
    _representative_cards,
)
from .run_benchmark_v2 import validate_report


DECK_ORDER = ("069", "007", "023", "003", "009", "066", "067")


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def _art(cards: list[dict[str, Any]]) -> str:
    return '<span class="art">' + "".join(
        f'<img src="{html.escape(str(card["image_url"]), quote=True)}" '
        f'alt="{html.escape(str(card["name"]), quote=True)}" loading="lazy" '
        f'title="{html.escape(str(card["name"]), quote=True)}">'
        for card in cards
    ) + "</span>"


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(rows)
    wins = sum(row["outcome"] == 1 for row in rows)
    losses = sum(row["outcome"] == -1 for row in rows)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": games - wins - losses,
        "win_rate": wins / games if games else None,
    }


def _load_reports(output_root: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for deck_id in DECK_ORDER:
        path = output_root / "reports" / deck_id / "report.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            validate_report(report)
            reports[deck_id] = report
    return reports


def render(output_root: Path) -> str:
    status_path = output_root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    reports = _load_reports(output_root)
    taxonomy = _meta_taxonomy()
    classes = {int(row["archetype_id"]): row for row in taxonomy["classes"]}
    art = _deck_representative_art()

    cards = []
    for deck_id in DECK_ORDER:
        report = reports.get(deck_id)
        state = "complete" if report else "running" if status.get("current_deck") == deck_id else "pending"
        if report:
            summary = report["summary"]
            focal_art = _representative_cards(report["focal_deck_cards"])
            detail = (
                f'<strong>{_pct(summary["win_rate"])}</strong>'
                f'<span>{summary["wins"]}-{summary["losses"]}-{summary["draws"]} · n={summary["games"]}</span>'
                f'<span>先手 {_pct(summary["focal_first_win_rate"])} · 后手 {_pct(summary["focal_second_win_rate"])}</span>'
            )
        else:
            focal_art = art.get(deck_id, [])
            detail = '<strong>测试中</strong><span>CUDA-2048 正在运行</span>' if state == "running" else '<strong>待测试</strong><span>固定队列等待</span>'
        cards.append(
            f'<article class="deck-card {state}"><header><span class="deck-id">{deck_id}</span>{_art(focal_art)}</header>'
            f'<div>{detail}</div><footer>{"PASS" if state == "complete" else "RUNNING" if state == "running" else "PENDING"}</footer></article>'
        )

    selected = list(range(14)) + [17, 27]
    meta_rows = []
    for class_id in selected:
        row = classes[class_id]
        cells = []
        for deck_id in DECK_ORDER:
            report = reports.get(deck_id)
            if not report:
                cells.append('<td class="muted">—</td>')
                continue
            subset = [entry for entry in report["entries"] if entry["opponent_meta_archetype_id"] == class_id]
            stat = _aggregate(subset)
            cells.append(f'<td><b>{_pct(stat["win_rate"])}</b><small>{stat["wins"]}-{stat["losses"]}-{stat["draws"]}</small></td>')
        representative = []
        for deck_id in map(str, row["deck_ids"]):
            representative.extend(art.get(deck_id, []))
            if len(representative) >= 2:
                break
        focus_class = (
            "focus-a" if class_id in {2, 3, 5}
            else "focus-b" if class_id in {0, 1, 4, 27}
            else ""
        )
        meta_rows.append(
            f'<tr class="{focus_class}"><th><span class="meta-id">{class_id:02d}</span>{_art(representative[:2])}'
            f'<span><b>{html.escape(str(row["display_name"]))}</b><small>{html.escape(str(row["definition"]))}</small></span></th>'
            + "".join(cells) + "</tr>"
        )

    completed = len(reports)
    total_games = sum(report["summary"]["games"] for report in reports.values())
    generated = datetime.now(UTC).isoformat()
    error = status.get("error")
    error_html = f'<div class="error"><b>运行失败：</b>{html.escape(str(error))}</div>' if error else ""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>0045 Frozen Init · Cross-deck Benchmark V2</title>
<style>
:root{{--bg:#09111f;--panel:#111d30;--panel2:#16243a;--line:#2a3a53;--text:#edf5ff;--muted:#8fa4be;--cyan:#54d4ff;--green:#54e59a;--amber:#ffc857;--red:#ff6f7d}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 18% 0,#19365b 0,transparent 35%),var(--bg);color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:34px 28px 70px}}nav{{color:var(--muted);margin-bottom:18px}}h1{{font-size:34px;line-height:1.12;margin:0 0 10px}}.lead{{color:var(--muted);max-width:960px;margin:0 0 24px}}.statusbar{{display:flex;gap:24px;align-items:center;padding:15px 18px;background:#0e1a2b;border:1px solid var(--line);border-radius:14px;margin-bottom:20px}}.statusbar b{{color:var(--cyan);font-size:20px}}.statusbar span{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(7,minmax(145px,1fr));gap:12px}}.deck-card{{min-height:190px;background:linear-gradient(145deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:16px;padding:14px;display:flex;flex-direction:column;justify-content:space-between}}.deck-card.running{{border-color:var(--amber);box-shadow:0 0 0 1px #ffc85733,0 0 24px #ffc85718}}.deck-card.complete{{border-color:#3e9871}}.deck-card header{{display:flex;align-items:center;justify-content:space-between}}.deck-id{{font-size:22px;font-weight:800;color:var(--cyan)}}.deck-card strong{{font-size:27px;display:block;margin-top:12px}}.deck-card span{{display:block;color:var(--muted)}}.deck-card footer{{margin-top:10px;font-size:11px;letter-spacing:.12em;color:var(--muted)}}.complete footer{{color:var(--green)}}.running footer{{color:var(--amber)}}.art{{display:inline-flex;align-items:center}}.art img{{height:50px;width:36px;object-fit:cover;border-radius:5px;margin-left:-5px;border:1px solid #ffffff44;background:#fff}}section{{margin-top:28px;background:#0e1929;border:1px solid var(--line);border-radius:16px;overflow:hidden}}section h2{{margin:0;padding:18px 20px;border-bottom:1px solid var(--line);font-size:19px}}.legend{{display:flex;gap:12px;padding:12px 20px;border-bottom:1px solid var(--line);color:var(--muted)}}.legend span{{padding:4px 9px;border-radius:7px}}.legend-a{{background:#163f43;color:#73f0d2!important}}.legend-b{{background:#3b2e59;color:#e7bdff!important}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:1120px}}th,td{{padding:11px 12px;border-bottom:1px solid #24354b;text-align:center}}thead th{{position:sticky;top:0;background:#142238;color:#b8cae0}}tbody th{{text-align:left;display:flex;align-items:center;gap:10px;min-width:310px}}tbody th .art img{{height:43px;width:31px}}tbody th span:last-child{{display:flex;flex-direction:column}}tbody tr.focus-a th,tbody tr.focus-a td{{background:#12363a;border-bottom-color:#28625f}}tbody tr.focus-a th{{box-shadow:inset 4px 0 #48e0bd}}tbody tr.focus-a .meta-id{{color:#62f0cf}}tbody tr.focus-b th,tbody tr.focus-b td{{background:#302744;border-bottom-color:#564377}}tbody tr.focus-b th{{box-shadow:inset 4px 0 #c98cff}}tbody tr.focus-b .meta-id{{color:#e0b0ff}}small{{display:block;color:var(--muted);font-weight:400}}td b{{font-size:15px}}.meta-id{{font:700 12px ui-monospace;color:var(--cyan);min-width:24px}}.muted{{color:#52657c}}.error{{margin:18px 0;padding:14px;border:1px solid var(--red);border-radius:12px;color:#ffd8dc;background:#411923}}.foot{{color:var(--muted);margin-top:18px;font-size:12px}}@media(max-width:1050px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:620px){{main{{padding:22px 14px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><nav>Combat Mat / Benchmark V2 / 0045</nav>
<h1>Frozen-0045-Init · 七卡组 CUDA-2048</h1>
<p class="lead">同一份冻结 0045 U0 权重分别绑定 069、007、023、003、009、066、067 exact deck。对手始终为 immutable Policy-0809；greedy official-engine；Core-16 Meta 每类 128 局；common random numbers。</p>
<div class="statusbar"><div><b>{completed}/7</b><span>已完成卡组</span></div><div><b>{total_games:,}</b><span>有效对局</span></div><div><b>{html.escape(str(status.get("state", "INITIALIZING")).upper())}</b><span>当前状态 · {html.escape(str(status.get("current_deck") or "—"))}</span></div></div>
{error_html}<div class="grid">{''.join(cards)}</div>
<section><h2>Core-16 Meta 胜率矩阵</h2><div class="legend"><span class="legend-a">重点组 A · 02 / 03 / 05</span><span class="legend-b">重点组 B · 00 / 01 / 04 / 27</span></div><div class="scroll"><table><thead><tr><th>对手 Meta Archetype</th>{''.join(f'<th>{deck}</th>' for deck in DECK_ORDER)}</tr></thead><tbody>{''.join(meta_rows)}</tbody></table></div></section>
<p class="foot">页面按每套 deck 完成时原子刷新。生成时间（UTC）：{generated}。完整原始报告位于同目录 reports/&lt;deck_id&gt;/report.json。</p>
</main></body></html>"""


def publish(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "index.html"
    temporary = target.with_suffix(f".html.{os.getpid()}.tmp")
    temporary.write_text(render(output_root), encoding="utf-8")
    os.replace(temporary, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    last_status_mtime = -1
    while True:
        status_path = root / "status.json"
        mtime = status_path.stat().st_mtime_ns if status_path.exists() else 0
        if mtime != last_status_mtime:
            publish(root)
            last_status_mtime = mtime
        if not args.watch:
            return 0
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        if status.get("state") in {"complete", "failed"}:
            return 0
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DECK_ORDER", "publish", "render"]
