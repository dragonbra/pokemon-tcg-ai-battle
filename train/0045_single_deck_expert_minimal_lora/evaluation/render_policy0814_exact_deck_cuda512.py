"""Render the V13/V14 Policy-0814 exact-deck CUDA-512 evaluation report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from ..assets import AssetRegistry
from .run_policy0814_exact_deck_cuda512 import BENCHMARK_ID, GAMES, validate_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render(report: dict[str, Any]) -> str:
    validate_report(report)
    if report.get("summary", {}).get("games") != GAMES:
        raise RuntimeError("HTML requires a passing Policy-0814 eval512 report")
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    names = {row.deck_id: row.name for row in registry.decks}
    schedule = report["schedule"]
    fixed = schedule["fixed_deck_quotas"]
    realized = schedule["realized_deck_counts"]
    per_deck = report["per_deck"]
    rows = []
    for deck_id, count in sorted(realized.items()):
        result = per_deck[deck_id]
        extra = count - fixed[deck_id]
        rows.append(
            "<tr>"
            f"<td><code>{deck_id}</code></td>"
            f"<td>{html.escape(names[deck_id])}</td>"
            f"<td>{fixed[deck_id]}</td><td>{extra}</td><td><b>{count}</b></td>"
            f"<td>{count / GAMES * 100:.2f}%</td>"
            f"<td>{result['wins']}-{result['losses']}-{result['draws']}</td>"
            f"<td><b>{_pct(result['win_rate'])}</b></td>"
            "</tr>"
        )
    first = [row for row in report["entries"] if row["focal_first"]]
    second = [row for row in report["entries"] if not row["focal_first"]]
    first_wins = sum(row["outcome"] == 1 for row in first)
    second_wins = sum(row["outcome"] == 1 for row in second)
    summary = report["summary"]
    candidate = report["focal_policy_identity_audit"]
    opponent = report["opponent_policy_identity_audit"]
    metrics = report["collector_metrics"]
    embedded = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    style = """
:root{--bg:#f5f7f6;--ink:#17211d;--muted:#66716c;--brand:#176b4b;--line:#d7dfdb;--paper:#fff;--accent:#b32835}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;letter-spacing:0}main{max-width:1180px;margin:auto;padding:28px 20px 72px}header{background:#163e32;color:#fff;padding:30px;border-left:6px solid #e3b341}header p{margin:0;color:#d4e5de}h1{margin:5px 0;font-size:34px}section{margin-top:24px}h2{margin:0 0 10px;font-size:21px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.metric{background:var(--paper);border:1px solid var(--line);padding:16px}.metric small{display:block;color:var(--muted)}.metric b{display:block;font-size:25px}.table-wrap{overflow:auto;background:var(--paper);border:1px solid var(--line)}table{width:100%;border-collapse:collapse;min-width:850px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:10px 12px}th{background:#edf2ef;color:#3d4944;font-size:13px}.note{color:var(--muted)}.audit{display:grid;grid-template-columns:220px minmax(0,1fr);gap:8px 14px;background:var(--paper);border:1px solid var(--line);padding:18px}.audit code{overflow-wrap:anywhere}.pass{color:var(--brand);font-weight:700}@media(max-width:760px){.metrics{grid-template-columns:1fr 1fr}.audit{grid-template-columns:1fr}h1{font-size:27px}}
"""
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0045 Deck 003 · V14-style Exact-Deck Eval512</title><style>{style}</style></head><body><main>
<header><p>0045 · {BENCHMARK_ID} · OFFICIAL ENGINE · GREEDY</p><h1>FINAL DANCE V1 · Exact-Deck Eval512</h1><p>Deck 003 · Mega Lopunny ex / Mega Froslass ex · U{report['focal_checkpoint_update']} · frozen Policy-0814 opponent</p></header>
<section class="metrics"><div class="metric"><small>总胜率</small><b>{_pct(summary['win_rate'])}</b><span>{summary['wins']}-{summary['losses']}-{summary['draws']}</span></div><div class="metric"><small>实际先手</small><b>{_pct(first_wins / len(first))}</b><span>{first_wins}-{len(first)-first_wins} · {len(first)} 局</span></div><div class="metric"><small>实际后手</small><b>{_pct(second_wins / len(second))}</b><span>{second_wins}-{len(second)-second_wins} · {len(second)} 局</span></div><div class="metric"><small>吞吐</small><b>{summary['games_per_second']:.2f}</b><span>games/s · {summary['elapsed_seconds']:.1f}s</span></div></section>
<section><h2>逐 exact deck 频次与胜率</h2><p class="note">固定配额共 390 局；其余 122 局仅在 071 / 008 / 009 / 011 中按冻结 seed 随机分配。实际总频次及顺序均来自 V14 同款 schedule。</p><div class="table-wrap"><table><thead><tr><th>Deck</th><th>对手构筑</th><th>固定配额</th><th>随机余量</th><th>实际局数</th><th>占比</th><th>W-L-D</th><th>胜率</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>Identity / health audit</h2><div class="audit"><span>Evaluation contract</span><code>{schedule['contract_id']}</code><span>Schedule SHA-256</span><code>{schedule['schedule_sha256']}</code><span>Master seed</span><code>{schedule['master_seed']}</code><span>Candidate deployment contract</span><code>{candidate['contract_id']}</code><span>Source FP32 checkpoint</span><code>{candidate['source_checkpoint_sha256']}</code><span>Portable FP16 artifact</span><code>{candidate['portable_checkpoint_sha256']}</code><span>Deployment-effective identity</span><code>{candidate['effective_candidate_sha256']}</code><span>Opponent</span><b class="pass">{opponent['requested_policy_id']} · identity PASS</b><span>Opponent effective identity</span><code>{opponent['effective_policy_sha256']}</code><span>Completion</span><b class="pass">512/512 valid · 0 error · 0 unfinished</b><span>Routing</span><b class="pass">audit failures {int(metrics['rollout/lane_routing_audit_failures'])} · feature D2H {int(metrics['rollout/cuda_feature_d2h_bytes'])}</b></div></section>
<script id="report-data" type="application/json">{embedded}</script></main></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = render(json.loads(args.report.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
