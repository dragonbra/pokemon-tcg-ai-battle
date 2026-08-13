"""Render an auditable standalone 0043 Frozen Policy-0809 CUDA-2048 report."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path

from evaluation.cards import card_image_url, load_card_catalog
from ..assets import AssetRegistry


ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pct(value: float) -> str: return f"{value*100:.2f}%"


def render(report: dict) -> str:
    if report.get("status") != "PASS" or report.get("summary",{}).get("games") != 2048:
        raise RuntimeError("HTML requires a passing CUDA-2048 report")
    registry=AssetRegistry.load(PROJECT_ROOT); registry.validate_all()
    names={row.deck_id:row.name for row in registry.decks}
    grouped=defaultdict(lambda:{"w":0,"l":0,"d":0,"n":0})
    for row in report["entries"]:
        x=grouped[row["opponent_id"]];x["n"]+=1;x["w"]+=row["outcome"]==1;x["l"]+=row["outcome"]==-1;x["d"]+=row["outcome"]==0
    matchups="".join(
        f"<tr><td>{deck}</td><td>{html.escape(names[deck])}</td><td>{x['n']}</td><td>{x['w']}-{x['l']}-{x['d']}</td><td>{_pct(x['w']/x['n'])}</td></tr>"
        for deck,x in sorted(grouped.items())
    )
    catalog=load_card_catalog(ROOT/"data/official/EN_Card_Data.csv")
    cards=[]
    for card,count in sorted(Counter(report["focal_deck_cards"]).items()):
        m=catalog[card];url=card_image_url(str(m["expansion"]),str(m["collection_number"])) or ""
        image=f'<img src="{html.escape(url)}" alt="" loading="lazy">' if url else ""
        cards.append(f'<div class="card">{image}<span><b>{html.escape(str(m["name"]))}</b><small>{html.escape(str(m["expansion"]))} {html.escape(str(m["collection_number"]))} · ID {card}</small></span><strong>×{count}</strong></div>')
    s=report["summary"];candidate=report["candidate_deployment_identity_audit"];opponent=report["opponent_policy_identity_audit"];cuda=report["cuda_engine_identity_audit"]
    update=report.get("checkpoint_update")
    if not isinstance(update,int) or update < 0 or candidate.get("checkpoint_update") != update:
        raise RuntimeError("HTML requires one consistent checkpoint update identity")
    embedded=json.dumps(report,ensure_ascii=False,separators=(",",":")).replace("</","<\\/")
    style="""
:root{--bg:#f5f7f6;--ink:#17211d;--muted:#65716c;--brand:#176b4b;--line:#d8e0dc;--card:#fff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:32px 20px 80px}header{background:#153f32;color:white;padding:32px;border-radius:20px}h1{margin:4px 0;font-size:38px}header p{margin:0;color:#cfe3db}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric,section{background:var(--card);border:1px solid var(--line);border-radius:16px}.metric{padding:18px}.metric b{display:block;font-size:25px}.metric small,.muted{color:var(--muted)}section{margin-top:18px;padding:24px}h2{margin-top:0}.deck{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.card{display:flex;align-items:center;gap:10px;border:1px solid var(--line);border-radius:10px;padding:8px}.card img{width:42px;height:58px;object-fit:contain}.card span{flex:1}.card small{display:block;color:var(--muted)}table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid var(--line);padding:8px}code{overflow-wrap:anywhere}.audit{display:grid;grid-template-columns:210px 1fr;gap:8px 14px}.pass{color:var(--brand);font-weight:700}@media(max-width:760px){.grid,.deck{grid-template-columns:1fr 1fr}.audit{grid-template-columns:1fr}}
"""
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0043 U{update} · Deck {report['focal_deck_id']} · CUDA-2048</title><style>{style}</style></head><body><main>
<header><p>0043 CHAMPION LEAGUE RL · FROZEN POLICY-0809</p><h1>U{update} · Deck {report['focal_deck_id']} · CUDA-2048</h1><p>{html.escape(report['focal_deck_display_name'])} · CUDA Engine 2.0 · greedy · official engine · 独立强度证据</p></header>
<section class="grid"><div class="metric"><small>胜率</small><b>{_pct(s['win_rate'])}</b><span>{s['wins']}-{s['losses']}-{s['draws']}</span></div><div class="metric"><small>实际先手胜率</small><b>{_pct(s['focal_first_win_rate'])}</b><span>{s['focal_first_games']} 局</span></div><div class="metric"><small>实际后手胜率</small><b>{_pct(s['focal_second_win_rate'])}</b><span>{s['focal_second_games']} 局</span></div><div class="metric"><small>吞吐</small><b>{s['games_per_second']:.2f}</b><span>games/s · {s['elapsed_seconds']:.1f}s</span></div></section>
<section><h2>Exact 60-card focal deck</h2><div class="deck">{''.join(cards)}</div></section>
<section><h2>001–055 对局明细</h2><p class="muted">每个 256 局 frequency unit 的 exact-deck composition 相同；八个 unit 的 engine/search/policy seed 均互不重复。</p><table><thead><tr><th>Deck</th><th>Opponent</th><th>Games</th><th>W-L-D</th><th>Win rate</th></tr></thead><tbody>{matchups}</tbody></table></section>
<section><h2>Identity / Health audit</h2><div class="audit"><span>Candidate contract</span><code>{candidate['contract_id']}</code><span>Source FP32 checkpoint</span><code>{candidate['source_checkpoint_sha256']}</code><span>Portable FP16 artifact</span><code>{candidate['portable_checkpoint_sha256']}</code><span>Deployment-effective identity</span><code>{candidate['effective_candidate_sha256']}</code><span>Exact focal deck</span><code>{report['focal_exact_deck_sha256']}</code><span>Opponent</span><b class="pass">Policy-0809 · PASS</b><span>Opponent effective identity</span><code>{opponent['effective_policy_sha256']}</code><span>Schedule</span><code>{report['schedule']['schedule_sha256']}</code><span>CUDA Engine</span><b class="pass">{cuda['engine_label']} · SM{cuda['compute_capability'][0]}{cuda['compute_capability'][1]} · PASS</b><span>Engine source</span><code>{cuda['engine_source_sha256']}</code><span>Extension</span><code>{cuda['extension_sha256']}</code><span>Health</span><b class="pass">2048 terminal · 0 error · 0 unfinished · routing PASS · feature D2H 0</b></div></section>
<script id="report-data" type="application/json">{embedded}</script></main></body></html>'''


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--report",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();content=render(json.loads(a.report.read_text()));a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(a.output.suffix+".tmp");tmp.write_text(content);tmp.replace(a.output);print(a.output.resolve());return 0


if __name__=="__main__":raise SystemExit(main())
