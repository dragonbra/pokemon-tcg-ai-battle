"""Render the V12 public-information routed CUDA-2048 comparison report."""

from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path

from ..own_archetype import OwnArchetypeVocabulary
from .run_public_meta_router_v1_cuda2048 import validate_report


ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions"
OUTPUT = ROOT / "experiments/0045_single_deck_expert_minimal_lora/evaluation/V12_dragapult_007_public_meta_router_v3_grass_fold_u200_cuda2048.html"
REPORTS = {
    "V9 oracle": VERSIONS / "V9_dragapult_007_meta_oracle_v1_cuda2048/artifact/oracle_evaluation/report.json",
    "V10 public": VERSIONS / "V10_dragapult_007_public_meta_router_v1_cuda2048/artifact/public_router_evaluation/report.json",
    "V11 grass→U282": VERSIONS / "V11_dragapult_007_public_meta_router_v2_meganium_fold_cuda2048/artifact/public_router_evaluation/report.json",
    "V12 grass→U200": VERSIONS / "V12_dragapult_007_public_meta_router_v3_grass_fold_u200_cuda2048/artifact/public_router_evaluation/report.json",
}


def _stats(report: dict, meta_id: int | None = None) -> tuple[int, int, int]:
    rows = report["entries"]
    if meta_id is not None:
        rows = [row for row in rows if row["opponent_meta_archetype_id"] == meta_id]
    return (
        sum(row["outcome"] == 1 for row in rows),
        sum(row["outcome"] == -1 for row in rows),
        sum(row["outcome"] == 0 for row in rows),
    )


def _pct(wins: int, games: int) -> str:
    return "—" if not games else f"{wins / games * 100:.2f}%"


def render() -> str:
    reports = {label: json.loads(path.read_text()) for label, path in REPORTS.items()}
    current = reports["V12 grass→U200"]
    validate_report(current)
    vocab = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    selected = sorted({row["opponent_meta_archetype_id"] for row in current["entries"]})
    rows = []
    for meta_id in selected:
        cells = []
        values = {}
        for label, report in reports.items():
            w, l, d = _stats(report, meta_id)
            values[label] = w
            cells.append(f"<td><b>{_pct(w, w+l+d)}</b><small>{w}-{l}-{d}</small></td>")
        delta = values["V12 grass→U200"] - values["V11 grass→U282"]
        row_class = " low" if meta_id in (2, 3, 5) else " priority" if meta_id in (0, 1, 4, 27) else ""
        rows.append(
            f'<tr class="{row_class}"><td><code>{meta_id:02d}</code></td>'
            f"<td>{html.escape(vocab.classes[meta_id].display_name)}</td>"
            + "".join(cells)
            + f'<td class="{"up" if delta > 0 else "down" if delta < 0 else ""}">{delta:+d}</td></tr>'
        )
    summary_cards = []
    for label, report in reports.items():
        w, l, d = _stats(report)
        summary_cards.append(
            f"<div class=card><small>{html.escape(label)}</small><b>{_pct(w,w+l+d)}</b><span>{w}-{l}-{d}</span></div>"
        )
    rules = current["focal_public_router_identity_audit"]["rules"]
    route_counts = current["summary"]["final_route_counts"]
    embedded = json.dumps(current, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    style = """
:root{--bg:#f4f7f5;--ink:#18231e;--muted:#66736d;--line:#d7e0dc;--green:#176b4b;--blue:#e5f1ff;--amber:#fff1cc}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:28px 20px 80px}header,section,.card{background:#fff;border:1px solid var(--line);border-radius:16px}header{padding:28px;background:#153f32;color:#fff}h1{margin:4px 0}header p{margin:0;color:#cfe3db}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.card{padding:16px}.card b,.card span,.card small{display:block}.card b{font-size:25px}.card small{color:var(--muted)}section{padding:22px;margin-top:18px}h2{margin-top:0}.note{color:var(--muted)}table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid var(--line);text-align:left}td small{display:block;color:var(--muted)}tr.low{background:var(--blue)}tr.priority{background:var(--amber)}code{overflow-wrap:anywhere}.up{color:#087345;font-weight:700}.down{color:#b13a32;font-weight:700}.audit{display:grid;grid-template-columns:220px 1fr;gap:8px 14px}.pass{color:var(--green);font-weight:700}@media(max-width:760px){.cards{grid-template-columns:1fr 1fr}.audit{grid-template-columns:1fr}.table-wrap{overflow:auto}}
"""
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0045 V12 Public Meta Router · CUDA-2048</title><style>{style}</style></head><body><main>
<header><p>0045 · PUBLIC-INFORMATION EXPERT ROUTER · POLICY-0809</p><h1>V12 草系家族尽早切 U200 · CUDA-2048</h1><p>official observation only · Benchmark V2 common seeds · 2,048 terminal games</p></header>
<div class=cards>{''.join(summary_cards)}</div>
<section><h2>结论</h2><p>公开看到 Festival Dipplin 线、Teal Mask Ogerpon，或 08/27/28 任一登记 Pokémon 线时，不等待细分类完成，立即路由到 U200。V12 比 V11 多 12 胜，并在 Meta 06/08/27 上分别达到 116/128、109/128、95/128，与作弊版 V9 对应行相同。Meta 28 不在冻结 Core-16 赛程中。</p><p class=note>蓝色为 02/03/05；黄色为 00/01/04/27。V9 使用隐藏真值，仅作上界诊断；V10–V12 使用公开 observation。</p></section>
<section><h2>逐 29-way reporting Meta</h2><div class=table-wrap><table><thead><tr><th>Meta</th><th>Archetype</th><th>V9 oracle</th><th>V10 public</th><th>V11 →282</th><th>V12 →200</th><th>V12−V11 wins</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>路由与身份审计</h2><div class=audit><span>Policy ID</span><code>{html.escape(current['focal_policy_id'])}</code><span>最终 route counts</span><code>{html.escape(json.dumps(route_counts,sort_keys=True))}</code><span>U200 early triggers</span><code>{html.escape(str(rules['u200_early_route']['any']))}</code><span>Candidate identity</span><b class=pass>PASS</b><span>Opponent</span><b class=pass>complete immutable Policy-0809 · PASS</b><span>CUDA Engine</span><b class=pass>official identity PASS · source unchanged</b><span>Completion</span><b class=pass>2048/2048 terminal · 0 error · 0 unfinished</b><span>Evidence boundary</span><b>experimental public router · not automatic Promote/Kaggle evidence</b></div></section>
<script id="report-data" type="application/json">{embedded}</script></main></body></html>'''


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    content = render()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(OUTPUT)
    artifact = VERSIONS / "V12_dragapult_007_public_meta_router_v3_grass_fold_u200_cuda2048/artifact/evaluation.json"
    artifact.write_text(json.dumps({
        "schema_version": "0045_formal_evaluation_link_v1",
        "version": "V12_dragapult_007_public_meta_router_v3_grass_fold_u200_cuda2048",
        "authoritative_report": str(OUTPUT.relative_to(ROOT)),
    }, indent=2) + "\n")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
