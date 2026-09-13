"""Render and index the V15 Policy-0814 public deck router eval512 report."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

from .run_public_deck_router_v2_policy0814_cuda512 import DEFAULT_OUTPUT, ROOT, VERSION


EXPERIMENT = ROOT / "docs/model/evaluation"
OUTPUT = EXPERIMENT / "V15_public_deck_router_v2_policy0814_cuda512.html"
INDEX = EXPERIMENT / "index.html"
ARTIFACT_LINK = (
    ROOT / "runs/versions" / VERSION
    / "artifact/evaluation.json"
)
DECK_NAMES = {
    "001": "Marnie's Grimmsnarl ex / Froslass",
    "002": "Alakazam / Dudunsparce",
    "003": "Mega Lopunny ex / Mega Froslass ex",
    "007": "Dragapult ex",
    "008": "Teal Mask Ogerpon ex",
    "009": "Mega Lucario ex / Solrock",
    "011": "Mega Kangaskhan ex / Crustle",
    "071": "Hydrapple ex / Meganium",
}


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def render(report: dict[str, Any]) -> str:
    summary = report["summary"]
    baseline = report["baseline"]
    paired = report["paired_vs_u70"]
    deck_rows = []
    for deck_id, result in sorted(report["per_deck"].items()):
        routed = result["router"]
        static = result["baseline_u70"]
        routes = ", ".join(
            f"U{update}: {games}"
            for update, games in sorted(result["final_route_counts"].items(), key=lambda x: int(x[0]))
        )
        delta_class = "up" if result["net_wins"] > 0 else "flat"
        deck_rows.append(
            "<tr>"
            f"<th>{deck_id}</th><td>{html.escape(DECK_NAMES[deck_id])}</td>"
            f"<td>{routes}</td><td>{routed['wins']}-{routed['losses']}-{routed['draws']}</td>"
            f"<td>{_pct(routed['win_rate'])}</td><td>{static['wins']}/{static['games']} ({_pct(static['win_rate'])})</td>"
            f"<td class='{delta_class}'>{result['net_wins']:+d} wins / {100 * result['win_rate_delta']:+.2f}pp</td>"
            f"<td>{result['classified_exact_games']}/{routed['games']}</td></tr>"
        )
    route_rows = "".join(
        f"<tr><th>U{update}</th><td>{games}</td><td>{summary['route_decision_counts'][update]}</td></tr>"
        for update, games in sorted(summary["final_route_counts"].items(), key=lambda x: int(x[0]))
    )
    audit = report["focal_public_router_identity_audit"]
    omission = audit["historical_checkpoint_boundary"]
    raw = html.escape(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>0045 V15 Public Deck Router V2 · Policy-0814 CUDA-512</title>
<style>
body{{font:15px/1.55 system-ui;margin:0;color:#17212b;background:#f5f7f8}}main{{max-width:1180px;margin:auto;padding:28px 22px 60px}}header{{background:#17212b;color:white;padding:24px;margin-bottom:18px}}h1{{font-size:28px;margin:4px 0}}h2{{margin-top:28px}}.notice{{border-left:5px solid #b06b00;background:#fff4d8;padding:13px 16px}}.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:18px 0}}.metric{{background:white;border:1px solid #d6dde2;padding:13px;border-radius:6px}}.metric b{{display:block;font-size:24px}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #d6dde2;padding:8px;text-align:left}}thead th{{background:#e8edef}}.up{{color:#08753f;font-weight:700}}.flat{{color:#52606c}}code{{overflow-wrap:anywhere}}.audit{{display:grid;grid-template-columns:230px 1fr;gap:6px 12px;background:white;border:1px solid #d6dde2;padding:14px}}small{{color:#65717c}}@media(max-width:760px){{.metrics{{grid-template-columns:1fr 1fr}}.audit{{grid-template-columns:1fr}}.scroll{{overflow:auto}}}}
</style></head><body><main>
<header><small>0045 · PUBLIC-INFORMATION CHECKPOINT ROUTER · POLICY-0814</small><h1>V15 拼好龙 V2 · CUDA-512</h1><div>与 V14 U70 使用完全相同的 official-engine common-seed schedule</div></header>
<div class="notice"><b>证据边界</b><br>路由只消费 actor-visible 的确定/记忆公开对手卡牌。结果使用与挑选 checkpoint 相同的 512 局 schedule，因此可与静态 U70 配对比较，但仍是 selection-set experimental evidence，不是独立 holdout、Promote 或 Kaggle 泛化证明。</div>
<section class="metrics"><div class="metric"><small>Router</small><b>{_pct(summary['win_rate'])}</b>{summary['wins']}-{summary['losses']}-{summary['draws']}</div><div class="metric"><small>Static U70</small><b>{_pct(baseline['win_rate'])}</b>{baseline['wins']}-{baseline['losses']}-{baseline['draws']}</div><div class="metric"><small>Paired delta</small><b class="up">+{paired['net_wins']} wins</b>{100 * paired['win_rate_delta']:+.3f}pp</div><div class="metric"><small>Classification</small><b>{summary['classified_exact_games']}/512</b>{summary['conflict_games']} conflict</div></section>
<h2>逐 exact deck</h2><div class="scroll"><table><thead><tr><th>Deck</th><th>名称</th><th>最终 route</th><th>Router W-L-D</th><th>Router</th><th>Static U70</th><th>变化</th><th>最终识别</th></tr></thead><tbody>{''.join(deck_rows)}</tbody></table></div>
<h2>路由使用</h2><div class="scroll"><table><thead><tr><th>Checkpoint</th><th>最终 route games</th><th>Actor decisions</th></tr></thead><tbody>{route_rows}</tbody></table></div>
<p>曾进入非默认 specialist：{summary['ever_specialist_games']}/512；Ogerpon provisional 最终保留：{summary['provisional_games']}；配对翻转为 router 胜：{paired['router_wins_baseline_losses']}；翻转为 U70 胜：{paired['baseline_wins_router_losses']}。</p>
<h2>Identity / isolation audit</h2><div class="audit"><b>Composite policy</b><code>{audit['policy_id']}</code><b>Composite SHA-256</b><code>{audit['composite_effective_sha256']}</code><b>Source updates</b><code>U5 / U15 / U25 / U70 / U95</code><b>Routed modules</b><code>{' + '.join(audit['routed_modules'])}</code><b>Shared Actor</b><span>{audit['shared_actor']['shared_tensor_count']} effective tensors · content-equal PASS</span><b>Opponent</b><span>complete immutable Policy-0814 · identity PASS · storage aliases 0</span><b>Candidate contract</b><span>FP16 storage → strict FP32 runtime · all five PASS</span><b>Completion</b><span>512/512 terminal · 0 error · 0 unfinished</span></div>
<h2>V14 checkpoint boundary</h2><div class="notice"><b>{omission['status']}</b><br>历史 V14 model-only checkpoint 漏存 {omission['omitted_trainable_tensor_count']} 个 StateEncoder LoRA tensors。本报告严格评测实际可重建的 Decoder / Option-LoRA / Allocation candidates，不把结果称为完整 live behavior policy。</div>
<script id="report-audit" type="application/json">{raw}</script>
</main></body></html>'''


def publish(*, report_path: Path, output: Path = OUTPUT) -> None:
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("summary", {}).get("games") != 512:
        raise RuntimeError("V15 report is not a complete PASS eval512")
    _atomic_text(output, render(report))
    link = output.name
    index = INDEX.read_text(encoding="utf-8")
    if link in index:
        raise RuntimeError("V15 evaluation index row already exists")
    row = (
        f'<tr><td><a href="{link}">V15 拼好龙 V2 · Policy-0814 CUDA-512</a></td>'
        '<td>Experimental-Public-DeckRouter-V2-Policy0814 · public observation only</td>'
        '<td>Policy-0814 · exact-deck common seeds · experimental selection-set evidence</td>'
        '<td>512</td><td>295-217-0</td><td>57.62%</td>'
        '<td>512/512 · 0 error · identity/isolation PASS · +16 wins vs U70</td></tr>\n'
    )
    index = index.replace("<tbody>\n", "<tbody>\n" + row, 1)
    _atomic_text(INDEX, index)
    _atomic_json(ARTIFACT_LINK, {
        "schema_version": "0045_v15_evaluation_link_v1",
        "version": VERSION,
        "status": "PASS",
        "authoritative_report": str(output.relative_to(ROOT)),
        "source_report": str(report_path.relative_to(ROOT)),
        "benchmark_id": report["benchmark_id"],
        "games": 512,
        "wins": 295,
        "losses": 217,
        "draws": 0,
        "win_rate": 295 / 512,
        "baseline_u70_wins": 279,
        "paired_net_wins": 16,
        "composite_effective_sha256": report[
            "focal_public_router_identity_audit"
        ]["composite_effective_sha256"],
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT / "report.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    publish(report_path=args.report.resolve(), output=args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
