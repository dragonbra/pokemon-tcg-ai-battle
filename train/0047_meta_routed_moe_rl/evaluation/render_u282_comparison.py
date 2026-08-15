"""Render the audited U282 Benchmark V2 comparison against today's checkpoints."""

from __future__ import annotations

from collections import defaultdict
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROJECT = "0047_meta_routed_moe_rl"
VERSIONS = ROOT / "rl_runs" / PROJECT / "versions"
OUTPUT = ROOT / "experiments" / PROJECT / "evaluation" / "V7_dragapult_007_u282_vs_0040_u270_policy0809_cuda2048.html"
POINTER = VERSIONS / "V7_dragapult_007_u276_aggressive_meta_quota_policy0809" / "artifact" / "evaluation_comparison_u270.json"
REPORTS = {
    40: VERSIONS / "V2_dragapult_007_expert_cold_start_lr" / "artifact/formal_evaluation/update-000040_benchmark_v2_cuda2048/report.json",
    90: VERSIONS / "V3_dragapult_007_expert_continue_u45" / "artifact/formal_evaluation/update-000090_benchmark_v2_cuda2048/report.json",
    95: VERSIONS / "V3_dragapult_007_expert_continue_u45" / "artifact/formal_evaluation/update-000095_benchmark_v2_cuda2048/report.json",
    100: VERSIONS / "V3_dragapult_007_expert_continue_u45" / "artifact/formal_evaluation/update-000100_benchmark_v2_cuda2048/report.json",
    190: VERSIONS / "V3_dragapult_007_expert_continue_u45" / "artifact/formal_evaluation/update-000190_benchmark_v2_cuda2048/report.json",
    200: VERSIONS / "V3_dragapult_007_expert_continue_u45" / "artifact/formal_evaluation/update-000200_benchmark_v2_cuda2048/report.json",
    270: ROOT / "rl_runs/0040_dragapult_0809_action_boundary_rl/versions/V6_u270_benchmark_v2_policy0809_engine2_cuda2048/artifact/report.json",
    282: VERSIONS / "V7_dragapult_007_u276_aggressive_meta_quota_policy0809" / "artifact/formal_evaluation/update-000282_benchmark_v2_cuda2048/report.json",
}
LABELS = {270: "0040 U270", 282: "0045 U282"}
LOW = {2, 3, 5}
PRIORITY = {0, 1, 4, 27}


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def delta(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def per_meta(report: dict) -> dict[int, dict[str, float | int]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in report["entries"]:
        grouped[int(row["opponent_meta_archetype_id"])].append(row)
    result = {}
    for meta, rows in grouped.items():
        wins = sum(row["outcome"] == 1 for row in rows)
        losses = sum(row["outcome"] == -1 for row in rows)
        draws = len(rows) - wins - losses
        result[meta] = {
            "games": len(rows), "wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / len(rows),
        }
    return result


def validate(reports: dict[int, dict]) -> None:
    target = reports[282]
    common_schedule = target["schedule"]["common_random_schedule_sha256"]
    opponent = target["opponent_policy_identity_audit"]["effective_policy_sha256"]
    for update, report in reports.items():
        if (
            report.get("status") != "PASS"
            or report.get("benchmark_id") != "Benchmark-V2"
            or report.get("focal_checkpoint_update") != update
            or report.get("focal_deck_id") != "007"
            or report.get("summary", {}).get("games") != 2048
            or len(report.get("entries", [])) != 2048
            or report["schedule"].get("common_random_schedule_sha256") != common_schedule
            or report["schedule"].get("opponent_policy_id") != "Policy-0809"
            or report["opponent_policy_identity_audit"].get("effective_policy_sha256") != opponent
        ):
            raise RuntimeError(f"U{update} is not comparable Policy-0809 Benchmark V2 evidence")
        if update == 270:
            groups = report.get("collector", {}).get("group_metrics", [])
            if (
                len(groups) != 50
                or sum(group.get("games", 0) for group in groups) != 2048
                or any(group["metrics"].get("rollout/lane_routing_audit_failures", 0) for group in groups)
                or any(group["metrics"].get("rollout/cuda_feature_d2h_bytes", 0) for group in groups)
            ):
                raise RuntimeError("0040 U270 CUDA group health gate failed")
        else:
            metrics = report["collector_metrics"]
            if (
                metrics.get("rollout/lane_routing_audit_failures", 0) != 0
                or metrics.get("rollout/unfinished_games", 0) != 0
                or metrics.get("rollout/cuda_feature_d2h_bytes", 0) != 0
            ):
                raise RuntimeError(f"U{update} CUDA health gate failed")
        if any(row["valid"] is not True or row["error"] is not None for row in report["entries"]):
            raise RuntimeError(f"U{update} contains invalid games")
        if set(row["opponent_meta_archetype_id"] for row in report["entries"]) != set(target["schedule"]["selected_class_ids"]):
            raise RuntimeError(f"U{update} meta set differs")
        if set(item["games"] for item in per_meta(report).values()) != {128}:
            raise RuntimeError(f"U{update} is not exactly 128 games per meta")


def render(reports: dict[int, dict]) -> str:
    validate(reports)
    updates = list(reports)
    metas = reports[282]["schedule"]["selected_class_ids"]
    stats = {update: per_meta(report) for update, report in reports.items()}
    overall_rows = []
    for update, report in reports.items():
        summary = report["summary"]
        difference = summary["win_rate"] - reports[200]["summary"]["win_rate"]
        overall_rows.append(
            f"<tr class={'current' if update == 282 else ''}><td>{LABELS.get(update, f'U{update}')}</td>"
            f"<td>{summary['wins']}-{summary['losses']}-{summary['draws']}</td>"
            f"<td><b>{pct(summary['win_rate'])}</b></td><td>{delta(difference)}</td>"
            f"<td>{pct(summary['focal_first_win_rate'])}</td><td>{pct(summary['focal_second_win_rate'])}</td></tr>"
        )
    meta_rows = []
    for meta in metas:
        group = " low" if meta in LOW else " priority" if meta in PRIORITY else ""
        cells = []
        for update in updates:
            item = stats[update][meta]
            cells.append(f"<td><b>{pct(item['win_rate'])}</b><small>{item['wins']}-{item['losses']}-{item['draws']}</small></td>")
        d190 = stats[282][meta]["win_rate"] - stats[190][meta]["win_rate"]
        d200 = stats[282][meta]["win_rate"] - stats[200][meta]["win_rate"]
        meta_rows.append(
            f"<tr class='{group.strip()}'><td><span class='badge'>{meta:02d}</span></td>{''.join(cells)}"
            f"<td class={'up' if d190 > 0 else 'down' if d190 < 0 else ''}>{delta(d190)}</td>"
            f"<td class={'up' if d200 > 0 else 'down' if d200 < 0 else ''}>{delta(d200)}</td></tr>"
        )
    target = reports[282]
    candidate = target["focal_policy_identity_audit"]
    opponent = target["opponent_policy_identity_audit"]
    embedded = json.dumps(
        {
            "reports": {str(update): str(REPORTS[update].relative_to(ROOT)) for update in updates},
            "summaries": {str(update): report["summary"] for update, report in reports.items()},
            "per_meta": {str(update): data for update, data in stats.items()},
        }, ensure_ascii=False, separators=(",", ":"),
    ).replace("</", "<\\/")
    headers = "".join(f"<th>{LABELS.get(update, f'U{update}')}</th>" for update in updates)
    style = """
:root{--bg:#eef3f1;--ink:#17241f;--muted:#68766f;--line:#d3dfda;--green:#0c7450;--red:#b83b4b;--low:#fff2c7;--priority:#dff2ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:28px 20px 80px}header{padding:30px;border-radius:20px;background:linear-gradient(135deg,#123c30,#176a50);color:#fff}header p{margin:0;color:#cde4db}h1{font-size:38px;margin:4px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.card,section{background:#fff;border:1px solid var(--line);border-radius:15px}.card{padding:17px}.card b{display:block;font-size:26px}.card small,small,.muted{color:var(--muted)}section{padding:22px;margin-top:16px;overflow:auto}h2{margin-top:0}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}td small{display:block}.current{background:#e6f6ef}.low{background:var(--low)}.priority{background:var(--priority)}.up{color:var(--green);font-weight:700}.down{color:var(--red);font-weight:700}.badge{font-weight:800;font-variant-numeric:tabular-nums}.legend{display:flex;gap:10px;flex-wrap:wrap}.legend span{padding:5px 9px;border-radius:8px}.legend .l{background:var(--low)}.legend .p{background:var(--priority)}.audit{display:grid;grid-template-columns:220px 1fr;gap:8px 14px}.audit code{overflow-wrap:anywhere;white-space:normal}.pass{color:var(--green);font-weight:800}@media(max-width:850px){.cards{grid-template-columns:1fr 1fr}.audit{grid-template-columns:1fr}}
"""
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0045 U282 · Policy-0809 CUDA-2048 对比</title><style>{style}</style></head><body><main>
<header><p>0045 SINGLE-DECK EXPERT · BENCHMARK V2 · COMMON RANDOM NUMBERS</p><h1>U282 · CUDA-2048 对比体检</h1><p>Deck 007 Dragapult ex · complete immutable Policy-0809 · Core-16 × 128 games · greedy · official engine</p></header>
<div class="cards"><div class="card"><small>U282 overall</small><b>{pct(target['summary']['win_rate'])}</b><span>{target['summary']['wins']}-{target['summary']['losses']}-0</span></div><div class="card"><small>vs U200</small><b class="down">{delta(target['summary']['win_rate']-reports[200]['summary']['win_rate'])}</b><span>同一 2,048-game schedule</span></div><div class="card"><small>Actual first / second</small><b>{pct(target['summary']['focal_first_win_rate'])} / {pct(target['summary']['focal_second_win_rate'])}</b><span>{target['summary']['focal_first_games']} / {target['summary']['focal_second_games']} games</span></div><div class="card"><small>Health</small><b class="pass">PASS</b><span>2048 terminal · 0 error · 0 unfinished</span></div></div>
<section><h2>总体对比</h2><table><thead><tr><th>Checkpoint</th><th>W-L-D</th><th>Win rate</th><th>vs U200</th><th>First</th><th>Second</th></tr></thead><tbody>{''.join(overall_rows)}</tbody></table></section>
<section><h2>各 Meta 对比</h2><div class="legend"><span class="l">02 / 03 / 05 · 本轮激进训练目标</span><span class="p">00 / 01 / 04 / 27 · 重点保护组</span><span>每格固定 n=128；W-L-D 显示在胜率下方</span></div><table><thead><tr><th>Meta</th>{headers}<th>U282 − U190</th><th>U282 − U200</th></tr></thead><tbody>{''.join(meta_rows)}</tbody></table></section>
<section><h2>0040 U270 可比性边界</h2><p><code>archive/submission/0040_dragapult_ex_007_0809_rl_update270</code> 已作为 focal 主视角，使用当前 CUDA Engine 2.0、完整 Policy-0809 和同一组 2,048 common-random games 重跑，因此可以进入本页逐 meta 对比。它 manifest 中旧的 67.82% 仍属于 <code>frozen_0806_seeded_agent_first_player_v3</code> 历史合同，没有混入本页数值。</p></section>
<section><h2>Identity / 可比性审计</h2><div class="audit"><span>Candidate contract</span><code>{html.escape(candidate['contract_id'])}</code><span>Source U282 FP32</span><code>{candidate['source_checkpoint_sha256']}</code><span>Portable FP16</span><code>{candidate['portable_checkpoint_sha256']}</code><span>Deployment-effective</span><code>{candidate['effective_candidate_sha256']}</code><span>Opponent</span><b class="pass">Policy-0809 · identity PASS</b><span>Opponent effective</span><code>{opponent['effective_policy_sha256']}</code><span>Common schedule</span><code>{target['schedule']['common_random_schedule_sha256']}</code><span>Meta allocation</span><code>00,01,02,03,04,05,06,07,08,09,10,11,12,13,17,27 × 128</code><span>CUDA</span><b class="pass">CUDA Engine 2.0 · routing PASS · feature D2H 0</b></div></section>
<script id="comparison-data" type="application/json">{embedded}</script></main></body></html>'''


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    reports = {update: json.loads(path.read_text(encoding="utf-8")) for update, path in REPORTS.items()}
    atomic_write(OUTPUT, render(reports))
    pointer = {
        "schema_version": "0045_evaluation_pointer_v1",
        "version": "V7_dragapult_007_u276_aggressive_meta_quota_policy0809",
        "checkpoint_update": 282,
        "authoritative_report": str(OUTPUT.relative_to(ROOT)),
        "raw_report": str(REPORTS[282].relative_to(ROOT)),
        "status": "PASS",
    }
    atomic_write(POINTER, json.dumps(pointer, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
