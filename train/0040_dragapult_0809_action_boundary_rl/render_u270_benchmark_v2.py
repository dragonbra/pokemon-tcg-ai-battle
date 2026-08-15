"""Render the formal 0040 U270 Engine-2 Benchmark V2 report."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERSION = "V6_u270_benchmark_v2_policy0809_engine2_cuda2048"
VERSION_ROOT = ROOT / "rl_runs/0040_dragapult_0809_action_boundary_rl/versions" / VERSION
REPORT = VERSION_ROOT / "artifact/report.json"
OUTPUT = ROOT / "experiments/0040_dragapult_0809_action_boundary_rl/evaluation" / f"{VERSION}.html"
EXTENSION = ROOT / "engine_cuda_2_0/build/native/_ptcg_cuda.so"
EXPECTED_EXTENSION = "62723caa175bf3b7bcf2dce13ce9064b2baeabeaee298be64ffadbf2ac9cd4d7"
LOW = {2, 3, 5}
PRIORITY = {0, 1, 4, 27}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def render(report: dict) -> str:
    groups = report["collector"]["group_metrics"]
    opponent = report["opponent_policy_identity_audit"]
    candidate = report["candidate_deployment_identity_audit"]
    extension_sha = sha256(EXTENSION)
    if (
        report.get("status") != "PASS"
        or report.get("benchmark_id") != "Benchmark-V2"
        or report.get("focal_policy_id") != "0040-U270"
        or report.get("summary", {}).get("games") != 2048
        or len(report.get("entries", [])) != 2048
        or len(groups) != 50
        or sum(group["games"] for group in groups) != 2048
        or any(row["valid"] is not True or row["error"] is not None for row in report["entries"])
        or any(group["metrics"].get("rollout/lane_routing_audit_failures", 0) for group in groups)
        or candidate.get("status") != "PASS"
        or opponent.get("status") != "PASS"
        or opponent.get("requested_policy_id") != "Policy-0809"
        or extension_sha != EXPECTED_EXTENSION
    ):
        raise RuntimeError("0040 U270 formal Benchmark V2 HTML gate failed")
    grouped = defaultdict(list)
    for row in report["entries"]:
        grouped[int(row["opponent_meta_archetype_id"])].append(row)
    rows = []
    for meta in report["schedule"]["selected_class_ids"]:
        games = grouped[meta]
        wins = sum(row["outcome"] == 1 for row in games)
        losses = sum(row["outcome"] == -1 for row in games)
        if len(games) != 128:
            raise RuntimeError(f"meta {meta} does not contain exactly 128 games")
        css = "low" if meta in LOW else "priority" if meta in PRIORITY else ""
        rows.append(
            f"<tr class='{css}'><td><b>{meta:02d}</b></td><td>128</td>"
            f"<td>{wins}-{losses}-0</td><td><b>{pct(wins/128)}</b></td></tr>"
        )
    summary = report["summary"]
    embedded = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    style = """
:root{--bg:#eef3f1;--ink:#17241f;--muted:#69766f;--line:#d4dfda;--green:#0c7450;--low:#fff2c7;--priority:#dff2ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:30px 20px 80px}header{background:linear-gradient(135deg,#133d31,#176c51);color:white;padding:30px;border-radius:20px}header p{margin:0;color:#cfe4dc}h1{margin:4px 0;font-size:36px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:17px 0}.card,section{background:white;border:1px solid var(--line);border-radius:15px}.card{padding:17px}.card b{font-size:25px;display:block}.card small,.muted{color:var(--muted)}section{padding:22px;margin-top:15px}table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid var(--line);text-align:left}.low{background:var(--low)}.priority{background:var(--priority)}.audit{display:grid;grid-template-columns:210px 1fr;gap:8px 14px}.audit code{overflow-wrap:anywhere}.pass{color:var(--green);font-weight:800}@media(max-width:760px){.cards{grid-template-columns:1fr 1fr}.audit{grid-template-columns:1fr}}
"""
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0040 U270 · Policy-0809 CUDA-2048</title><style>{style}</style></head><body><main>
<header><p>0040 DRAGAPULT · BENCHMARK V2 · CURRENT CUDA ENGINE 2.0</p><h1>U270 · Policy-0809 CUDA-2048</h1><p>Deck 007 main perspective · Core-16 × 128 · common random numbers · greedy · official engine</p></header>
<div class="cards"><div class="card"><small>Overall</small><b>{pct(summary['win_rate'])}</b><span>{summary['wins']}-{summary['losses']}-0</span></div><div class="card"><small>Actual first</small><b>{pct(summary['focal_first_win_rate'])}</b><span>n={summary['focal_first_games']}</span></div><div class="card"><small>Actual second</small><b>{pct(summary['focal_second_win_rate'])}</b><span>n={summary['focal_second_games']}</span></div><div class="card"><small>Health</small><b class="pass">PASS</b><span>2048 terminal · 0 error</span></div></div>
<section><h2>各 Meta</h2><p class="muted">黄色：02/03/05；蓝色：00/01/04/27。每个 meta 固定 128 局。</p><table><thead><tr><th>Meta</th><th>Games</th><th>W-L-D</th><th>Win rate</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section><h2>Identity / Health audit</h2><div class="audit"><span>Candidate</span><code>archive/submission/0040_dragapult_ex_007_0809_rl_update270</code><span>Source FP32</span><code>{candidate['source_checkpoint_sha256']}</code><span>Portable FP16</span><code>{candidate['portable_checkpoint_sha256']}</code><span>Deployment effective</span><code>{candidate['effective_candidate_sha256']}</code><span>Opponent</span><b class="pass">Policy-0809 · PASS</b><span>Opponent effective</span><code>{opponent['effective_policy_sha256']}</code><span>Common schedule</span><code>{report['schedule']['common_random_schedule_sha256']}</code><span>CUDA runtime</span><b class="pass">engine_cuda_2_0 · extension PASS</b><span>Extension</span><code>{extension_sha}</code><span>Routing</span><b class="pass">50 exact-deck groups · 0 error · feature D2H 0</b></div></section>
<script id="report-data" type="application/json">{embedded}</script></main></body></html>'''


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    atomic_text(OUTPUT, render(report))
    pointer = {
        "version": VERSION, "status": "PASS", "checkpoint_update": 270,
        "authoritative_report": str(OUTPUT.relative_to(ROOT)),
        "raw_report": str(REPORT.relative_to(ROOT)),
    }
    atomic_text(
        VERSION_ROOT / "artifact/evaluation.json",
        json.dumps(pointer, indent=2, sort_keys=True) + "\n",
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
