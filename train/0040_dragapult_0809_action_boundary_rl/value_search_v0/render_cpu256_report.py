"""Render the immutable Phase 4 A/B report and evaluation backlink."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VERSION = "V3_u270_value_search_v0_cpu256"
EXPERIMENT = ROOT / "experiments/0040_dragapult_0809_action_boundary_rl"
RUN = ROOT / "rl_runs/0040_dragapult_0809_action_boundary_rl/versions" / VERSION
RAW = ROOT / ".tmp/evaluation/0040_u270_value_search_v0"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def main() -> int:
    report_path = EXPERIMENT / "evaluation" / f"{VERSION}.html"
    index_path = EXPERIMENT / "evaluation/index.html"
    artifact = RUN / "artifact"
    collisions = [path for path in (report_path, artifact / "evaluation.json") if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite formal artifacts: {collisions}")
    result = _json(RAW / "ab_result.json")
    baseline_games = _json(RAW / "cpu256_baseline/games.json")
    search_games = _json(RAW / "cpu256_search/games.json")
    stats = result["search_statistics"]
    family_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td><td>{row['searches']}</td>"
        f"<td>{row['candidate_states']}</td><td>{row['overrides']}</td>"
        f"<td>{_pct(row['agreement_rate'])}</td>"
        f"<td>{row['margin']['mean'] if row['margin']['mean'] is not None else 'N/A'}</td>"
        "</tr>"
        for name, row in result["family_breakdown"].items()
    )
    paired_rows = "".join(
        f"<tr><td>{key}</td><td>{value}</td></tr>"
        for key, value in result["paired"].items() if value or key in {"W->W", "W->L", "L->W", "L->L"}
    )
    examples = "".join(
        "<details><summary>"
        f"{html.escape(row['game_id'])} · {row['baseline_result']}->{row['search_result']} · "
        f"{html.escape(str(row['family']))} · margin {row['margin']:.6f}</summary>"
        f"<pre>{html.escape(json.dumps(row, indent=2, sort_keys=True))}</pre></details>"
        for row in result["representative_disagreements"]
    )
    embedded = html.escape(
        json.dumps(
            {"analysis": result, "baseline_games": baseline_games, "search_games": search_games},
            sort_keys=True,
        )
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>0040 U270 Value Search V0 CPU256</title><style>
body{{font:14px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 22px;color:#182230}}
h1,h2{{color:#102a43}} .metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.metric{{border:1px solid #d5dde8;padding:12px;border-radius:6px}} .metric strong{{display:block;font-size:24px}}
table{{border-collapse:collapse;width:100%;margin:12px 0 24px}}th,td{{border:1px solid #d5dde8;padding:7px;text-align:left}}
th{{background:#edf2f7}}code,pre{{background:#f3f6f9}}pre{{padding:12px;overflow:auto;max-height:420px}}
.negative{{color:#b42318}}@media(max-width:720px){{.metrics{{grid-template-columns:1fr 1fr}}}}
</style></head><body>
<h1>0040 U270 Value Search V0 — CPU256</h1>
<p>Official CPU paired evaluation, canonical Frozen-0806 v3 256-game manifest, evaluation seed <code>341512806</code>. Manifest exact: <strong>{result['manifest_exact']}</strong>.</p>
<div class="metrics"><div class="metric"><strong>162-94-0</strong>Baseline</div><div class="metric"><strong>120-136-0</strong>Value Search</div><div class="metric"><strong class="negative">-16.406 pp</strong>Win-rate delta</div><div class="metric"><strong>858</strong>Real overrides</div></div>
<h2>Paired outcomes</h2><table><thead><tr><th>Transition</th><th>Games</th></tr></thead><tbody>{paired_rows}</tbody></table>
<h2>Coverage</h2><p>{stats['total_policy_decisions']:,} Policy decisions; {stats['eligible_decisions']:,} searches ({_pct(stats['eligible_rate_of_policy_decisions'])}); {stats['candidate_states_evaluated']:,} afterstates; {stats['overrides']:,} overrides ({_pct(stats['override_rate'])} of searches); zero fallback.</p>
<table><thead><tr><th>Family</th><th>Searches</th><th>Candidates</th><th>Overrides</th><th>Agreement</th><th>Mean margin</th></tr></thead><tbody>{family_rows}</tbody></table>
<h2>Performance boundary</h2><p>Baseline wall {result['performance']['baseline_wall_seconds']:.3f}s; Search wall {result['performance']['search_wall_seconds']:.3f}s. RUN A was GPU-contention affected, so this wall delta is not intrinsic overhead. Recorded CPU Search: {result['performance']['recorded_search_seconds']:.3f}s; branch encode/Value request time: {result['performance']['recorded_value_seconds']:.3f}s summed across concurrent workers.</p>
<h2>Representative disagreements</h2>{examples}
<h2>Evidence</h2><p>The full narrative, assets, tests, exact whitelist and failure counts are in <a href="../VALUE_SEARCH_V0_CPU256_RESULT.md">VALUE_SEARCH_V0_CPU256_RESULT.md</a>. Both official evaluation reports and every decision JSONL remain under <code>.tmp/evaluation/0040_u270_value_search_v0/</code>.</p>
<details><summary>Embedded A/B analysis and all 512 lightweight game records</summary><pre id="embedded-data">{embedded}</pre></details>
</body></html>"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(document, encoding="utf-8")
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    artifact.mkdir(parents=True, exist_ok=True)
    status = {
        "version": VERSION,
        "stage": "official_cpu_evaluation_only_value_search_v0",
        "state": "completed",
        "source_checkpoint": "rl_runs/0040_dragapult_0809_action_boundary_rl/versions/V2_snapshot_loader_fix_long_run/checkpoint/update-000270.pt",
        "games_per_arm": 256,
        "baseline": {"wins": 162, "losses": 94, "draws": 0, "win_rate": 0.6328125},
        "search": {"wins": 120, "losses": 136, "draws": 0, "win_rate": 0.46875},
        "delta": -0.1640625,
        "errors": 0,
        "report": str(report_path.relative_to(ROOT)),
        "run_ids": [result["baseline"]["run_id"], result["search"]["run_id"]],
        "wandb": {"state": "disabled", "reason": "evaluation_only_no_training"},
    }
    (artifact / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    backlink = {"version": VERSION, "report": status["report"], "report_sha256": digest, "run_ids": status["run_ids"]}
    (artifact / "evaluation.json").write_text(json.dumps(backlink, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    index = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>0040 evaluation</title><style>body{{font:14px/1.5 system-ui;max-width:1000px;margin:32px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d5dde8;padding:8px}}th{{background:#edf2f7}}</style></head><body><h1>0040 Evaluation</h1><table><thead><tr><th>Version</th><th>Candidate / runs</th><th>Games</th><th>Baseline</th><th>Search</th><th>Delta</th><th>Error</th><th>Completion</th></tr></thead><tbody><tr><td><a href="{VERSION}.html">{VERSION}</a></td><td>U270 policy-conditioned Value Search V0<br>{status['run_ids'][0]} / {status['run_ids'][1]}</td><td>256 + 256</td><td>162-94-0 · 63.28%</td><td>120-136-0 · 46.88%</td><td>-16.41 pp</td><td>0</td><td>100%</td></tr></tbody></table></body></html>"""
    index_path.write_text(index, encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
